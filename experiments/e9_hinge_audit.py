#!/usr/bin/env python
"""E9a: what actually holds the segments together?

The design intent is a hinge-free machine in which the prism cells at the
joints ARE the joints. The implementation is not that. This script (i) audits
the kinematic constraints the model imposes, and (ii) removes them --- turning
every joint into a free 6-DoF connection so that only the cable network holds
the robot together --- and reports what happens.

If the cable network can carry the joint, the hinge-free robot stands. If it
cannot, the collapse mode says what geometry change would be needed.
"""
import json
import re
import numpy as np
import mujoco

MJ = "/Users/ncorrell/Downloads/tensegrity/mujoco"
RES = "/Users/ncorrell/Downloads/tensegrity/experiments/results"
OUT = f"{RES}/e9_hinge_audit.json"
TORQUE = f"{MJ}/humanoid_27dof_tensegrity.xml"

# physically motivated cable properties (Sec. member specs): a series-elastic
# element per group, and prestress specified as a TENSION rather than a strain
K_SER = {"hip": 165e3, "knee": 165e3, "ankle": 100e3, "waist": 110e3,
         "shoulder": 61e3, "elbow": 57e3, "wrist": 20e3, "neck": 25e3,
         "cell": 40e3}
T0 = {"hip": 200.0, "knee": 200.0, "ankle": 200.0, "waist": 150.0,
      "shoulder": 60.0, "elbow": 40.0, "wrist": 20.0, "neck": 15.0,
      "cell": 100.0}


def group_of(name):
    for g in ("hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist",
              "neck"):
        if name.startswith(g):
            return g
    return "cell"


def audit(m):
    """What does each joint constrain?"""
    types = {mujoco.mjtJoint.mjJNT_FREE: "free", mujoco.mjtJoint.mjJNT_BALL:
             "ball", mujoco.mjtJoint.mjJNT_SLIDE: "slide",
             mujoco.mjtJoint.mjJNT_HINGE: "hinge"}
    per_body = {}
    for j in range(m.njnt):
        b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.jnt_bodyid[j])
        per_body.setdefault(b, []).append(types[m.jnt_type[j]])
    rows = []
    for b, js in per_body.items():
        if js == ["free"]:
            continue
        n_rot = sum(1 for x in js if x == "hinge")
        n_tr = sum(1 for x in js if x == "slide")
        rows.append(dict(body=b, joints=js, rot_dof=n_rot, trans_dof=n_tr,
                         constrained_dof=6 - n_rot - n_tr))
    return rows


def physical_cables(m, scale=1.0):
    """Replace the modelled soft springs with a series-elastic element per
    group and a prestress specified in newtons."""
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    for t in range(m.ntendon):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TENDON, t)
        g = group_of(n if not n.startswith(("pw", "rt")) else "cell")
        k = K_SER[g]
        m.tendon_stiffness[t] = k
        m.tendon_damping[t] = 0.02 * np.sqrt(k)
        L = float(d.ten_length[t])
        m.tendon_lengthspring[t, 1] = max(1e-4, L - scale * T0[g] / k)
    return m


def free_xml(path):
    """Delete every kinematic constraint at the joints: each articulated body
    gets a free connection to its parent, so only cables hold it."""
    xml = open(path).read()
    xml = re.sub(r"<keyframe>.*?</keyframe>", "", xml, flags=re.S)
    xml = re.sub(r"<actuator>.*?</actuator>", "", xml, flags=re.S)
    # Replace each run of named hinge joints with a full 6-DoF connection.
    # MuJoCo forbids <freejoint> below the top level, so the equivalent is
    # three unlimited slides plus three unlimited hinges.
    FREE6 = ('\n        <joint type="slide" axis="1 0 0" limited="false" '
             'damping="0.5" armature="0.001"/>'
             '\n        <joint type="slide" axis="0 1 0" limited="false" '
             'damping="0.5" armature="0.001"/>'
             '\n        <joint type="slide" axis="0 0 1" limited="false" '
             'damping="0.5" armature="0.001"/>'
             '\n        <joint type="hinge" axis="1 0 0" limited="false" '
             'damping="0.05" armature="0.001"/>'
             '\n        <joint type="hinge" axis="0 1 0" limited="false" '
             'damping="0.05" armature="0.001"/>'
             '\n        <joint type="hinge" axis="0 0 1" limited="false" '
             'damping="0.05" armature="0.001"/>')
    xml = re.sub(r'(?:[ \t]*<joint name="[^"]+"[^/]*/>\s*)+',
                 lambda mo: FREE6 + '\n', xml)
    return xml


def wrench_closure(model_path):
    """Can the cables alone hold each joint?

    For each articulated segment, collect the cables that cross its joint and
    compute the wrench each applies to the child about the joint origin,
    w_c = [u_c ; r_c x u_c], with u_c the unit vector along the cable at the
    child attachment. Because cables only pull, the reachable wrench set is
    the POSITIVE span of {w_c}. The joint is held by cables alone iff that
    positive span is all of R^6 -- the wrench-closure condition of the
    cable-driven parallel robot literature. Testing it is an LP:

        max t   s.t.   W lambda = 0,   lambda >= t 1,   t <= 1

    with t* > 0 iff closure holds. We report the 6-DoF (full) and 3-DoF
    (translation-only) tests separately, because a joint can resist forces
    while being free to rotate -- which is what a hinge does mechanically.
    """
    from scipy.optimize import linprog
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    def subtree(b):
        out, stack = set(), [b]
        while stack:
            x = stack.pop()
            out.add(x)
            stack += [c for c in range(m.nbody) if m.body_parentid[c] == x]
        return out

    site_body = m.site_bodyid
    rows = []
    for b in range(m.nbody):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
        js = [j for j in range(m.njnt) if m.jnt_bodyid[j] == b]
        if not js or m.jnt_type[js[0]] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        below = subtree(b)
        origin = d.xanchor[js[0]]
        W = []
        for t in range(m.ntendon):
            adr, num = m.tendon_adr[t], m.tendon_num[t]
            sids = [int(m.wrap_objid[adr + i]) for i in range(num)
                    if m.wrap_type[adr + i] == mujoco.mjtWrap.mjWRAP_SITE]
            if len(sids) < 2:
                continue
            side = [site_body[s] in below for s in sids]
            if all(side) or not any(side):
                continue                      # does not cross this joint
            for k, s in enumerate(sids):      # attachment(s) on the child
                if not side[k]:
                    continue
                nb = sids[k + 1] if k + 1 < len(sids) else sids[k - 1]
                v = d.site_xpos[nb] - d.site_xpos[s]
                n = np.linalg.norm(v)
                if n < 1e-9:
                    continue
                u = v / n                     # pull direction on the child
                r = d.site_xpos[s] - origin
                W.append(np.concatenate([u, np.cross(r, u)]))
        W = np.array(W).T if W else np.zeros((6, 0))

        def closure(A):
            if A.shape[1] == 0 or np.linalg.matrix_rank(A) < A.shape[0]:
                return 0.0
            nc = A.shape[1]
            c = np.zeros(nc + 1)
            c[-1] = -1.0                                   # maximise t
            Aeq = np.hstack([A, np.zeros((A.shape[0], 1))])
            Aub = np.hstack([-np.eye(nc), np.ones((nc, 1))])   # t - lam <= 0
            r = linprog(c, A_ub=Aub, b_ub=np.zeros(nc), A_eq=Aeq,
                        b_eq=np.zeros(A.shape[0]),
                        bounds=[(0, None)] * nc + [(0, 1)])
            return float(r.x[-1]) if r.success else 0.0

        rows.append(dict(body=name, n_cables=int(W.shape[1]),
                         rank=int(np.linalg.matrix_rank(W)) if W.size else 0,
                         t6=closure(W), t3=closure(W[:3]) if W.size else 0.0))
    return rows


def settle(m, seconds=2.0, drop=0.0):
    m.opt.timestep = 1e-4          # stiff cables need a fine step
    d = mujoco.MjData(m)
    d.qpos[2] += drop
    mujoco.mj_forward(m, d)
    z0 = float(d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                        "pelvis")][2])
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
        if not np.all(np.isfinite(d.qpos)):
            return dict(ok=False, reason="diverged", z0=z0, z=float("nan"))
    z = float(d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                       "pelvis")][2])
    # how far did adjacent segments separate / interpenetrate?
    pairs = [("pelvis", "thigh_l"), ("thigh_l", "shank_l"),
             ("shank_l", "foot_l"), ("pelvis", "torso"),
             ("torso", "upper_arm_l")]
    drift = {}
    for a, b in pairs:
        ia = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, a)
        ib = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
        drift[f"{a}->{b}"] = float(np.linalg.norm(d.xpos[ib] - d.xpos[ia]))
    return dict(ok=True, z0=z0, z=z, drift=drift,
                collapsed=bool(z < 0.6 * z0))


def main():
    m = mujoco.MjModel.from_xml_path(TORQUE)
    rows = audit(m)
    n_hinge = sum(r["rot_dof"] for r in rows)
    print(f"KINEMATIC AUDIT of {TORQUE.split('/')[-1]}")
    print(f"{'body':12s} {'joints':28s} {'rot':>4s} {'trans':>6s} "
          f"{'constrained DoF':>16s}")
    for r in sorted(rows, key=lambda x: x["body"]):
        print(f"{r['body']:12s} {'+'.join(r['joints']):28s} {r['rot_dof']:4d} "
              f"{r['trans_dof']:6d} {r['constrained_dof']:16d}")
    print(f"\n{len(rows)} articulated segments, {n_hinge} hinge DoF, "
          f"{sum(r['constrained_dof'] for r in rows)} constrained DoF total.")
    print("Every joint fixes ALL THREE relative translations: the segments "
          "cannot\nseparate, shear or interpenetrate. That is a physical "
          "bearing, not a cable.")

    # --- can the cables alone hold each joint? (integration-free)
    wc = wrench_closure(TORQUE)
    print(f"\nWRENCH-CLOSURE TEST: can the cable network alone hold a joint?")
    print(f"{'segment':12s} {'cables':>7s} {'rank':>5s} {'6-DoF':>7s} "
          f"{'3-DoF':>7s}  verdict")
    for r in sorted(wc, key=lambda x: x["body"]):
        v = ("held by cables" if r["t6"] > 1e-6 else
             "forces only" if r["t3"] > 1e-6 else "NOT held")
        print(f"{r['body']:12s} {r['n_cables']:7d} {r['rank']:5d} "
              f"{r['t6']:7.4f} {r['t3']:7.4f}  {v}")
    n_ok = sum(1 for r in wc if r["t6"] > 1e-6)
    print(f"{n_ok}/{len(wc)} joints have full 6-DoF wrench closure; "
          f"{sum(1 for r in wc if r['t3'] > 1e-6)}/{len(wc)} can resist "
          f"forces alone.")

    # --- reference: hinged model with physical cables, does it stand?
    m1 = physical_cables(mujoco.MjModel.from_xml_path(TORQUE))
    r1 = settle(m1)
    print(f"\nhinged + physical cables: pelvis {r1['z0']:.3f} -> "
          f"{r1['z']:.3f} m  {'OK' if not r1.get('collapsed') else 'COLLAPSED'}")

    # --- the hinge-free variant
    import os
    os.chdir(MJ)
    xml = free_xml(TORQUE)
    open(f"{MJ}/humanoid_27dof_tensegrity_free.xml", "w").write(xml)
    m2 = mujoco.MjModel.from_xml_string(xml)
    print(f"\nhinge-free variant: nq={m2.nq} nv={m2.nv} "
          f"(vs {m.nq}/{m.nv}), joints = "
          f"{sum(1 for t in m2.jnt_type if t == mujoco.mjtJoint.mjJNT_FREE)} free")
    m2 = physical_cables(m2)
    r2 = settle(m2)
    print(f"hinge-free + physical cables: pelvis {r2['z0']:.3f} -> "
          f"{r2['z']:.3f} m  "
          f"{'COLLAPSED' if r2.get('collapsed') else 'OK'}")
    if r2["ok"]:
        print("  segment separations after settling (home distance in "
              "brackets):")
        m0 = mujoco.MjModel.from_xml_path(TORQUE)
        d0 = mujoco.MjData(m0)
        mujoco.mj_forward(m0, d0)
        for k, v in r2["drift"].items():
            a, b = k.split("->")
            ia = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_BODY, a)
            ib = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_BODY, b)
            home = float(np.linalg.norm(d0.xpos[ib] - d0.xpos[ia]))
            print(f"    {k:22s} {v:.3f} m  [{home:.3f} m]")

    # --- is there any compression path across a joint at all?
    m0 = mujoco.MjModel.from_xml_path(TORQUE)
    d0 = mujoco.MjData(m0)
    mujoco.mj_forward(m0, d0)
    contype = m0.geom_contype
    struts = [g for g in range(m0.ngeom)
              if m0.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE
              and np.allclose(m0.geom_rgba[g][:3], (0.22, 0.24, 0.28), atol=1e-3)]
    colliding = [g for g in struts if contype[g] != 0]
    print(f"\ncompression path check: {len(struts)} struts, "
          f"{len(colliding)} of them can collide.")
    print("With no strut-strut contact and no strut overlap across a joint, a "
          "cable\nnetwork can pull the segments together but nothing pushes "
          "them apart.")

    json.dump(dict(audit=rows, wrench_closure=wc, n_hinge_dof=int(n_hinge),
                   hinged=r1, hinge_free=r2,
                   n_struts=len(struts), n_colliding=len(colliding)),
              open(OUT, "w"), indent=1)
    print("\nE9a ->", OUT)


if __name__ == "__main__":
    main()
