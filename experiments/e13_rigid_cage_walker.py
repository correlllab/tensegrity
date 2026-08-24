#!/usr/bin/env python
"""E13: a hinge-free humanoid that is small enough to control.

E12 builds the whole robot from floating struts: 210 degrees of freedom, which
no predictive controller in this paper's régime can plan over. But the floating
struts inside a SEGMENT are not what the tensegrity argument needs. Within a
prestressed cage the struts barely move relative to one another; the compliance
that matters -- and the force closure that replaces the bearing -- lives at the
joints.

So this builds the intermediate: each limb segment is one rigid prestressed
cage (a class-2 tensegrity element, struts meeting at nodes), and adjacent
segments are joined by nothing but overlapping cables (class-1 joints). That is
6 DoF per joint instead of 6 per strut.

Reported: degrees of freedom, force closure at every joint, whether bounded
tension distribution can hold it standing, and the actuator count a planner
would face.
"""
import json
import sys
import numpy as np
import mujoco
from scipy.optimize import lsq_linear, linprog

sys.path.insert(0, "/Users/ncorrell/Downloads/tensegrity/experiments")
from e12_cell_graph import ring, TWIST, STRUT_R, STRUT_DENS, closure  # noqa

MJ = "/Users/ncorrell/Downloads/tensegrity/mujoco"
RES = "/Users/ncorrell/Downloads/tensegrity/experiments/results"
OUT = f"{RES}/e13_rigid_cage.json"
K_CABLE, T_PRE, FMAX = 40e3, 400.0, 1500.0
DT = 2e-4


class Body:
    def __init__(self, name, rings, mass_kg, contact=False, motors=None):
        """mass_kg is STRUCTURAL mass. `motors` is a list of
        (mesh, mass_kg, count, ring) -- the drives for the joint at that ring,
        mounted on this segment. Tendon transmission is the whole point of the
        architecture: a joint's motors sit on the segment proximal to it, so
        the distal segments stay light."""
        self.name, self.rings, self.mass, self.contact = name, rings, mass_kg, contact
        self.motors = motors or []


def build(segments, interfaces, actuate, t_pre=T_PRE, standing=True,
          rests=None):
    """segments: name -> Body; interfaces: list of (p_near, p_far, d_near, d_far)
    given as (segment, ring) pairs."""
    nodes, bodies = {}, []
    for b in segments.values():
        for rname, pts in b.rings.items():
            for i, p in enumerate(pts):
                nodes[f"{b.name}_{rname}{i}"] = (np.array(p), b.name)
    for b in segments.values():
        pts = [nodes[f"{b.name}_{r}{i}"][0] for r in b.rings for i in range(3)]
        c = np.mean(pts, axis=0)
        keys = list(b.rings)
        seg = []
        for a in range(len(keys)):
            for q in range(a + 1, len(keys)):
                for i in range(3):
                    pa = nodes[f"{b.name}_{keys[a]}{i}"][0]
                    pb = nodes[f"{b.name}_{keys[q]}{(i+1)%3}"][0]
                    seg.append(
                        f'<geom type="capsule" size="{STRUT_R}" mass="0.001" '
                        f'contype="0" conaffinity="0" rgba="0.28 0.31 0.36 1" '
                        f'fromto="{pa[0]-c[0]:.5f} {pa[1]-c[1]:.5f} '
                        f'{pa[2]-c[2]:.5f} {pb[0]-c[0]:.5f} {pb[1]-c[1]:.5f} '
                        f'{pb[2]-c[2]:.5f}"/>')
        sites = [f'<site name="{n}" pos="{p[0]-c[0]:.5f} {p[1]-c[1]:.5f} '
                 f'{p[2]-c[2]:.5f}" size="0.006"/>'
                 for n, (p, owner) in nodes.items() if owner == b.name]
        # Distribute the segment mass over the cage. Lumping it into a small
        # sphere gives a rotational inertia one to two orders of magnitude too
        # small, and the structure then rings at tens of rad/s against the
        # 40 kN/m cables -- which no derivative term can be tuned against.
        # Inertia of a ring-and-strut cage of radius rr and height hh.
        rr = max(np.linalg.norm(np.array(p[:2]) - c[:2]) for p in pts)
        hh = max(p[2] for p in pts) - min(p[2] for p in pts)
        ixx = b.mass * (0.5 * rr ** 2 + hh ** 2 / 12.0)
        izz = b.mass * rr ** 2
        # Motor placements first: an explicit <inertial> OVERRIDES geom
        # masses in MuJoCo, so the drives have to be folded into it or they
        # weigh nothing.
        motor_geoms, motor_pm = [], []
        for mesh, mmass, count, rkey in b.motors:
            rp = [nodes[f"{b.name}_{rkey}{i}"][0] for i in range(3)]
            rc = np.mean(rp, axis=0)
            axis = rc - c
            na = np.linalg.norm(axis)
            axis = axis / na if na > 1e-9 else np.array([0.0, 0.0, 1.0])
            rr_m = max(np.linalg.norm(np.array(p) - rc) for p in rp)
            for k in range(count):
                th = 2 * np.pi * k / count + np.pi / count
                u = np.array([1.0, 0.0, 0.0])
                if abs(axis @ u) > 0.9:
                    u = np.array([0.0, 1.0, 0.0])
                e1 = np.cross(axis, u)
                e1 /= np.linalg.norm(e1)
                e2 = np.cross(axis, e1)
                radial = np.cos(th) * e1 + np.sin(th) * e2
                tang = -np.sin(th) * e1 + np.cos(th) * e2
                pos = rc - 0.10 * na * axis + (rr_m + 0.035) * radial - c
                motor_pm.append((mmass, pos))
                motor_geoms.append(
                    f'<geom type="mesh" mesh="{mesh}" mass="0" '
                    f'contype="0" conaffinity="0" rgba="0.32 0.33 0.36 1" '
                    f'pos="{pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f}" '
                    f'xyaxes="{tang[0]:.5f} {tang[1]:.5f} {tang[2]:.5f} '
                    f'{radial[0]:.5f} {radial[1]:.5f} {radial[2]:.5f}"/>')

        I = np.diag([ixx, ixx, izz])                 # cage, about the origin
        M = b.mass
        com = np.zeros(3)
        for mmass, pos in motor_pm:
            M += mmass
            com += mmass * pos
            I += mmass * (float(pos @ pos) * np.eye(3) - np.outer(pos, pos))
        if M > 1e-9:
            com /= M
        I -= M * (float(com @ com) * np.eye(3) - np.outer(com, com))
        seg.append(f'<inertial pos="{com[0]:.5f} {com[1]:.5f} {com[2]:.5f}" '
                   f'mass="{M:.4f}" fullinertia="{I[0,0]:.6f} {I[1,1]:.6f} '
                   f'{I[2,2]:.6f} {I[0,1]:.6f} {I[0,2]:.6f} {I[1,2]:.6f}"/>')
        seg += motor_geoms
        if b.contact:
            # A foot needs a support POLYGON. One pad is a point contact: it
            # transmits force but no moment, so the ground cannot resist
            # toppling and no ankle strategy has anything to push against.
            # Put a pad on every node of the lowest ring.
            zmin = min(p[2] for p in pts)
            low = [p for p in pts if p[2] < zmin + 1e-6]
            if len(low) < 3:
                ring_of = {}
                for rn, rp in b.rings.items():
                    ring_of[rn] = np.mean([q[2] for q in rp])
                lowest = min(ring_of, key=ring_of.get)
                low = list(b.rings[lowest])
            for k, lo in enumerate(low):
                seg.append(
                    f'<geom name="pad_{b.name}_{k}" type="sphere" size="0.025" '
                    f'mass="0.04" contype="1" conaffinity="1" '
                    f'friction="1.2 0.01 0.001" pos="{lo[0]-c[0]:.5f} '
                    f'{lo[1]-c[1]:.5f} {lo[2]-c[2]:.5f}"/>')
        bodies.append(f'''
    <body name="{b.name}" pos="{c[0]:.5f} {c[1]:.5f} {c[2]:.5f}">
      <freejoint/>
      {chr(10).join("      " + x for x in seg + sites)}
    </body>''')

    # Cable families per interface (E15). The original set ran its cross
    # family with the same handedness as the cell twist, which leaves the
    # joint with force closure but ZERO moment authority about its worst axis
    # -- no self-stress, no co-contraction, and nothing for a balance
    # controller to push with. Counter-winding the cross family closes the
    # wrench set with three cables fewer.
    #   A  proximal near ring -> distal near ring, aligned
    #   B  proximal near ring -> distal far  ring, aligned   (anti-pull-out)
    #   C  proximal near ring -> distal near ring, offset +1 (clockwise)
    #   D  proximal near ring -> distal near ring, offset -1 (anticlockwise)
    # C and D are the counter-wound pair. Winding both ways is what closes the
    # wrench set, and it is required at every joint for the same reason the
    # yaw drums need opposite-handed windings: one handedness alone can only
    # generate moment of one sign.
    cables = []
    for p_near, p_far, d_near, d_far in interfaces:
        for i in range(3):
            cables += [(f"{p_near[0]}_{p_near[1]}{i}",
                        f"{d_near[0]}_{d_near[1]}{i}", True),
                       (f"{p_near[0]}_{p_near[1]}{i}",
                        f"{d_far[0]}_{d_far[1]}{i}", True),
                       (f"{p_near[0]}_{p_near[1]}{i}",
                        f"{d_near[0]}_{d_near[1]}{(i + 1) % 3}", True),
                       (f"{p_near[0]}_{p_near[1]}{i}",
                        f"{d_near[0]}_{d_near[1]}{(i - 1) % 3}", True)]

    def emit(rests):
        ten, act = [], []
        for i, (a, b_, drivable) in enumerate(cables):
            r = rests[i] if rests else 0.001
            ten.append(f'<spatial name="t{i}" width="0.004" stiffness="{K_CABLE}"'
                       f' damping="{0.02*np.sqrt(K_CABLE):.2f}"'
                       f' springlength="0 {r:.6f}" rgba="0.95 0.97 1 1">'
                       f'\n      <site site="{a}"/>\n      <site site="{b_}"/>'
                       f'\n    </spatial>')
            if drivable and actuate:
                act.append(f'<motor name="a{i}" tendon="t{i}" gear="-{FMAX}" '
                           f'ctrlrange="0 1" ctrllimited="true"/>')
        return f'''<mujoco model="rigid_cage_walker">
  <compiler angle="radian" autolimits="true" meshdir="assets"/>
  <option timestep="{DT}" integrator="implicitfast" solver="Newton"
          iterations="50"/>
  <visual><global offwidth="1600" offheight="1200"/></visual>
  <asset>
    <mesh name="ak70" file="ak70_10.stl"/>
    <mesh name="ak60" file="ak60_6.stl"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.88 0.89 0.90"
             rgb2="0.96 0.96 0.97" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="10 10"/>
  </asset>
  <worldbody>
    <light pos="1.2 -1.2 2.6" dir="-0.4 0.4 -1" directional="true"/>
    <geom name="floor" type="plane" size="6 6 0.1" material="grid"
          contype="1" conaffinity="1" friction="1.2 0.01 0.001"/>
    {"".join(bodies)}
  </worldbody>
  <tendon>
    {(chr(10) + "    ").join(ten)}
  </tendon>
  <actuator>
    {(chr(10) + "    ").join(act)}
  </actuator>
</mujoco>'''

    if rests is None:
        m0 = mujoco.MjModel.from_xml_string(emit(None))
        d0 = mujoco.MjData(m0)
        mujoco.mj_forward(m0, d0)
        rests = [max(1e-4, float(d0.ten_length[i]) - t_pre / K_CABLE)
                 for i in range(m0.ntendon)]
    return emit(rests)


def humanoid(H=0.30, R=0.09, ov=0.09, torso_kg=22.0):
    segs, ifs = {}, []

    def cage(name, centre, h, r, phase, mass, contact=False):
        c = np.array(centre, float)
        segs[name] = Body(name,
                          {"b": ring(r, c - [0, 0, h / 2], phase),
                           "t": ring(r, c + [0, 0, h / 2], phase + TWIST)},
                          mass, contact)
        return name

    z = 0.0
    for side, y in (("l", +0.10), ("r", -0.10)):
        prev = None
        zc = H / 2
        for k, (nm, mass) in enumerate((("foot", 0.4), ("shank", 1.0),
                                        ("thigh", 1.0))):
            t = cage(f"{nm}_{side}", (0, y, zc), H, R, k * np.pi / 3, mass,
                     contact=(k == 0))
            if prev:
                ifs.append(((prev, "t"), (prev, "b"), (t, "b"), (t, "t")))
            prev = t
            zc += H - ov
        segs[f"__hip_{side}"] = None
        del segs[f"__hip_{side}"]
        ifs.append(((f"thigh_{side}", "t"), (f"thigh_{side}", "b"),
                    ("pelvis", "b"), ("pelvis", "t")))
    z_pel = H / 2 + 2 * (H - ov) + (H - ov) / 2
    cage("pelvis", (0, 0, z_pel), 0.22, 0.11, 0.0, 3.0)
    cage("torso", (0, 0, z_pel + 0.42), 0.44, 0.13, np.pi / 3, torso_kg)
    ifs.append((("pelvis", "t"), ("pelvis", "b"), ("torso", "b"),
                ("torso", "t")))
    cage("head", (0, 0, z_pel + 0.78), 0.20, 0.07, 0.0, 0.6)
    ifs.append((("torso", "t"), ("torso", "b"), ("head", "b"), ("head", "t")))
    for side, y in (("l", +0.20), ("r", -0.20)):
        prev = "torso"
        zc = z_pel + 0.42 + 0.22 - H / 2
        for k, (nm, mass) in enumerate((("uarm", 0.5), ("farm", 0.3))):
            t = cage(f"{nm}_{side}", (0, y, zc), H * 0.85, R * 0.8,
                     k * np.pi / 3, mass)
            if prev == "torso":
                ifs.append((("torso", "t"), ("torso", "b"), (t, "t"), (t, "b")))
            else:
                ifs.append(((prev, "b"), (prev, "t"), (t, "t"), (t, "b")))
            prev = t
            zc -= H * 0.85 - ov
    return segs, ifs


def balance(xml, seconds=3.0, push=None):
    """Hold the home pose by bounded tension distribution over all 6-DoF
    joints -- the hierarchical scheme of the paper, applied to a hinge-free
    machine."""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    q0, nv = d.qpos.copy(), m.nv
    fmax = -m.actuator_gear[:, 0]
    reg = np.sqrt(0.02) * np.eye(m.nu)
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
    head = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "head")
    z0 = float(d.xpos[head][2])
    decim = max(1, int(round(1.0 / (250.0 * m.opt.timestep))))
    x = np.zeros(m.nu)
    for step in range(int(seconds / m.opt.timestep)):
        if step % decim == 0:
            err = np.zeros(nv)
            k = 0
            for j in range(m.njnt):
                a = m.jnt_qposadr[j]
                err[k:k + 3] = d.qpos[a:a + 3] - q0[a:a + 3]
                dq = np.zeros(3)
                mujoco.mju_subQuat(dq, d.qpos[a + 3:a + 7], q0[a + 3:a + 7])
                err[k + 3:k + 6] = dq
                k += 6
            tau = -600.0 * err - 40.0 * d.qvel + d.qfrc_bias
            mom = np.zeros((m.nu, m.nv))
            mujoco.mju_sparse2dense(mom, d.actuator_moment, d.moment_rownnz,
                                    d.moment_rowadr, d.moment_colind)
            sol = lsq_linear(np.vstack([mom.T, reg]),
                             np.concatenate([tau, np.sqrt(0.02) * x]),
                             bounds=(0.0, 1.0), max_iter=12, tol=1e-5)
            x = sol.x
            d.ctrl[:] = x
        if push and push[0] <= d.time < push[0] + 0.2:
            d.xfrc_applied[torso, 0] = push[1]
        else:
            d.xfrc_applied[torso, :3] = 0.0
        mujoco.mj_step(m, d)
        if not np.all(np.isfinite(d.qpos)):
            return dict(ok=False, z=float("nan"), reason="diverged")
    z = float(d.xpos[head][2])
    return dict(ok=bool(z > 0.75 * z0), z0=z0, z=z, peak_T=float((fmax * x).max()))


def main():
    segs, ifs = humanoid()
    xml = build(segs, ifs, actuate=True)
    open(f"{MJ}/humanoid_hingefree_walker.xml", "w").write(xml)
    m = mujoco.MjModel.from_xml_string(xml)
    print(f"HINGE-FREE HUMANOID, RIGID CAGES")
    print(f"  {len(segs)} segments, {m.nv} DoF, {m.ntendon} cables, "
          f"{m.nu} actuators, {m.nbody-1} bodies")
    print(f"  (E12's floating-strut version: 210 DoF, 306 cables)")

    d = mujoco.MjData(m)
    for name in ("foot_l", "thigh_l", "torso", "head"):
        b = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)]
        ids = {b[0]}
        W = []
        mujoco.mj_forward(m, d)
        print(f"  force closure at {name:8s}: "
              f"{closure(m, d, [name]):.4f}")

    print("\n  bounded tension distribution holding the home pose:")
    r = balance(xml, seconds=3.0)
    print(f"    3 s quiet stand: head {r.get('z0', 0):.3f} -> {r['z']:.3f} m  "
          f"{'HOLDS' if r['ok'] else 'FALLS'}"
          + (f", peak cable {r['peak_T']:.0f} N" if r["ok"] else ""))
    rp = balance(xml, seconds=3.0, push=(1.5, 40.0))
    print(f"    + 40 N torso push: {'HOLDS' if rp['ok'] else 'FALLS'}")

    json.dump(dict(n_segments=len(segs), nv=int(m.nv), ntendon=int(m.ntendon),
                   nu=int(m.nu), stand=r, push=rp),
              open(OUT, "w"), indent=1)
    print(f"\n  model -> {MJ}/humanoid_hingefree_walker.xml")
    print("E13 ->", OUT)


if __name__ == "__main__":
    main()
