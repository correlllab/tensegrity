#!/usr/bin/env python
"""E10: what a hinge-free joint would have to look like.

E9 shows the humanoid's joints have no wrench closure: the cables crossing a
joint all pull the distal segment toward the proximal one, so their positive
span is a half-space and nothing resists the segments approaching. The
geometric reason is that the two cages meet end to end. In a tensegrity tower
the stages OVERLAP: the distal cage's bottom ring sits below the proximal
cage's top ring, so cables run downward-outward as well as upward, and the
positive span closes.

This builds that joint parametrically -- a fixed proximal cage and a floating
distal cage of three free struts, joined only by cables -- and sweeps the
overlap. For each overlap it reports

  closure    the wrench-closure LP of E9, on the interface
  hold       does it stand under gravity and a distal payload
  k_bend     lateral endpoint stiffness, swept over prestress
  impact     peak reaction in a drop, against a pin-jointed equivalent

so the two properties the hinged humanoid failed to show (prestress-tunable
stiffness, impact attenuation) can be tested where the mechanism is present.
"""
import json
import re
import numpy as np
import mujoco
from scipy.optimize import linprog

RES = "/Users/ncorrell/Downloads/tensegrity/experiments/results"
OUT = f"{RES}/e10_tensegrity_joint.json"

R, H, TWIST = 0.09, 0.30, np.deg2rad(29.8)
STRUT_R, STRUT_M = 0.009, 0.15
TIP_MASS = 1.0                       # distal payload, ~ a shank + foot
K_CABLE = 40e3                       # series-elastic element, N/m


def ring(radius, z, phase):
    return [(radius * np.cos(phase + 2 * np.pi * i / 3),
             radius * np.sin(phase + 2 * np.pi * i / 3), z) for i in range(3)]


def build(overlap, prestress_N=100.0, pin=False, k_cable=None):
    """Proximal cage fixed to the world; distal cage = 3 free struts.
    `pin=True` replaces the cable interface with a ball joint at the axis, the
    hinged equivalent used as the control condition."""
    K = K_CABLE if k_cable is None else k_cable
    p_bot = ring(R, 0.0, 0.0)
    p_top = ring(R, H, TWIST)
    d_bot = ring(R, H - overlap, TWIST + np.pi / 3)
    d_top = ring(R, 2 * H - overlap, 2 * TWIST + np.pi / 3)

    def site(name, p):
        return (f'<site name="{name}" pos="{p[0]:.5f} {p[1]:.5f} {p[2]:.5f}" '
                f'size="0.005"/>')

    world_sites = "\n      ".join(
        [site(f"p_bot{i}", p_bot[i]) for i in range(3)] +
        [site(f"p_top{i}", p_top[i]) for i in range(3)])
    # proximal struts, drawn as world geoms (the fixed segment)
    prox = "\n      ".join(
        f'<geom type="capsule" size="{STRUT_R}" contype="0" conaffinity="0" '
        f'rgba="0.22 0.24 0.28 1" fromto="{p_bot[i][0]:.5f} {p_bot[i][1]:.5f} '
        f'{p_bot[i][2]:.5f} {p_top[(i+1)%3][0]:.5f} {p_top[(i+1)%3][1]:.5f} '
        f'{p_top[(i+1)%3][2]:.5f}"/>' for i in range(3))

    bodies = []
    for i in range(3):
        a, b = d_bot[i], d_top[(i + 1) % 3]
        c = tuple(0.5 * (np.array(a) + np.array(b)))
        bodies.append(f'''
    <body name="ds{i}" pos="{c[0]:.5f} {c[1]:.5f} {c[2]:.5f}">
      {'<joint type="ball"/>' if pin else '<freejoint/>'}
      <geom type="capsule" size="{STRUT_R}" mass="{STRUT_M}" contype="0"
            conaffinity="0" rgba="0.30 0.33 0.38 1"
            fromto="{a[0]-c[0]:.5f} {a[1]-c[1]:.5f} {a[2]-c[2]:.5f}
                    {b[0]-c[0]:.5f} {b[1]-c[1]:.5f} {b[2]-c[2]:.5f}"/>
      <site name="d_bot{i}" pos="{a[0]-c[0]:.5f} {a[1]-c[1]:.5f} {a[2]-c[2]:.5f}"
            size="0.005"/>
      <site name="d_top{(i+1)%3}" pos="{b[0]-c[0]:.5f} {b[1]-c[1]:.5f} {b[2]-c[2]:.5f}"
            size="0.005"/>
    </body>''')
    # a payload plate rigidly tied to strut 0 stands in for the next segment
    bodies.append(f'''
    <body name="tip" pos="0 0 {2*H-overlap:.5f}">
      <freejoint/>
      <geom type="sphere" size="0.02" mass="{TIP_MASS}" contype="0"
            conaffinity="0" rgba="0.8 0.5 0.2 1"/>
      <site name="tip" pos="0 0 0" size="0.008"/>
    </body>''')

    cab = []

    def cable(name, s1, s2, group):
        cab.append((name, s1, s2, group))

    for i in range(3):                                   # distal cell
        cable(f"d_bot{i}", f"d_bot{i}", f"d_bot{(i+1)%3}", "cell")
        cable(f"d_topp{i}", f"d_top{i}", f"d_top{(i+1)%3}", "cell")
        cable(f"d_vert{i}", f"d_bot{i}", f"d_top{i}", "cell")
    for i in range(3):                                   # interface: saddle
        cable(f"if_down{i}", f"p_top{i}", f"d_bot{i}", "joint")
        cable(f"if_up{i}", f"p_top{i}", f"d_top{i}", "joint")
        cable(f"if_x{i}", f"p_top{i}", f"d_bot{(i+1)%3}", "joint")
    for i in range(3):                                   # tie the tip on
        cable(f"tip{i}", "tip", f"d_top{i}", "tip")

    def make(rests):
        ten = "\n    ".join(
            f'<spatial name="{n}" width="0.003" rgba="0.95 0.97 1 1" '
            f'stiffness="{K}" damping="{0.02*np.sqrt(K):.2f}" '
            f'springlength="0 {rests.get(n, 0.001):.6f}">'
            f'\n      <site site="{a}"/>'
            f'\n      <site site="{b}"/>\n    </spatial>'
            for n, a, b, g in cab)
        return TEMPLATE.replace("__TENDONS__", ten)

    TEMPLATE = f'''<mujoco model="tensegrity_joint">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="5e-5" integrator="implicitfast" gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1" directional="true"/>
      {prox}
      {world_sites}
    {"".join(bodies)}
  </worldbody>
  <tendon>
    __TENDONS__
  </tendon>
</mujoco>'''
    # rest lengths from the built geometry, so prestress is a TENSION
    m0 = mujoco.MjModel.from_xml_string(make({}))
    d0 = mujoco.MjData(m0)
    mujoco.mj_forward(m0, d0)
    rests = {n: max(1e-4, float(d0.ten_length[i]) - prestress_N / K)
             for i, (n, a, b, g) in enumerate(cab)}
    return make(rests), cab


def closure_lp(W):
    if W.shape[1] == 0 or np.linalg.matrix_rank(W) < W.shape[0]:
        return 0.0
    nc = W.shape[1]
    c = np.zeros(nc + 1)
    c[-1] = -1.0
    Aub = np.hstack([-np.eye(nc), np.ones((nc, 1))])
    r = linprog(c, A_ub=Aub, b_ub=np.zeros(nc),
                A_eq=np.hstack([W, np.zeros((W.shape[0], 1))]),
                b_eq=np.zeros(W.shape[0]),
                bounds=[(0, None)] * nc + [(0, 1)])
    return float(r.x[-1]) if r.success else 0.0


def interface_closure(m, d):
    """Wrench closure of the distal cage against the fixed proximal cage."""
    mujoco.mj_forward(m, d)
    distal = {b for b in range(m.nbody)
              if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
              and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                                    b).startswith(("ds", "tip"))}
    origin = np.array([0.0, 0.0, H])
    W = []
    for t in range(m.ntendon):
        adr, num = m.tendon_adr[t], m.tendon_num[t]
        sids = [int(m.wrap_objid[adr + i]) for i in range(num)
                if m.wrap_type[adr + i] == mujoco.mjtWrap.mjWRAP_SITE]
        side = [m.site_bodyid[s] in distal for s in sids]
        if all(side) or not any(side):
            continue
        for k, s in enumerate(sids):
            if not side[k]:
                continue
            nb = sids[1 - k]
            v = d.site_xpos[nb] - d.site_xpos[s]
            n = np.linalg.norm(v)
            if n < 1e-9:
                continue
            u = v / n
            r = d.site_xpos[s] - origin
            W.append(np.concatenate([u, np.cross(r, u)]))
    W = np.array(W).T if W else np.zeros((6, 0))
    return closure_lp(W), closure_lp(W[:3]) if W.size else 0.0, W.shape[1]


def settle(m, seconds=0.6, ctrl=None, force=None, damp_boost=0.0):
    """damp_boost raises tendon damping during a static probe. It does not
    move the static equilibrium, but without it a lightly damped cable network
    is still ringing after any affordable settle time."""
    if damp_boost:
        m.tendon_damping[:] = m.tendon_damping + damp_boost
    d = mujoco.MjData(m)
    tip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tip")
    for _ in range(int(seconds / m.opt.timestep)):
        if force is not None:
            d.xfrc_applied[tip, :3] = force
        mujoco.mj_step(m, d)
        if not np.all(np.isfinite(d.qpos)):
            return None
    return d


def build_drop(overlap, pin, prestress_N=100.0, k_cable=None):
    """Falling assembly for the impact test.

    Both conditions carry identical masses: a proximal cage with a contact
    foot, and a 1.45 kg distal segment. They differ only in the interface ---
    floating struts held by cables, or one rigid body on a ball joint, which
    is the bearing this design currently uses.
    """
    K = K_CABLE if k_cable is None else k_cable
    p_bot, p_top = ring(R, 0.0, 0.0), ring(R, H, TWIST)
    d_bot = ring(R, H - overlap, TWIST + np.pi / 3)
    d_top = ring(R, 2 * H - overlap, 2 * TWIST + np.pi / 3)
    head = '''<mujoco model="joint_drop">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="5e-5" integrator="implicitfast" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" contype="1" conaffinity="1"/>
    <body name="prox" pos="0 0 0">
      <freejoint/>
      <geom name="foot" type="sphere" size="0.03" mass="1.0" contype="1"
            conaffinity="1" pos="0 0 -0.03"/>'''
    for i in range(3):
        a_, b_ = p_bot[i], p_top[(i + 1) % 3]
        head += (f'\n      <geom type="capsule" size="{STRUT_R}" mass="{STRUT_M}"'
                 f' contype="0" conaffinity="0" rgba="0.22 0.24 0.28 1"'
                 f' fromto="{a_[0]:.5f} {a_[1]:.5f} {a_[2]:.5f} '
                 f'{b_[0]:.5f} {b_[1]:.5f} {b_[2]:.5f}"/>')
    for i in range(3):
        head += (f'\n      <site name="p_top{i}" pos="{p_top[i][0]:.5f} '
                 f'{p_top[i][1]:.5f} {p_top[i][2]:.5f}" size="0.005"/>')

    cab = []
    if pin:
        # one rigid distal body on a ball joint at the joint origin
        body = f'\n      <body name="distal" pos="0 0 {H:.5f}">\n'
        body += '        <joint type="ball" damping="0.05"/>\n'
        for i in range(3):
            a_, b_ = d_bot[i], d_top[(i + 1) % 3]
            body += (f'        <geom type="capsule" size="{STRUT_R}" '
                     f'mass="{STRUT_M}" contype="0" conaffinity="0" '
                     f'rgba="0.30 0.33 0.38 1" fromto="{a_[0]:.5f} {a_[1]:.5f} '
                     f'{a_[2]-H:.5f} {b_[0]:.5f} {b_[1]:.5f} {b_[2]-H:.5f}"/>\n')
        body += (f'        <geom type="sphere" size="0.02" mass="{TIP_MASS}" '
                 f'contype="0" conaffinity="0" rgba="0.8 0.5 0.2 1" '
                 f'pos="0 0 {2*H-overlap-H:.5f}"/>\n      </body>')
        xml = head + body + "\n    </body>\n  </worldbody>\n</mujoco>"
        return xml, []

    body = ""
    for i in range(3):
        a_, b_ = d_bot[i], d_top[(i + 1) % 3]
        c = tuple(0.5 * (np.array(a_) + np.array(b_)))
        body += f'''
    <body name="ds{i}" pos="{c[0]:.5f} {c[1]:.5f} {c[2]:.5f}">
      <freejoint/>
      <geom type="capsule" size="{STRUT_R}" mass="{STRUT_M}" contype="0"
            conaffinity="0" rgba="0.30 0.33 0.38 1"
            fromto="{a_[0]-c[0]:.5f} {a_[1]-c[1]:.5f} {a_[2]-c[2]:.5f}
                    {b_[0]-c[0]:.5f} {b_[1]-c[1]:.5f} {b_[2]-c[2]:.5f}"/>
      <site name="d_bot{i}" pos="{a_[0]-c[0]:.5f} {a_[1]-c[1]:.5f} {a_[2]-c[2]:.5f}" size="0.005"/>
      <site name="d_top{(i+1)%3}" pos="{b_[0]-c[0]:.5f} {b_[1]-c[1]:.5f} {b_[2]-c[2]:.5f}" size="0.005"/>
    </body>'''
    body += f'''
    <body name="tip" pos="0 0 {2*H-overlap:.5f}">
      <freejoint/>
      <geom type="sphere" size="0.02" mass="{TIP_MASS}" contype="0"
            conaffinity="0" rgba="0.8 0.5 0.2 1"/>
      <site name="tip" pos="0 0 0" size="0.008"/>
    </body>'''
    for i in range(3):
        cab += [(f"d_bot{i}", f"d_bot{i}", f"d_bot{(i+1)%3}"),
                (f"d_topp{i}", f"d_top{i}", f"d_top{(i+1)%3}"),
                (f"d_vert{i}", f"d_bot{i}", f"d_top{i}"),
                (f"if_down{i}", f"p_top{i}", f"d_bot{i}"),
                (f"if_up{i}", f"p_top{i}", f"d_top{i}"),
                (f"if_x{i}", f"p_top{i}", f"d_bot{(i+1)%3}"),
                (f"tip{i}", "tip", f"d_top{i}")]

    def make(rests):
        ten = "\n    ".join(
            f'<spatial name="{n}" width="0.003" stiffness="{K}" '
            f'damping="{0.02*np.sqrt(K):.2f}" '
            f'springlength="0 {rests.get(n, 0.001):.6f}">'
            f'\n      <site site="{x}"/>\n      <site site="{y}"/>'
            f'\n    </spatial>' for n, x, y in cab)
        return (head + "\n    </body>" + body +
                f"\n  </worldbody>\n  <tendon>\n    {ten}\n  </tendon>\n</mujoco>")

    m0 = mujoco.MjModel.from_xml_string(make({}))
    d0 = mujoco.MjData(m0)
    mujoco.mj_forward(m0, d0)
    rests = {n: max(1e-4, float(d0.ten_length[i]) - prestress_N / K)
             for i, (n, x, y) in enumerate(cab)}
    return make(rests), cab


def drop_test(overlap, h, pin, prestress_N=100.0):
    """Settle the assembly standing on the floor first, then lift it by h and
    release. Without the settling phase the cable variant starts out of
    equilibrium and its ring-down swamps the landing transient."""
    xml, _ = build_drop(overlap, pin, prestress_N)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    for _ in range(int(0.8 / m.opt.timestep)):      # settle in contact
        mujoco.mj_step(m, d)
    if not np.all(np.isfinite(d.qpos)):
        return float("nan"), float(m.body_mass.sum())
    # lift every free body by h and release from rest
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            d.qpos[m.jnt_qposadr[j] + 2] += h
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    peak, f6 = 0.0, np.zeros(6)
    mass = float(m.body_mass.sum())
    for _ in range(int(0.35 / m.opt.timestep)):
        mujoco.mj_step(m, d)
        if not np.all(np.isfinite(d.qpos)):
            return float("nan"), mass
        for i in range(d.ncon):
            c = d.contact[i]
            if floor in (c.geom[0], c.geom[1]):
                mujoco.mj_contactForce(m, d, i, f6)
                peak = max(peak, abs(float(f6[0])))
    return peak, mass


def main():
    res = {"overlap_sweep": [], "prestress_sweep": []}

    print("OVERLAP SWEEP  (does the cable interface close, and does it hold?)")
    print(f"{'overlap/H':>9s} {'cables':>7s} {'6-DoF':>8s} {'3-DoF':>8s} "
          f"{'tip sag mm':>11s}  verdict")
    for frac in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        ov = frac * H
        xml, cab = build(ov)
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        t6, t3, nc = interface_closure(m, d)
        z0 = float(d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                            "tip")][2])
        dd = settle(m)
        sag = float("nan") if dd is None else 1e3 * (
            z0 - float(dd.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                                 "tip")][2]))
        held = dd is not None and abs(sag) < 50
        print(f"{frac:9.2f} {nc:7d} {t6:8.4f} {t3:8.4f} {sag:11.1f}  "
              f"{'force closure, holds' if held and t3 > 1e-6 else 'collapses'}")
        res["overlap_sweep"].append(dict(frac=frac, overlap=ov, n_cables=nc,
                                         t6=t6, t3=t3, sag_mm=sag,
                                         held=bool(held)))

    # Force closure (t3) is the criterion, not full wrench closure: a JOINT
    # must be free to rotate, so t6 > 0 would mean it is rigid. What the hinge
    # provides structurally is exactly the three translational constraints,
    # and t3 > 0 means the cables provide them instead.
    ok = [r for r in res["overlap_sweep"] if r["held"] and r["t3"] > 1e-6]
    if not ok:
        print("\nNo overlap in the swept range produces a closed, "
              "load-bearing joint.")
        json.dump(res, open(OUT, "w"), indent=1)
        return
    best = ok[0]
    print(f"\nsmallest overlap with force closure and load capacity: "
          f"{best['frac']:.2f} H = {1e3*best['overlap']:.0f} mm. "
          f"6-DoF closure is 0 throughout, which is correct: the joint is free "
          f"to rotate.")

    print("\nPRESTRESS SWEEP at that overlap, at three cable stiffnesses.")
    print("The tangent stiffness of a prestressed network is an elastic term "
          "(EA/L)\nplus a geometric term (T/L). Prestress only tunes "
          "stiffness when the two\nare comparable, so the sweep is repeated "
          "for soft, series-elastic and rope\ncables.")
    for KC, tag in ((2.0e3, "soft (as modelled)"),
                    (40.0e3, "series-elastic"),
                    (600.0e3, "near-rope")):
        print(f"\n  k_cable = {KC/1e3:.0f} kN/m  [{tag}]")
        print(f"  {'T0 (N)':>7s} {'k_lat (N/m)':>12s} {'geom/elastic':>13s}")
        ks_this = []
        for T0 in (25.0, 50.0, 100.0, 200.0, 400.0):
            xml, _ = build(best["overlap"], prestress_N=T0, k_cable=KC)
            m = mujoco.MjModel.from_xml_string(xml)
            tip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tip")
            d0 = settle(m, seconds=2.0, damp_boost=60.0)
            if d0 is None:
                print(f"  {T0:7.0f} {'unstable':>12s}")
                continue
            p0 = d0.xpos[tip].copy()
            PROBE = 2.0
            d1 = settle(m, seconds=2.0, force=np.array([PROBE, 0, 0]))
            if d1 is None:
                continue
            dx = float(np.linalg.norm(d1.xpos[tip] - p0))
            k = PROBE / max(dx, 1e-9)
            ratio = (T0 / H) / (KC)          # geometric / elastic term
            print(f"  {T0:7.0f} {k:12.1f} {ratio:13.4f}")
            ks_this.append(k)
            res["prestress_sweep"].append(
                dict(k_cable=KC, T0=T0, k_lat=k, geom_over_elastic=ratio))
        if len(ks_this) > 1:
            print(f"  -> prestress 25-400 N moves stiffness "
                  f"{min(ks_this):.1f} -> {max(ks_this):.1f} N/m "
                  f"({max(ks_this)/max(min(ks_this),1e-9):.2f}x)")
            res.setdefault("prestress_ratio_by_k", {})[str(int(KC))] = \
                max(ks_this) / max(min(ks_this), 1e-9)
    print("\nIMPACT AT JOINT SCALE (drop, peak ground reaction)")
    print(f"{'drop (m)':>9s} {'cable joint':>12s} {'pin joint':>10s} "
          f"{'delta':>7s}")
    for h in (0.05, 0.10, 0.20):
        a, ma = drop_test(best["overlap"], h, pin=False)
        b, mb = drop_test(best["overlap"], h, pin=True)
        if abs(ma - mb) > 1e-6:
            print(f"  (mass mismatch {ma:.3f} vs {mb:.3f} kg)")
        if np.isfinite(a) and np.isfinite(b) and b > 0:
            print(f"{h:9.2f} {a:11.0f}N {b:9.0f}N {100*(a/b-1):+6.0f}%")
            res.setdefault("impact", []).append(
                dict(h=h, cable=a, pin=b, delta_pct=100 * (a / b - 1)))
        else:
            print(f"{h:9.2f} {'unstable':>12s}")

    res["best_overlap_frac"] = best["frac"]
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nE10 ->", OUT)


if __name__ == "__main__":
    main()
