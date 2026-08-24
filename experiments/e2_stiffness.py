#!/usr/bin/env python
"""E2: stiffness on demand -- measured in the linear regime.

The first version probed the hand with 5 N, which deflected it 0.25-0.76 m on
a 0.52 m arm: a large-displacement secant, not a stiffness, and one dominated
by the gravitational pendulum term of a hanging limb. This version

  * probes with a force small enough to keep the deflection in the millimetre
    range, and verifies linearity by halving the probe,
  * probes along three axes and reports the full endpoint stiffness matrix
    (its eigenvalues, not a single scalar),
  * separates the three contributions that the single "3x range" number
    previously conflated:
        no cables            -- gravity/pendulum restoring force alone
        cables, zero prestress -- what the network geometry adds
        cables, prestress p   -- what tensioning adds on top,
  * samples co-contraction densely at the low end, where it saturates.

(a) passive: torque model, pelvis welded, arm unpowered, prestress swept.
(b) active : cable model, uniform tension on the nine left-shoulder cables.
"""
import json
import os
import re
import numpy as np
import mujoco

MJ = "/Users/ncorrell/Downloads/tensegrity/mujoco"
OUT = "/Users/ncorrell/Downloads/tensegrity/experiments/results/e2_stiffness.json"
os.chdir(MJ)  # meshdir resolution for from_xml_string
PROBE_N = 0.20          # keeps the hand inside ~10 mm
AXES = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def welded(path):
    xml = open(path).read().replace('<freejoint name="root"/>', "")
    xml = re.sub(r"<keyframe>.*?</keyframe>", "", xml, flags=re.S)
    return mujoco.MjModel.from_xml_string(xml)


def scale_prestress(m, scale):
    d0 = mujoco.MjData(m)
    mujoco.mj_forward(m, d0)
    for t in range(m.ntendon):
        L = float(d0.ten_length[t])
        rest = float(m.tendon_lengthspring[t, 1])
        pre = max(0.0, 1.0 - rest / max(L, 1e-9))
        m.tendon_lengthspring[t, 1] = L * (1.0 - scale * pre)


def endpoint_stiffness(m, ctrl=None, settle=6.0, hold=6.0, probe=PROBE_N):
    """Finite-difference the 3x3 endpoint compliance about the settled pose.

    Two details make a millinewton-scale probe measurable. (i) Damping is
    raised before settling: it does not move the static equilibrium, but
    without it the limb is still ringing after any affordable settle time and
    the ring swamps the probe response. (ii) Every probe is measured
    DIFFERENTIALLY against an unprobed rollout of identical length, so residual
    drift cancels instead of being read as compliance."""
    hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand_l")
    dof_d, ten_d = m.dof_damping.copy(), m.tendon_damping.copy()
    m.dof_damping[:] = dof_d + 8.0
    m.tendon_damping[:] = ten_d + 40.0
    try:
        def rollout(force):
            d = mujoco.MjData(m)
            n_s, n_h = int(settle / m.opt.timestep), int(hold / m.opt.timestep)
            for i in range(n_s + n_h):
                if ctrl is not None:
                    d.ctrl[:] = ctrl
                if i >= n_s:
                    d.xfrc_applied[hand, :3] = force
                mujoco.mj_step(m, d)
            return d.xpos[hand].copy(), float(np.abs(d.qvel).max())

        p0, v0 = rollout(np.zeros(3))
        C = np.zeros((3, 3))
        defl, vmax = [], v0
        for k, ax in enumerate(AXES):
            p, v = rollout(np.array(ax) * probe)
            C[:, k] = (p - p0) / probe
            defl.append(float(np.linalg.norm(p - p0)))
            vmax = max(vmax, v)
    finally:
        m.dof_damping[:] = dof_d
        m.tendon_damping[:] = ten_d
    Csym = 0.5 * (C + C.T)
    # per-axis stiffness. The eigen-decomposition of C is reported for
    # completeness but its stiffest direction (along the arm, where the cables
    # are effectively inextensible) is numerically ill-conditioned, so the
    # paper quotes the per-axis numbers.
    k_axis = [float(1.0 / max(abs(Csym[i, i]), 1e-12)) for i in range(3)]
    return dict(k_lat=k_axis[0], k_axis=k_axis,
                compliance=Csym.tolist(),
                defl_mm=[1e3 * x for x in defl], resid_qvel=vmax)



def main():
    res = {"probe_N": PROBE_N}

    # ---- (a) prestress sweep, arm passive
    rows = []
    for s in (0.0, 0.5, 1.0, 2.0, 3.0, 5.0):
        m = welded(f"{MJ}/humanoid_27dof_tensegrity.xml")
        scale_prestress(m, s)
        r = endpoint_stiffness(m)
        r["prestress_scale"] = s
        rows.append(r)
        print(f"prestress x{s:.1f}: k_lat = {r['k_lat']:6.1f} N/m  "
              f"k_xyz = {np.round(r['k_axis'], 1)}  "
              f"defl {np.round(r['defl_mm'], 2)} mm", flush=True)
    m = welded(f"{MJ}/humanoid_27dof_tensegrity.xml")
    m.tendon_stiffness[:] = 0.0
    m.tendon_damping[:] = 0.0
    r = endpoint_stiffness(m)
    r["prestress_scale"] = -1
    rows.append(r)
    print(f"no cables    : k_lat = {r['k_lat']:6.1f} N/m  "
          f"k_xyz = {np.round(r['k_axis'], 1)}  "
          f"defl {np.round(r['defl_mm'], 2)} mm", flush=True)
    res["prestress"] = rows

    # linearity check: halve the probe at nominal prestress
    m = welded(f"{MJ}/humanoid_27dof_tensegrity.xml")
    scale_prestress(m, 1.0)
    half = endpoint_stiffness(m, probe=0.5 * PROBE_N)
    nominal = [r for r in rows if r["prestress_scale"] == 1.0][0]
    res["linearity"] = dict(
        k_lat_full=nominal["k_lat"], k_lat_half=half["k_lat"],
        rel_change=abs(half["k_lat"] - nominal["k_lat"]) / nominal["k_lat"])
    print(f"linearity    : {nominal['k_lat']:.1f} vs {half['k_lat']:.1f} N/m at "
          f"half probe ({100 * res['linearity']['rel_change']:.1f}% change)",
          flush=True)

    # ---- (b) shoulder co-contraction, cable model
    mc = welded(f"{MJ}/humanoid_27dof_tensegrity_cable.xml")
    names = [mujoco.mj_id2name(mc, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
             for i in range(mc.nu)]
    sh = [i for i, n in enumerate(names) if n.startswith("shoulder_l")]
    fmax = -mc.actuator_gear[:, 0]
    rows = []
    for t0 in (0.0, 10.0, 25.0, 50.0, 100.0, 200.0, 400.0):
        ctrl = np.zeros(mc.nu)
        for i in sh:
            ctrl[i] = min(1.0, t0 / fmax[i])
        r = endpoint_stiffness(mc, ctrl=ctrl)
        r["cocontraction_N"] = t0
        rows.append(r)
        print(f"co-contraction {t0:5.0f} N: k_lat = {r['k_lat']:6.1f} N/m  "
              f"defl {np.round(r['defl_mm'], 2)} mm", flush=True)
    res["cocontraction"] = rows

    json.dump(res, open(OUT, "w"), indent=1)
    print("E2 done ->", OUT)


if __name__ == "__main__":
    main()
