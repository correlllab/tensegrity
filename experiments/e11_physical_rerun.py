#!/usr/bin/env python
"""E11: do the conclusions survive a physically stiff cable network?

The campaign was run with cables modelled at 600-2500 N/m. A rope of the
specified rating, or the series-elastic element the transmission study
assumes, is 20-1700x stiffer, and at those stiffnesses the network carries a
different share of the load. This re-runs the two experiments whose
conclusions could plausibly depend on it -- E1 (impact) and E2 (stiffness) --
on the same model with cables replaced by a series-elastic element per group
and prestress specified as a TENSION rather than a strain fraction.

A 4% strain prestress is meaningless at rope stiffness (it is past breaking
load), which is itself the reason the original model used soft springs.
"""
import json
import os
import re
import sys
import numpy as np
import mujoco

sys.path.insert(0, "/Users/ncorrell/Downloads/tensegrity/experiments")
from e9_hinge_audit import physical_cables, K_SER, T0   # noqa: E402

MJ = "/Users/ncorrell/Downloads/tensegrity/mujoco"
RES = "/Users/ncorrell/Downloads/tensegrity/experiments/results"
OUT = f"{RES}/e11_physical.json"
TORQUE = f"{MJ}/humanoid_27dof_tensegrity.xml"
HEIGHTS = (0.05, 0.10, 0.20, 0.30, 0.50, 1.00)
DT = 1e-4                       # stiff cables need a fine step


def load(rigid=False, physical=True, pre_scale=1.0):
    m = mujoco.MjModel.from_xml_path(TORQUE)
    if physical:
        physical_cables(m, scale=pre_scale)
    if rigid:
        m.tendon_stiffness[:] = 0.0
        m.tendon_damping[:] = 0.0
    m.opt.timestep = DT
    return m


def drop(m, h, held):
    d = mujoco.MjData(m)
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "walk_ready")
    mujoco.mj_resetDataKeyframe(m, d, key)
    d.qpos[2] += h
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    jids = m.actuator_trnid[:, 0]
    qadr, vadr = m.jnt_qposadr[jids], m.jnt_dofadr[jids]
    gear = m.actuator_gear[:, 0]
    qref = d.qpos[qadr].copy()
    peak, f6, t_contact = 0.0, np.zeros(6), None
    for _ in range(int(1.2 / m.opt.timestep)):
        if held:
            tau = -400.0 * (d.qpos[qadr] - qref) - 15.0 * d.qvel[vadr]
            d.ctrl[:] = np.clip(tau / gear, -1, 1)
        mujoco.mj_step(m, d)
        if not np.all(np.isfinite(d.qpos)):
            return float("nan")
        grf = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            if floor in (c.geom[0], c.geom[1]):
                mujoco.mj_contactForce(m, d, i, f6)
                grf += abs(float(f6[0]))
        if grf > 1.0 and t_contact is None:
            t_contact = d.time
        if t_contact is not None and d.time <= t_contact + 0.25:
            peak = max(peak, grf)
    return peak


def endpoint_stiffness(m, probe=2.0, settle=3.0, hold=3.0):
    """Same differential, damped protocol as E2, with a probe scaled to the
    stiffer network."""
    xml = open(TORQUE).read().replace('<freejoint name="root"/>', "")
    xml = re.sub(r"<keyframe>.*?</keyframe>", "", xml, flags=re.S)
    cwd = os.getcwd()
    os.chdir(MJ)                     # meshdir is relative to the model file
    try:
        mw = mujoco.MjModel.from_xml_string(xml)
    finally:
        os.chdir(cwd)
    physical_cables(mw)
    mw.opt.timestep = DT
    mw.tendon_stiffness[:] = m.tendon_stiffness
    mw.tendon_lengthspring[:] = m.tendon_lengthspring
    mw.dof_damping[:] = mw.dof_damping + 8.0
    mw.tendon_damping[:] = mw.tendon_damping + 400.0
    hand = mujoco.mj_name2id(mw, mujoco.mjtObj.mjOBJ_BODY, "hand_l")

    def rollout(force):
        d = mujoco.MjData(mw)
        ns, nh = int(settle / mw.opt.timestep), int(hold / mw.opt.timestep)
        for i in range(ns + nh):
            if i >= ns:
                d.xfrc_applied[hand, :3] = force
            mujoco.mj_step(mw, d)
        return d.xpos[hand].copy()

    p0 = rollout(np.zeros(3))
    p1 = rollout(np.array([probe, 0, 0]))
    dx = float(abs((p1 - p0)[0]))
    return probe / max(dx, 1e-12), 1e3 * dx


def main():
    res = {}

    print("E1 AT PHYSICAL CABLE STIFFNESS (matched pairs, peak GRF in N)")
    print(f"{'h (m)':>6s} {'tsg limp':>9s} {'rigid limp':>11s} {'delta':>7s} "
          f"{'tsg held':>9s} {'rigid held':>11s} {'delta':>7s}")
    rows = []
    for h in HEIGHTS:
        a = drop(load(), h, held=False)
        b = drop(load(rigid=True), h, held=False)
        c = drop(load(), h, held=True)
        e = drop(load(rigid=True), h, held=True)
        da = 100 * (a / b - 1) if b else float("nan")
        dh = 100 * (c / e - 1) if e else float("nan")
        print(f"{h:6.2f} {a:8.0f}N {b:10.0f}N {da:+6.0f}% "
              f"{c:8.0f}N {e:10.0f}N {dh:+6.0f}%")
        rows.append(dict(h=h, tsg_limp=a, rigid_limp=b, tsg_held=c,
                         rigid_held=e, delta_limp=da, delta_held=dh))
    res["e1"] = rows

    print("\nE2 AT PHYSICAL CABLE STIFFNESS (lateral endpoint stiffness)")
    print(f"{'prestress':>10s} {'k_lat (N/m)':>12s} {'defl (mm)':>10s}")
    pres = []
    for s in (0.0, 0.5, 1.0, 2.0, 4.0):
        m = load(pre_scale=s)
        k, dx = endpoint_stiffness(m)
        print(f"{s:9.1f}x {k:12.1f} {dx:10.3f}")
        pres.append(dict(scale=s, k_lat=k, defl_mm=dx))
    m = load(physical=True)
    m.tendon_stiffness[:] = 0.0
    m.tendon_damping[:] = 0.0
    k0, dx0 = endpoint_stiffness(m)
    print(f"{'no cables':>10s} {k0:12.1f} {dx0:10.3f}")
    pres.append(dict(scale=-1, k_lat=k0, defl_mm=dx0))
    res["e2"] = pres

    ks = [r["k_lat"] for r in pres if r["scale"] >= 0]
    res["e1_delta_limp_stumble"] = rows[1]["delta_limp"]
    res["e2_prestress_ratio"] = max(ks) / max(min(ks), 1e-9)
    res["e2_cable_ratio"] = min(ks) / max(k0, 1e-9)
    print(f"\nprestress 0-4x moves stiffness "
          f"{min(ks):.1f} -> {max(ks):.1f} N/m "
          f"({res['e2_prestress_ratio']:.2f}x); cables present vs removed "
          f"= {res['e2_cable_ratio']:.1f}x")
    json.dump(res, open(OUT, "w"), indent=1)
    print("E11 ->", OUT)


if __name__ == "__main__":
    main()
