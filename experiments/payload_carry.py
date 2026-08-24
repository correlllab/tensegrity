#!/usr/bin/env python
"""Payload II: single-arm static carry. Box welded under the left hand, arm
PD-held in a carry pose (elbow flexed to horizontal forearm), fixed base.
Sweep box mass; record settled arm torques against limits and the tension
mapping on the cable model. Max carry = heaviest box with pose held and all
constraints met.
"""
import json
import os
import re
import numpy as np
import mujoco
from scipy.optimize import lsq_linear

MJ = "/Users/ncorrell/Downloads/tensegrity/mujoco"
OUT = "/Users/ncorrell/Downloads/tensegrity/experiments/results/payload_carry.json"
os.chdir(MJ)
CARRY = {"shoulder_pitch_l": -0.35, "elbow_l": -1.45}
KP, KD = 300.0, 12.0
MASSES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]


def build(path, mass):
    xml = open(path).read().replace('<freejoint name="root"/>', "")
    xml = re.sub(r"<keyframe>.*?</keyframe>", "", xml, flags=re.S)
    box = (f'<body name="carrybox" pos="0 0 -0.14">'
           f'<geom name="carrybox_g" type="box" size="0.06 0.08 0.06" '
           f'mass="{mass}" rgba="0.72 0.55 0.30 1" contype="0" conaffinity="0"/>'
           f'</body>')
    xml = xml.replace('<site name="ee_l" pos="0 0 -0.08" size="0.01"/>',
                      '<site name="ee_l" pos="0 0 -0.08" size="0.01"/>' + box)
    return mujoco.MjModel.from_xml_string(xml)


def carry(mass):
    m = build(f"{MJ}/humanoid_27dof_tensegrity.xml", mass)
    d = mujoco.MjData(m)
    jids = m.actuator_trnid[:, 0]
    qadr, vadr = m.jnt_qposadr[jids], m.jnt_dofadr[jids]
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
             for i in range(m.nu)]
    gear = m.actuator_gear[:, 0]
    qref = np.zeros(m.nu)
    for n, v in CARRY.items():
        qref[names.index(n)] = v
    hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand_l")
    # nominal hand position: settle without payload influence first is moot;
    # measure sag as drift over the last second instead
    tau_hold = None
    for step in range(int(6.0 / m.opt.timestep)):
        tau = -KP * (d.qpos[qadr] - qref) - KD * d.qvel[vadr]
        d.ctrl[:] = np.clip(tau / gear, -1, 1)
        mujoco.mj_step(m, d)
        if step == int(5.0 / m.opt.timestep):
            p_1s = d.xpos[hand].copy()
    sag = float(np.linalg.norm(d.xpos[hand] - p_1s))
    q_err = float(np.abs(d.qpos[qadr] - qref)[
        [names.index(n) for n in CARRY]].max())
    tau_hold = d.ctrl * gear
    arm = {n: float(abs(tau_hold[i])) for i, n in enumerate(names)
           if n.endswith("_l") and ("shoulder" in n or "elbow" in n or "wrist" in n)}
    lims = {n: abs(float(gear[i])) for i, n in enumerate(names)}
    util = max(arm[n] / lims[n] for n in arm)
    held = bool(q_err < 0.35 and sag < 0.02)

    # tension mapping at the settled pose on the cable model
    mc = build(f"{MJ}/humanoid_27dof_tensegrity_cable.xml", mass)
    dc = mujoco.MjData(mc)
    dc.qpos[:] = d.qpos
    mujoco.mj_forward(mc, dc)
    mom = np.zeros((mc.nu, mc.nv))
    mujoco.mju_sparse2dense(mom, dc.actuator_moment, dc.moment_rownnz,
                            dc.moment_rowadr, dc.moment_colind)
    M = mom.T                                   # welded base: nv == 27
    sol = lsq_linear(np.vstack([M, 0.05 * np.eye(mc.nu)]),
                     np.concatenate([tau_hold, np.zeros(mc.nu)]),
                     bounds=(0.0, 1.0), max_iter=40, tol=1e-7)
    fmax = -mc.actuator_gear[:, 0]
    tension = fmax * sol.x
    resid = float(np.abs(M @ sol.x - tau_hold).max())
    cn = [mujoco.mj_id2name(mc, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
          for i in range(mc.nu)]
    sh_T = max(tension[i] for i, n in enumerate(cn) if n.startswith("shoulder"))
    return dict(mass=mass, held=held, sag_mm=sag * 1000, torque_util=util,
                arm_torques=arm, peak_shoulder_T=float(sh_T),
                tension_resid=resid)


def main():
    rows = []
    print(f"{'kg':>4s} {'held':>5s} {'util':>6s} {'worst arm torque':>30s} "
          f"{'shoulder T':>10s} {'resid':>7s}")
    for mass in MASSES:
        r = carry(mass)
        rows.append(r)
        worst = max(r["arm_torques"], key=lambda n: r["arm_torques"][n])
        print(f"{mass:4.0f} {'y' if r['held'] else 'N':>5s} "
              f"{r['torque_util']:6.2f} "
              f"{worst + ' ' + format(r['arm_torques'][worst], '.1f') + ' Nm':>30s} "
              f"{r['peak_shoulder_T']:9.0f}N {r['tension_resid']:7.2f}",
              flush=True)
    json.dump(rows, open(OUT, "w"), indent=1)
    print("carry done ->", OUT)


if __name__ == "__main__":
    main()
