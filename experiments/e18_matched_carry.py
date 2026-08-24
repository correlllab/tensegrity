#!/usr/bin/env python
"""E18: single-arm static carry under ONE protocol for both machines.

Protocol (identical for the fully hinged 27-DoF machine and the hybrid):
  - pelvis welded to the world (the E2 methodology)
  - all hinge joints PD-held (Kp 300, Kd 15) at the carry pose:
    shoulder_pitch_l = -0.25, elbow_l = -1.30, everything else home
  - settle 1.5 s unloaded, record the hand position
  - apply -9.81*m N at the hand for 3 s
  - failure = hand drop from the unloaded settled position; report the
    max held mass at 8, 12 and 16 cm thresholds (threshold sensitivity)

The hybrid additionally holds its tension actuators at a light 0.06
co-contraction, its standing convention.
"""
import json
import os
import numpy as np
import mujoco

ROOT = "/Users/ncorrell/Downloads/tensegrity"
MJ = f"{ROOT}/mujoco"
OUT = f"{ROOT}/experiments/results/e18_matched_carry.json"
CARRY = {"shoulder_pitch_l": -0.25, "elbow_l": -1.30}
MASSES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
THRESH = [0.08, 0.12, 0.16]


def load_welded(path):
    import re
    xml = open(path).read().replace('<freejoint name="root"/>', '')
    xml = re.sub(r"<keyframe>.*?</keyframe>", "", xml, flags=re.S)
    return mujoco.MjModel.from_xml_string(xml)


def run_machine(tag, path, n_hinge):
    os.chdir(MJ)
    out = []
    for m_kg in MASSES:
        m = load_welded(path)
        d = mujoco.MjData(m)
        jid = m.actuator_trnid[:n_hinge, 0]
        qadr, vadr = m.jnt_qposadr[jid], m.jnt_dofadr[jid]
        gear = m.actuator_gear[:, 0]
        jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(j))
              for j in jid]
        mujoco.mj_forward(m, d)
        q0 = d.qpos[qadr].copy()
        for k, nm in enumerate(jn):
            if nm in CARRY:
                q0[k] = CARRY[nm]
        hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand_l")
        an = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
              for i in range(m.nu)]
        shp = an.index("shoulder_pitch_l")

        def pd_step(load):
            tau = 300.0 * (q0 - d.qpos[qadr]) - 15.0 * d.qvel[vadr]
            d.ctrl[:n_hinge] = np.clip(tau / gear[:n_hinge], -1, 1)
            if m.nu > n_hinge:
                d.ctrl[n_hinge:] = 0.06
            d.xfrc_applied[hand, 2] = load
            mujoco.mj_step(m, d)

        for _ in range(int(1.5 / m.opt.timestep)):
            pd_step(0.0)
        p0 = d.xpos[hand].copy()
        tau_pk = 0.0
        for _ in range(int(3.0 / m.opt.timestep)):
            pd_step(-9.81 * m_kg)
            tau_pk = max(tau_pk, abs(float(d.ctrl[shp] * gear[shp])))
        drop = float(np.linalg.norm(d.xpos[hand] - p0))
        lim = abs(float(gear[shp]))
        out.append(dict(mass=m_kg, drop_m=drop, shoulder_Nm=tau_pk,
                        shoulder_util=tau_pk / lim,
                        finite=bool(np.all(np.isfinite(d.qpos)))))
        print(f"  {tag} {m_kg:.1f} kg: drop {1e3*drop:5.0f} mm  "
              f"shoulder {tau_pk:5.1f}/{lim:.0f} Nm", flush=True)
    return out


def main():
    res = {}
    print("hybrid (XM540 shoulder, 15 Nm)")
    res["hybrid"] = run_machine("hyb", f"{MJ}/humanoid_hybrid.xml", 12)
    print("fully hinged (30 Nm shoulder)")
    res["hinged"] = run_machine("hin", f"{MJ}/humanoid_27dof_tensegrity.xml",
                                27)
    for tag, rows in res.items():
        for th in THRESH:
            held = [r["mass"] for r in rows if r["drop_m"] < th and
                    r["finite"]]
            print(f"{tag}: max hold at {100*th:.0f} cm threshold = "
                  f"{max(held) if held else 0:.1f} kg")
    json.dump(res, open(OUT, "w"), indent=1)
    print("E18 ->", OUT)


if __name__ == "__main__":
    main()
