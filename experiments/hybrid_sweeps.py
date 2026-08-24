#!/usr/bin/env python
"""Re-run the Fig. 7 capability sweeps on the hybrid platform.

  (a) speed envelope   commanded 0.10-0.60 m/s, n=6 per point
  (b) trunk payload    box welded to the torso (riding the tensegrity waist),
                       2-20 kg, n=4 per point
  (c) arm carry        static hold with the 7-DoF arm (XM540 shoulder), mass
                       swept until the arm drops the pose; also records what
                       the carry does to the waist cables -- a load path that
                       did not exist on the fully hinged machine

(a) and (b) go through testspeed with the shipped controller configuration;
(c) is a direct simulation with the arm joints PD-held at their limits.
"""
import json
import os
import re
import subprocess
import numpy as np
import mujoco

ROOT = "/Users/ncorrell/Downloads/tensegrity"
MJ = f"{ROOT}/mujoco"
TD = f"{ROOT}/mujoco_mpc/build/mjpc/tasks/tensegrity"
BIN = f"{ROOT}/mujoco_mpc/build/bin/testspeed"
RES = f"{ROOT}/experiments/results"
LOGD = f"{RES}/hybrid_sweeps"
os.makedirs(LOGD, exist_ok=True)

SPEEDS = [0.10, 0.20, 0.30, 0.45, 0.60]
MASSES = [2, 5, 8, 12, 16, 20]
N_SPEED = 6
N_PAY = {2: 10, 5: 10, 8: 10, 12: 10, 16: 4, 20: 4}
LIVE = f"{TD}/task_hybrid_walk.xml"


def payload_model(mass):
    xml = open(f"{TD}/humanoid_hybrid.xml").read()
    xml = xml.replace('<body name="head"', f'''<body name="payload" pos="0 0 0.06">
        <geom name="payload_box" type="box" size="0.07 0.10 0.07"
              mass="{mass}" rgba="0.72 0.55 0.30 1" contype="0"
              conaffinity="0"/>
      </body>
      <body name="head"''', 1)
    path = f"{TD}/humanoid_pay_{mass:02d}.xml"
    open(path, "w").write(xml)
    return f"humanoid_pay_{mass:02d}.xml"


def run(tag, task_xml, n):
    backup = open(LIVE).read()
    rows = []
    try:
        open(LIVE, "w").write(task_xml)
        for t in range(n):
            log = f"{LOGD}/log_{tag}_t{t}.csv"
            if not os.path.exists(log):
                env = dict(os.environ, TESTSPEED_LOG=log)
                subprocess.run([BIN, "--task=Hybrid Walk",
                                "--total_time=16",
                                "--steps_per_planning_iteration=4"],
                               capture_output=True, text=True, env=env,
                               cwd=TD)
            try:
                raw = np.genfromtxt(log, delimiter=",", invalid_raise=False)
                raw = raw[~np.isnan(raw).any(axis=1)]
                tt, x, z = raw[:, 0], raw[:, 1], raw[:, 3]
                w = (tt >= 3.0) & (tt <= 15.0)
                ok = bool(z[w].min() > 0.65 and (x[w][-1] - x[w][0]) > 1.0)
                rows.append(dict(ok=ok, v=(x[w][-1] - x[w][0]) / 12.0))
            except Exception:
                rows.append(dict(ok=False, v=0.0))
            print(f"  {tag} t{t}: {'ok' if rows[-1]['ok'] else 'FELL'} "
                  f"v={rows[-1]['v']:.3f}", flush=True)
    finally:
        open(LIVE, "w").write(backup)
    ok = sum(r["ok"] for r in rows)
    v = float(np.mean([r["v"] for r in rows if r["ok"]])) if ok else 0.0
    return dict(ok=ok, n=len(rows), v=v)


def arm_carry():
    """Static hold: PD all hinge joints at home, gravity load at the hand,
    sweep the load. Reports shoulder-pitch utilisation and the busiest waist
    cable -- the arm's load path runs through the tensegrity waist."""
    os.chdir(MJ)
    # weld the pelvis (the paper's E2 methodology): the question is arm and
    # waist strength, not standing balance. The torso still rides the waist
    # cables, so the carry's load path through them is preserved.
    xml = open("humanoid_hybrid.xml").read().replace(
        '<freejoint name="root"/>', '')
    out = []
    for m_kg in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]:
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
              for i in range(m.nu)]
        nj = 12
        jid = m.actuator_trnid[:nj, 0]
        qadr, vadr = m.jnt_qposadr[jid], m.jnt_dofadr[jid]
        gear = m.actuator_gear[:, 0]
        mujoco.mj_forward(m, d)
        q0 = d.qpos[qadr].copy()
        # carry pose, matching the original experiment: forearm forward
        jnames = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(j))
                  for j in jid]
        for k, nm in enumerate(jnames):
            if nm == "shoulder_pitch_l":
                q0[k] = -0.25
            if nm == "elbow_l":
                q0[k] = -1.30
        hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand_l")
        # settle into the carry pose unloaded first, then measure from there
        for _ in range(int(1.5 / m.opt.timestep)):
            tau = 300.0 * (q0 - d.qpos[qadr]) - 15.0 * d.qvel[vadr]
            d.ctrl[:nj] = np.clip(tau / gear[:nj], -1, 1)
            d.ctrl[nj:] = 0.06
            mujoco.mj_step(m, d)
        p0 = d.xpos[hand].copy()
        shp = jn.index("shoulder_pitch_l")
        waist_t = [t for t in range(m.ntendon)
                   if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TENDON, t)
                       or "").startswith("waist_")]
        tau_pk, T_pk = 0.0, 0.0
        for _ in range(int(3.0 / m.opt.timestep)):
            tau = 300.0 * (q0 - d.qpos[qadr]) - 15.0 * d.qvel[vadr]
            d.ctrl[:nj] = np.clip(tau / gear[:nj], -1, 1)
            d.ctrl[nj:] = 0.06               # light ankle/waist co-contraction
            d.xfrc_applied[hand, 2] = -9.81 * m_kg
            mujoco.mj_step(m, d)
            tau_pk = max(tau_pk, abs(float(d.ctrl[shp] * gear[shp])))
            T = [m.tendon_stiffness[t] * max(0.0, float(
                d.ten_length[t]) - m.tendon_lengthspring[t, 1])
                for t in waist_t]
            T_pk = max(T_pk, max(T))
        drop = float(np.linalg.norm(d.xpos[hand] - p0))
        held = bool(drop < 0.12 and np.all(np.isfinite(d.qpos)))
        out.append(dict(mass=m_kg, held=held, drop_m=drop,
                        shoulder_Nm=tau_pk,
                        shoulder_util=tau_pk / 15.0,
                        waist_T=T_pk))
        print(f"  carry {m_kg:.1f} kg: {'held' if held else 'DROPPED'} "
              f"(drop {1e3*drop:.0f} mm, shoulder {tau_pk:.1f}/15 Nm, "
              f"waist cable {T_pk:.0f} N)", flush=True)
        if not held:
            break
    return out


def main():
    out = {}
    base = open(LIVE).read()

    print("(a) speed envelope")
    out["speed"] = {}
    for sp in SPEEDS:
        xml = re.sub(r'(name="residual_Speed" data=")[\d.+-]+',
                     rf'\g<1>{sp}', base)
        out["speed"][str(sp)] = run(f"spd{int(sp*100):02d}", xml, N_SPEED)
        print(f"speed {sp:.2f}: {out['speed'][str(sp)]['ok']}/{N_SPEED} at "
              f"{out['speed'][str(sp)]['v']:.3f} m/s", flush=True)

    print("(b) trunk payload")
    out["payload"] = {}
    for mkg in MASSES:
        mdl = payload_model(mkg)
        xml = base.replace("humanoid_hybrid.xml", mdl)
        out["payload"][str(mkg)] = run(f"pay{mkg:02d}", xml, N_PAY[mkg])
        print(f"payload {mkg} kg: {out['payload'][str(mkg)]['ok']}/{N_PAY[mkg]} "
              f"at {out['payload'][str(mkg)]['v']:.3f} m/s", flush=True)

    print("(c) static arm carry")
    out["carry"] = arm_carry()

    json.dump(out, open(f"{RES}/hybrid_sweeps.json", "w"), indent=1)
    print("SWEEPS DONE ->", f"{RES}/hybrid_sweeps.json")


if __name__ == "__main__":
    main()
