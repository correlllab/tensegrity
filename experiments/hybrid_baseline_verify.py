#!/usr/bin/env python
"""Scripted verification of the hybrid walker's headline numbers.

Supersedes an earlier results/hybrid_full.json that was written without a
producing script or logs. Every condition here runs through testspeed with
logs kept, and the JSON records everything the paper quotes:

  base      full model (7-DoF arms, tensegrity waist, packs) at the
            pre-retune baseline configuration, n = 10
            -> success rate, speed, realtime factor, average running cost,
               actuator duty, commanded-torque peaks, torso attitude
  welded    ankle cable interfaces replaced by welds (E17's rigid_ankle
            variant), same configuration, n = 10 -> the A/B control the
            gait-synthesis narrative compares against, with running cost
  passive   ankle tension actuators removed, cables kept, n = 2 x 4
            configurations -> does the planner ever lift the swing foot?
  battery   one 2.55 kg battery pack removed outright, n = 5 -> gait
            robustness to an asymmetric mass change (testspeed cannot
            remove mass mid-run, so this is a from-start removal, which
            the paper must state)

Baseline configuration (the shipped task now carries the E21/E24 retune, so
every parameter is overridden explicitly): speed 0.45, cadence 1.1 Hz, step
height 0.08, sway 0.06, torso 1.20, Gait weight 25, horizon 0.42 s.
Success = min pelvis z > 0.65 m and > 1.0 m travelled in t in [3, 15] s
(the hybrid_sweeps criterion).
"""
import json
import os
import re
import subprocess
import sys
import numpy as np
import mujoco

ROOT = "/Users/ncorrell/Downloads/tensegrity"
TD = f"{ROOT}/mujoco_mpc/build/mjpc/tasks/tensegrity"
BIN = f"{ROOT}/mujoco_mpc/build/bin/testspeed"
RES = f"{ROOT}/experiments/results"
LOGD = f"{RES}/hybrid_full"
os.makedirs(LOGD, exist_ok=True)
LIVE = f"{TD}/task_hybrid_walk.xml"
sys.path.insert(0, f"{ROOT}/experiments")
import e17_joint_count as e17                            # noqa: E402

BASECFG = dict(speed=0.45, cadence=1.1, step=0.08, sway=0.06, torso=1.20,
               w_gait=25.0, horizon=0.42)


def configure(task_xml, speed, cadence, step, sway, torso, w_gait, horizon):
    x = task_xml

    def num(name, val, s):
        return re.sub(rf'(name="residual_{name}" data=")[\d.+-]+',
                      rf'\g<1>{val}', s)
    x = num("Speed", speed, x)
    x = num("Cadence", cadence, x)
    x = num("Step Height", step, x)
    x = num("Sway", sway, x)
    x = num("Torso", torso, x)
    x = re.sub(r'(<user name="Gait"\s+dim="\d+"\s+user="\d+) [\d.]+',
               rf'\g<1> {w_gait}', x)
    x = re.sub(r'(name="agent_horizon" data=")[\d.]+', rf'\g<1>{horizon}', x)
    return x


def trial(task_xml, tag, t):
    """One testspeed run; returns (log_path, realtime_factor, avg_cost)."""
    log = f"{LOGD}/log_{tag}_t{t}.csv"
    backup = open(LIVE).read()
    rt, cost = None, None
    try:
        open(LIVE, "w").write(task_xml)
        if not os.path.exists(log):
            env = dict(os.environ, TESTSPEED_LOG=log)
            out = subprocess.run([BIN, "--task=Hybrid Walk",
                                  "--total_time=16",
                                  "--steps_per_planning_iteration=4"],
                                 capture_output=True, text=True, env=env,
                                 cwd=TD).stdout
            m_rt = re.search(r"\(([\d.]+)x realtime\)", out)
            m_c = re.search(r"Average cost per step[^:]*: ([\d.eE+-]+)", out)
            rt = float(m_rt.group(1)) if m_rt else None
            cost = float(m_c.group(1)) if m_c else None
    finally:
        open(LIVE, "w").write(backup)
    return log, rt, cost


def walk_metrics(log):
    raw = np.genfromtxt(log, delimiter=",", invalid_raise=False)
    raw = raw[~np.isnan(raw).any(axis=1)]
    tt, x, z = raw[:, 0], raw[:, 1], raw[:, 3]
    w = (tt >= 3.0) & (tt <= 15.0)
    ok = bool(z[w].min() > 0.65 and (x[w][-1] - x[w][0]) > 1.0)
    return ok, float((x[w][-1] - x[w][0]) / 12.0), raw


def quat_rel_euler(q_parent, q_child):
    """Euler xyz (deg) of child relative to parent, from wxyz quaternions."""
    qp = np.zeros(4)
    mujoco.mju_negQuat(qp, q_parent)
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, qp, q_child)
    R9 = np.zeros(9)
    mujoco.mju_quat2Mat(R9, q)
    R = R9.reshape(3, 3)
    sy = np.hypot(R[0, 0], R[1, 0])
    return np.rad2deg([np.arctan2(R[2, 1], R[2, 2]),      # roll
                       np.arctan2(-R[2, 0], sy),          # pitch
                       np.arctan2(R[1, 0], R[0, 0])])     # yaw


def gait_analysis(m, logs):
    """Duty, commanded torque, and torso attitude over the success window,
    with all indexing derived from the compiled model."""
    an = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
          for i in range(m.nu)]
    leg = [i for i, n in enumerate(an) if n.startswith(("hip", "knee"))]
    ank = [i for i, n in enumerate(an) if n.startswith("ankm")]
    gear = m.actuator_gear[:, 0]
    jt = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.jnt_bodyid[j]):
          m.jnt_qposadr[j] for j in range(m.njnt)
          if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE}
    duty_leg, duty_ank, torso_rpy = [], [], []
    ank_T, leg_Nm, sat = [], [], 0
    n_samp = 0
    for log in logs:
        raw = np.genfromtxt(log, delimiter=",", invalid_raise=False)
        raw = raw[~np.isnan(raw).any(axis=1)]
        tt = raw[:, 0]
        w = (tt >= 3.0) & (tt <= 15.0)
        qpos = raw[w, 1:1 + m.nq]
        ctrl = raw[w, 1 + m.nq + m.nv:1 + m.nq + m.nv + m.nu]
        duty_leg.append(np.abs(ctrl[:, leg]).mean())
        duty_ank.append(ctrl[:, ank].mean())
        ank_T.append(np.abs(gear[ank[0]]) * ctrl[:, ank])
        leg_Nm.append(np.abs(ctrl[:, leg] * gear[leg]))
        sat += int((ctrl[:, ank] > 0.999).sum())
        n_samp += ctrl.shape[0] * len(ank)
        for row in qpos:
            qp = row[jt["pelvis"] + 3:jt["pelvis"] + 7]
            qt = row[jt["torso"] + 3:jt["torso"] + 7]
            torso_rpy.append(quat_rel_euler(qp, qt))
    ank_T = np.concatenate([a.ravel() for a in ank_T])
    leg_Nm = np.concatenate([a.ravel() for a in leg_Nm])
    rpy = np.array(torso_rpy)
    return dict(
        duty_leg_pct=float(100 * np.mean(duty_leg)),
        duty_ankle_pct=float(100 * np.mean(duty_ank)),
        ankle_T_p99=float(np.percentile(ank_T, 99)),
        ankle_T_max=float(ank_T.max()),
        ankle_sat_frac=float(sat / max(n_samp, 1)),
        leg_Nm_p99=float(np.percentile(leg_Nm, 99)),
        leg_Nm_max=float(leg_Nm.max()),
        torso_roll_max=float(np.abs(rpy[:, 0]).max()),
        torso_pitch_max=float(np.abs(rpy[:, 1]).max()),
        torso_yaw_max=float(np.abs(rpy[:, 2]).max()),
        torso_rp_max=float(np.abs(rpy[:, :2]).max()))


def foot_lift(m, raw):
    """Max swing-foot rise above its settled height, before any fall."""
    jt = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.jnt_bodyid[j]):
          m.jnt_qposadr[j] for j in range(m.njnt)
          if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE}
    tt, z = raw[:, 0], raw[:, 3]
    fell = np.where(z < 0.65)[0]
    end = fell[0] if len(fell) else len(tt)
    t_fall = float(tt[fell[0]]) if len(fell) else None
    lifts = []
    for side in ("foot_l", "foot_r"):
        fz = raw[:end, 1 + jt[side] + 2]
        if len(fz):
            lifts.append(float(fz.max() - np.percentile(fz, 10)))
    return (max(lifts) if lifts else 0.0), t_fall


def run_condition(tag, task_xml, n):
    rows, rts, costs, logs = [], [], [], []
    for t in range(n):
        log, rt, cost = trial(task_xml, tag, t)
        ok, v, _ = walk_metrics(log)
        rows.append(dict(ok=ok, v=v))
        logs.append(log)
        if rt:
            rts.append(rt)
        if cost is not None:
            costs.append(cost)
        print(f"  {tag} t{t}: {'ok  ' if ok else 'FELL'} v={v:.3f}"
              + (f" rt={rt:.2f}" if rt else "")
              + (f" cost={cost:.3f}" if cost is not None else ""), flush=True)
    ok = sum(r["ok"] for r in rows)
    v = float(np.mean([r["v"] for r in rows if r["ok"]])) if ok else 0.0
    return dict(ok=ok, n=len(rows), v=v,
                realtime=float(np.mean(rts)) if rts else None,
                realtime_n=len(rts),
                cost=float(np.mean(costs)) if costs else None,
                trials=rows), logs


def main():
    os.chdir(f"{ROOT}/mujoco")          # model asset paths are relative
    live0 = open(LIVE).read()
    res = dict(model="full: 7-DoF arms + tensegrity waist + packs",
               config=BASECFG)

    print("BASE: full model at the pre-retune baseline configuration, n=10")
    base_task = configure(live0, **BASECFG)
    res["base"], base_logs = run_condition("base", base_task, 10)
    # keys the rest of the pipeline reads (make_numbers.py)
    res.update(ok=res["base"]["ok"], n=res["base"]["n"], v=res["base"]["v"])
    m_full = mujoco.MjModel.from_xml_path(f"{ROOT}/mujoco/humanoid_hybrid.xml")
    ok_logs = [lg for lg, r in zip(base_logs, res["base"]["trials"])
               if r["ok"]]
    if ok_logs:
        res["gait"] = gait_analysis(m_full, ok_logs)
        print("  gait:", {k: round(v, 2) for k, v in res["gait"].items()},
              flush=True)

    print("WELDED ankles (E17 rigid_ankle variant), same configuration, n=10")
    mv, mpath, tpath = e17.make_variant("rigid_ankle", True, False)
    welded_task = configure(open(tpath).read(), **BASECFG)
    res["welded"], _ = run_condition("welded", welded_task, 10)

    print("PASSIVE ankles (actuators removed, cables kept), 4 cfgs x 2")
    xml = open(f"{ROOT}/mujoco/humanoid_hybrid.xml").read()
    xml = re.sub(r'<motor name="ankm_[^>]*/>\n?', '', xml)
    open(f"{TD}/humanoid_passive_ankle.xml", "w").write(xml)
    mp = mujoco.MjModel.from_xml_string(xml)
    ptask = live0.replace("humanoid_hybrid.xml", "humanoid_passive_ankle.xml")
    ptask = re.sub(r'(<user name="Control"\s+dim=")\d+', rf'\g<1>{mp.nu}',
                   ptask)
    pas = []
    for ptag, over in (("p_base", {}),
                       ("p_c13", dict(cadence=1.3)),
                       ("p_sw8", dict(sway=0.08)),
                       ("p_s30", dict(speed=0.30))):
        cfg = dict(BASECFG, **over)
        ptx = configure(ptask, **cfg)
        for t in range(2):
            log, _, _ = trial(ptx, ptag, t)
            ok, v, raw = walk_metrics(log)
            lift, t_fall = foot_lift(mp, raw)
            pas.append(dict(tag=ptag, ok=ok, v=v, max_lift_mm=1e3 * lift,
                            t_fall=t_fall))
            print(f"  {ptag} t{t}: {'ok' if ok else 'FELL'} "
                  f"lift={1e3*lift:.0f} mm fall_t={t_fall}", flush=True)
    res["passive"] = pas

    print("BATTERY: one pack removed from the start, n=5")
    xml = open(f"{ROOT}/mujoco/humanoid_hybrid.xml").read()
    a, b = e17.body_block(xml, "battery_l")
    xml = xml[:a] + xml[b:]
    open(f"{TD}/humanoid_one_battery.xml", "w").write(xml)
    mb = mujoco.MjModel.from_xml_string(xml)
    res["battery_removed_kg"] = float(m_full.body_mass.sum()
                                      - mb.body_mass.sum())
    btask = configure(live0.replace("humanoid_hybrid.xml",
                                    "humanoid_one_battery.xml"), **BASECFG)
    res["battery"], _ = run_condition("battery", btask, 5)

    json.dump(res, open(f"{RES}/hybrid_full.json", "w"), indent=1)
    print("->", f"{RES}/hybrid_full.json")


if __name__ == "__main__":
    main()
