#!/usr/bin/env python
"""E6: does the TENDON-DRIVEN robot walk?

Every walking result in the first campaign was produced on the joint-torque
model -- 27 ideal motors at the hinges. Whether the cable network can deliver
those torques was only checked offline, quasi-statically, on logged states.
This experiment closes that loop: the same reference-tracking controller drives
two plants,

    torque : humanoid_27dof_tensegrity.xml       (27 joint motors)
    cable  : humanoid_27dof_tensegrity_cable.xml (114 tension-only actuators)

and on the cable plant the desired joint torques are realised through the
bounded tension distribution

    min_{0<=x<=1} || M(q) x - tau_des ||^2 + lambda ||x - x0||^2

solved online at 250 Hz. The controller is the analytic gait reference of the
MJPC task (ported from walk.cc: lateral sway, treadmill footsteps, capture-
point landing correction, closed-form flat-foot leg IK) tracked by joint PD
with gravity/Coriolis feedforward. It is deliberately NOT iLQG: the point is a
matched comparison between the two plants, not the best possible gait.

Actuation sets on the cable plant:
    all   : all 114 cables driven independently (upper bound on the network)
    loops : only the 36 loop drives of the bill of materials

Reports achieved speed, survival, realised-vs-demanded torque error, and cable
saturation for each condition.
"""
import json
import sys
import numpy as np
import mujoco
from scipy.optimize import lsq_linear

sys.path.insert(0, "/Users/ncorrell/Downloads/tensegrity/mujoco")
from verify_humanoid27 import gains_for                        # noqa: E402

MJ = "/Users/ncorrell/Downloads/tensegrity/mujoco"
RES = "/Users/ncorrell/Downloads/tensegrity/experiments/results"
OUT = f"{RES}/e6_cable_walk.json"
TORQUE = f"{MJ}/humanoid_27dof_tensegrity.xml"
CABLE = f"{MJ}/humanoid_27dof_tensegrity_cable.xml"

# --- gait reference constants, identical to mjpc/tasks/tensegrity/walk/walk.cc
FOOT_Z = 0.076
DUTY = 0.65
CAPTURE_GAIN = 0.28
SWAY_LEAD = 0.07
ARM_SWING = 0.25
ELBOW_REF = -0.3
HIP_Y, HIP_Z = 0.09, -0.06
THIGH, SHANK = 0.40, 0.40
TORSO_ABOVE_PELVIS = 0.10
# task parameters from task_walk.xml
TORSO_REF, CADENCE, STEP_H, STANCE_W, SWAY = 0.98, 1.1, 0.05, 0.20, 0.05
CTRL_HZ = 250.0


def leg_ik(side, ankle):
    d = np.array([ankle[0], ankle[1] - side * HIP_Y, ankle[2] - HIP_Z])
    roll = np.arctan2(d[1], -d[2])
    D2 = min(float(d @ d), (THIGH + SHANK - 1e-6) ** 2)
    c = (THIGH ** 2 + SHANK ** 2 - D2) / (2 * THIGH * SHANK)
    knee = np.pi - np.arccos(np.clip(c, -1.0, 1.0))
    D = np.sqrt(D2)
    alpha = np.arctan2(d[0], np.sqrt(d[1] ** 2 + d[2] ** 2))
    cb = (THIGH ** 2 + D2 - SHANK ** 2) / (2 * THIGH * D)
    beta = np.arccos(np.clip(cb, -1.0, 1.0))
    hip_pitch = -(alpha + beta)
    return roll, hip_pitch, knee, -(hip_pitch + knee), -roll


def foot_target(phase, speed):
    stride = speed * DUTY / CADENCE
    if phase < DUTY:
        s = phase / DUTY
        return 0.5 * stride - s * stride, 0.0, 0.0
    s = (phase - DUTY) / (1.0 - DUTY)
    return -0.5 * stride + s * stride, STEP_H * np.sin(np.pi * s), s


def qref_at(t, speed, v_fwd, v_lat, pelvis_z_ref):
    phase = np.mod(t * CADENCE, 1.0)
    y_ref = SWAY * np.sin(2 * np.pi * (phase - SWAY_LEAD))
    t_stance = DUTY / CADENCE
    corr_f = np.clip(0.5 * (v_fwd - speed) * t_stance
                     + CAPTURE_GAIN * (v_fwd - speed), -0.15, 0.15)
    corr_l = np.clip(0.5 * v_lat * t_stance + CAPTURE_GAIN * v_lat, -0.10, 0.10)
    q = np.zeros(27)
    for f, (off, side) in enumerate(((0.0, +1.0), (0.5, -1.0))):
        ph = np.mod(phase + off, 1.0)
        fwd, lift, swing = foot_target(ph, speed)
        fwd += swing * corr_f
        lane = side * 0.5 * STANCE_W + swing * corr_l
        ankle = (fwd, lane - y_ref, FOOT_Z + lift - pelvis_z_ref)
        q[6 * f + 1:6 * f + 6] = leg_ik(side, ankle)
    s_arm = np.sin(2 * np.pi * phase)
    q[17], q[20] = -ARM_SWING * s_arm, ELBOW_REF
    q[22], q[25] = +ARM_SWING * s_arm, ELBOW_REF
    return q, y_ref


def run(plant, speed=0.30, seconds=16.0, driven=None, seed=0):
    """plant in {'torque','cable'}; `driven` restricts the cable columns."""
    path = TORQUE if plant == "torque" else CABLE
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    mt = mujoco.MjModel.from_xml_path(TORQUE)
    jn = [mujoco.mj_id2name(mt, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
          for i in range(mt.nu)]
    lim = np.abs(mt.actuator_gear[:, 0])
    kp = np.array([gains_for(n)[0] for n in jn])
    kd = np.array([gains_for(n)[1] for n in jn])

    jids = mt.actuator_trnid[:, 0]
    qadr = mt.jnt_qposadr[jids]
    vadr = mt.jnt_dofadr[jids]
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "walk_ready")
    mujoco.mj_resetDataKeyframe(m, d, key)
    pelvis_z_ref = TORSO_REF - TORSO_ABOVE_PELVIS

    if plant == "cable":
        fmax = -m.actuator_gear[:, 0]
        cols = np.array(driven if driven is not None else range(m.nu))
        reg = np.sqrt(0.05) * np.eye(len(cols))
        x_prev = np.zeros(len(cols))

    decim = max(1, int(round(1.0 / (CTRL_HZ * m.opt.timestep))))
    n = int(seconds / m.opt.timestep)
    x0 = float(d.qpos[0])
    tau_des = np.zeros(mt.nu)
    err, sat, peakT, zmin, traj = [], [], 0.0, 9.9, []
    for step in range(n):
        if step % decim == 0:
            v = d.qvel[:3].copy()
            q, _ = qref_at(d.time, speed, float(v[0]), float(v[1]),
                           pelvis_z_ref)
            tau_des = np.clip(kp * (q - d.qpos[qadr]) - kd * d.qvel[vadr]
                              + d.qfrc_bias[vadr], -lim, lim)
            if plant == "cable":
                mom = np.zeros((m.nu, m.nv))
                mujoco.mju_sparse2dense(mom, d.actuator_moment,
                                        d.moment_rownnz, d.moment_rowadr,
                                        d.moment_colind)
                M = mom[:, 6:].T[:, cols]
                sol = lsq_linear(np.vstack([M, reg]),
                                 np.concatenate([tau_des,
                                                 np.sqrt(0.05) * x_prev]),
                                 bounds=(0.0, 1.0), max_iter=20, tol=1e-6)
                x_prev = sol.x
                d.ctrl[:] = 0.0
                d.ctrl[cols] = sol.x
                err.append(float(np.abs(M @ sol.x - tau_des).max()))
                sat.append(float((sol.x > 0.99).mean()))
                peakT = max(peakT, float((fmax[cols] * sol.x).max()))
            else:
                d.ctrl[:] = tau_des / mt.actuator_gear[:, 0]
        mujoco.mj_step(m, d)
        if d.time >= 3.0:
            zmin = min(zmin, float(d.qpos[2]))
        traj.append((float(d.time), float(d.qpos[0]), float(d.qpos[2])))
    tr = np.array(traj)
    w = tr[:, 0] >= 3.0
    speed_ach = ((tr[w][-1, 1] - tr[w][0, 1]) / (tr[w][-1, 0] - tr[w][0, 0]))
    return dict(plant=plant, driven=("all" if driven is None else "loops"),
                speed_cmd=speed, speed_ach=float(speed_ach),
                dist=float(tr[-1, 1] - x0), min_z=zmin,
                success=bool(zmin > 0.55 and speed_ach > 0.02),
                torque_err_p95=float(np.percentile(err, 95)) if err else 0.0,
                torque_err_max=float(np.max(err)) if err else 0.0,
                sat_frac=float(np.mean(sat)) if sat else 0.0,
                peak_tension=peakT)


def main():
    auth = json.load(open(f"{RES}/e5_authority.json"))
    mc = mujoco.MjModel.from_xml_path(CABLE)
    cn = [mujoco.mj_id2name(mc, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
          for i in range(mc.nu)]
    loops = [cn.index(x) for x in auth["driven_home"]]

    rows = []
    for speed in (0.20, 0.30):
        rows.append(run("torque", speed))
        rows.append(run("cable", speed))
        rows.append(run("cable", speed, driven=loops))
    for r in rows:
        print(f"{r['plant']:6s}/{r['driven']:5s} cmd {r['speed_cmd']:.2f} -> "
              f"{r['speed_ach']:+.3f} m/s  minz {r['min_z']:.2f}  "
              f"{'OK' if r['success'] else 'FELL'}  "
              f"tau_err p95 {r['torque_err_p95']:5.1f} Nm  "
              f"sat {r['sat_frac']*100:4.1f}%  peakT {r['peak_tension']:.0f} N",
              flush=True)
    json.dump(rows, open(OUT, "w"), indent=1)
    print("E6 ->", OUT)


if __name__ == "__main__":
    main()
