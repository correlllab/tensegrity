#!/usr/bin/env python
"""Balance verification for the cable-driven model (humanoid_27dof_tensegrity_cable.xml).

Control law: the same joint-space PD + CoM-feedback ankle strategy as
verify_humanoid27.py computes desired joint torques tau (27), which are then
distributed to the 51 cable tensions by bounded least squares each control
tick:

    min || A t - tau ||^2 + lambda || t - t_cocon ||^2,   0 <= t <= F_max

where A[dof, cable] = -dL/dq (tension pulls along the negative length
gradient) from MuJoCo's tendon Jacobian, and t_cocon is a small co-contraction
floor that keeps cables taut. ctrl_j = t_j / F_max_j (actuator gear is -F_max).

This is the existence proof that pure tension control can balance the robot —
MJPC replaces the outer PD, not this distribution geometry.
"""
import argparse
import numpy as np
import mujoco
from scipy.optimize import lsq_linear

from verify_humanoid27 import gains_for, remove_battery

CTRL_DECIM = 1          # tension solve at the full 250 Hz physics rate: the
                        # joint PD's ~15 Hz natural frequency is unstable
                        # through a 62.5 Hz zero-order hold (bounce-and-fall);
                        # real spool drives close tension loops faster still
LAMBDA = 0.1            # regularization toward co-contraction floor (ctrl units)
T_COCON = 20.0          # N, co-contraction floor target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--push-fwd", type=float, default=60.0)
    ap.add_argument("--push-lat", type=float, default=40.0)
    ap.add_argument("--model", default="humanoid_27dof_tensegrity_cable.xml")
    ap.add_argument("--remove-battery", default=None,
                    choices=("battery_l", "battery_r"),
                    help="hot-swap case: run with one pack pulled out")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.model)
    if args.remove_battery:
        remove_battery(model, args.remove_battery)
    data = mujoco.MjData(model)
    assert model.nu >= 27, f"expected cable actuators, got {model.nu}"

    # cable actuators -> tendon ids and max tensions
    tid = model.actuator_trnid[:, 0]
    fmax = -model.actuator_gear[:, 0]
    assert np.all(fmax > 0), "cable gears must be negative (tension-only)"

    # joint-space PD setup over all 27 hinges (by joint, not actuator)
    hinge_j = [j for j in range(model.njnt)
               if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    qadr = model.jnt_qposadr[hinge_j]
    vadr = model.jnt_dofadr[hinge_j]
    jnames = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
              for j in hinge_j]
    kp = np.array([gains_for(n)[0] for n in jnames])
    kd = np.array([gains_for(n)[1] for n in jnames])

    idx = {n: i for i, n in enumerate(jnames)}
    ankle_p = [idx["ankle_pitch_l"], idx["ankle_pitch_r"]]
    ankle_r = [idx["ankle_roll_l"], idx["ankle_roll_r"]]
    sfl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "foot_l_site")
    sfr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "foot_r_site")
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    KC_P, KC_D = 400.0, 150.0

    mujoco.mj_resetDataKeyframe(model, data, 0)
    qref = data.qpos[qadr].copy()
    mujoco.mj_forward(model, data)
    com_prev = data.subtree_com[0].copy()

    # solve in normalized ctrl space: actuator_moment already includes gear,
    # so tau = M x with x = ctrl in [0,1]; tension = fmax * x
    x0 = T_COCON / fmax                      # co-contraction floor in ctrl units
    reg = np.sqrt(LAMBDA) * np.eye(model.nu)
    x_prev = x0.copy()

    z0 = data.qpos[2]
    min_z = z0
    peak_t = np.zeros(model.nu)
    n_steps = int(args.duration / model.opt.timestep)
    for step in range(n_steps):
        t = step * model.opt.timestep
        com = data.subtree_com[0]
        com_v = (com - com_prev) / model.opt.timestep
        com_prev = com.copy()

        if step % CTRL_DECIM == 0:
            sup = 0.5 * (data.site_xpos[sfl] + data.site_xpos[sfr])
            tau = -kp * (data.qpos[qadr] - qref) - kd * data.qvel[vadr]
            fb_x = KC_P * (com[0] - sup[0]) + KC_D * com_v[0]
            fb_y = KC_P * (com[1] - sup[1]) + KC_D * com_v[1]
            for i in ankle_p:
                tau[i] += fb_x
            for i in ankle_r:
                tau[i] -= fb_y

            # tension distribution: M x = tau, 0 <= x <= 1
            moment = np.zeros((model.nu, model.nv))
            mujoco.mju_sparse2dense(moment, data.actuator_moment,
                                    data.moment_rownnz, data.moment_rowadr,
                                    data.moment_colind)
            M = moment[:, 6:].T                    # (27, 51), gear included
            stacked_A = np.vstack([M, reg])
            stacked_b = np.concatenate([tau, np.sqrt(LAMBDA) * x0])
            sol = lsq_linear(stacked_A, stacked_b, bounds=(0.0, 1.0),
                             max_iter=30, tol=1e-6)
            x_prev = sol.x
        data.ctrl[:] = x_prev
        peak_t = np.maximum(peak_t, fmax * x_prev)

        data.xfrc_applied[pelvis, 0] = args.push_fwd if 2.0 <= t < 2.2 else 0.0
        data.xfrc_applied[pelvis, 1] = args.push_lat if 5.0 <= t < 5.2 else 0.0
        mujoco.mj_step(model, data)
        min_z = min(min_z, data.qpos[2])
        if data.qpos[2] < 0.4:
            break

    print(f"cable-driven stand: initial z={z0:.3f}  min z={min_z:.3f}  "
          f"final z={data.qpos[2]:.3f}")
    print(f"pushes: {args.push_fwd:.0f} N fwd @2s, {args.push_lat:.0f} N lat @5s")
    anames = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
              for i in range(model.nu)]
    order = np.argsort(-peak_t / fmax)
    print("peak cable tension utilization:")
    for i in order[:8]:
        print(f"  {anames[i]:12s} {peak_t[i]:7.0f} / {fmax[i]:.0f} N "
              f"({100 * peak_t[i] / fmax[i]:3.0f}%)")

    ok = data.qpos[2] > 0.85 and min_z > 0.80
    print("\nRESULT:", "PASS — balances on cable tension alone" if ok else "FAIL — fell")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
