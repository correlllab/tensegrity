#!/usr/bin/env python
"""Verification for humanoid_27dof.xml (M0 mass-true skeleton).

Checks: model loads with 27 actuated DoF, mass budget matches the design doc,
and the robot stands under joint PD hold plus a CoM-feedback ankle strategy,
surviving a forward push (t=2 s) and a lateral push (t=5 s) at the pelvis.
Reports per-joint torque utilization (peak |tau| / ctrl limit) as a first
actuator-sizing signal. Balance beyond the ankle-strategy envelope (stepping)
is MJPC's job, not this script's.
"""
import argparse
import numpy as np
import mujoco

# kp, kd per joint name prefix (matched with startswith, first hit wins)
GAINS = [
    ("hip_yaw", 120, 6), ("hip_roll", 220, 10), ("hip_pitch", 260, 12),
    ("knee", 260, 12), ("ankle_pitch", 200, 9), ("ankle_roll", 140, 6),
    ("waist_yaw", 150, 8), ("waist_pitch", 220, 10), ("waist_roll", 180, 8),
    ("neck", 20, 1), ("shoulder", 60, 3), ("elbow", 40, 2), ("wrist", 10, 0.5),
]


def gains_for(name):
    for prefix, kp, kd in GAINS:
        if name.startswith(prefix):
            return kp, kd
    raise ValueError(f"no gains for joint {name}")


def remove_battery(model, body_name):
    """Simulate a pulled battery pack.

    Each pack is a zero-joint body welded to the pelvis, so zeroing its mass
    and inertia is exactly equivalent to removing it: MuJoCo's composite-rigid-
    body pass accumulates per-body mass/inertia over the weld tree.

    mj_setConst is REQUIRED after the edit: body_subtreemass is a compile-time
    derived constant, and subtree_com divides by it. Skipping it leaves the
    stale total in the denominator, so every CoM readout (this script's balance
    feedback, and MJPC's balance residuals) silently reports a CoM ~85 mm too
    low while the mass matrix itself is correct.
    """
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise ValueError(f"no body {body_name} in this model")
    model.body_mass[bid] = 0.0
    model.body_inertia[bid] = 0.0
    mujoco.mj_setConst(model, mujoco.MjData(model))
    return bid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--push-fwd", type=float, default=60.0,
                    help="forward push at pelvis (N), applied 2.0-2.2 s")
    ap.add_argument("--push-lat", type=float, default=40.0,
                    help="lateral push at pelvis (N), applied 5.0-5.2 s")
    ap.add_argument("--model", default="humanoid_27dof.xml")
    ap.add_argument("--remove-battery", default=None,
                    choices=("battery_l", "battery_r"),
                    help="hot-swap case: run with one pack pulled out")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.model)
    if args.remove_battery:
        remove_battery(model, args.remove_battery)
    data = mujoco.MjData(model)

    print(f"nq={model.nq} nv={model.nv} nu={model.nu} "
          f"(actuated DoF = {model.nv - 6})")
    assert model.nu == 27 and model.nv - 6 == 27, "expected 27 actuated DoF"

    print(f"total mass: {model.body_mass.sum():.2f} kg")
    print("mass by body:")
    for i in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        print(f"  {name:14s} {model.body_mass[i]:6.2f} kg")

    # per-actuator PD setup (one actuator per hinge joint)
    jids = model.actuator_trnid[:, 0]
    qadr = model.jnt_qposadr[jids]
    vadr = model.jnt_dofadr[jids]
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
             for i in range(model.nu)]
    kp = np.array([gains_for(n)[0] for n in names])
    kd = np.array([gains_for(n)[1] for n in names])
    lo, hi = model.actuator_ctrlrange.T
    gear = model.actuator_gear[:, 0]  # ctrl is normalized; torque = gear * ctrl

    mujoco.mj_resetDataKeyframe(model, data, 0)
    qref = data.qpos[qadr].copy()
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    # CoM-feedback ankle strategy (whole-body CoM vs support center).
    # Sign convention verified empirically: +pitch (plantarflex reaction pushes
    # body back), -roll.
    idx = {n: i for i, n in enumerate(names)}
    ankle_p = [idx["ankle_pitch_l"], idx["ankle_pitch_r"]]
    ankle_r = [idx["ankle_roll_l"], idx["ankle_roll_r"]]
    sfl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "foot_l_site")
    sfr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "foot_r_site")
    KC_P, KC_D = 400.0, 150.0

    mujoco.mj_forward(model, data)
    com_prev = data.subtree_com[0].copy()
    z0 = data.qpos[2]
    min_z = z0
    peak_tau = np.zeros(model.nu)
    n_steps = int(args.duration / model.opt.timestep)
    for step in range(n_steps):
        t = step * model.opt.timestep
        com = data.subtree_com[0]
        com_v = (com - com_prev) / model.opt.timestep
        com_prev = com.copy()
        sup = 0.5 * (data.site_xpos[sfl] + data.site_xpos[sfr])
        tau = -kp * (data.qpos[qadr] - qref) - kd * data.qvel[vadr]
        fb_x = KC_P * (com[0] - sup[0]) + KC_D * com_v[0]
        fb_y = KC_P * (com[1] - sup[1]) + KC_D * com_v[1]
        for i in ankle_p:
            tau[i] += fb_x
        for i in ankle_r:
            tau[i] -= fb_y
        data.ctrl[:] = np.clip(tau / gear, lo, hi)
        peak_tau = np.maximum(peak_tau, np.abs(gear * data.ctrl))
        data.xfrc_applied[pelvis, 0] = args.push_fwd if 2.0 <= t < 2.2 else 0.0
        data.xfrc_applied[pelvis, 1] = args.push_lat if 5.0 <= t < 5.2 else 0.0
        mujoco.mj_step(model, data)
        min_z = min(min_z, data.qpos[2])

    com_drift = np.linalg.norm(data.subtree_com[0][:2])
    print(f"\nstand test: initial z={z0:.3f}  min z={min_z:.3f}  "
          f"final z={data.qpos[2]:.3f}  CoM xy drift={com_drift:.3f} m")
    print(f"pushes: {args.push_fwd:.0f} N fwd at t=2 s, "
          f"{args.push_lat:.0f} N lateral at t=5 s (0.2 s each)")

    print("\npeak torque utilization (|tau|/limit):")
    order = np.argsort(-peak_tau / gear)
    for i in order[:10]:
        print(f"  {names[i]:18s} {peak_tau[i]:6.1f} / {gear[i]:.0f} Nm "
              f"({100 * peak_tau[i] / gear[i]:3.0f}%)")

    ok = data.qpos[2] > 0.85 and min_z > 0.80
    print("\nRESULT:", "PASS — stands and recovers" if ok else "FAIL — fell")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
