#!/usr/bin/env python
"""E1: impact tolerance. Unpowered drops from increasing clearance, tensegrity
(prestress network intact) vs rigid-equivalent (cable stiffness/damping
zeroed). Start pose = walk_ready crouch; metrics per drop:

  - peak total ground reaction force (sum of contact normal forces vs floor)
  - peak internal structural load at the knee interface (norm of the
    translational part of cfrc_int at the shank bodies) -- the load the
    joint hardware must survive
  - fallen flag (pelvis below 0.45 m at the end)
"""
import json
import numpy as np
import mujoco

MJ = "/Users/ncorrell/Downloads/tensegrity/mujoco"
OUT = "/Users/ncorrell/Downloads/tensegrity/experiments/results/e1_drop.json"
HEIGHTS = [0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00]


def drop(h, rigid, held, prestress_scale=1.0):
    """rigid: cables severed. held: joints servo-held at the pose (strong PD,
    torque-limited) -- the conventional position-controlled robot. Limp +
    rigid = ragdoll floor reference; limp + cables = passive tensegrity."""
    m = mujoco.MjModel.from_xml_path(f"{MJ}/humanoid_27dof_tensegrity.xml")
    if rigid:
        m.tendon_stiffness[:] = 0.0
        m.tendon_damping[:] = 0.0
    if prestress_scale != 1.0:
        # deepen the deadband: rest length shrinks toward prestress_scale x
        d0 = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d0, 0)
        mujoco.mj_forward(m, d0)
        for tdx in range(m.ntendon):
            L = float(d0.ten_length[tdx])
            rest = float(m.tendon_lengthspring[tdx, 1])
            pre = max(0.0, 1.0 - rest / max(L, 1e-9))
            m.tendon_lengthspring[tdx, 1] = L * (1.0 - prestress_scale * pre)
    d = mujoco.MjData(m)
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "walk_ready")
    mujoco.mj_resetDataKeyframe(m, d, key)
    d.qpos[2] += h
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    shanks = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
              for b in ("shank_l", "shank_r")]
    jids = m.actuator_trnid[:, 0]
    qadr, vadr = m.jnt_qposadr[jids], m.jnt_dofadr[jids]
    gear = m.actuator_gear[:, 0]
    qref = d.qpos[qadr].copy()
    peak_grf = 0.0
    peak_knee = 0.0
    f6 = np.zeros(6)
    t_contact = None
    WINDOW = 0.25          # landing transient only; excludes later toppling
    for step in range(int(1.5 / m.opt.timestep)):
        if held:
            tau = -400.0 * (d.qpos[qadr] - qref) - 15.0 * d.qvel[vadr]
            d.ctrl[:] = np.clip(tau / gear, -1, 1)
        else:
            d.ctrl[:] = 0.0
        mujoco.mj_step(m, d)
        grf = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            if floor in (c.geom[0], c.geom[1]):
                mujoco.mj_contactForce(m, d, i, f6)
                grf += abs(f6[0])
        if grf > 1.0 and t_contact is None:
            t_contact = d.time
        in_window = t_contact is not None and d.time <= t_contact + WINDOW
        if in_window:
            peak_grf = max(peak_grf, grf)
            mujoco.mj_rnePostConstraint(m, d)
            for b in shanks:
                peak_knee = max(peak_knee,
                                float(np.linalg.norm(d.cfrc_int[b][3:])))
    return dict(h=h, rigid=rigid, held=held, peak_grf=peak_grf, peak_knee=peak_knee,
                final_z=float(d.qpos[2]), fell=bool(d.qpos[2] < 0.45))


def main():
    rows = []
    W = 27.6 * 9.81
    modes = [("tensegrity-passive", False, False, 1.0),
             ("tsg-passive-3xpre", False, False, 3.0),
             ("tensegrity-held", False, True, 1.0),
             ("rigid-held", True, True, 1.0),
             ("rigid-limp", True, False, 1.0)]
    print(f"{'h (m)':>6s} {'variant':>19s} {'peak GRF':>9s} {'xBW':>5s} "
          f"{'knee load':>10s} {'final z':>8s}")
    for h in HEIGHTS:
        for name, rigid, held, ps in modes:
            r = drop(h, rigid, held, ps)
            r["mode"] = name
            rows.append(r)
            print(f"{h:6.2f} {name:>19s} {r['peak_grf']:8.0f}N "
                  f"{r['peak_grf']/W:5.1f} {r['peak_knee']:9.0f}N "
                  f"{r['final_z']:8.2f}")
    json.dump(rows, open(OUT, "w"), indent=1)
    print("E1 done ->", OUT)


if __name__ == "__main__":
    main()
