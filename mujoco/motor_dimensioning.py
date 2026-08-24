#!/usr/bin/env python
"""Motor dimensioning + transmission (spool vs lever-arm) study.

Extracts actuation requirements from the two verified balance simulations
(stand + 60 N forward / 40 N lateral push recovery):
  1. joint-torque model  -> per-joint peak/RMS torque, peak speed, peak power
  2. cable-driven model  -> per-cable-group peak/RMS tension, peak cable
     velocity, peak tension slew (dF/dt), required stroke (from the model's
     own moment arms x joint ranges)

Then evaluates transmission options for the spool-vs-lever question:
  - QDD motor + spool (radius r_s): unlimited stroke, but reflected cable mass
    m_eff = I_rotor G^2 / r_s^2 is large -> force-control resonance through
    the series (cable+textile) compliance k_ser is low.
  - QDD motor + lever arm (radius R > r_s): m_eff smaller by (R/r_s)^2 ->
    higher force bandwidth, but stroke limited to ~ +/- R sin(60 deg) and
    tension capacity G tau / R is lower.

MOTOR DATA marked ~ are estimates from public datasheets — re-verify before
freezing a BOM (rotor inertia especially; vendors rarely publish it).
"""
import numpy as np
import mujoco
from scipy.optimize import lsq_linear
from verify_humanoid27 import gains_for

# --------------------------------------------------------------- motor table
# I_rotor: motor-side rotor inertia (kg m^2), ~estimated from rotor mass/size
MOTORS = {
    "AK70-10": dict(G=10.0, tau_peak=24.8, tau_rated=8.3, rpm_free=475.0,
                    mass=0.52, I_rotor=1.2e-4),
    "AK60-6":  dict(G=6.0,  tau_peak=9.0,  tau_rated=3.0, rpm_free=420.0,
                    mass=0.31, I_rotor=6.0e-5),
}
K_SER = {"textile channel (soft)": 4.0e4,   # N/m: anchors + long sleeve run
         "short stiff run": 3.0e5}          # N/m: 3mm Dyneema, ~1 m, EA/L
SPOOL_RADII = (0.010, 0.012, 0.015)         # m
LEVER_RADII = (0.030, 0.040, 0.050)         # m
LEVER_SWEEP = np.radians(120.0)             # usable lever rotation (+/-60 deg)
SF = 1.5                                    # safety factor on sim peaks

GROUPS = ("hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist", "neck")


def group_of(name):
    for g in GROUPS:
        if name.startswith(g):
            return g
    raise ValueError(name)


def dense_moment(model, data):
    m = np.zeros((model.nu, model.nv))
    mujoco.mju_sparse2dense(m, data.actuator_moment, data.moment_rownnz,
                            data.moment_rowadr, data.moment_colind)
    return m


def pd_setup(model):
    hinge = [j for j in range(model.njnt)
             if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    qadr, vadr = model.jnt_qposadr[hinge], model.jnt_dofadr[hinge]
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in hinge]
    kp = np.array([gains_for(n)[0] for n in names])
    kd = np.array([gains_for(n)[1] for n in names])
    return hinge, qadr, vadr, names, kp, kd


def balance_tau(model, data, qadr, vadr, names, kp, kd, com_v, sfl, sfr):
    idx = {n: i for i, n in enumerate(names)}
    sup = 0.5 * (data.site_xpos[sfl] + data.site_xpos[sfr])
    com = data.subtree_com[0]
    tau = -kp * (data.qpos[qadr] - np.zeros_like(qadr, dtype=float)) - kd * data.qvel[vadr]
    fb_x = 400 * (com[0] - sup[0]) + 150 * com_v[0]
    fb_y = 400 * (com[1] - sup[1]) + 150 * com_v[1]
    for n in ("ankle_pitch_l", "ankle_pitch_r"):
        tau[idx[n]] += fb_x
    for n in ("ankle_roll_l", "ankle_roll_r"):
        tau[idx[n]] -= fb_y
    return tau


def run_joint_model():
    model = mujoco.MjModel.from_xml_path("humanoid_27dof_tensegrity.xml")
    data = mujoco.MjData(model)
    hinge, qadr, vadr, names, kp, kd = pd_setup(model)
    gear = model.actuator_gear[:, 0]
    anames = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
              for i in range(model.nu)]
    sfl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "foot_l_site")
    sfr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "foot_r_site")
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    com_prev = data.subtree_com[0].copy()
    stats = {}
    taus, vels = [], []
    for step in range(2000):
        t = step * model.opt.timestep
        com = data.subtree_com[0]
        com_v = (com - com_prev) / model.opt.timestep
        com_prev = com.copy()
        tau = balance_tau(model, data, qadr, vadr, names, kp, kd, com_v, sfl, sfr)
        data.ctrl[:] = np.clip(tau / gear, -1, 1)
        data.xfrc_applied[pelvis, 0] = 60.0 if 2.0 <= t < 2.2 else 0.0
        data.xfrc_applied[pelvis, 1] = 40.0 if 5.0 <= t < 5.2 else 0.0
        mujoco.mj_step(model, data)
        taus.append(gear * data.ctrl)
        vels.append(data.qvel[vadr].copy())
    taus, vels = np.abs(taus), np.abs(vels)
    for g in GROUPS:
        cols = [i for i, n in enumerate(anames) if group_of(n) == g]
        stats[g] = dict(tau_pk=taus[:, cols].max(),
                        tau_rms=np.sqrt((taus[:, cols] ** 2).mean()),
                        om_pk=vels[:, cols].max(),
                        p_pk=(taus[:, cols] * vels[:, cols]).max())
    return stats


def run_cable_model():
    model = mujoco.MjModel.from_xml_path("humanoid_27dof_tensegrity_cable.xml")
    data = mujoco.MjData(model)
    hinge, qadr, vadr, names, kp, kd = pd_setup(model)
    fmax = -model.actuator_gear[:, 0]
    anames = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
              for i in range(model.nu)]
    sfl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "foot_l_site")
    sfr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "foot_r_site")
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    com_prev = data.subtree_com[0].copy()
    x0 = 20.0 / fmax
    reg = np.sqrt(0.1) * np.eye(model.nu)
    dt = model.opt.timestep

    # stroke requirement: |dL/dq| at home x joint range span, summed over the
    # joint's dofs (conservative; moment arms change with pose)
    dLdq = dense_moment(model, data) / model.actuator_gear[:, [0]]  # 51 x nv
    span = np.zeros(model.nv)
    for j in hinge:
        span[model.jnt_dofadr[j]] = model.jnt_range[j, 1] - model.jnt_range[j, 0]
    stroke = (np.abs(dLdq[:, :]) * span).sum(axis=1)

    x = x0.copy()
    tensions, tvels = [], []
    for step in range(2000):
        t = step * dt
        com = data.subtree_com[0]
        com_v = (com - com_prev) / dt
        com_prev = com.copy()
        tau = balance_tau(model, data, qadr, vadr, names, kp, kd, com_v, sfl, sfr)
        M = dense_moment(model, data)[:, 6:].T
        sol = lsq_linear(np.vstack([M, reg]),
                         np.concatenate([tau, np.sqrt(0.1) * x0]),
                         bounds=(0.0, 1.0), max_iter=30, tol=1e-6)
        x = sol.x
        data.ctrl[:] = x
        data.xfrc_applied[pelvis, 0] = 60.0 if 2.0 <= t < 2.2 else 0.0
        data.xfrc_applied[pelvis, 1] = 40.0 if 5.0 <= t < 5.2 else 0.0
        mujoco.mj_step(model, data)
        tensions.append(fmax * x)
        # actuator_velocity is gear-scaled (= gear * dL/dt); undo the gear
        tvels.append(np.abs(data.actuator_velocity / fmax))
    tensions = np.array(tensions)
    tvels = np.array(tvels)
    slew = np.abs(np.diff(tensions, axis=0)) / dt
    stats = {}
    for g in GROUPS:
        cols = [i for i, n in enumerate(anames) if group_of(n) == g]
        stats[g] = dict(F_pk=tensions[:, cols].max(),
                        F_rms=np.sqrt((tensions[:, cols] ** 2).mean()),
                        v_pk=tvels[:, cols].max(),
                        dFdt_pk=slew[:, cols].max(),
                        stroke=stroke[cols].max())
    return stats


def transmission_table(F_req, v_req, stroke_req):
    rows = []
    for mname, m in MOTORS.items():
        w_out = m["rpm_free"] * 2 * np.pi / 60.0        # gearbox output rad/s
        I_out = m["I_rotor"] * m["G"] ** 2              # reflected at output
        for kind, radii in (("spool", SPOOL_RADII), ("lever", LEVER_RADII)):
            for r in radii:
                F_cap = m["tau_peak"] / r
                F_cont = m["tau_rated"] / r
                v_cap = w_out * r
                stroke_cap = np.inf if kind == "spool" else r * LEVER_SWEEP
                m_eff = I_out / r ** 2
                fbw = {lbl: np.sqrt(k / m_eff) / (2 * np.pi)
                       for lbl, k in K_SER.items()}
                ok = (F_cap >= F_req and v_cap >= v_req
                      and stroke_cap >= stroke_req)
                rows.append((mname, kind, r, F_cap, F_cont, v_cap, stroke_cap,
                             m_eff, fbw, ok))
    return rows


def main():
    print("=" * 78)
    print("1) JOINT-TORQUE MODEL: stand + push recovery (proxy for balance ops)")
    print("=" * 78)
    js = run_joint_model()
    print(f"{'group':10s} {'tau_pk Nm':>10s} {'tau_rms':>9s} {'om_pk rad/s':>12s} {'P_pk W':>8s}")
    for g in GROUPS:
        s = js[g]
        print(f"{g:10s} {s['tau_pk']:10.1f} {s['tau_rms']:9.1f} "
              f"{s['om_pk']:12.2f} {s['p_pk']:8.1f}")

    print()
    print("=" * 78)
    print("2) CABLE MODEL: per-cable-group requirements (same scenario)")
    print("=" * 78)
    cs = run_cable_model()
    print(f"{'group':10s} {'F_pk N':>8s} {'F_rms':>7s} {'v_pk m/s':>9s} "
          f"{'dF/dt kN/s':>11s} {'stroke m':>9s}")
    for g in GROUPS:
        s = cs[g]
        print(f"{g:10s} {s['F_pk']:8.0f} {s['F_rms']:7.0f} {s['v_pk']:9.3f} "
              f"{s['dFdt_pk'] / 1e3:11.1f} {s['stroke']:9.3f}")

    # leg-group requirements drive the design; add walking headroom on speed
    # (stance recovery is slow; swing-phase cable speed will be ~r_j * 8 rad/s)
    F_req = SF * max(cs[g]["F_pk"] for g in ("hip", "knee", "ankle"))
    v_req = max(0.05 * 8.0,  # swing: moment arm x expected joint speed
                SF * max(cs[g]["v_pk"] for g in ("hip", "knee", "ankle")))
    stroke_req = max(cs[g]["stroke"] for g in ("hip", "knee", "ankle"))
    print(f"\nLEG requirement (SF {SF}): F >= {F_req:.0f} N, v >= {v_req:.2f} m/s, "
          f"stroke >= {stroke_req * 1000:.0f} mm")

    print()
    print("=" * 78)
    print("3) TRANSMISSION OPTIONS (leg requirement)   [~ = verify datasheet]")
    print("=" * 78)
    print(f"{'motor':9s} {'type':6s} {'r mm':>5s} {'F_pk N':>7s} {'F_cont':>7s} "
          f"{'v m/s':>6s} {'strk mm':>8s} {'m_eff kg':>9s} "
          f"{'f_bw soft':>10s} {'f_bw stiff':>11s} {'meets':>6s}")
    for (mn, kind, r, Fc, Fcont, vc, sc, me, fbw, ok) in \
            transmission_table(F_req, v_req, stroke_req):
        scs = "inf" if np.isinf(sc) else f"{sc * 1000:.0f}"
        vals = list(fbw.values())
        print(f"{mn:9s} {kind:6s} {r * 1000:5.0f} {Fc:7.0f} {Fcont:7.0f} "
              f"{vc:6.2f} {scs:>8s} {me:9.1f} {vals[0]:8.1f}Hz {vals[1]:9.1f}Hz "
              f"{'YES' if ok else 'no':>6s}")

    print("""
notes:
- f_bw = open-loop force-control resonance sqrt(k_ser/m_eff)/2pi through the
  series compliance; closed-loop force feedback can extend ~2-3x beyond it,
  but not orders of magnitude.
- 'soft' k_ser = 40 kN/m (textile-channel-dominated run), 'stiff' = 300 kN/m
  (short Dyneema run with rigid anchors).
- lever stroke assumes +/-60 deg usable crank rotation.
- The prestress network supplies baseline joint stiffness passively, so the
  MOTOR force bandwidth requirement applies to the modulation on top, not to
  the stiffness itself — this is the tensegrity advantage.""")


if __name__ == "__main__":
    main()
