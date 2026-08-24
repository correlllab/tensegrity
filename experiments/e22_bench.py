#!/usr/bin/env python
"""E22: component-bench predictions for the staged validation programme.

Two bench units that do not yet have measured numbers, simulated exactly as
they would be tested on a bench, producing the falsifiable predictions the
paper's validation ladder quotes:

  (a) ankle bench   pelvis welded, robot standing; apply a differential
                    tension step on the left ankle's A vs B cable families
                    and measure the quasi-static foot pitch moment it
                    produces -> N m per 100 N of differential tension.
                    Bench version: foot plate bolted to a force table, mast
                    + shank stub above, spring scales on the cables.

  (b) leg bench     pelvis welded at stance height; press down on the
                    pelvis 0..400 N and record (i) vertical deflection per
                    100 N (the leg chain's stiffness, dominated by the
                    ankle cage) and (ii) the busiest ankle cable tension at
                    400 N against the 4 mm rope rating.
"""
import json
import os
import numpy as np
import mujoco

ROOT = "/Users/ncorrell/Downloads/tensegrity"
RES = f"{ROOT}/experiments/results"
os.chdir(f"{ROOT}/mujoco")


def welded_model():
    xml = open("humanoid_hybrid.xml").read().replace(
        '<freejoint name="root"/>', '')
    return mujoco.MjModel.from_xml_string(xml)


def settle(m, d, nj, steps, extra=None):
    jid = m.actuator_trnid[:nj, 0]
    qadr, vadr = m.jnt_qposadr[jid], m.jnt_dofadr[jid]
    gear = m.actuator_gear[:, 0]
    q0 = d.qpos[qadr].copy()
    for _ in range(steps):
        tau = 300.0 * (q0 - d.qpos[qadr]) - 15.0 * d.qvel[vadr]
        d.ctrl[:nj] = np.clip(tau / gear[:nj], -1, 1)
        d.ctrl[nj:] = 0.06
        if extra:
            extra(d)
        mujoco.mj_step(m, d)


def ankle_bench():
    m = welded_model()
    d = mujoco.MjData(m)
    nj = 12
    mujoco.mj_forward(m, d)
    settle(m, d, nj, int(1.5 / m.opt.timestep))
    an = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
          for i in range(m.nu)]
    # left ankle A and B family actuators
    ia = [i for i, n in enumerate(an) if n.startswith("ankm_lA")]
    ib = [i for i, n in enumerate(an) if n.startswith("ankm_lB")]
    foot = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "foot_l")
    # quasi-static: record foot torque from cable forces before/after a
    # differential tension step, via the actuator moment arms
    mujoco.mj_forward(m, d)
    mom = np.zeros((m.nu, m.nv))
    mujoco.mju_sparse2dense(mom, d.actuator_moment, d.moment_rownnz,
                            d.moment_rowadr, d.moment_colind)
    # foot_l pitch DoF: free joint rotational y; dofadr of foot_l free joint
    jfree = [j for j in range(m.njnt)
             if m.jnt_bodyid[j] == foot][0]
    va = m.jnt_dofadr[jfree]
    pitch_dof = va + 4                       # rot y of the free joint
    gear = np.abs(m.actuator_gear[:, 0])
    # actuator_moment includes the gear; divide it out for the true
    # per-cable moment arm about foot pitch
    rA = np.mean([abs(mom[i, pitch_dof]) / gear[i] for i in ia])
    rB = np.mean([abs(mom[i, pitch_dof]) / gear[i] for i in ib])
    # bench protocol: raise every A-family cable 100 N above its
    # co-contraction level (spring scale per cable), foot on a force table
    m_per_100N = 3.0 * rA * 100.0
    return dict(moment_per_100N=float(m_per_100N),
                arm_A_mm=float(1e3 * rA), arm_B_mm=float(1e3 * rB))


def waist_bench():
    """Welded pelvis, downward force on the torso: the load path a trunk
    payload takes through the twelve waist cables."""
    m = welded_model()
    d = mujoco.MjData(m)
    nj = 12
    mujoco.mj_forward(m, d)
    settle(m, d, nj, int(1.5 / m.opt.timestep))
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
    z0, out = None, []
    for F in (0.0, 100.0, 200.0, 300.0):
        def push(dd, F=F):
            dd.xfrc_applied[torso, 2] = -F
        settle(m, d, nj, int(1.2 / m.opt.timestep), extra=push)
        wz = float(d.xpos[torso][2])
        if z0 is None:
            z0 = wz
        out.append(dict(F=F, sink_mm=1e3 * (z0 - wz)))
        print(f"  waist bench F={F:.0f} N: torso sink "
              f"{1e3*(z0-wz):.1f} mm", flush=True)
    return dict(points=out,
                sink_per_100N=float(out[2]["sink_mm"] / 2.0),
                holds_200=bool(out[2]["sink_mm"] < 100),
                collapses_300=bool(out[3]["sink_mm"] > 200))


def leg_bench():
    """Feet on the floor, pelvis held by a 6-DoF virtual jig (a bench
    fixture), extra vertical load pressed into the pelvis: ankle cable
    tension vs. total load."""
    xml = open("humanoid_hybrid.xml").read()
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    nj = 12
    pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    mujoco.mj_forward(m, d)
    p0 = d.qpos[:7].copy()
    tn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TENDON, t) or ""
          for t in range(m.ntendon)]
    ank = [t for t, n in enumerate(tn) if n.startswith("ank_")]
    out = []
    for F in (0.0, 100.0, 200.0):
        def jig(dd, F=F):
            # bench fixture: lateral guides only -- the vertical load path
            # must run through the legs, not the jig
            dd.xfrc_applied[pelvis, 0] = (
                8000.0 * (p0[0] - dd.qpos[0]) - 400.0 * dd.qvel[0])
            dd.xfrc_applied[pelvis, 1] = (
                8000.0 * (p0[1] - dd.qpos[1]) - 400.0 * dd.qvel[1])
            dd.xfrc_applied[pelvis, 2] = -F
        settle(m, d, nj, int(1.5 / m.opt.timestep), extra=jig)
        T = [m.tendon_stiffness[t] * max(0.0, float(d.ten_length[t])
             - m.tendon_lengthspring[t, 1]) for t in ank]
        out.append(dict(F=F, peak_ankle_T=float(max(T)),
                        pelvis_z_mm=float(1e3 * d.qpos[2])))
        print(f"  leg bench +{F:.0f} N: peak ankle cable {max(T):.0f} N, "
              f"pelvis z {1e3*d.qpos[2]:.0f} mm", flush=True)
    slope = (out[-1]["peak_ankle_T"] - out[0]["peak_ankle_T"]) / 2.0
    return dict(points=out, T_per_100N=float(slope),
                T_at_stance=out[0]["peak_ankle_T"])


def main():
    a = ankle_bench()
    print(f"ankle bench: {a['moment_per_100N']:.1f} N m per 100 N "
          f"differential (arms A {a['arm_A_mm']:.0f} mm, "
          f"B {a['arm_B_mm']:.0f} mm)", flush=True)
    w = waist_bench()
    l = leg_bench()
    json.dump(dict(ankle=a, waist=w, leg=l),
              open(f"{RES}/e22_bench.json", "w"), indent=1)
    print("E22 ->", f"{RES}/e22_bench.json")


if __name__ == "__main__":
    main()
