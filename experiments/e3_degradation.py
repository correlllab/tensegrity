#!/usr/bin/env python
"""E3: graceful degradation under cable failure (cable-driven model).

(a) Authority after single-cable removal, exhaustively (linear algebra only).
(b) Stand + 40 N push survival with k cables severed (worst singles by
    authority margin, plus random k = 2, 4, 8).

A severed cable loses both its actuator (gear -> 0) and its passive spring
(stiffness/damping -> 0).
"""
import json
import sys
import numpy as np
import mujoco
from scipy.optimize import lsq_linear

sys.path.insert(0, "/Users/ncorrell/Downloads/tensegrity/mujoco")
from verify_humanoid27 import gains_for  # noqa: E402

MJ = "/Users/ncorrell/Downloads/tensegrity/mujoco"
OUT = "/Users/ncorrell/Downloads/tensegrity/experiments/results/e3_degradation.json"
CABLE = f"{MJ}/humanoid_27dof_tensegrity_cable.xml"
TORQUE = f"{MJ}/humanoid_27dof_tensegrity.xml"


def authority_margins():
    mc = mujoco.MjModel.from_xml_path(CABLE)
    dc = mujoco.MjData(mc)
    mujoco.mj_resetDataKeyframe(mc, dc, 0)
    mujoco.mj_forward(mc, dc)
    mom = np.zeros((mc.nu, mc.nv))
    mujoco.mju_sparse2dense(mom, dc.actuator_moment, dc.moment_rownnz,
                            dc.moment_rowadr, dc.moment_colind)
    M = mom[:, 6:].T                              # (27, ncable), gear folded
    mt = mujoco.MjModel.from_xml_path(TORQUE)
    need = np.abs(mt.actuator_gear[:, 0])         # (27,)
    pos = np.maximum(M, 0)
    neg = np.maximum(-M, 0)
    P, N = pos.sum(axis=1), neg.sum(axis=1)
    base = float(min((P / need).min(), (N / need).min()))
    margins = np.empty(mc.nu)
    for c in range(mc.nu):
        m1 = ((P - pos[:, c]) / need).min()
        m2 = ((N - neg[:, c]) / need).min()
        margins[c] = min(m1, m2)
    names = [mujoco.mj_id2name(mc, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
             for i in range(mc.nu)]
    return base, margins, names


def greedy_worst(margins, M, need, k, pool=None):
    """The k cables an adversary would cut: greedily remove whichever cable
    most reduces the worst per-DoF authority margin. Random draws over all 114
    cables mostly hit arm and neck cables that quiet standing never uses; this
    is the condition that actually tests the redundancy claim."""
    pos, neg = np.maximum(M, 0), np.maximum(-M, 0)
    P, N = pos.sum(axis=1), neg.sum(axis=1)
    cand = list(range(M.shape[1])) if pool is None else list(pool)
    chosen = []
    for _ in range(k):
        best, best_m = None, np.inf
        for c in cand:
            if c in chosen:
                continue
            m = min(((P - pos[:, c]) / need).min(), ((N - neg[:, c]) / need).min())
            if m < best_m:
                best, best_m = c, m
        chosen.append(best)
        P -= pos[:, best]
        N -= neg[:, best]
    return chosen


def run_stand(removed, seconds=5.0, push=(40.0, 0.0)):
    model = mujoco.MjModel.from_xml_path(CABLE)
    for c in removed:
        tid = model.actuator_trnid[c, 0]
        model.actuator_gear[c, 0] = -1e-6          # dead motor (keep sign)
        model.tendon_stiffness[tid] = 0.0          # severed cable
        model.tendon_damping[tid] = 0.0
    data = mujoco.MjData(model)
    fmax = -model.actuator_gear[:, 0]
    hinge = [j for j in range(model.njnt)
             if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    qadr, vadr = model.jnt_qposadr[hinge], model.jnt_dofadr[hinge]
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in hinge]
    kp = np.array([gains_for(n)[0] for n in names])
    kd = np.array([gains_for(n)[1] for n in names])
    idx = {n: i for i, n in enumerate(names)}
    ap = [idx["ankle_pitch_l"], idx["ankle_pitch_r"]]
    ar = [idx["ankle_roll_l"], idx["ankle_roll_r"]]
    sfl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "foot_l_site")
    sfr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "foot_r_site")
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    mujoco.mj_resetDataKeyframe(model, data, 0)
    qref = data.qpos[qadr].copy()
    mujoco.mj_forward(model, data)
    com_prev = data.subtree_com[0].copy()
    x0 = np.minimum(20.0 / np.maximum(fmax, 1e-6), 1.0)
    reg = np.sqrt(0.1) * np.eye(model.nu)
    for step in range(int(seconds / model.opt.timestep)):
        t = step * model.opt.timestep
        com = data.subtree_com[0]
        com_v = (com - com_prev) / model.opt.timestep
        com_prev = com.copy()
        sup = 0.5 * (data.site_xpos[sfl] + data.site_xpos[sfr])
        tau = -kp * (data.qpos[qadr] - qref) - kd * data.qvel[vadr]
        fx = 400 * (com[0] - sup[0]) + 150 * com_v[0]
        fy = 400 * (com[1] - sup[1]) + 150 * com_v[1]
        for i in ap:
            tau[i] += fx
        for i in ar:
            tau[i] -= fy
        mom = np.zeros((model.nu, model.nv))
        mujoco.mju_sparse2dense(mom, data.actuator_moment, data.moment_rownnz,
                                data.moment_rowadr, data.moment_colind)
        sol = lsq_linear(np.vstack([mom[:, 6:].T, reg]),
                         np.concatenate([tau, np.sqrt(0.1) * x0]),
                         bounds=(0.0, 1.0), max_iter=25, tol=1e-6)
        data.ctrl[:] = sol.x
        if 2.5 <= t < 2.7:
            data.xfrc_applied[pelvis, 0] = push[0]
            data.xfrc_applied[pelvis, 1] = push[1]
        else:
            data.xfrc_applied[pelvis, :2] = 0.0
        mujoco.mj_step(model, data)
        if data.qpos[2] < 0.4:
            return False, float(data.qpos[2])
    return bool(data.qpos[2] > 0.8), float(data.qpos[2])


def main():
    rng = np.random.default_rng(7)
    base, margins, names = authority_margins()
    order = np.argsort(margins)
    res = dict(base_margin=base,
               worst_singles=[dict(cable=names[c], margin=float(margins[c]))
                              for c in order[:12]],
               n_cables=len(names),
               singles_below_1=int((margins < 1.0).sum()),
               singles_below_05=int((margins < 0.5).sum()))
    print(f"baseline margin {base:.2f}; single-removal margins: "
          f"min {margins.min():.2f}, {res['singles_below_1']} cables drop a "
          f"DoF below 1.0x", flush=True)

    singles = []
    for c in order[:10]:
        ok, z = run_stand({int(c)})
        singles.append(dict(cable=names[c], margin=float(margins[c]),
                            survived=ok, final_z=z))
        print(f"single cut {names[c]:22s} margin {margins[c]:.2f} -> "
              f"{'OK' if ok else 'FELL'} (z={z:.2f})", flush=True)
    for c in rng.choice(len(names), 5, replace=False):
        ok, z = run_stand({int(c)})
        singles.append(dict(cable=names[c], margin=float(margins[c]),
                            survived=ok, final_z=z))
        print(f"single cut {names[c]:22s} (random) -> "
              f"{'OK' if ok else 'FELL'}", flush=True)
    res["singles"] = singles

    # ---- multi-cable failures, three selection rules
    mc = mujoco.MjModel.from_xml_path(CABLE)
    dc = mujoco.MjData(mc)
    mujoco.mj_resetDataKeyframe(mc, dc, 0)
    mujoco.mj_forward(mc, dc)
    mom = np.zeros((mc.nu, mc.nv))
    mujoco.mju_sparse2dense(mom, dc.actuator_moment, dc.moment_rownnz,
                            dc.moment_rowadr, dc.moment_colind)
    M = mom[:, 6:].T
    mt = mujoco.MjModel.from_xml_path(TORQUE)
    need = np.abs(mt.actuator_gear[:, 0])
    stance = [i for i, n in enumerate(names)
              if n.startswith(("hip", "knee", "ankle", "waist"))]
    PUSHES = ((40.0, 0.0), (60.0, 0.0), (-60.0, 0.0), (0.0, 40.0))

    multi = []
    for k in (2, 4, 8):
        wins = 0
        for trial in range(8):
            rm = set(int(x) for x in rng.choice(len(names), k, replace=False))
            ok, z = run_stand(rm)
            wins += ok
        multi.append(dict(k=k, rule="random-all", trials=8, survived=wins))
        print(f"k={k} random-all : {wins}/8 survived", flush=True)

        wins = 0
        for trial in range(8):
            rm = set(int(x) for x in rng.choice(stance, k, replace=False))
            ok, z = run_stand(rm)
            wins += ok
        multi.append(dict(k=k, rule="random-stance", trials=8, survived=wins))
        print(f"k={k} random-leg : {wins}/8 survived", flush=True)

        adv = set(greedy_worst(margins, M, need, k, pool=stance))
        wins = 0
        for p in PUSHES:
            ok, z = run_stand(adv, push=p)
            wins += ok
        multi.append(dict(k=k, rule="adversarial", trials=len(PUSHES),
                          survived=wins,
                          cables=[names[c] for c in sorted(adv)]))
        print(f"k={k} adversarial: {wins}/{len(PUSHES)} survived "
              f"({', '.join(names[c] for c in sorted(adv))})", flush=True)
    res["multi"] = multi

    json.dump(res, open(OUT, "w"), indent=1)
    print("E3 done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
