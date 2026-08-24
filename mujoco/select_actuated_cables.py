#!/usr/bin/env python
"""How many of the 102 cables actually need a motor?

Cables and motors are not the same thing. A cable is Dyneema plus two anchors:
grams, and it is what makes the structure a tensegrity, so having many is
fine. A motor is ~0.3-0.9 kg of QDD actuator. The 102 figure is the size of
the TENSION NETWORK, not a bill of materials for actuators.

This picks the minimum set of cables that must be actively driven so that
every one of the 27 DoFs still reaches the torque limits the joint-torque
model uses. Greedy set cover on the moment matrix at the home pose:

    tau_j = sum_c M[j,c] t_c,   0 <= t_c <= Fmax_c

Two architectures:
  A. fully antagonistic — active cables must cover BOTH directions of every
     DoF (motor count = number of selected cables).
  B. motor + elastic return — active cables cover the loaded direction; a
     passive prestress element returns the joint. Cheaper in motors, but the
     spring has to carry the return load, so it only applies where the return
     direction is lightly loaded.

Unselected cables stay in the build as passive prestress: elastic plus a
turnbuckle to tune tension at assembly.
"""
import numpy as np
import mujoco

CABLE_MODEL = "humanoid_27dof_tensegrity_cable.xml"
TORQUE_MODEL = "humanoid_27dof_tensegrity.xml"
MOTOR_KG = {"hip": 0.52, "knee": 0.52, "ankle": 0.96, "waist": 0.52,
            "shoulder": 0.31, "elbow": 0.31, "wrist": 0.17, "neck": 0.17}


def motor_mass(cable_name):
    for k, v in MOTOR_KG.items():
        if cable_name.startswith(k):
            return v
    raise ValueError(cable_name)


def greedy(M, fmax, need_pos, need_neg, both=True):
    """Select cables until every DoF/direction requirement is met."""
    ndof, ncab = M.shape
    pos = np.maximum(M, 0) * fmax          # per-cable contribution at full tension
    neg = -np.minimum(M, 0) * fmax
    have_p = np.zeros(ndof)
    have_n = np.zeros(ndof)
    chosen = []
    for _ in range(ncab):
        cov = (np.minimum(have_p, need_pos).sum() / need_pos.sum()
               + (np.minimum(have_n, need_neg).sum() / need_neg.sum() if both else 0))
        best, best_gain = None, 1e-12
        for c in range(ncab):
            if c in chosen:
                continue
            p = np.minimum(have_p + pos[:, c], need_pos).sum() / need_pos.sum()
            n = (np.minimum(have_n + neg[:, c], need_neg).sum() / need_neg.sum()
                 if both else 0)
            gain = (p + n) - cov
            if gain > best_gain:
                best, best_gain = c, gain
        if best is None:
            break
        chosen.append(best)
        have_p += pos[:, best]
        have_n += neg[:, best]
        done = np.all(have_p >= need_pos * 0.999) and (
            np.all(have_n >= need_neg * 0.999) if both else True)
        if done:
            break
    return chosen, have_p, have_n


def main():
    mc = mujoco.MjModel.from_xml_path(CABLE_MODEL)
    dc = mujoco.MjData(mc)
    mujoco.mj_resetDataKeyframe(mc, dc, 0)
    mujoco.mj_forward(mc, dc)
    moment = np.zeros((mc.nu, mc.nv))
    mujoco.mju_sparse2dense(moment, dc.actuator_moment, dc.moment_rownnz,
                            dc.moment_rowadr, dc.moment_colind)
    M = moment[:, 6:].T
    fmax = np.ones(mc.nu)          # gear already folded into actuator_moment
    cnames = [mujoco.mj_id2name(mc, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
              for i in range(mc.nu)]

    mt = mujoco.MjModel.from_xml_path(TORQUE_MODEL)
    jn = [mujoco.mj_id2name(mt, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(mt.nu)]
    need = np.array([abs(float(mt.actuator_gear[i, 0])) for i in range(mt.nu)])

    print(f"tension network: {mc.nu} cables")
    print(f"all-cables-driven mass: "
          f"{sum(motor_mass(c) for c in cnames):.1f} kg of motors "
          f"on a {mt.body_mass.sum():.1f} kg robot  <- not an option\n")

    sel_a, hp, hn = greedy(M, fmax, need, need, both=True)
    print(f"A. fully antagonistic: {len(sel_a)} driven cables, "
          f"{sum(motor_mass(cnames[c]) for c in sel_a):.1f} kg of motors")
    sel_b, _, _ = greedy(M, fmax, need, need, both=False)
    print(f"B. motor + elastic return: {len(sel_b)} driven cables, "
          f"{sum(motor_mass(cnames[c]) for c in sel_b):.1f} kg of motors")

    per = {}
    for c in sel_a:
        grp = "".join(ch for ch in cnames[c] if not ch.isdigit()).rstrip("_")
        per[grp] = per.get(grp, 0) + 1
    print("\ndriven cables per joint (architecture A):")
    for k in sorted(per):
        print(f"  {k:16s} {per[k]}")

    print(f"\npassive prestress cables (elastic + turnbuckle): "
          f"{mc.nu - len(sel_a)} of {mc.nu}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
