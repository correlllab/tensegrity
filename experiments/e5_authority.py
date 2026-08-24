#!/usr/bin/env python
"""E5: cable authority, evaluated honestly.

The first version of Eq. (5) was evaluated (a) at the home pose only and
(b) over all 114 cables, i.e. assuming every cable in the network is
independently commandable. Neither holds for the proposed robot: moment arms
are configuration dependent, and the bill of materials drives only 36 loops.

This script recomputes the bound

    tau_j^+ = sum_c max(0, M_jc) Fmax_c ,  tau_j^- = sum_c min(0, M_jc) Fmax_c

for three actuation sets

    all    : all 114 cables (the optimistic bound reported previously)
    loops  : the 36 loop drives of the BOM (2 cables each, 72 driven)
    loops+ : loops, plus what a passive prestress element can hold

over two configuration sets

    home   : the assembly keyframe
    gait   : poses sampled from a logged walking trajectory

and reports, per DoF and per direction, the margin against the joint-torque
model's limits. It also reports the effective moment arm each DoF would need
for a single loop to cover its requirement at the 1.5 kN cable limit -- the
geometric target for the next design iteration.
"""
import json
import glob
import numpy as np
import mujoco

MJ = "/Users/ncorrell/Downloads/tensegrity/mujoco"
RES = "/Users/ncorrell/Downloads/tensegrity/experiments/results"
OUT = f"{RES}/e5_authority.json"
CABLE = f"{MJ}/humanoid_27dof_tensegrity_cable.xml"
TORQUE = f"{MJ}/humanoid_27dof_tensegrity.xml"
# joints carrying a second (antagonist) motor for variable stiffness
COCONTRACT = ("waist", "shoulder")


def moment(mc, dc, q=None):
    if q is not None:
        dc.qpos[:] = q
    mujoco.mj_forward(mc, dc)
    mom = np.zeros((mc.nu, mc.nv))
    mujoco.mju_sparse2dense(mom, dc.actuator_moment, dc.moment_rownnz,
                            dc.moment_rowadr, dc.moment_colind)
    return mom[:, 6:].T                       # (27, ncable), gear folded in


def select_loops(Ms, jn):
    """One loop drive per DoF: the cable pair with the largest +/- moment arm
    about that DoF. Joints in COCONTRACT get a second loop (the next best
    pair) -- 36 motors, 2 cables each.

    `Ms` is a list of moment matrices. Scoring each candidate cable by its
    WORST value over that list is what makes the selection gait-aware: pass
    [M_home] to reproduce the assembly-pose choice, or the gait poses to pick
    cables that keep their moment arm through the stride."""
    stack = np.stack(Ms)                       # (npose, ndof, ncable)
    score_p = np.maximum(stack, 0).min(axis=0)
    score_n = np.maximum(-stack, 0).min(axis=0)
    driven, per_dof = set(), {}
    for j, name in enumerate(jn):
        # a cable belongs to exactly one loop: a motor cannot share it
        order_p = [c for c in np.argsort(-score_p[j]) if c not in driven]
        order_n = [c for c in np.argsort(-score_n[j]) if c not in driven]
        picks = []
        for k in range(2 if name.startswith(COCONTRACT) else 1):
            p_, n_ = int(order_p[k]), int(order_n[k])
            if n_ == p_:
                n_ = int(order_n[k + 1])
            picks += [p_, n_]
        per_dof[name] = picks
        driven.update(picks)
    return sorted(driven), per_dof


def bounds(M, cols):
    """M is d(torque)/d(ctrl) with the gear (i.e. Fmax) already folded in by
    MuJoCo, so the column sums are already in N m at full tension."""
    sub = np.zeros_like(M)
    sub[:, cols] = M[:, cols]
    return np.maximum(sub, 0).sum(axis=1), np.maximum(-sub, 0).sum(axis=1)


def main():
    mc = mujoco.MjModel.from_xml_path(CABLE)
    dc = mujoco.MjData(mc)
    mujoco.mj_resetDataKeyframe(mc, dc, 0)
    fmax = -mc.actuator_gear[:, 0]
    cnames = [mujoco.mj_id2name(mc, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
              for i in range(mc.nu)]
    mt = mujoco.MjModel.from_xml_path(TORQUE)
    jn = [mujoco.mj_id2name(mt, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
          for i in range(mt.nu)]
    need = np.abs(mt.actuator_gear[:, 0])

    M_home = moment(mc, dc)
    n_motors = sum(2 if n.startswith(COCONTRACT) else 1 for n in jn)
    allc = list(range(mc.nu))

    # ---- gait poses from the logged walks
    qs = []
    for f in sorted(glob.glob(f"{RES}/log_baseline_t*.csv"))[:3]:
        raw = np.genfromtxt(f, delimiter=",", invalid_raise=False)
        raw = raw[~np.isnan(raw).any(axis=1)]
        t = raw[:, 0]
        sel = np.where((t >= 3.0) & (t <= 15.0))[0][::40]
        qs.append(raw[sel, 1:1 + mt.nq])
    qs = np.vstack(qs) if qs else np.zeros((0, mt.nq))
    Ms_gait = [moment(mc, dc, q) for q in qs]

    driven, per_dof = select_loops([M_home], jn)              # assembly-pose BOM
    driven_g, per_dof_g = select_loops(Ms_gait or [M_home], jn)  # gait-aware BOM

    sets = {"all": allc, "loops": driven, "loopsgait": driven_g}
    res = {"n_cables": mc.nu, "n_driven": len(driven),
           "n_driven_gait": len(driven_g), "n_motors": n_motors,
           "dofs": jn, "need": need.tolist(),
           "driven_home": [cnames[c] for c in driven],
           "driven_gait": [cnames[c] for c in driven_g]}

    for sname, cols in sets.items():
        P, N = bounds(M_home, cols)
        res[f"home_{sname}"] = dict(pos=P.tolist(), neg=N.tolist(),
                                    margin=np.minimum(P, N).tolist())
        if Ms_gait:
            per = np.array([bounds(Mq, cols) for Mq in Ms_gait])  # (npose,2,ndof)
            Pw, Nw = per[:, 0].min(axis=0), per[:, 1].min(axis=0)
            res[f"gait_{sname}"] = dict(
                pos=Pw.tolist(), neg=Nw.tolist(),
                margin=np.minimum(Pw, Nw).tolist(),
                # a joint may lose authority at an instant without failing;
                # the p10 envelope is the usable-torque estimate, the min the
                # worst case
                pos_p10=np.percentile(per[:, 0], 10, axis=0).tolist(),
                neg_p10=np.percentile(per[:, 1], 10, axis=0).tolist(),
                pos_med=np.median(per[:, 0], axis=0).tolist(),
                neg_med=np.median(per[:, 1], axis=0).tolist())

    # ---- single-loop capacity and the moment arm that would fix it
    arms = np.abs(M_home) / fmax
    single, r_need, r_have, cbl = [], [], [], []
    for j, name in enumerate(jn):
        c = int(np.argmax(arms[j]))
        single.append(float(arms[j, c] * fmax[c]))
        r_have.append(float(arms[j, c]))
        r_need.append(float(need[j] / fmax[c]))
        cbl.append(cnames[c])
    res.update(single_loop=single, r_have=r_have, r_need=r_need,
               best_cable=cbl)

    json.dump(res, open(OUT, "w"), indent=1)

    print(f"{len(driven)} driven cables ({n_motors} loop motors), "
          f"{mc.nu - len(driven)} passive prestress elements\n")
    hdr = (f"{'DoF':18s} {'need':>5s} | {'all/home':>9s} {'all/gait':>9s} "
           f"| {'loop/gait':>9s} {'loopG/gait':>10s} | {'1 loop Nm':>9s} "
           f"{'r_eff mm':>8s} {'r_need mm':>9s}")
    print(hdr)
    print("-" * len(hdr))

    def mar(tag, j):
        return min(res[tag]["pos"][j], res[tag]["neg"][j]) / need[j]

    for j, name in enumerate(jn):
        lg = mar("gait_loops", j)
        lgg = mar("gait_loopsgait", j)
        flag = "  <-- short" if lgg < 1.0 else ""
        print(f"{name:18s} {need[j]:5.0f} | {mar('home_all', j):8.2f}x "
              f"{mar('gait_all', j):8.2f}x | {lg:8.2f}x {lgg:9.2f}x "
              f"| {single[j]:9.0f} {r_have[j]*1000:8.1f} "
              f"{r_need[j]*1000:9.1f}{flag}")

    for tag in ("home_all", "gait_all", "home_loops", "gait_loops",
                "home_loopsgait", "gait_loopsgait"):
        mar = np.array(res[tag]["margin"]) / need
        print(f"\n{tag:12s} worst margin {mar.min():.2f}x at "
              f"{jn[int(mar.argmin())]}; {(mar < 1).sum()}/{len(jn)} DoFs short")
    print("\nE5 ->", OUT)


if __name__ == "__main__":
    main()
