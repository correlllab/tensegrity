#!/usr/bin/env python
"""Paper plots from experiment JSONs and raw logs -> paper/figures/*.pdf

Walking metrics are recomputed from the logs with the current success
criterion rather than read from a cached summary, so the figures cannot
disagree with the text.
"""
import glob
import json
import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/ncorrell/Downloads/tensegrity"
RES = f"{ROOT}/experiments/results"
FIG = f"{ROOT}/paper/figures"
MJ = f"{ROOT}/mujoco"
sys.path.insert(0, f"{ROOT}/experiments")
import walklog                                          # noqa: E402

C = dict(tsg="#1f77b4", tsg3="#17becf", held="#c8781e", rheld="#d62728",
         limp="0.5", adv="#8c564b")
plt.rcParams.update({"font.size": 8.5, "axes.grid": True,
                     "grid.alpha": 0.3, "grid.linewidth": 0.5,
                     "legend.fontsize": 7.0})


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


_CACHE = {}


def agg(tag):
    if tag in _CACHE:
        return _CACHE[tag]
    rows = []
    for f in sorted(glob.glob(f"{RES}/log_{tag}_t*.csv")):
        pay = 0.0
        if tag.startswith("cpayload"):
            pay = float(tag[8:])
        elif tag.startswith("payload"):
            pay = float(tag[7:])
        try:
            rows.append(walklog.metrics(f, total_mass=27.6 + pay))
        except Exception:
            pass
    ok = [r for r in rows if r["success"]]
    out = dict(n=len(rows), ok=len(ok), rows=rows,
               v=np.mean([r["speed"] for r in ok]) if ok else np.nan,
               vsd=np.std([r["speed"] for r in ok]) if len(ok) > 1 else 0.0,
               ci=wilson(len(ok), len(rows)))
    for g in ("hip", "knee", "ankle", "waist"):
        out[f"peak_{g}"] = max([r["per_group"][g]["peak"] for r in rows],
                               default=np.nan)
        out[f"rms_{g}"] = np.mean([r["per_group"][g]["rms"] for r in rows]) \
            if rows else np.nan
    _CACHE[tag] = out
    return out


def rate_bar(ax, x, a, color, width):
    lo, hi = a["ci"]
    p = a["ok"] / max(a["n"], 1)
    ax.bar(x, p, width=width, color=color)
    ax.errorbar(x, p, yerr=[[p - lo], [hi - p]], fmt="none", ecolor="0.25",
                capsize=2.5, lw=1.0)


def fig_experiments():
    e1 = json.load(open(f"{RES}/e1_drop.json"))
    e2 = json.load(open(f"{RES}/e2_stiffness.json"))
    e3 = json.load(open(f"{RES}/e3_degradation.json"))
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.2))
    W = 27.6 * 9.81

    # (a) E1 -- matched pairs only; the mismatched pair is what misled us
    ax = axs[0, 0]
    pairs = [("tensegrity-passive", "cables, limp", C["tsg"], "-o"),
             ("rigid-limp", "no cables, limp", C["limp"], "--o"),
             ("tensegrity-held", "cables, servo-held", C["held"], "-^"),
             ("rigid-held", "no cables, servo-held", C["rheld"], "--^")]
    for mode, lab, c, ls in pairs:
        rows = sorted([r for r in e1 if r["mode"] == mode], key=lambda r: r["h"])
        ax.plot([r["h"] for r in rows], [r["peak_grf"] / W for r in rows],
                ls, color=c, label=lab, ms=3.5, lw=1.2)
    ax.set_xlabel("drop height [m]")
    ax.set_ylabel("peak GRF [body weights]")
    ax.set_title("(a) E1: matched-condition landing impact", fontsize=9)
    ax.legend(loc="upper left")
    ax.annotate("cables raise the peak in the\nlimp pair; neutral when held",
                xy=(0.26, 0.78), xycoords="axes fraction", fontsize=7,
                color="0.25")

    # (b) E2 -- prestress is flat; co-contraction is not
    ax = axs[0, 1]
    pr = sorted([r for r in e2["prestress"] if r["prestress_scale"] >= 0],
                key=lambda r: r["prestress_scale"])
    ax.plot([r["prestress_scale"] for r in pr], [r["k_lat"] for r in pr],
            "-o", color=C["tsg"], ms=3.5, lw=1.2, label="assembly prestress")
    nc = [r for r in e2["prestress"] if r["prestress_scale"] < 0][0]
    ax.axhline(nc["k_lat"], color=C["limp"], ls="--", lw=1,
               label="no cables (gravity only)")
    cc = sorted(e2["cocontraction"], key=lambda r: r["cocontraction_N"])
    ax2 = ax.twiny()
    ax2.plot([r["cocontraction_N"] for r in cc], [r["k_lat"] for r in cc],
             "-s", color=C["held"], ms=3.5, lw=1.2, label="co-contraction")
    ax2.set_xlabel("shoulder co-contraction [N]", color=C["held"])
    ax2.tick_params(axis="x", colors=C["held"])
    ax2.grid(False)
    ax.set_xlabel("assembly prestress [$\\times$ nominal]")
    ax.set_ylabel("lateral endpoint stiffness [N/m]")
    ax.set_title("(b) E2: what actually sets stiffness", fontsize=9)
    ax.legend(loc="center left")

    # (c) E3 -- random vs adversarial cable loss
    ax = axs[1, 0]
    ks = sorted({m["k"] for m in e3["multi"]})
    rules = [("random-all", "random, whole network", C["tsg"]),
             ("random-stance", "random, leg + waist", C["tsg3"]),
             ("adversarial", "adversarial (worst-margin)", C["adv"])]
    w = 0.26
    for i, (rule, lab, c) in enumerate(rules):
        xs, ys, los, his = [], [], [], []
        for j, k in enumerate(ks):
            row = [m for m in e3["multi"] if m["k"] == k and m["rule"] == rule]
            if not row:
                continue
            r = row[0]
            p = r["survived"] / r["trials"]
            lo, hi = wilson(r["survived"], r["trials"])
            xs.append(j + (i - 1) * w)
            ys.append(p)
            los.append(p - lo)
            his.append(hi - p)
        ax.bar(xs, ys, width=w, color=c, label=lab)
        ax.errorbar(xs, ys, yerr=[los, his], fmt="none", ecolor="0.25",
                    capsize=2, lw=0.9)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([f"k = {k}" for k in ks])
    ax.set_ylim(0, 1.62)
    ax.set_ylabel("stand + push survival rate")
    ax.set_xlabel("cables severed")
    ax.set_title("(c) E3: degradation depends on who chooses", fontsize=9)
    ax.legend(loc="upper center", ncol=1, framealpha=0.9)

    # (d) E4 -- proximal vs distal, n = 10
    ax = axs[1, 1]
    conds = [("baseline", "proximal\n(design)", C["tsg"]),
             ("distal", "distal\n(conventional)", C["rheld"])]
    for i, (tag, lab, c) in enumerate(conds):
        a = agg(tag)
        rate_bar(ax, i, a, c, 0.45)
        ax.text(i, 1.06, f"{a['ok']}/{a['n']} walks\n"
                         f"{a['v']:.2f} m/s", ha="center", fontsize=7.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([c[1] for c in conds])
    ax.set_ylim(0, 1.3)
    ax.set_ylabel("walk success rate (Wilson 95\\%)")
    ax.set_title("(d) E4: motor placement, $n=10$", fontsize=9)

    fig.tight_layout()
    fig.savefig(f"{FIG}/experiments.pdf")
    plt.close(fig)
    print("experiments.pdf")


SPEEDS = [0.10, 0.20, 0.30, 0.35, 0.40, 0.45]
MASSES = [0, 2, 4, 6, 8, 10, 12, 16, 20]


def speed_tag(v):
    return "baseline" if abs(v - 0.30) < 1e-9 else f"speed{v:.2f}"


def mass_tag(m):
    return "baseline" if m == 0 else f"cpayload{int(m):02d}"


def fig_sweeps():
    carry = json.load(open(f"{RES}/payload_carry.json"))
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))

    # (a) speed: achieved speed + success rate
    ax = axs[0]
    A = [agg(speed_tag(v)) for v in SPEEDS]
    good = [i for i, a in enumerate(A) if a["ok"]]
    ax.errorbar([SPEEDS[i] for i in good], [A[i]["v"] for i in good],
                yerr=[A[i]["vsd"] for i in good], fmt="-o", color=C["tsg"],
                ms=4, capsize=2, lw=1.2, label="achieved speed")
    ax.plot([0, 0.47], [0, 0.47], ":", color="0.6", lw=1, label="ideal")
    ax.set_xlabel("commanded speed [m/s]")
    ax.set_ylabel("achieved speed [m/s]", color=C["tsg"])
    ax2 = ax.twinx()
    for v, a in zip(SPEEDS, A):
        lo, hi = a["ci"]
        p = a["ok"] / max(a["n"], 1)
        ax2.errorbar(v, p, yerr=[[p - lo], [hi - p]], fmt="s", color=C["held"],
                     ms=3.5, capsize=2, lw=0.9)
    ax2.plot(SPEEDS, [a["ok"] / max(a["n"], 1) for a in A], "-", color=C["held"],
             lw=1.0, alpha=0.7)
    ax2.set_ylabel("success rate", color=C["held"])
    ax2.tick_params(axis="y", colors=C["held"])
    ax2.set_ylim(-0.05, 1.15)
    ax2.grid(False)
    ax.set_title("(a) speed envelope, $n=10$", fontsize=9)
    ax.legend(loc="upper left")

    # (b) payload
    ax = axs[1]
    A = [agg(mass_tag(m)) for m in MASSES]
    good = [i for i, a in enumerate(A) if a["ok"]]
    ax.errorbar([MASSES[i] for i in good], [A[i]["v"] for i in good],
                yerr=[A[i]["vsd"] for i in good], fmt="-o", color=C["tsg"],
                ms=4, capsize=2, lw=1.2, label="speed")
    ax.set_xlabel("centred trunk payload [kg]")
    ax.set_ylabel("walking speed [m/s]", color=C["tsg"])
    ax.set_ylim(0, 0.28)
    ax2 = ax.twinx()
    for m, a in zip(MASSES, A):
        lo, hi = a["ci"]
        p = a["ok"] / max(a["n"], 1)
        ax2.errorbar(m, p, yerr=[[p - lo], [hi - p]], fmt="s", color=C["held"],
                     ms=3.5, capsize=2, lw=0.9)
    ax2.plot(MASSES, [a["ok"] / max(a["n"], 1) for a in A], "-",
             color=C["held"], lw=1.0, alpha=0.7)
    ax2.set_ylabel("success rate", color=C["held"])
    ax2.tick_params(axis="y", colors=C["held"])
    ax2.set_ylim(-0.05, 1.15)
    ax2.grid(False)
    ax.set_title("(b) trunk payload (flat to 20 kg)", fontsize=9)

    # (c) arm carry
    ax = axs[2]
    mkg = [r["mass"] for r in carry]
    util = [r["torque_util"] for r in carry]
    held = [r["held"] for r in carry]
    Tsh = [r["peak_shoulder_T"] for r in carry]
    ax.plot(mkg, util, "-o", color=C["tsg"], ms=4, label="arm torque / limit")
    ax.axhline(1.0, color=C["rheld"], ls="--", lw=1, label="motor limit")
    ax2 = ax.twinx()
    ax2.plot(mkg, np.array(Tsh) / 700.0, "-s", color=C["held"], ms=3.5)
    ax2.set_ylabel("cable tension / limit", color=C["held"])
    ax2.tick_params(axis="y", colors=C["held"])
    ax2.set_ylim(0, 1.1)
    ax2.grid(False)
    hmax = max(m for m, h in zip(mkg, held) if h)
    ax.axvline(hmax + 0.5, color="0.4", lw=0.8, ls=":")
    ax.text(hmax + 0.6, 0.25, f"max hold\n{hmax:.0f} kg/arm", fontsize=7.5)
    ax.set_xlabel("box mass [kg] (single arm)")
    ax.set_ylabel("utilization")
    ax.set_ylim(0, 1.15)
    ax.set_title("(c) static arm carry", fontsize=9)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(f"{FIG}/sweeps.pdf")
    plt.close(fig)
    print("sweeps.pdf")


def fig_tradecurves():
    """Requirements vs operating point -- the output an outer design loop
    would consume."""
    ta = json.load(open(f"{RES}/tension_audit.json"))
    byp = ta.get("by_payload", {})
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.6))
    groups = [("hip", C["tsg"]), ("knee", C["held"]), ("ankle", C["rheld"]),
              ("waist", C["tsg3"])]

    for row, (xs, tags, xlabel) in enumerate([
            (SPEEDS, [speed_tag(v) for v in SPEEDS], "commanded speed [m/s]"),
            (MASSES, [mass_tag(m) for m in MASSES],
             "carried payload [kg] (total mass $=27.6+m$)")]):
        ax = axs[row, 0]
        for g, c in groups:
            A = [agg(t) for t in tags]
            ax.plot(xs, [a[f"peak_{g}"] for a in A], "-o", color=c, ms=3,
                    lw=1.1, label=f"{g} peak")
            ax.plot(xs, [a[f"rms_{g}"] for a in A], "--", color=c, lw=0.9,
                    alpha=0.7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("joint torque [N m]")
        ax.set_title("peak (solid) and RMS (dashed) joint torque", fontsize=9)
        if row == 0:
            ax.legend(ncol=2, loc="center right")
        ax.annotate("peak is pinned at the limit", xy=(0.03, 0.06),
                    xycoords="axes fraction", fontsize=7, color="0.3")

        ax = axs[row, 1]
        if row == 1 and byp:
            ms = sorted(float(k) for k in byp)
            ax.plot(ms, [byp[str(int(m)) if m else "0"]["fmax_needed"]
                         for m in ms], "-o", color=C["tsg"], ms=4, lw=1.2,
                    label="required cable rating (p99)")
            ax.axhline(1500, color="0.4", ls="--", lw=1)
            ax.text(0.3, 1560, "specified 1.5 kN limit", fontsize=7,
                    color="0.35")
            ax.set_xlabel("carried payload [kg]")
            ax.set_ylabel("required cable rating [N]", color=C["tsg"])
            ax.set_ylim(0, 3300)
            axb = ax.twinx()
            axb.plot(ms, [100 * byp[str(int(m)) if m else "0"]["sat_frac"]
                          for m in ms], "-s", color=C["rheld"], ms=3.5, lw=1.1)
            axb.set_ylabel("\\% of stride at the bound", color=C["rheld"])
            axb.tick_params(axis="y", colors=C["rheld"])
            axb.set_ylim(0, 105)
            axb.grid(False)
            ax.set_title("what the cables are asked for", fontsize=9)
            ax.legend(loc="lower right")
        else:
            A = [agg(t) for t in tags]
            ok = [a["ok"] / max(a["n"], 1) for a in A]
            ax.plot(xs, ok, "-o", color=C["adv"], ms=4, lw=1.2)
            ax.set_ylim(-0.05, 1.1)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("walk success rate")
            ax.set_title("reliability at the operating point", fontsize=9)

    fig.tight_layout()
    fig.savefig(f"{FIG}/tradecurves.pdf")
    plt.close(fig)
    print("tradecurves.pdf")


if __name__ == "__main__":
    fig_experiments()
    fig_sweeps()
    fig_tradecurves()
