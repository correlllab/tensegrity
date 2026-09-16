#!/usr/bin/env python
"""Render Fig. 7 (sweeps.pdf) from the hybrid platform data, with the fully
hinged machine's results as faint comparison curves where they exist."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/ncorrell/Downloads/tensegrity"
RES = f"{ROOT}/experiments/results"
FIG = f"{ROOT}/paper/figures"
C = dict(hyb="#1f77b4", old="0.62", rate="#c8781e", warn="#d62728")
plt.rcParams.update({"font.size": 8.5, "axes.grid": True, "grid.alpha": 0.3,
                     "grid.linewidth": 0.5, "legend.fontsize": 7.0})


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main():
    hs = json.load(open(f"{RES}/hybrid_sweeps.json"))
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))

    # (a) speed envelope
    ax = axs[0]
    sp = sorted(float(k) for k in hs["speed"])
    A = [hs["speed"][str(s)] for s in sp]
    good = [i for i, a in enumerate(A) if a["ok"]]
    ax.plot([sp[i] for i in good], [A[i]["v"] for i in good], "-o",
            color=C["hyb"], ms=4, lw=1.3, label="hybrid")
    # upright-but-undertracking commands (0/n "walks" but no falls): show the
    # achieved crawl speed as open markers so the failure mode is legible
    UNDER = {0.10: 0.017, 0.20: 0.051}
    ax.plot(list(UNDER), list(UNDER.values()), "o", mfc="none",
            color=C["hyb"], ms=4, mew=1.1)
    # fully hinged reference: achieved speed and success rate from the E-suite
    # (results/hinged_speed_sweep.json, written by make_numbers.py)
    old_cmd = [0.10, 0.20, 0.30, 0.35, 0.40, 0.45]
    old_v = [np.nan, 0.13, 0.21, 0.24, 0.26, 0.29]
    hinged_rate = None
    try:
        _h = json.load(open(f"{RES}/hinged_speed_sweep.json"))
        old_cmd = sorted(float(k) for k in _h)
        old_v = [(_h[f"{c:.2f}"]["v"] if _h[f"{c:.2f}"]["v"] is not None else np.nan)
                 for c in old_cmd]
        hinged_rate = [_h[f"{c:.2f}"]["ok"] / max(_h[f"{c:.2f}"]["n"], 1) for c in old_cmd]
    except FileNotFoundError:
        pass
    ax.plot(old_cmd, old_v, "--s", color=C["old"], ms=3, lw=1.0,
            label="fully hinged (27 DoF)")
    # falls at over-speed commands: mark mean achieved-before-fall
    falls = [(s_, a) for s_, a in zip(sp, A) if a["ok"] == 0 and s_ > 0.25]
    if falls:
        ax.plot([f[0] for f in falls], [0.21 for f in falls], "x",
                color=C["warn"], ms=5, mew=1.4)
    ax.plot([0, 0.62], [0, 0.62], ":", color="0.75", lw=1)
    ax2 = ax.twinx()
    for s, a in zip(sp, A):
        lo, hi = wilson(a["ok"], a["n"])
        p = a["ok"] / max(a["n"], 1)
        ax2.errorbar(s, p, yerr=[[max(0, p - lo)], [max(0, hi - p)]],
                     fmt="^", color=C["rate"], ms=3.5, capsize=2, lw=0.9)
    hs_line, = ax2.plot(sp, [a["ok"] / max(a["n"], 1) for a in A], "-^",
                        color=C["rate"], ms=3.5, lw=0.9, alpha=0.7,
                        label="success, hybrid")
    hh_line = None
    if hinged_rate is not None:
        hh_line, = ax2.plot(old_cmd, hinged_rate, "--v", color=C["old"], ms=3,
                            lw=0.9, alpha=0.9, label="success, hinged")
    ax2.set_ylabel("success rate", color=C["rate"])
    ax2.tick_params(axis="y", colors=C["rate"])
    ax2.set_ylim(-0.05, 1.45)               # headroom for the legend
    ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.grid(False)
    ax.set_ylim(0, 0.8)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.set_xlabel("commanded speed [m/s]")
    ax.set_ylabel("achieved speed [m/s]", color=C["hyb"])
    ax.set_title(f"(a) speed envelope, $n={A[0]['n']}$", fontsize=9)
    handles = [h for h in ax.get_legend_handles_labels()[0]] + [hs_line] + \
              ([hh_line] if hh_line is not None else [])
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.01),
              ncol=2, fontsize=6.2, handletextpad=0.4, labelspacing=0.25,
              columnspacing=0.8, framealpha=0.9)

    # (b) payload
    ax = axs[1]
    mk = sorted(float(k) for k in hs["payload"])
    P = [hs["payload"][str(int(m))] for m in mk]
    good = [i for i, a in enumerate(P) if a["ok"]]
    ax.plot([mk[i] for i in good], [P[i]["v"] for i in good], "-o",
            color=C["hyb"], ms=4, lw=1.3, label="speed")
    ax.set_xlabel("trunk payload on the waist cables [kg]")
    ax.set_ylabel("walking speed [m/s]", color=C["hyb"])
    ax.set_ylim(0, 0.28)
    ax2 = ax.twinx()
    for m, a in zip(mk, P):
        lo, hi = wilson(a["ok"], a["n"])
        p = a["ok"] / max(a["n"], 1)
        ax2.errorbar(m, p, yerr=[[max(0, p - lo)], [max(0, hi - p)]],
                     fmt="^", color=C["rate"], ms=3.5, capsize=2, lw=0.9)
    ax2.plot(mk, [a["ok"] / max(a["n"], 1) for a in P], "-",
             color=C["rate"], lw=0.9, alpha=0.7)
    ax2.set_ylabel("success rate", color=C["rate"])
    ax2.tick_params(axis="y", colors=C["rate"])
    ax2.set_ylim(-0.05, 1.15)
    ax2.grid(False)
    ax.set_title(f"(b) trunk payload, $n={P[0]['n']}$", fontsize=9)

    # (c) matched arm carry (E18): both machines, one protocol
    e18 = json.load(open(f"{RES}/e18_matched_carry.json"))
    ax = axs[2]
    for key, col, sty, lab in (
            ("hybrid", C["hyb"], "-o", "hybrid (15 Nm XM540)"),
            ("hinged", C["old"], "--s", "fully hinged (30 Nm)"),
            ("hinged_half", C["old"], ":^", "hinged, 15 Nm shoulder")):
        if key not in e18:
            continue
        rows = e18[key]
        ax.plot([r["mass"] for r in rows],
                [1e3 * r["drop_m"] for r in rows],
                sty, color=col, ms=3.5, lw=1.2, label=lab)
    for th in (80, 120, 160):
        ax.axhline(th, color="0.75", lw=0.7, ls=":")
    ax.text(0.55, 168, "thresholds 8/12/16 cm", fontsize=6.5, color="0.45")
    hyb = e18["hybrid"]
    sat = next(r["mass"] for r in hyb if r["shoulder_util"] >= 0.99)
    ax.annotate("shoulder\nsaturates", xy=(sat, 276), xytext=(5.4, 205),
                fontsize=6.5, arrowprops=dict(arrowstyle="-", lw=0.7))
    ax.set_xlabel("box mass [kg] (single arm)")
    ax.set_ylabel("hand drop under load [mm]")
    ax.set_ylim(0, 430)
    ax.set_title("(c) static arm carry", fontsize=9)
    ax.legend(loc="upper left", fontsize=6.2, handlelength=1.6)

    fig.tight_layout()
    fig.savefig(f"{FIG}/sweeps.pdf")
    print("sweeps.pdf (hybrid)")


if __name__ == "__main__":
    main()
