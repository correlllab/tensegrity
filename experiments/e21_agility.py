#!/usr/bin/env python
"""E21: push the hybrid's gait from safe shuffling toward agility.

The shipped gait undertracks (0.45 commanded -> 0.19 achieved) with a short
stride (cadence 1.1 Hz). Stride = v*d/f_c, so the same speed at lower
cadence means longer steps; clearance and cost shaping must follow. Sweep:

  stage A  cadence x speed grid, n=2, shipped weights, step 0.05
  stage B  top configs +- step height 0.08 and 2x Speed weight, n=2
  stage C  n=6 on the best two by achieved speed (success >= 5/6)

Success = min pelvis z in [3,15] s > 0.65 and distance > 1.2 m.
Ranking = achieved speed among configs with full success at n=2.
"""
import json
import os
import re
import subprocess
import numpy as np

ROOT = "/Users/ncorrell/Downloads/tensegrity"
TD = f"{ROOT}/mujoco_mpc/build/mjpc/tasks/tensegrity"
BIN = f"{ROOT}/mujoco_mpc/build/bin/testspeed"
RES = f"{ROOT}/experiments/results"
LOGD = f"{RES}/e21_agility"
os.makedirs(LOGD, exist_ok=True)
LIVE = f"{TD}/task_hybrid_walk.xml"
BASE = open(LIVE).read()


def configure(cadence, speed, step=0.05, w_speed=None, horizon=None):
    xml = BASE

    def num(name, val):
        return re.sub(rf'(name="residual_{name}" data=")[\d.+-]+',
                      rf'\g<1>{val}', xml)
    xml = num("Speed", speed)
    xml = num("Cadence", cadence)
    xml = num("Step Height", step)
    if w_speed is not None:
        xml = re.sub(r'(<user name="Speed"\s+dim="\d+"\s+user="\d+) [\d.]+',
                     rf'\g<1> {w_speed}', xml)
    if horizon is not None:
        xml = re.sub(r'(name="agent_horizon" data=")[\d.]+',
                     rf'\g<1>{horizon}', xml)
    return xml


def trial(xml, tag, t):
    log = f"{LOGD}/log_{tag}_t{t}.csv"
    try:
        open(LIVE, "w").write(xml)
        if not os.path.exists(log):
            env = dict(os.environ, TESTSPEED_LOG=log)
            subprocess.run([BIN, "--task=Hybrid Walk", "--total_time=16",
                            "--steps_per_planning_iteration=4"],
                           capture_output=True, text=True, env=env, cwd=TD)
    finally:
        open(LIVE, "w").write(BASE)
    try:
        raw = np.genfromtxt(log, delimiter=",", invalid_raise=False)
        raw = raw[~np.isnan(raw).any(axis=1)]
        tt, x, z = raw[:, 0], raw[:, 1], raw[:, 3]
        w = (tt >= 3.0) & (tt <= 15.0)
        ok = bool(z[w].min() > 0.65 and (x[w][-1] - x[w][0]) > 1.2)
        return ok, (x[w][-1] - x[w][0]) / 12.0
    except Exception:
        return False, 0.0


def screen(cfgs, n):
    out = []
    for tag, kw in cfgs:
        xml = configure(**kw)
        rows = [trial(xml, tag, t) for t in range(n)]
        ok = sum(r[0] for r in rows)
        v = float(np.mean([r[1] for r in rows if r[0]])) if ok else 0.0
        out.append(dict(tag=tag, cfg=kw, ok=ok, n=n, v=v))
        print(f"  {tag} {kw}: {ok}/{n} at {v:.3f} m/s", flush=True)
    return out


def main():
    print("stage A: cadence x speed, n=2")
    cfgs = [(f"a_c{c:.1f}_s{s:.2f}".replace(".", ""),
             dict(cadence=c, speed=s))
            for c in (0.8, 0.9, 1.0, 1.1) for s in (0.45, 0.55, 0.65)]
    A = screen(cfgs, 2)

    good = sorted([r for r in A if r["ok"] == 2],
                  key=lambda r: r["v"], reverse=True)[:2]
    print("stage B: step height and speed-weight variants on the top configs")
    cfgs = []
    for r in good:
        # NB suffixes must stay distinct on a case-insensitive filesystem:
        # an earlier "h"/"H" pair collided in the log filenames, so the
        # horizon variant silently reused the step-height variant's logs
        for suffix, extra in (("h", dict(step=0.08)),
                              ("w", dict(w_speed=2.0)),
                              ("hw", dict(step=0.08, w_speed=2.0)),
                              ("hz", dict(horizon=0.6))):
            cfgs.append((r["tag"] + suffix, dict(r["cfg"], **extra)))
    B = screen(cfgs, 2)

    pool = sorted([r for r in (good + B) if r["ok"] == r["n"]],
                  key=lambda r: r["v"], reverse=True)[:2]
    print("stage C: n=6 on the finalists")
    C = screen([(r["tag"] + "F", r["cfg"]) for r in pool], 6)

    best = max(C, key=lambda r: (r["ok"], r["v"])) if C else None
    json.dump(dict(A=A, B=B, C=C, best=best),
              open(f"{RES}/e21_agility.json", "w"), indent=1)
    print("E21 ->", f"{RES}/e21_agility.json")


if __name__ == "__main__":
    main()
