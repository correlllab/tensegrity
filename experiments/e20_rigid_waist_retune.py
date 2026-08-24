#!/usr/bin/env python
"""E20: retune the rigid-waist variant with the CURRENT 7-DoF arms.

The joint-count table's rigid-waist row fails (0/5) under the hybrid's
tuning, and the only retuned number available came from an older build with
simplified arms -- a confound the review flagged. This retunes the
rigid-waist variant of the current machine: coordinate screen around the
hybrid's shipped gait parameters (n=2), then n=5 on the best configs.

Success = min pelvis z in [3,15] s > 0.65 m AND net forward travel > 1.2 m.
"""
import itertools
import json
import os
import re
import subprocess
import sys
import numpy as np

sys.path.insert(0, "/Users/ncorrell/Downloads/tensegrity/experiments")
import e17_joint_count as e17

ROOT = "/Users/ncorrell/Downloads/tensegrity"
TD = f"{ROOT}/mujoco_mpc/build/mjpc/tasks/tensegrity"
BIN = f"{ROOT}/mujoco_mpc/build/bin/testspeed"
RES = f"{ROOT}/experiments/results"
LOGD = f"{RES}/e20_retune"
os.makedirs(LOGD, exist_ok=True)
LIVE = f"{TD}/task_hybrid_walk.xml"

BASE = dict(speed=0.45, cadence=1.1, sway=0.06, torso=1.20)
# coordinate variations around the hybrid winner
VARIANTS = [dict()] + \
    [dict(torso=v) for v in (1.16, 1.24, 1.28)] + \
    [dict(sway=v) for v in (0.05, 0.08)] + \
    [dict(cadence=v) for v in (1.0, 1.2)] + \
    [dict(speed=v) for v in (0.35, 0.55)]


def configure(xml, cfg):
    def num(name, val):
        return re.sub(rf'(name="residual_{name}" data=")[\d.+-]+',
                      rf'\g<1>{val}', xml)
    xml = num("Speed", cfg["speed"])
    xml = num("Cadence", cfg["cadence"])
    xml = num("Sway", cfg["sway"])
    xml = num("Torso", cfg["torso"])
    return xml


def trial(task_xml, tag, t):
    backup = open(LIVE).read()
    log = f"{LOGD}/log_{tag}_t{t}.csv"
    try:
        open(LIVE, "w").write(task_xml)
        if not os.path.exists(log):
            env = dict(os.environ, TESTSPEED_LOG=log)
            subprocess.run([BIN, "--task=Hybrid Walk", "--total_time=16",
                            "--steps_per_planning_iteration=4"],
                           capture_output=True, text=True, env=env, cwd=TD)
    finally:
        open(LIVE, "w").write(backup)
    try:
        raw = np.genfromtxt(log, delimiter=",", invalid_raise=False)
        raw = raw[~np.isnan(raw).any(axis=1)]
        tt, x, z = raw[:, 0], raw[:, 1], raw[:, 3]
        w = (tt >= 3.0) & (tt <= 15.0)
        ok = bool(z[w].min() > 0.65 and (x[w][-1] - x[w][0]) > 1.2)
        return ok, (x[w][-1] - x[w][0]) / 12.0
    except Exception:
        return False, 0.0


def main():
    mv, mpath, tpath = e17.make_variant("rigid_waist", False, True)
    print(f"rigid_waist rebuilt: nv={mv.nv} nu={mv.nu} "
          f"mass={mv.body_mass.sum():.1f} kg", flush=True)
    task0 = open(tpath).read()

    print("stage 1: coordinate screen, n=2")
    screen = []
    for i, dv in enumerate(VARIANTS):
        cfg = dict(BASE, **dv)
        tag = f"c{i:02d}"
        xml = configure(task0, cfg)
        rows = [trial(xml, tag, t) for t in range(2)]
        ok = sum(r[0] for r in rows)
        v = np.mean([r[1] for r in rows if r[0]]) if ok else 0.0
        screen.append(dict(cfg=cfg, tag=tag, ok=ok, v=float(v)))
        print(f"  {tag} {dv or 'base'}: {ok}/2 at {v:.3f} m/s", flush=True)

    screen.sort(key=lambda r: (r["ok"], r["v"]), reverse=True)
    best = None
    print("stage 2: n=5 on the top configs")
    for cand in screen[:3]:
        if cand["ok"] == 0:
            break
        xml = configure(task0, cand["cfg"])
        rows = [trial(xml, cand["tag"], t) for t in range(2, 7)]
        ok = sum(r[0] for r in rows)
        v = np.mean([r[1] for r in rows if r[0]]) if ok else 0.0
        print(f"  {cand['tag']} {cand['cfg']}: {ok}/5 at {v:.3f} m/s",
              flush=True)
        if best is None or (ok, v) > (best["ok"], best["v"]):
            best = dict(cfg=cand["cfg"], ok=ok, n=5, v=float(v))
    out = dict(screen=screen, best=best)
    json.dump(out, open(f"{RES}/e20_rigid_waist_retune.json", "w"), indent=1)
    print("E20 ->", f"{RES}/e20_rigid_waist_retune.json")


if __name__ == "__main__":
    main()
