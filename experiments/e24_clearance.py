#!/usr/bin/env python
"""E24: from shuffling to stepping.

The commanded swing height is 0.08 m but achieved clearance is 8-27 mm:
the planner under-tracks the cartesian foot reference (Gait weight 25)
in favour of balance and control costs. Screen higher step-height
references and stronger Gait tracking weights; select on success rate,
then achieved clearance, then speed. Verify the winner at n=10.
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
LOGD = f"{RES}/e24_clearance"
os.makedirs(LOGD, exist_ok=True)
LIVE = f"{TD}/task_hybrid_walk.xml"
BASE = open(LIVE).read()


def configure(step=None, w_gait=None, horizon=None, w_ctrl=None):
    xml = BASE
    if step is not None:
        xml = re.sub(r'(name="residual_Step Height" data=")[\d.+-]+',
                     rf'\g<1>{step}', xml)
    if w_gait is not None:
        xml = re.sub(r'(<user name="Gait"\s+dim="\d+"\s+user="\d+) [\d.]+',
                     rf'\g<1> {w_gait}', xml)
    if horizon is not None:
        xml = re.sub(r'(name="agent_horizon" data=")[\d.]+',
                     rf'\g<1>{horizon}', xml)
    if w_ctrl is not None:
        xml = re.sub(r'(<user name="Control"\s+dim="\d+"\s+user="\d+) [\d.]+',
                     rf'\g<1> {w_ctrl}', xml)
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
        v = (x[w][-1] - x[w][0]) / 12.0
        # achieved swing clearance: mean of per-cycle foot-z peaks above
        # stance level, both feet
        cls = []
        for zc in (raw[w, 1 + 17], raw[w, 1 + 24]):
            base = np.percentile(zc, 10)
            from scipy.signal import find_peaks
            pk, _ = find_peaks(zc, height=base + 0.005, distance=80)
            if len(pk):
                cls.append(float(np.mean(zc[pk] - base)))
        clr = 1e3 * float(np.mean(cls)) if cls else 0.0
        return ok, v, clr
    except Exception:
        return False, 0.0, 0.0


def screen(cfgs, n):
    out = []
    for tag, kw in cfgs:
        xml = configure(**kw)
        rows = [trial(xml, tag, t) for t in range(n)]
        ok = sum(r[0] for r in rows)
        v = float(np.mean([r[1] for r in rows if r[0]])) if ok else 0.0
        c = float(np.mean([r[2] for r in rows if r[0]])) if ok else 0.0
        out.append(dict(tag=tag, cfg=kw, ok=ok, n=n, v=v, clr=c))
        print(f"  {tag}: {ok}/{n} at {v:.3f} m/s, clearance {c:.0f} mm",
              flush=True)
    return out


def main():
    print("E24 round 3: horizon/control, n=3")
    cfgs = [
        ("h60",    dict(step=0.16, w_gait=50.0, horizon=0.60)),
        ("h75",    dict(step=0.16, w_gait=50.0, horizon=0.75)),
        ("s24",    dict(step=0.24, w_gait=50.0)),
        ("h60s24", dict(step=0.24, w_gait=50.0, horizon=0.60)),
        ("ctrl",   dict(step=0.16, w_gait=50.0, w_ctrl=0.04)),
    ]
    A = screen(cfgs, 3)
    good = sorted([r for r in A if r["ok"] == 3],
                  key=lambda r: (r["clr"], r["v"]), reverse=True)[:2]
    print("verify n=10 on:", [r["tag"] for r in good])
    V = []
    for r in good:
        xml = configure(**r["cfg"])
        rows = [trial(xml, r["tag"] + "v", t) for t in range(10)]
        ok = sum(x[0] for x in rows)
        v = float(np.mean([x[1] for x in rows if x[0]])) if ok else 0.0
        c = float(np.mean([x[2] for x in rows if x[0]])) if ok else 0.0
        V.append(dict(tag=r["tag"], cfg=r["cfg"], ok=ok, n=10, v=v, clr=c))
        print(f"  VERIFY {r['tag']}: {ok}/10 at {v:.3f} m/s, "
              f"clearance {c:.0f} mm", flush=True)
    json.dump(dict(screen=A, verify=V),
              open(f"{RES}/e24_round3.json", "w"), indent=1)
    print("E24 r3 ->", f"{RES}/e24_round3.json")


if __name__ == "__main__":
    main()
