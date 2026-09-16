#!/usr/bin/env python
"""Rerun E21's stage-B horizon rows, which never actually ran.

The original run named the step-height variant "h" and the horizon variant
"H"; on a case-insensitive filesystem the log filenames collide, and since
trials are skipped when their log exists, the "H" rows silently reused the
"h" logs (they are byte-identical in e21_agility.json). This reruns the two
horizon configs with a distinct "hz" suffix, at the stage-B configuration
(pre-retune task: Gait weight 25), and rewrites the affected rows in
e21_agility.json in place.
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
LIVE = f"{TD}/task_hybrid_walk.xml"
JS = f"{RES}/e21_agility.json"

# stage-B era task: current LIVE carries the later retune's Gait weight 50;
# E21 ran at 25
BASE = re.sub(r'(<user name="Gait"\s+dim="\d+"\s+user="\d+) [\d.]+',
              r'\g<1> 25.0', open(LIVE).read())


def configure(cadence, speed, step=0.05, w_speed=None, horizon=None):
    xml = BASE

    def num(name, val, s):
        return re.sub(rf'(name="residual_{name}" data=")[\d.+-]+',
                      rf'\g<1>{val}', s)
    xml = num("Speed", speed, xml)
    xml = num("Cadence", cadence, xml)
    xml = num("Step Height", step, xml)
    if w_speed is not None:
        xml = re.sub(r'(<user name="Speed"\s+dim="\d+"\s+user="\d+) [\d.]+',
                     rf'\g<1> {w_speed}', xml)
    if horizon is not None:
        xml = re.sub(r'(name="agent_horizon" data=")[\d.]+',
                     rf'\g<1>{horizon}', xml)
    return xml


def trial(xml, tag, t):
    log = f"{LOGD}/log_{tag}_t{t}.csv"
    backup = open(LIVE).read()
    try:
        open(LIVE, "w").write(xml)
        if not os.path.exists(log):
            env = dict(os.environ, TESTSPEED_LOG=log)
            subprocess.run([BIN, "--task=Hybrid Walk", "--total_time=16",
                            "--steps_per_planning_iteration=4"],
                           capture_output=True, text=True, env=env, cwd=TD)
    finally:
        open(LIVE, "w").write(backup)
    raw = np.genfromtxt(log, delimiter=",", invalid_raise=False)
    raw = raw[~np.isnan(raw).any(axis=1)]
    tt, x, z = raw[:, 0], raw[:, 1], raw[:, 3]
    w = (tt >= 3.0) & (tt <= 15.0)
    ok = bool(z[w].min() > 0.65 and (x[w][-1] - x[w][0]) > 1.2)
    return ok, (x[w][-1] - x[w][0]) / 12.0


def main():
    js = json.load(open(JS))
    bad = [r for r in js.get("B", []) if r["tag"].endswith("H")]
    if not bad:
        print("no colliding horizon rows found; nothing to do")
        return
    for r in bad:
        tag = r["tag"][:-1] + "hz"
        xml = configure(**r["cfg"])
        rows = []
        for t in range(r["n"]):
            ok, v = trial(xml, tag, t)
            rows.append((ok, v))
            print(f"  {tag} t{t}: {'ok' if ok else 'FELL'} v={v:.3f}",
                  flush=True)
        ok = sum(x[0] for x in rows)
        v = float(np.mean([x[1] for x in rows if x[0]])) if ok else 0.0
        r.update(tag=tag, ok=ok, v=v,
                 note="rerun; original 'H' row reused the 'h' logs via a "
                      "case-insensitive filename collision")
    json.dump(js, open(JS, "w"), indent=1)
    print("patched ->", JS)


if __name__ == "__main__":
    main()
