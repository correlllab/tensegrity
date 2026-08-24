#!/usr/bin/env python
"""Overnight tuning loop for the Hybrid Walk task.

Stage 1: coarse screen over gait parameters and cost weights, n=2 each.
Stage 2: n=10 on the best configs; the winner is written back into
         mjpc/tasks/tensegrity/task_hybrid_walk.xml (source AND build) so the
         GUI opens with the walking configuration.

Success = min pelvis z in [3,15] s > 0.65 m AND net forward travel > 1.2 m.
"""
import itertools
import json
import os
import re
import subprocess
import numpy as np

ROOT = "/Users/ncorrell/Downloads/tensegrity"
TD = f"{ROOT}/mujoco_mpc/build/mjpc/tasks/tensegrity"
SRC = f"{ROOT}/mujoco_mpc/mjpc/tasks/tensegrity/task_hybrid_walk.xml"
BIN = f"{ROOT}/mujoco_mpc/build/bin/testspeed"
RES = f"{ROOT}/experiments/results"
LOGD = f"{RES}/hybrid"
os.makedirs(LOGD, exist_ok=True)
TASK = f"{TD}/task_hybrid_walk.xml"
SECONDS, STEPS = 16, 4
Z_HOME = 0.939

orig = open(TASK).read()


def configure(xml, cfg):
    def num(name, val):
        return re.sub(rf'(name="residual_{name}" data=")[\d.+-]+',
                      rf'\g<1>{val}', xml)
    xml = num("Speed", cfg["speed"])
    xml = num("Cadence", cfg["cadence"])
    xml = num("Step Height", cfg["step"])
    xml = num("Sway", cfg["sway"])
    xml = num("Stance Width", cfg["width"])
    xml = num("Torso", cfg["torso"])
    for sensor, w in (("Gait", cfg["w_gait"]), ("Balance", cfg["w_bal"]),
                      ("Height", cfg["w_h"]), ("Posture", cfg["w_post"])):
        xml = re.sub(
            rf'(<user name="{sensor}"\s+dim="\d+"\s+user="\d+) [\d.]+',
            rf'\g<1> {w}', xml)
    return xml


def run(cfg, tag, trial):
    log = f"{LOGD}/log_{tag}_t{trial}.csv"
    if not os.path.exists(log):
        open(TASK, "w").write(configure(orig, cfg))
        env = dict(os.environ, TESTSPEED_LOG=log)
        subprocess.run([BIN, "--task=Hybrid Walk",
                        f"--total_time={SECONDS}",
                        f"--steps_per_planning_iteration={STEPS}"],
                       capture_output=True, text=True, env=env, cwd=TD)
    try:
        raw = np.genfromtxt(log, delimiter=",", invalid_raise=False)
        raw = raw[~np.isnan(raw).any(axis=1)]
        t, x, z = raw[:, 0], raw[:, 1], raw[:, 3]
        w = (t >= 3.0) & (t <= 15.0)
        minz = float(z[w].min())
        net = float(x[w][-1] - x[w][0])
        v = net / (t[w][-1] - t[w][0])
        ok = bool(minz > 0.65 and net > 1.2)
        return dict(ok=ok, v=v, minz=minz, net=net)
    except Exception:
        return dict(ok=False, v=0.0, minz=0.0, net=0.0)


def main():
    # the failure mode is a refusal to lift the swing foot: with passive
    # ankles the planner cannot steer the CoP under a foot, so weight shift
    # must come from sway + narrow stance + slow cadence, and lifting from
    # posture tracking
    grid = dict(
        speed=[0.12, 0.20],
        cadence=[0.8, 1.0],
        step=[0.06],
        sway=[0.07, 0.09],
        width=[0.14, 0.18],
        torso=[1.20, 1.24],
        w_gait=[25.0],
        w_bal=[3.0],
        w_h=[5.0],
        w_post=[2.0, 4.0],
    )
    keys = list(grid)
    combos = list(itertools.product(*grid.values()))
    print(f"stage 1: {len(combos)} configs x 2 trials")
    results = []
    try:
        for ci, vals in enumerate(combos):
            cfg = dict(zip(keys, vals))
            tag = f"c{ci:03d}"
            rr = [run(cfg, tag, t) for t in range(2)]
            n_ok = sum(r["ok"] for r in rr)
            v = np.mean([r["v"] for r in rr if r["ok"]]) if n_ok else 0.0
            results.append(dict(cfg=cfg, tag=tag, n_ok=n_ok, v=float(v),
                                minz=min(r["minz"] for r in rr)))
            print(f"{tag} {cfg}  -> {n_ok}/2 ok  v={v:.2f}  "
                  f"minz={results[-1]['minz']:.2f}", flush=True)
        results.sort(key=lambda r: (-r["n_ok"], -r["v"]))
        json.dump(results, open(f"{RES}/hybrid_stage1.json", "w"), indent=1)

        print("\nstage 2: top 3 configs x 10 trials")
        finals = []
        for r in results[:3]:
            rr = [run(r["cfg"], r["tag"], t) for t in range(10)]
            n_ok = sum(x["ok"] for x in rr)
            v = np.mean([x["v"] for x in rr if x["ok"]]) if n_ok else 0.0
            finals.append(dict(cfg=r["cfg"], tag=r["tag"], n_ok=n_ok,
                               v=float(v)))
            print(f"{r['tag']} -> {n_ok}/10 ok  v={v:.2f}", flush=True)
        finals.sort(key=lambda r: (-r["n_ok"], -r["v"]))
        json.dump(finals, open(f"{RES}/hybrid_stage2.json", "w"), indent=1)
        if finals and finals[0]["n_ok"] > 0:
            best = configure(orig, finals[0]["cfg"])
            open(TASK, "w").write(best)
            open(SRC, "w").write(best)
            print(f"\nWINNER {finals[0]['tag']}: {finals[0]['n_ok']}/10 at "
                  f"{finals[0]['v']:.2f} m/s -> written to task xml")
        else:
            open(TASK, "w").write(orig)
    finally:
        pass
    print("TUNING DONE")


if __name__ == "__main__":
    main()
