#!/usr/bin/env python
"""E25: long-duration runs of the fully hinged walker.

The walking campaign runs 16 s trials; nothing in it supports a statement
about how far the machine walks in one continuous run. This runs the
baseline configuration for 30 s, n = 3, and records distance and duration
so any "longest run" figure is a measurement.
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
LOGD = f"{RES}/e25_longrun"
os.makedirs(LOGD, exist_ok=True)
SECONDS = 30

rows = []
for t in range(3):
    log = f"{LOGD}/log_long_t{t}.csv"
    if not os.path.exists(log):
        env = dict(os.environ, TESTSPEED_LOG=log)
        subprocess.run([BIN, "--task=Tensegrity Walk",
                        f"--total_time={SECONDS}",
                        "--steps_per_planning_iteration=4"],
                       capture_output=True, text=True, env=env, cwd=TD)
    raw = np.genfromtxt(log, delimiter=",", invalid_raise=False)
    raw = raw[~np.isnan(raw).any(axis=1)]
    tt, x, z = raw[:, 0], raw[:, 1], raw[:, 3]
    up = z > 0.55
    t_up = float(tt[up][-1]) if up.any() else 0.0     # last upright time
    d_up = float(x[up][-1] - x[0]) if up.any() else 0.0
    rows.append(dict(trial=t, upright_s=t_up, dist_m=d_up,
                     final_x=float(x[-1]), min_z=float(z.min())))
    print(f"t{t}: upright {t_up:.1f} s, distance {d_up:.2f} m "
          f"(min z {z.min():.2f})", flush=True)

best = max(rows, key=lambda r: r["dist_m"])
json.dump(dict(seconds=SECONDS, rows=rows, best=best),
          open(f"{RES}/e25_longrun.json", "w"), indent=1)
print(f"best: {best['dist_m']:.2f} m in {best['upright_s']:.0f} s ->",
      f"{RES}/e25_longrun.json")
