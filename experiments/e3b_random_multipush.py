#!/usr/bin/env python
"""E3b: de-confound the adversarial cable-cut result.

In E3 the random k = 8 condition was tested against a single 40 N forward
push, while the adversarial condition faced four pushes (40/60/-60 N
forward, 40 N lateral) -- so its k = 8 failure changed both the selection
rule and the disturbance. This runs random stance-cable draws at k = 8
under the SAME four-push protocol as the adversarial condition: each draw
must survive all four pushes.
"""
import json
import sys
import numpy as np
import mujoco

sys.path.insert(0, "/Users/ncorrell/Downloads/tensegrity/experiments")
from e3_degradation import run_stand, CABLE  # noqa: E402

OUT = ("/Users/ncorrell/Downloads/tensegrity/experiments/results/"
       "e3b_random_multipush.json")
PUSHES = ((40.0, 0.0), (60.0, 0.0), (-60.0, 0.0), (0.0, 40.0))
K = 8
DRAWS = 4

rng = np.random.default_rng(11)
mc = mujoco.MjModel.from_xml_path(CABLE)
names = [mujoco.mj_id2name(mc, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
         for i in range(mc.nu)]
stance = [i for i, n in enumerate(names)
          if n.startswith(("hip", "knee", "ankle", "waist"))]

rows = []
for trial in range(DRAWS):
    rm = set(int(x) for x in rng.choice(stance, K, replace=False))
    wins = 0
    for p in PUSHES:
        ok, z = run_stand(rm, push=p)
        wins += ok
    rows.append(dict(k=K, cables=[names[c] for c in sorted(rm)],
                     pushes=len(PUSHES), survived=wins,
                     all_survived=bool(wins == len(PUSHES))))
    print(f"draw {trial}: {wins}/{len(PUSHES)} pushes survived "
          f"({', '.join(names[c] for c in sorted(rm))})", flush=True)

ok_draws = sum(r["all_survived"] for r in rows)
json.dump(dict(k=K, draws=DRAWS, pushes=list(PUSHES), rows=rows,
               draws_surviving_all=ok_draws),
          open(OUT, "w"), indent=1)
print(f"{ok_draws}/{DRAWS} random draws survive all four pushes -> {OUT}")
