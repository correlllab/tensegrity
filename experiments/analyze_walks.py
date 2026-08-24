#!/usr/bin/env python
"""Aggregate the walk batch: per-tag gait metrics from CSV logs, plus tension
mapping for payload runs. Writes experiments/results/walk_summary.json."""
import glob
import json
import os
import re
import sys
import numpy as np

sys.path.insert(0, "/Users/ncorrell/Downloads/tensegrity/experiments")
import walklog                            # noqa: E402
import tension_map                        # noqa: E402

RES = "/Users/ncorrell/Downloads/tensegrity/experiments/results"
MJ = "/Users/ncorrell/Downloads/tensegrity/mujoco"


def main():
    logs = sorted(glob.glob(f"{RES}/log_*.csv"))
    out = {}
    for log in logs:
        tag = re.match(r"log_(.+)_t(\d+)\.csv", os.path.basename(log))
        name, trial = tag.group(1), int(tag.group(2))
        payload = 0.0
        if name.startswith("payload"):
            payload = float(name[7:])
        elif name.startswith("cpayload"):
            payload = float(name[8:])
        mm = walklog.metrics(log, total_mass=27.6 + payload)
        rec = out.setdefault(name, dict(trials=[]))
        row = dict(trial=trial, speed=mm["speed"], success=mm["success"],
                   cot=mm["cot"], min_z=mm["min_z"], final_x=mm["final_x"],
                   peak=dict((g, mm["per_group"][g]["peak"])
                             for g in ("hip", "knee", "ankle", "waist")),
                   p95=dict((g, mm["per_group"][g]["p95"])
                            for g in ("hip", "knee", "ankle", "waist")),
                   rms=dict((g, mm["per_group"][g]["rms"])
                            for g in ("hip", "knee", "ankle", "waist")))
        # tension mapping: trial 0 of payload runs + baseline t0
        if trial == 0 and ("payload" in name or name == "baseline"):
            if name.startswith("cpayload"):
                cx = f"{RES}/cable_cpayload_{int(payload)}.xml"
            elif payload:
                cx = f"{RES}/cable_payload_{int(payload)}.xml"
            else:
                cx = f"{MJ}/humanoid_27dof_tensegrity_cable.xml"
            if mm["success"]:
                row["tension"] = tension_map.analyze(log, cx, stride=12)
        rec["trials"].append(row)
        print(name, trial, "v=%.3f" % mm["speed"], "ok" if mm["success"]
              else "FELL", flush=True)
    json.dump(out, open(f"{RES}/walk_summary.json", "w"), indent=1)
    print("->", f"{RES}/walk_summary.json")


if __name__ == "__main__":
    main()
