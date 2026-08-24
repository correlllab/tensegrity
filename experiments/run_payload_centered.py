#!/usr/bin/env python
"""Payload sweep, take two: the rear backpack mount (-0.16 m) destabilized the
gait at 2-4 kg by shifting the CoM behind the support. Design response: mount
the payload on the CoM column (spine-hugging, x = -0.02, z = 0.30). Same
sweep, same task, same controller — only the mount moved.
"""
import json
import os
import subprocess

ROOT = "/Users/ncorrell/Downloads/tensegrity"
TD = f"{ROOT}/mujoco_mpc/build/mjpc/tasks/tensegrity"
BIN = f"{ROOT}/mujoco_mpc/build/bin/testspeed"
RES = f"{ROOT}/experiments/results"
MODEL = f"{TD}/humanoid_27dof_tensegrity.xml"

orig_model = open(MODEL).read()
results = open(f"{RES}/walk_runs_centered.jsonl", "w")


def payload_xml(xml, mass):
    body = (f'<body name="payload" pos="-0.02 0 0.30">'
            f'<geom name="payload_box" type="box" size="0.09 0.13 0.11" '
            f'mass="{mass}" rgba="0.72 0.55 0.30 1"/></body>\n        ')
    return xml.replace('<body name="head" pos="0 0 0.50">',
                       body + '<body name="head" pos="0 0 0.50">')


def run(tag, model_xml, trial):
    open(MODEL, "w").write(model_xml)
    log = f"{RES}/log_{tag}_t{trial}.csv"
    env = dict(os.environ, TESTSPEED_LOG=log)
    out = subprocess.run(
        [BIN, "--task=Tensegrity Walk", "--total_time=16",
         "--steps_per_planning_iteration=4"],
        capture_output=True, text=True, env=env).stdout
    tail = [l for l in out.splitlines() if "sim time" in l][-1:]
    rec = dict(tag=tag, trial=trial, log=log, last=tail[0] if tail else "NONE")
    results.write(json.dumps(rec) + "\n")
    results.flush()
    print(f"[{tag} t{trial}] {rec['last']}", flush=True)


cable_orig = open(f"{ROOT}/mujoco/humanoid_27dof_tensegrity_cable.xml").read()
try:
    for m in (2, 4, 6, 8, 10, 12, 16, 20):
        pm = payload_xml(orig_model, m)
        open(f"{RES}/cable_cpayload_{m}.xml", "w").write(
            payload_xml(cable_orig, m))
        for t in range(2):
            run(f"cpayload{m:02d}", pm, t)
finally:
    open(MODEL, "w").write(orig_model)
    results.close()
print("CENTERED PAYLOAD BATCH DONE", flush=True)
