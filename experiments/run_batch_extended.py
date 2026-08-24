#!/usr/bin/env python
"""Top up every walking sweep to an adequate trial count.

The first campaign used n = 2-4 per point, which cannot separate a capability
boundary from planner stochasticity (4 kg failed 4/4 while 20 kg succeeded
2/2). This script fills each condition to a target n, skipping trials whose
logs already exist, and records the testspeed realtime factor per run so the
planner's wall-clock cost is reported rather than asserted.

Conditions:
  baseline / distal (E4)                 n = 10
  commanded speed 0.10-0.45              n = 10 (0.35, 0.45 are new points)
  centered trunk payload 2-20 kg         n = 10
  rear trunk payload 2-20 kg             n =  6
"""
import json
import os
import re
import subprocess
import sys

ROOT = "/Users/ncorrell/Downloads/tensegrity"
TD = f"{ROOT}/mujoco_mpc/build/mjpc/tasks/tensegrity"
BIN = f"{ROOT}/mujoco_mpc/build/bin/testspeed"
RES = f"{ROOT}/experiments/results"
MODEL = f"{TD}/humanoid_27dof_tensegrity.xml"
TASK = f"{TD}/task_walk.xml"
SECONDS = 16
STEPS = 4

orig_model = open(MODEL).read()
orig_task = open(TASK).read()
results = open(f"{RES}/walk_runs_extended.jsonl", "a")

N_WALK = 10        # baseline, distal, speed, centered payload
N_REAR = 6         # rear-mount payload (secondary comparison)
MASSES = (2, 4, 6, 8, 10, 12, 16, 20)
SPEEDS = (0.10, 0.20, 0.35, 0.40, 0.45)


def payload_xml(xml, mass, pos):
    body = (f'<body name="payload" pos="{pos}">'
            f'<geom name="payload_box" type="box" size="0.09 0.13 0.11" '
            f'mass="{mass}" rgba="0.72 0.55 0.30 1"/></body>\n        ')
    return xml.replace('<body name="head" pos="0 0 0.50">',
                       body + '<body name="head" pos="0 0 0.50">')


def distal_xml(xml):
    moves = [
        ('name="knee_motor_l" class="motormesh" mesh="ak70" pos="0.055 0 -0.09"',
         'name="knee_motor_l" class="motormesh" mesh="ak70" pos="0.05 0 -0.375"'),
        ('name="knee_motor_r" class="motormesh" mesh="ak70" pos="0.055 0 -0.09"',
         'name="knee_motor_r" class="motormesh" mesh="ak70" pos="0.05 0 -0.375"'),
        ('name="ankle_motor_l_a" class="motormesh" mesh="ak70" pos="0.05 0 -0.055"',
         'name="ankle_motor_l_a" class="motormesh" mesh="ak70" pos="0.045 0 -0.30"'),
        ('name="ankle_motor_r_a" class="motormesh" mesh="ak70" pos="0.05 0 -0.055"',
         'name="ankle_motor_r_a" class="motormesh" mesh="ak70" pos="0.045 0 -0.30"'),
        ('name="ankle_motor_l_b" class="motormesh" mesh="ak70" pos="0.05 0 -0.15"',
         'name="ankle_motor_l_b" class="motormesh" mesh="ak70" pos="0.045 0 -0.36"'),
        ('name="ankle_motor_r_b" class="motormesh" mesh="ak70" pos="0.05 0 -0.15"',
         'name="ankle_motor_r_b" class="motormesh" mesh="ak70" pos="0.045 0 -0.36"'),
        ('name="arm_motor_l_a" class="motormesh" mesh="ak60" pos="0.045 0 -0.05"',
         'name="arm_motor_l_a" class="motormesh" mesh="ak60" pos="0.045 0 -0.20"'),
        ('name="arm_motor_r_a" class="motormesh" mesh="ak60" pos="0.045 0 -0.05"',
         'name="arm_motor_r_a" class="motormesh" mesh="ak60" pos="0.045 0 -0.20"'),
        ('name="arm_motor_l_b" class="motormesh" mesh="ak60" pos="0.045 0 -0.14"',
         'name="arm_motor_l_b" class="motormesh" mesh="ak60" pos="0.045 0 -0.26"'),
        ('name="arm_motor_r_b" class="motormesh" mesh="ak60" pos="0.045 0 -0.14"',
         'name="arm_motor_r_b" class="motormesh" mesh="ak60" pos="0.045 0 -0.26"'),
    ]
    for a, b in moves:
        assert a in xml, a
        xml = xml.replace(a, b)
    return xml


def set_speed(task, v):
    return task.replace('<numeric name="residual_Speed" data="0.30 -0.5 1.0" />',
                        f'<numeric name="residual_Speed" data="{v} -0.5 1.0" />')


def run(tag, model_xml, task_xml, trial):
    log = f"{RES}/log_{tag}_t{trial}.csv"
    if os.path.exists(log):
        return False
    open(MODEL, "w").write(model_xml)
    open(TASK, "w").write(task_xml)
    env = dict(os.environ, TESTSPEED_LOG=log)
    out = subprocess.run(
        [BIN, "--task=Tensegrity Walk", f"--total_time={SECONDS}",
         f"--steps_per_planning_iteration={STEPS}"],
        capture_output=True, text=True, env=env).stdout
    tail = [l for l in out.splitlines() if "sim time" in l][-1:]
    rt = re.search(r"\(([\d.]+)x realtime\)", out)
    wall = re.search(r"Total wall time \((\d+) planning steps\): ([\d.]+) s", out)
    rec = dict(tag=tag, trial=trial, log=log, last=tail[0] if tail else "NONE",
               realtime_factor=float(rt.group(1)) if rt else None,
               planning_steps=int(wall.group(1)) if wall else None,
               wall_s=float(wall.group(2)) if wall else None)
    results.write(json.dumps(rec) + "\n")
    results.flush()
    print(f"[{tag} t{trial}] rt={rec['realtime_factor']} {rec['last']}",
          flush=True)
    return True


def fill(tag, model_xml, task_xml, n):
    for t in range(n):
        run(tag, model_xml, task_xml, t)


cable_orig = open(f"{ROOT}/mujoco/humanoid_27dof_tensegrity_cable.xml").read()
try:
    fill("baseline", orig_model, orig_task, N_WALK)
    fill("distal", distal_xml(orig_model), orig_task, N_WALK)
    for v in SPEEDS:
        fill(f"speed{v:.2f}", orig_model, set_speed(orig_task, v), N_WALK)
    for m in MASSES:                                   # centered mount
        open(f"{RES}/cable_cpayload_{m}.xml", "w").write(
            payload_xml(cable_orig, m, "-0.02 0 0.30"))
        fill(f"cpayload{m:02d}", payload_xml(orig_model, m, "-0.02 0 0.30"),
             orig_task, N_WALK)
    for m in MASSES:                                   # rear mount
        open(f"{RES}/cable_payload_{m}.xml", "w").write(
            payload_xml(cable_orig, m, "-0.16 0 0.24"))
        fill(f"payload{m:02d}", payload_xml(orig_model, m, "-0.16 0 0.24"),
             orig_task, N_REAR)
finally:
    open(MODEL, "w").write(orig_model)
    open(TASK, "w").write(orig_task)
    results.close()
print("EXTENDED BATCH DONE", flush=True)
