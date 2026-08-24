#!/usr/bin/env python
"""Sequential walk-experiment batch on the MJPC testspeed harness.

Runs, in order, writing per-step CSV logs and a JSONL results index:
  1. baseline walk x3           (also provides the film-strip frames)
  2. E4 distal-motor variant x3 (motor masses moved from proximal mounts to
                                 the joints they drive; total mass unchanged)
  3. commanded-speed sweep      {0.10, 0.20, 0.40} x2   (0.30 = baseline)
  4. payload sweep              backpack box, {2,...,20} kg x2

The harness reads build/mjpc/tasks/tensegrity/{humanoid_27dof_tensegrity.xml,
task_walk.xml}; both are backed up and restored. Matching cable-model payload
variants are saved for offline tension mapping.
"""
import json
import os
import re
import shutil
import subprocess

ROOT = "/Users/ncorrell/Downloads/tensegrity"
TD = f"{ROOT}/mujoco_mpc/build/mjpc/tasks/tensegrity"
BIN = f"{ROOT}/mujoco_mpc/build/bin/testspeed"
RES = f"{ROOT}/experiments/results"
MODEL = f"{TD}/humanoid_27dof_tensegrity.xml"
TASK = f"{TD}/task_walk.xml"
SECONDS = 16
STEPS = 4

os.makedirs(RES, exist_ok=True)
orig_model = open(MODEL).read()
orig_task = open(TASK).read()
results = open(f"{RES}/walk_runs.jsonl", "w")


def payload_xml(xml, mass):
    body = (f'<body name="payload" pos="-0.16 0 0.24">'
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
    open(MODEL, "w").write(model_xml)
    open(TASK, "w").write(task_xml)
    log = f"{RES}/log_{tag}_t{trial}.csv"
    env = dict(os.environ, TESTSPEED_LOG=log)
    out = subprocess.run(
        [BIN, "--task=Tensegrity Walk", f"--total_time={SECONDS}",
         f"--steps_per_planning_iteration={STEPS}"],
        capture_output=True, text=True, env=env).stdout
    tail = [l for l in out.splitlines() if "sim time" in l][-1:]
    rec = dict(tag=tag, trial=trial, log=log, last=tail[0] if tail else "NONE")
    results.write(json.dumps(rec) + "\n")
    results.flush()
    print(f"[{tag} t{trial}] {rec['last']}", flush=True)


try:
    for t in range(3):
        run("baseline", orig_model, orig_task, t)
    dm = distal_xml(orig_model)
    for t in range(3):
        run("distal", dm, orig_task, t)
    for v in (0.10, 0.20, 0.40):
        for t in range(2):
            run(f"speed{v:.2f}", orig_model, set_speed(orig_task, v), t)
    cable_orig = open(f"{ROOT}/mujoco/humanoid_27dof_tensegrity_cable.xml").read()
    for m in (2, 4, 6, 8, 10, 12, 16, 20):
        pm = payload_xml(orig_model, m)
        open(f"{RES}/cable_payload_{m}.xml", "w").write(payload_xml(cable_orig, m))
        for t in range(2):
            run(f"payload{m:02d}", pm, orig_task, t)
finally:
    open(MODEL, "w").write(orig_model)
    open(TASK, "w").write(orig_task)
    results.close()
print("WALK BATCH DONE", flush=True)
