#!/usr/bin/env python
"""E23: carry 10 kg on the trunk without losing reliability.

The 12 kg payload failures are dynamic, not static (the waist bench holds
200 N): the loaded waist pitches the trunk aft faster than the planner
corrects, under a controller tuned for the unloaded machine. Candidate
interventions, screened at n=3 and verified at n=10:

  torso   lower the torso height target by the loaded waist's static sag
  stiff   waist cables stiffened 2x for the loaded condition (the cable
          analogue of tightening a backpack hip belt; raises prestress too)
  fwd     payload centre of mass shifted 4 cm forward to counter aft pitch
  slow    reduced speed command under load
  pelvis  the placement alternative: same 10 kg on the pelvis, below the
          waist joint (the load path that bypasses the waist cables)

Success = min pelvis z in [3,15] s > 0.65 and distance > 1.2 m.
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
LOGD = f"{RES}/e23_payload10"
os.makedirs(LOGD, exist_ok=True)
LIVE = f"{TD}/task_hybrid_walk.xml"
BASE_TASK = open(LIVE).read()
BASE_MODEL = open(f"{TD}/humanoid_hybrid.xml").read()
MASS = 10


def model_variant(tag, dx=0.0, waist_k=1.0, pelvis_mount=False):
    xml = BASE_MODEL
    if pelvis_mount:
        # payload on the pelvis rails, between the battery packs
        xml = xml.replace('<body name="battery_l"',
                          f'''<body name="payload" pos="-0.10 0 0.06">
        <geom name="payload_box" type="box" size="0.07 0.10 0.07"
              mass="{MASS}" rgba="0.72 0.55 0.30 1" contype="0"
              conaffinity="0"/>
      </body>
      <body name="battery_l"''', 1)
    else:
        xml = xml.replace('<body name="head"',
                          f'''<body name="payload" pos="{dx:.3f} 0 0.06">
        <geom name="payload_box" type="box" size="0.07 0.10 0.07"
              mass="{MASS}" rgba="0.72 0.55 0.30 1" contype="0"
              conaffinity="0"/>
      </body>
      <body name="head"''', 1)
    if waist_k != 1.0:
        xml = re.sub(
            r'(<spatial name="waist_[^"]*" stiffness=")([\d.]+)(" '
            r'damping=")([\d.]+)',
            lambda m: f'{m.group(1)}{float(m.group(2))*waist_k:.1f}'
                      f'{m.group(3)}{float(m.group(4))*waist_k:.1f}',
            xml)
    path = f"{TD}/humanoid_p10_{tag}.xml"
    open(path, "w").write(xml)
    return f"humanoid_p10_{tag}.xml"


def task_variant(model_file, torso=None, speed=None):
    xml = BASE_TASK.replace("humanoid_hybrid.xml", model_file)
    if torso is not None:
        xml = re.sub(r'(name="residual_Torso" data=")[\d.+-]+',
                     rf'\g<1>{torso}', xml)
    if speed is not None:
        xml = re.sub(r'(name="residual_Speed" data=")[\d.+-]+',
                     rf'\g<1>{speed}', xml)
    return xml


def trial(task_xml, tag, t):
    log = f"{LOGD}/log_{tag}_t{t}.csv"
    try:
        open(LIVE, "w").write(task_xml)
        if not os.path.exists(log):
            env = dict(os.environ, TESTSPEED_LOG=log)
            subprocess.run([BIN, "--task=Hybrid Walk", "--total_time=16",
                            "--steps_per_planning_iteration=4"],
                           capture_output=True, text=True, env=env, cwd=TD)
    finally:
        open(LIVE, "w").write(BASE_TASK)
    try:
        raw = np.genfromtxt(log, delimiter=",", invalid_raise=False)
        raw = raw[~np.isnan(raw).any(axis=1)]
        tt, x, z = raw[:, 0], raw[:, 1], raw[:, 3]
        w = (tt >= 3.0) & (tt <= 15.0)
        ok = bool(z[w].min() > 0.65 and (x[w][-1] - x[w][0]) > 1.2)
        return ok, (x[w][-1] - x[w][0]) / 12.0
    except Exception:
        return False, 0.0


def screen(cfgs, n):
    out = []
    for tag, mkw, tkw in cfgs:
        mdl = model_variant(tag, **mkw)
        xml = task_variant(mdl, **tkw)
        rows = [trial(xml, tag, t) for t in range(n)]
        ok = sum(r[0] for r in rows)
        v = float(np.mean([r[1] for r in rows if r[0]])) if ok else 0.0
        out.append(dict(tag=tag, model=mkw, task=tkw, ok=ok, n=n, v=v))
        print(f"  {tag}: {ok}/{n} at {v:.3f} m/s", flush=True)
    return out


def main():
    print(f"E23 round 2: {MASS} kg, refined screen n=3")
    cfgs = [
        ("fwd6",   dict(dx=0.06), dict()),
        ("fwd8",   dict(dx=0.08), dict()),
        ("fwd4s",  dict(dx=0.04), dict(speed=0.40)),
        ("fwd4k",  dict(dx=0.04, waist_k=1.5), dict()),
    ]
    A = screen(cfgs, 3)
    best = sorted([r for r in A if r["ok"] == 3],
                  key=lambda r: r["v"], reverse=True)[:2]
    print("verify n=10 on:", [r["tag"] for r in best])
    V = []
    for r in best:
        mdl = model_variant(r["tag"] + "v", **r["model"])
        xml = task_variant(mdl, **r["task"])
        rows = [trial(xml, r["tag"] + "v", t) for t in range(10)]
        ok = sum(x[0] for x in rows)
        v = float(np.mean([x[1] for x in rows if x[0]])) if ok else 0.0
        V.append(dict(tag=r["tag"], model=r["model"], task=r["task"],
                      ok=ok, n=10, v=v))
        print(f"  VERIFY {r['tag']}: {ok}/10 at {v:.3f} m/s", flush=True)
    json.dump(dict(screen=A, verify=V),
              open(f"{RES}/e23_round2.json", "w"), indent=1)
    print("E23 r2 ->", f"{RES}/e23_round2.json")


if __name__ == "__main__":
    main()
