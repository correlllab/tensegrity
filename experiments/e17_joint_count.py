#!/usr/bin/env python
"""E17: the joint-count study -- how much hinge does the machine need?

Variants of the SAME full-body robot (7-DoF arms, packs, identical masses and
controller), differing only in which joints are bearings and which are cable
interfaces:

  rigid_waist   waist cable interface replaced by a weld (torso rigidly on
                the pelvis) -> hinged DoF unchanged, tensegrity joints 2
  rigid_ankle   ankle cable interfaces replaced by welds (feet rigidly on
                the shanks) -> tensegrity joints 1 (waist only)
  rigid_both    both welded -> tensegrity joints 0 (fully articulated-rigid
                except the arm servos)
  full          the shipped hybrid (waist + 2 ankles cable) -> 3 tensegrity
                joints [reference, already measured 10/10]

Together with the pure-tensegrity endpoints (E13: 0 hinges, stands but no
controller) and the original 27-hinge machine (7/10 at 0.21 m/s, 1.89 kg of
bearings, E1: impacts worse), these span the axis the paper's new study
section discusses.

Welding is done by post-processing the generated XML: the welded child keeps
its geoms and inertial, loses its freejoint, and the corresponding cables and
tension actuators are removed. Control dim changes, so a matching task xml is
emitted per variant.
"""
import json
import os
import re
import subprocess
import numpy as np
import mujoco

ROOT = "/Users/ncorrell/Downloads/tensegrity"
MJ = f"{ROOT}/mujoco"
TD = f"{ROOT}/mujoco_mpc/build/mjpc/tasks/tensegrity"
BIN = f"{ROOT}/mujoco_mpc/build/bin/testspeed"
RES = f"{ROOT}/experiments/results"
LOGD = f"{RES}/jointcount"
os.makedirs(LOGD, exist_ok=True)
N_TRIALS = 5


def body_block(xml, name):
    start = xml.index(f'<body name="{name}"')
    i, depth = start, 0
    pat = re.compile(r'<body\b|</body>')
    while True:
        mo = pat.search(xml, i)
        if mo.group(0) == '<body':
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                return start, mo.end()
        i = mo.end()


def weld(xml, child, parent, m, d):
    """Move `child` (a top-level free body) inside `parent`, dropping its
    freejoint; reposition to parent-local coordinates."""
    cb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, child)
    pb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, parent)
    off = d.xpos[cb] - d.xpos[pb]
    a, b = body_block(xml, child)
    block = xml[a:b]
    xml = xml[:a] + xml[b:]
    block = block.replace('<freejoint/>', '')
    block = re.sub(rf'(<body name="{child}" pos=")[^"]+',
                   rf'\g<1>{off[0]:.5f} {off[1]:.5f} {off[2]:.5f}', block)
    pa, pbend = body_block(xml, parent)
    close = xml.rindex('</body>', pa, pbend)
    return xml[:close] + block + '\n      ' + xml[close:]


def strip_cables(xml, prefix):
    xml = re.sub(rf'<spatial name="{prefix}[^>]*>\s*<site[^/]*/>\s*'
                 rf'<site[^/]*/>\s*</spatial>', '', xml)
    xml = re.sub(rf'<motor name="[^"]*" tendon="{prefix}[^>]*/>', '', xml)
    return xml


def make_variant(tag, weld_ankle, weld_waist):
    src = f"{MJ}/humanoid_hybrid.xml"
    xml = open(src).read()
    os.chdir(MJ)
    m = mujoco.MjModel.from_xml_path(src)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    if weld_waist:
        xml = weld(xml, "torso", "pelvis", m, d)
        xml = strip_cables(xml, "waist_")
    if weld_ankle:
        for side in ("l", "r"):
            xml = weld(xml, f"foot_{side}", f"shank_{side}", m, d)
            xml = strip_cables(xml, f"ank_{side}_")
        # actuator names for ankles use ankm_
        xml = re.sub(r'<motor name="ankm_[^>]*/>', '', xml)
    xml = re.sub(r'<motor name="waistm_[^>]*/>', '', xml) if weld_waist else xml
    mpath = f"{TD}/humanoid_jc_{tag}.xml"
    open(mpath, "w").write(xml)
    mv = mujoco.MjModel.from_xml_string(xml)

    # task xml with matching control dim
    task = open(f"{TD}/task_hybrid_walk.xml").read()
    task = task.replace('humanoid_hybrid.xml', f'humanoid_jc_{tag}.xml')
    task = re.sub(r'(<user name="Control"\s+dim=")\d+',
                  rf'\g<1>{mv.nu}', task)
    tpath = f"{TD}/task_jc_{tag}.xml"
    open(tpath, "w").write(task)
    return mv, mpath, tpath


def run_variant(tag, tpath):
    """testspeed reads the registered task xml path, so swap the file in."""
    live = f"{TD}/task_hybrid_walk.xml"
    backup = open(live).read()
    rows = []
    try:
        open(live, "w").write(open(tpath).read().replace(
            f'humanoid_jc_{tag}.xml', f'humanoid_jc_{tag}.xml'))
        for t in range(N_TRIALS):
            log = f"{LOGD}/log_{tag}_t{t}.csv"
            env = dict(os.environ, TESTSPEED_LOG=log)
            subprocess.run([BIN, "--task=Hybrid Walk", "--total_time=16",
                            "--steps_per_planning_iteration=4"],
                           capture_output=True, text=True, env=env, cwd=TD)
            raw = np.genfromtxt(log, delimiter=",", invalid_raise=False)
            raw = raw[~np.isnan(raw).any(axis=1)]
            tt, x, z = raw[:, 0], raw[:, 1], raw[:, 3]
            w = (tt >= 3.0) & (tt <= 15.0)
            ok = bool(z[w].min() > 0.65 and (x[w][-1] - x[w][0]) > 1.2)
            rows.append(dict(ok=ok, v=(x[w][-1] - x[w][0]) / 12.0,
                             minz=float(z[w].min())))
            print(f"  {tag} t{t}: {'ok  ' if ok else 'FELL'} "
                  f"v={rows[-1]['v']:.3f} minz={rows[-1]['minz']:.2f}",
                  flush=True)
    finally:
        open(live, "w").write(backup)
    ok = sum(r["ok"] for r in rows)
    v = float(np.mean([r["v"] for r in rows if r["ok"]])) if ok else 0.0
    return dict(ok=ok, n=len(rows), v=v)


def main():
    out = {}
    variants = [("rigid_waist", False, True),
                ("rigid_ankle", True, False),
                ("rigid_both", True, True)]
    for tag, wa, ww in variants:
        mv, mpath, tpath = make_variant(tag, wa, ww)
        print(f"{tag}: nv={mv.nv} nu={mv.nu} tendons={mv.ntendon} "
              f"mass={mv.body_mass.sum():.1f} kg")
        out[tag] = run_variant(tag, tpath)
        out[tag].update(nv=int(mv.nv), nu=int(mv.nu))
        print(f"{tag}: {out[tag]['ok']}/{out[tag]['n']} at "
              f"{out[tag]['v']:.3f} m/s\n", flush=True)
    json.dump(out, open(f"{RES}/e17_joint_count.json", "w"), indent=1)
    print("E17 ->", f"{RES}/e17_joint_count.json")


if __name__ == "__main__":
    main()
