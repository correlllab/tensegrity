#!/usr/bin/env python
"""Open any model in the interactive viewer.

    .venv/bin/mjpython experiments/view.py mujoco/tensegrity_column.xml

On macOS the MuJoCo viewer must run under `mjpython`, not `python` -- plain
python raises before it ever reads the file, which looks like a load failure
but is not one. This loads the model first and reports what it found, so a
genuine XML problem is distinguishable from a launcher problem.

Options:
    --stand   drive the model with e14_stand_controller.StandController
    --tree a:b,b:c   interface tree for the controller (default: humanoid)
    --ground seg0    grounded bodies, comma separated
"""
import os
import sys

import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_flags(argv):
    args, flags = [], {}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            key = a[2:]
            if "=" in key:
                key, val = key.split("=", 1)
                flags[key] = val
            elif i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                flags[key] = argv[i + 1]
                i += 1
            else:
                flags[key] = True
        else:
            args.append(a)
        i += 1
    return args, flags


def main():
    args, flags = parse_flags(sys.argv[1:])
    if not args:
        print(__doc__)
        return 1
    path = args[0]
    if not os.path.exists(path):
        print(f"no such file: {path}")
        return 1

    try:
        m = mujoco.MjModel.from_xml_path(path)
    except ValueError as e:
        print(f"model does NOT parse:\n  {e}")
        return 1
    print(f"parsed OK: {m.nbody - 1} bodies, {m.nv} DoF, {m.ntendon} cables, "
          f"{m.nu} actuators")

    d = mujoco.MjData(m)
    ctrl = None
    if flags.get("stand"):
        from e14_stand_controller import StandController
        tree, ground = None, None
        if isinstance(flags.get("tree"), str):
            tree = [tuple(p.split(":")) for p in flags["tree"].split(",")]
        if isinstance(flags.get("ground"), str):
            ground = flags["ground"].split(",")
        ctrl = StandController(m, d, tree=tree, grounded=ground)
        print(f"standing controller active over {len(ctrl.iface)} interfaces")

    try:
        with mujoco.viewer.launch_passive(m, d) as v:
            while v.is_running():
                if ctrl is not None:
                    ctrl.step()
                mujoco.mj_step(m, d)
                v.sync()
    except RuntimeError as e:
        print(f"\nviewer could not start: {e}")
        print("on macOS this must run under mjpython:")
        print(f"  .venv/bin/mjpython experiments/view.py {path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
