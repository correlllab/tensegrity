#!/usr/bin/env python
"""Verify the Hybrid Walk at n=10, then a focused speed/cadence sweep, then
n=10 on the best. Writes the winner into the task xml (source + build)."""
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
LOGD = f"{RES}/hybrid_v2"
os.makedirs(LOGD, exist_ok=True)
TASK = f"{TD}/task_hybrid_walk.xml"
orig = open(TASK).read()


def cfg_xml(xml, **kw):
    for name, val in kw.items():
        label = {"speed": "Speed", "cadence": "Cadence", "sway": "Sway",
                 "step": "Step Height"}[name]
        xml = re.sub(rf'(name="residual_{label}" data=")[\d.+-]+',
                     rf'\g<1>{val}', xml)
    return xml


def run(tag, trial, xml):
    log = f"{LOGD}/log_{tag}_t{trial}.csv"
    if not os.path.exists(log):
        open(TASK, "w").write(xml)
        env = dict(os.environ, TESTSPEED_LOG=log)
        subprocess.run([BIN, "--task=Hybrid Walk", "--total_time=16",
                        "--steps_per_planning_iteration=4"],
                       capture_output=True, text=True, env=env, cwd=TD)
    try:
        raw = np.genfromtxt(log, delimiter=",", invalid_raise=False)
        raw = raw[~np.isnan(raw).any(axis=1)]
        t, x, z = raw[:, 0], raw[:, 1], raw[:, 3]
        w = (t >= 3.0) & (t <= 15.0)
        minz, net = float(z[w].min()), float(x[w][-1] - x[w][0])
        return dict(ok=bool(minz > 0.65 and net > 1.2),
                    v=net / 12.0, minz=minz)
    except Exception:
        return dict(ok=False, v=0.0, minz=0.0)


def batch(tag, xml, n):
    rr = [run(tag, t, xml) for t in range(n)]
    ok = sum(r["ok"] for r in rr)
    v = float(np.mean([r["v"] for r in rr if r["ok"]])) if ok else 0.0
    sd = float(np.std([r["v"] for r in rr if r["ok"]])) if ok > 1 else 0.0
    print(f"{tag}: {ok}/{n} ok  v={v:.3f}±{sd:.3f}", flush=True)
    return dict(tag=tag, ok=ok, n=n, v=v, sd=sd)


def main():
    out = {}
    print("baseline (speed 0.25, cadence 1.1, sway 0.05, step 0.05), n=10")
    out["baseline"] = batch("base", orig, 10)

    print("\nspeed/cadence screen, n=3")
    screen = []
    for sp, ca, sw in itertools.product((0.25, 0.35), (1.1, 1.3),
                                        (0.05, 0.07)):
        tag = f"s{int(sp*100)}c{int(ca*10)}w{int(sw*100)}"
        r = batch(tag, cfg_xml(orig, speed=sp, cadence=ca, sway=sw), 3)
        r["cfg"] = dict(speed=sp, cadence=ca, sway=sw)
        screen.append(r)
    out["screen"] = screen

    good = [r for r in screen if r["ok"] == r["n"] and r["v"] >
            out["baseline"]["v"] + 0.02]
    good.sort(key=lambda r: -r["v"])
    if good:
        print("\nbest candidates at n=10")
        finals = []
        for r in good[:2]:
            rr = batch(r["tag"] + "F", cfg_xml(orig, **r["cfg"]), 10)
            rr["cfg"] = r["cfg"]
            finals.append(rr)
        finals.sort(key=lambda r: (-(r["ok"] >= 8), -r["v"]))
        out["finals"] = finals
        if finals and finals[0]["ok"] >= 8:
            best = cfg_xml(orig, **finals[0]["cfg"])
            open(TASK, "w").write(best)
            open(SRC, "w").write(best)
            print(f"\nWINNER {finals[0]['tag']}: {finals[0]['ok']}/10 at "
                  f"{finals[0]['v']:.3f} m/s -> written to task xml")
        else:
            open(TASK, "w").write(orig)
            open(SRC, "w").write(orig)
            print("\nkeeping baseline config")
    else:
        open(TASK, "w").write(orig)
        open(SRC, "w").write(orig)
        print("\nno candidate beat baseline; keeping it")
    json.dump(out, open(f"{RES}/hybrid_verify.json", "w"), indent=1)
    print("VERIFY DONE")


if __name__ == "__main__":
    main()
