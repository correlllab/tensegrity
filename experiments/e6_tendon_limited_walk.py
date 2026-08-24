#!/usr/bin/env python
"""E6: walking inside the torque envelope the cable network can actually
deliver.

The gait was synthesised on the joint-torque model, whose 27 motors reach the
Winter-derived limits (80 N m at hip pitch and knee) at every instant. The
authority analysis (E5) shows the tension network does not: over the logged
gait poses the achievable torque falls well below those limits at several
DoFs. The offline tension mapping could only say that the demanded torques are
not realisable; it could not say whether a gait exists inside the realisable
set.

This experiment answers that directly. It re-runs the SAME MJPC planner and
task on models whose joint torque limits have been capped at the tendon-
achievable envelope from E5:

  full      original limits (the published result)
  tendonall p10 of the gait-wide bound over all 114 cables
  tendonbom p10 of the same bound over the 36 loop drives of the BOM
  tendonmin the worst-case (min) bound over all 114 cables

If the robot still walks under a cap, the gait fits inside what the cables can
deliver and the tendon-driven robot walks in a meaningful sense. If it does
not, the published gait is an artefact of ideal joint motors.
"""
import json
import os
import re
import subprocess
import sys
import numpy as np

ROOT = "/Users/ncorrell/Downloads/tensegrity"
TD = f"{ROOT}/mujoco_mpc/build/mjpc/tasks/tensegrity"
BIN = f"{ROOT}/mujoco_mpc/build/bin/testspeed"
RES = f"{ROOT}/experiments/results"
MODEL = f"{TD}/humanoid_27dof_tensegrity.xml"
TASK = f"{TD}/task_walk.xml"
N_TRIAL = {'tendonall': 10, 'tendonbom': 10, 'tendonmin': 6}
FLOOR_NM = 1.0          # keep gear strictly positive for MuJoCo

orig_model = open(MODEL).read()
orig_task = open(TASK).read()


def capped_xml(xml, limits, names):
    for n, g in zip(names, limits):
        pat = f'<motor joint="{n}" name="{n}" gear="'
        i = xml.index(pat) + len(pat)
        j = xml.index('"', i)
        xml = xml[:i] + f"{max(g, FLOOR_NM):.4g}" + xml[j:]
    return xml


def run(tag, model_xml, trial):
    log = f"{RES}/log_{tag}_t{trial}.csv"
    if os.path.exists(log):
        return
    open(MODEL, "w").write(model_xml)
    env = dict(os.environ, TESTSPEED_LOG=log)
    out = subprocess.run(
        [BIN, "--task=Tensegrity Walk", "--total_time=16",
         "--steps_per_planning_iteration=4"],
        capture_output=True, text=True, env=env).stdout
    tail = [l for l in out.splitlines() if "sim time" in l][-1:]
    print(f"[{tag} t{trial}] {tail[0] if tail else 'NONE'}", flush=True)


def main():
    a = json.load(open(f"{RES}/e5_authority.json"))
    names, need = a["dofs"], np.array(a["need"])
    caps = {
        "tendonall": np.minimum(a["gait_all"]["pos_p10"],
                                a["gait_all"]["neg_p10"]),
        "tendonbom": np.minimum(a["gait_loops"]["pos_p10"],
                                a["gait_loops"]["neg_p10"]),
        "tendonmin": np.array(a["gait_all"]["margin"]),
    }
    try:
        for tag, cap in caps.items():
            lim = np.minimum(need, cap)
            print(f"\n{tag}: capped DoFs -> " + ", ".join(
                f"{n}:{need[i]:.0f}->{max(lim[i], FLOOR_NM):.0f}"
                for i, n in enumerate(names) if lim[i] < need[i] - 1e-9),
                flush=True)
            xml = capped_xml(orig_model, lim, names)
            for t in range(N_TRIAL[tag]):
                run(tag, xml, t)
    finally:
        open(MODEL, "w").write(orig_model)
        open(TASK, "w").write(orig_task)

    # ---- analysis
    sys.path.insert(0, f"{ROOT}/experiments")
    import walklog                                        # noqa: E402
    import glob                                           # noqa: E402
    out = {}
    for tag in caps:
        rows = []
        for f in sorted(glob.glob(f"{RES}/log_{tag}_t*.csv")):
            try:
                rows.append(walklog.metrics(f))
            except Exception:
                pass
        ok = [r for r in rows if r["success"]]
        v = float(np.mean([r["speed"] for r in ok])) if ok else float("nan")
        out[tag] = dict(n=len(rows), ok=len(ok), v=v)
        print(f"{tag}: {len(ok)}/{len(rows)} walk, "
              f"v={v:.3f} m/s", flush=True)

    def rate(t):
        return f"{out[t]['ok']}/{out[t]['n']}"

    va, vb = out["tendonall"]["v"], out["tendonbom"]["v"]
    summary = (
        f"Capping every joint at the torque the full cable network can deliver "
        f"over the gait leaves the planner {rate('tendonall')} successful "
        f"walks"
        + (f" at {va:.2f}\\,m/s" if out["tendonall"]["ok"] else "")
        + f"; capping at the worst-case bound gives {rate('tendonmin')}; and "
        f"capping at what the {36}-motor bill of materials can deliver gives "
        f"{rate('tendonbom')}. "
        + ("A gait therefore exists inside the envelope of the full network "
           "but not inside the envelope of the actuation set we specified: "
           "the shortfall is in the bill of materials, not in the topology."
           if out["tendonall"]["ok"] and not out["tendonbom"]["ok"] else
           "No gait was found inside either envelope, so the shortfall is in "
           "the cage geometry rather than in the number of motors."
           if not out["tendonall"]["ok"] else
           "Both envelopes admit a gait, so the offline tension mapping "
           "overstates the problem.")
    )
    js = dict(Summary=summary, **{f"{k}Rate": rate(k) for k in out})
    json.dump(js, open(f"{RES}/e6_summary.json", "w"), indent=1)
    print("\nE6 DONE ->", f"{RES}/e6_summary.json", flush=True)


if __name__ == "__main__":
    main()
