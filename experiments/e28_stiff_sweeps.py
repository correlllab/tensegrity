#!/usr/bin/env python
"""E28: capability and robustness sweeps at series-elastic cable stiffness.

Two arms, both built on the E27 stiff-cable variants of the hybrid walker
(prestress and damping ratio preserved, planner/plant steps from E27):

  payload K mass n     trunk payload box (hybrid_sweeps.payload_model) on the
                       K kN/m walker, n trials     e.g. payload 40 12 3
  random  K n          Monte Carlo robustness on the K kN/m walker, n seeds:
                       per-cable interface stiffness x U(0.7, 1.3),
                       per-cable prestress x U(0.8, 1.2) (rest length
                       recomputed), every geom/inertial mass x U(0.85, 1.15),
                       floor + pad friction mu ~ U(0.3, 0.9).  Planner and
                       plant share the perturbed model (testspeed has one
                       model), so this tests robustness of the gait TUNING to
                       parameter variation, not planner-model mismatch.

Results accumulate in results/e28_stiff_sweeps.json; logs in
results/e28_stiff_sweeps/.
"""
import json
import os
import re
import subprocess
import sys
import numpy as np
import mujoco

ROOT = "/Users/ncorrell/Downloads/tensegrity"
MJ = f"{ROOT}/mujoco"
TD = f"{ROOT}/mujoco_mpc/build/mjpc/tasks/tensegrity"
BIN = f"{ROOT}/mujoco_mpc/build/bin/testspeed"
RES = f"{ROOT}/experiments/results"
LOGD = f"{RES}/e28_stiff_sweeps"
OUT = f"{RES}/e28_stiff_sweeps.json"
LIVE = f"{TD}/task_hybrid_walk.xml"
PRISTINE = f"{TD}/task_hybrid_walk.xml.e27orig"
os.makedirs(LOGD, exist_ok=True)
sys.path.insert(0, f"{ROOT}/experiments")
import e27_walker_stiffness as e27                          # noqa: E402
import hybrid_baseline_verify as hbv                        # noqa: E402

TEND_RE = e27.TEND_RE


def load():
    return json.load(open(OUT)) if os.path.exists(OUT) else {}


def save(res):
    json.dump(res, open(OUT, "w"), indent=1)


LOCK = f"{TD}/.task_swap.lock"


def trial(task, log, spp):
    """Swap the live task in under a lock, launch testspeed, hold the lock
    until it has loaded, then restore the pristine task. Several arms run
    concurrently, and testspeed reads the task file only at startup."""
    import fcntl
    import time
    rt = cost = None
    if not os.path.exists(log):
        env = dict(os.environ, TESTSPEED_LOG=log)
        with open(LOCK, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            open(LIVE, "w").write(task)
            proc = subprocess.Popen([BIN, "--task=Hybrid Walk", "--total_time=16",
                                     f"--steps_per_planning_iteration={spp}"],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, env=env, cwd=TD)
            time.sleep(8)                      # model compile + task load
            open(LIVE, "w").write(open(PRISTINE).read())
            fcntl.flock(lf, fcntl.LOCK_UN)
        out = proc.communicate()[0]
        m_rt = re.search(r"\(([\d.]+)x realtime\)", out)
        m_c = re.search(r"Average cost per step[^:]*: ([\d.eE+-]+)", out)
        rt = float(m_rt.group(1)) if m_rt else None
        cost = float(m_c.group(1)) if m_c else None
    ok, v, raw = hbv.walk_metrics(log)
    t_fall = None
    z = raw[:, 3]
    fell = np.where(z < 0.65)[0]
    if len(fell):
        t_fall = float(raw[fell[0], 0])
    return dict(ok=ok, v=v, rt=rt, cost=cost, t_fall=t_fall)


def record(key, info, t, row):
    res = load()
    cond = res.get(key, dict(info, trials={}))
    cond["trials"][str(t)] = row
    trials = list(cond["trials"].values())
    okn = sum(tr["ok"] for tr in trials)
    cond.update(ok=okn, n=len(trials),
                falls=sum(1 for tr in trials if tr.get("t_fall")),
                v=float(np.mean([tr["v"] for tr in trials if tr["ok"]])) if okn else 0.0)
    res[key] = cond
    save(res)
    return cond


def payload(K, mass, n, overrides=None):
    info, task = e27.build(K, overrides)      # stiff model + task at right steps
    xml = open(info["model"]).read()
    assert '<body name="head"' in xml
    xml = xml.replace('<body name="head"', f'''<body name="payload" pos="0 0 0.06">
        <geom name="payload_box" type="box" size="0.07 0.10 0.07"
              mass="{mass}" rgba="0.72 0.55 0.30 1" contype="0"
              conaffinity="0"/>
      </body>
      <body name="head"''', 1)
    mname = f"humanoid_e28_{info['tag']}_pay{mass:02d}.xml"
    open(f"{TD}/{mname}", "w").write(xml)
    task = task.replace(os.path.basename(info["model"]), mname)
    key = f"pay_{info['tag']}_{mass:02d}"
    meta = dict(K=K, mass=mass, plan_dt=info["plan_dt"], plant_dt=info["plant_dt"], spp=info["spp"])
    print(f"[{key}] steps {info['plan_dt']*1e3:.2f}/{info['plant_dt']*1e3:.2f} ms spp {info['spp']}", flush=True)
    for t in range(n):
        row = trial(task, f"{LOGD}/log_{key}_t{t}.csv", info["spp"])
        cond = record(key, meta, t, row)
        print(f"  {key} t{t}: {'ok  ' if row['ok'] else 'FELL'} v={row['v']:.3f}"
              + (f" fall_t={row['t_fall']:.1f}" if row['t_fall'] else "")
              + f"   [{cond['ok']}/{cond['n']}]", flush=True)


def randomized(K, n):
    info, task = e27.build(K)
    base = open(info["model"]).read()
    os.chdir(MJ)
    m0 = mujoco.MjModel.from_xml_path(info["model"])
    d0 = mujoco.MjData(m0)
    mujoco.mj_forward(m0, d0)
    length = {mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_TENDON, i): d0.ten_length[i]
              for i in range(m0.ntendon)}
    key = f"rand_{info['tag']}"
    meta = dict(K=K, plan_dt=info["plan_dt"], plant_dt=info["plant_dt"], spp=info["spp"],
                ranges=dict(k="U(0.7,1.3) per cable", pre="U(0.8,1.2) per cable",
                            mass="U(0.85,1.15) per geom/inertial", mu="U(0.3,0.9)"))
    print(f"[{key}] steps {info['plan_dt']*1e3:.2f}/{info['plant_dt']*1e3:.2f} ms spp {info['spp']}", flush=True)
    for seed in range(n):
        rng = np.random.RandomState(1000 + seed)
        mu = float(rng.uniform(0.3, 0.9))
        kf, pf = [], []

        def sub(mo):
            name, k0, c0, l0 = mo.group(1), float(mo.group(2)), float(mo.group(3)), float(mo.group(4))
            fk, fp = rng.uniform(0.7, 1.3), rng.uniform(0.8, 1.2)
            kf.append(fk); pf.append(fp)
            F = max(0.0, k0 * (length[name] - l0)) * fp
            k = k0 * fk
            return (f'<spatial name="{name}" stiffness="{k:.1f}" damping="{c0*np.sqrt(fk):.2f}"'
                    f' springlength="0 {length[name] - F / k:.6f}"')
        xml, ncab = TEND_RE.subn(sub, base)
        assert ncab == 36
        mf = []

        def msub(mo):
            f = rng.uniform(0.85, 1.15); mf.append(f)
            return f'mass="{float(mo.group(1)) * f:.5f}"'
        xml = re.sub(r'mass="([\d.eE+-]+)"', msub, xml)
        xml = re.sub(r'friction="[\d.eE+-]+ ', f'friction="{mu:.3f} ', xml)
        mname = f"humanoid_e28_{info['tag']}_rand{seed:02d}.xml"
        open(f"{TD}/{mname}", "w").write(xml)
        mm = mujoco.MjModel.from_xml_string(xml)
        ttask = task.replace(os.path.basename(info["model"]), mname)
        row = trial(ttask, f"{LOGD}/log_{key}_s{seed:02d}.csv", info["spp"])
        row.update(seed=seed, mu=mu, total_mass=float(mm.body_mass.sum()),
                   k_factor_mean=float(np.mean(kf)), pre_factor_mean=float(np.mean(pf)),
                   mass_factor_mean=float(np.mean(mf)))
        cond = record(key, meta, seed, row)
        print(f"  {key} s{seed}: {'ok  ' if row['ok'] else 'FELL'} v={row['v']:.3f} mu={mu:.2f} "
              f"m={row['total_mass']:.1f}kg" + (f" fall_t={row['t_fall']:.1f}" if row['t_fall'] else "")
              + f"   [{cond['ok']}/{cond['n']}]", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "payload":
        ov = {k: float(v) for k, v in (a.split("=") for a in sys.argv[5:] if "=" in a)}
        payload(float(sys.argv[2]) * 1e3, int(sys.argv[3]), int(sys.argv[4]), ov)
    elif sys.argv[1] == "random":
        randomized(float(sys.argv[2]) * 1e3, int(sys.argv[3]))
