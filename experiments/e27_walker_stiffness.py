#!/usr/bin/env python
"""E27: the whole-body hybrid walker at series-elastic cable stiffness.

The review's remaining objection: the joint testbed (E10/E19) runs at the
specified 40 kN/m series-elastic stiffness, but the walking model's ankle and
waist cables are 2.4 kN/m. Here the SAME walker, SAME controller configuration
(hybrid_baseline_verify BASECFG), is run with the ankle and waist interface
cables at 20, 40 and 168 kN/m -- the ends and the testbed point of the
specified 20-168 kN/m band.

What is held fixed, and how:
  * prestress: every interface cable keeps the tension it carries at the
    model's start pose (F = K0 (L - L0) at qpos0), by recomputing its rest
    length for the new K;
  * damping ratio: per-cable damping scales with sqrt(K);
  * everything else (masses, geometry, actuators, gear, cost weights, horizon
    0.42 s, planning period 16 ms of simulated time).

What must change: the integration steps. The stiffest cable mode of each
variant is computed from J^T K J against the mass matrix at the start pose,
and the plant / planner steps are chosen so that (omega_max * dt) does not
exceed the ratio the shipped baseline already runs at (calibrated here, not
assumed). testspeed's steps_per_planning_iteration is rescaled so the planner
still replans every 16 ms of simulated time.

Usage:
  e27_walker_stiffness.py plan                 # build variants, print steps
  e27_walker_stiffness.py run  K_kN n [t0]     # e.g. run 40 10
Results accumulate in results/e27_walker_stiffness.json after every trial.
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
LOGD = f"{RES}/e27_walker_stiffness"
OUT = f"{RES}/e27_walker_stiffness.json"
LIVE = f"{TD}/task_hybrid_walk.xml"
PRISTINE = f"{TD}/task_hybrid_walk.xml.e27orig"   # restored after every trial
if not os.path.exists(PRISTINE):
    open(PRISTINE, "w").write(open(LIVE).read())
os.makedirs(LOGD, exist_ok=True)
sys.path.insert(0, f"{ROOT}/experiments")
import hybrid_baseline_verify as hbv                       # noqa: E402

K0 = 2400.0                    # shipped interface-cable stiffness
PLAN_PERIOD = 0.016            # s of simulated time between planner iterations
STEP_CHOICES = [0.008, 0.004, 0.002, 0.001, 0.0005]
TEND_RE = re.compile(r'<spatial name="((?:ank|waist)_[^"]+)" stiffness="([\d.eE+-]+)"'
                     r' damping="([\d.eE+-]+)" springlength="0 ([\d.eE+-]+)"')


def omega_max(xml):
    """Stiffest tendon-spring mode at qpos0: max generalized eigenvalue of
    (J^T diag(k) J, M) over the interface cables."""
    os.chdir(MJ)                       # mesh assets are relative to mujoco/
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    idx = [i for i in range(m.ntendon)
           if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TENDON, i)
           .startswith(("ank_", "waist_"))]
    # dense tendon Jacobian by central differences of ten_length (robust to
    # the sparse ten_J layout of recent MuJoCo versions)
    eps = 1e-6
    Jd = np.zeros((m.ntendon, m.nv))
    q0 = d.qpos.copy()
    for k in range(m.nv):
        dq = np.zeros(m.nv); dq[k] = eps
        qp = q0.copy(); mujoco.mj_integratePos(m, qp, dq, 1.0)
        qm = q0.copy(); mujoco.mj_integratePos(m, qm, -dq, 1.0)
        d.qpos[:] = qp; mujoco.mj_forward(m, d); lp = d.ten_length.copy()
        d.qpos[:] = qm; mujoco.mj_forward(m, d); lm = d.ten_length.copy()
        Jd[:, k] = (lp - lm) / (2 * eps)
    d.qpos[:] = q0
    mujoco.mj_forward(m, d)
    J = Jd[idx]
    k = m.tendon_stiffness[idx]
    Kq = J.T @ (k[:, None] * J)
    M = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, M)                # (model, data, dst) in MuJoCo 3.11
    # generalized eigenproblem via Cholesky of M
    L = np.linalg.cholesky(M + 1e-9 * np.eye(m.nv))
    Li = np.linalg.inv(L)
    w = np.linalg.eigvalsh(Li @ Kq @ Li.T)
    return float(np.sqrt(max(w.max(), 0.0))), m


def make_variant(K):
    """Model file with interface cables at K, prestress and damping ratio
    preserved; returns (model_path, omega_max, plant_dt)."""
    xml = open(f"{MJ}/humanoid_hybrid.xml").read()
    os.chdir(MJ)
    m = mujoco.MjModel.from_xml_path(f"{MJ}/humanoid_hybrid.xml")
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    length = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TENDON, i): d.ten_length[i]
              for i in range(m.ntendon)}

    def sub(mo):
        name, k0, c0, l0 = mo.group(1), float(mo.group(2)), float(mo.group(3)), float(mo.group(4))
        F = max(0.0, k0 * (length[name] - l0))          # tension at start pose
        l_new = length[name] - F / K
        c_new = c0 * np.sqrt(K / k0)
        return (f'<spatial name="{name}" stiffness="{K:.1f}" damping="{c_new:.2f}"'
                f' springlength="0 {l_new:.6f}"')
    xml2, n = TEND_RE.subn(sub, xml)
    assert n == 36, n
    om, _ = omega_max(xml2)
    return xml2, om


MJPC_MAX_KNOTS = 512           # mjpc/trajectory.h kMaxTrajectoryHorizon


def choose_steps(om, ratio, horizon=0.42):
    """Largest step in STEP_CHOICES with omega*dt <= ratio, but never so
    small that the horizon exceeds MJPC's knot cap (which would silently
    shorten the horizon -- this clipped the first 168 kN/m arm to 0.26 s)."""
    plan_dt = next((s for s in STEP_CHOICES if om * s <= ratio), STEP_CHOICES[-1])
    floor = horizon / (MJPC_MAX_KNOTS - 1)
    if plan_dt < floor:
        plan_dt = round(floor, 5)
    plant_dt = plan_dt / 2
    return plan_dt, plant_dt


def build(K, overrides=None):
    overrides = overrides or {}
    om0, _ = omega_max(open(f"{MJ}/humanoid_hybrid.xml").read())
    ratio = om0 * 0.008                     # the shipped planner step is stable here
    xml, om = make_variant(K)
    plan_dt, plant_dt = choose_steps(om, ratio, cfg_h := dict(hbv.BASECFG, **overrides)["horizon"])
    xml = re.sub(r'<option timestep="[\d.]+"', f'<option timestep="{plant_dt}"', xml)
    ktag = f"k{int(K/1000)}"
    tag = ktag + "".join(f"_{k}{v}" for k, v in sorted(overrides.items()))
    mpath = f"{TD}/humanoid_e27_{ktag}.xml"
    open(mpath, "w").write(xml)
    task = open(PRISTINE).read().replace("humanoid_hybrid.xml", f"humanoid_e27_{ktag}.xml")
    task = re.sub(r'(name="agent_timestep" data=")[\d.]+', rf'\g<1>{plan_dt}', task)
    cfg = dict(hbv.BASECFG, **overrides)
    task = hbv.configure(task, **cfg)
    spp = int(round(PLAN_PERIOD / plant_dt))
    return dict(tag=tag, K=K, config=cfg, omega_max=om, omega_base=om0, ratio=ratio,
                ratio_used=om * plan_dt, knots=int(round(cfg_h / plan_dt)) + 1,
                plan_dt=plan_dt, plant_dt=plant_dt, spp=spp, model=mpath), task


def load():
    return json.load(open(OUT)) if os.path.exists(OUT) else {}


def save(res):
    json.dump(res, open(OUT, "w"), indent=1)


def run(K, n, t0=0, overrides=None):
    info, task = build(K, overrides)
    tag = info["tag"]
    res = load()
    cond = res.get(tag, dict(info, trials={}))
    cond.update(info)
    print(f"[{tag}] omega_max {info['omega_max']:.0f} rad/s (base {info['omega_base']:.0f}), "
          f"plan_dt {info['plan_dt']*1e3:.1f} ms, plant_dt {info['plant_dt']*1e3:.2f} ms, "
          f"spp {info['spp']}", flush=True)
    os.chdir(MJ)
    m = mujoco.MjModel.from_xml_path(info["model"])
    for t in range(t0, t0 + n):
        log = f"{LOGD}/log_{tag}_t{t}.csv"
        backup = open(PRISTINE).read()
        rt = cost = None
        try:
            open(LIVE, "w").write(task)
            if not os.path.exists(log):
                env = dict(os.environ, TESTSPEED_LOG=log)
                out = subprocess.run([BIN, "--task=Hybrid Walk", "--total_time=16",
                                      f"--steps_per_planning_iteration={info['spp']}"],
                                     capture_output=True, text=True, env=env, cwd=TD).stdout
                m_rt = re.search(r"\(([\d.]+)x realtime\)", out)
                m_c = re.search(r"Average cost per step[^:]*: ([\d.eE+-]+)", out)
                rt = float(m_rt.group(1)) if m_rt else None
                cost = float(m_c.group(1)) if m_c else None
        finally:
            open(LIVE, "w").write(backup)
        ok, v, raw = hbv.walk_metrics(log)
        lift, t_fall = hbv.foot_lift(m, raw)
        cond["trials"][str(t)] = dict(ok=ok, v=v, rt=rt, cost=cost, t_fall=t_fall,
                                      max_lift_mm=1e3 * lift)
        print(f"  {tag} t{t}: {'ok  ' if ok else 'FELL'} v={v:.3f}"
              + (f" rt={rt:.3f}" if rt else "") + (f" cost={cost:.3f}" if cost else "")
              + (f" fall_t={t_fall:.1f}" if t_fall else ""), flush=True)
        trials = cond["trials"].values()
        okn = sum(tr["ok"] for tr in trials)
        cond.update(ok=okn, n=len(cond["trials"]),
                    v=float(np.mean([tr["v"] for tr in trials if tr["ok"]])) if okn else 0.0,
                    realtime=float(np.mean([tr["rt"] for tr in trials if tr["rt"]])) if any(tr["rt"] for tr in trials) else None)
        ok_logs = [f"{LOGD}/log_{tag}_t{k}.csv" for k, tr in cond["trials"].items() if tr["ok"]]
        if ok_logs:
            cond["gait"] = hbv.gait_analysis(m, ok_logs)
        res = load()
        res[tag] = cond
        save(res)


if __name__ == "__main__":
    if sys.argv[1] == "plan":
        for K in (20e3, 40e3, 168e3):
            info, _ = build(K)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in info.items()})
    else:
        # run K_kN n [t0] [key=value ...]   e.g. run 168 5 0 torso=1.28
        extra = [a for a in sys.argv[4:] if "=" in a]
        pos = [a for a in sys.argv[4:] if "=" not in a]
        ov = {k: float(v) for k, v in (a.split("=") for a in extra)}
        run(float(sys.argv[2]) * 1e3, int(sys.argv[3]), int(pos[0]) if pos else 0, ov)
