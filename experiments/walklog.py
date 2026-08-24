"""Parse testspeed CSV logs into gait metrics."""
import numpy as np
import mujoco

MJ = "/Users/ncorrell/Downloads/tensegrity/mujoco"
GROUPS = ("hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist", "neck")


def group_of(name):
    for g in GROUPS:
        if name.startswith(g):
            return g
    raise ValueError(name)


def load(csv, model_path=f"{MJ}/humanoid_27dof_tensegrity.xml"):
    m = mujoco.MjModel.from_xml_path(model_path)
    raw = np.genfromtxt(csv, delimiter=",", invalid_raise=False)
    raw = raw[~np.isnan(raw).any(axis=1)]
    nq, nv, nu = m.nq, m.nv, m.nu
    t = raw[:, 0]
    qpos = raw[:, 1:1 + nq]
    qvel = raw[:, 1 + nq:1 + nq + nv]
    ctrl = raw[:, 1 + nq + nv:1 + nq + nv + nu]
    gear = m.actuator_gear[:, 0]
    tau = ctrl * gear                      # applied joint torques
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
             for i in range(nu)]
    return dict(m=m, t=t, qpos=qpos, qvel=qvel, tau=tau, names=names)


def metrics(csv, model_path=f"{MJ}/humanoid_27dof_tensegrity.xml",
            t0=3.0, t1=15.0, total_mass=None):
    d = load(csv, model_path)
    t, qpos, qvel, tau = d["t"], d["qpos"], d["qvel"], d["tau"]
    m = d["m"]
    if total_mass is None:
        total_mass = float(m.body_mass.sum())
    w = (t >= t0) & (t <= t1)
    x = qpos[:, 0]
    z = qpos[:, 2]
    speed = (x[w][-1] - x[w][0]) / (t[w][-1] - t[w][0])
    ok = bool(z[w].min() > 0.55 and (x[w][-1] - x[w][0]) > 0.10 * (t1 - t0))
    # joint velocities: skip the 6 free-joint dofs
    jvel = qvel[:, 6:]
    power = np.abs(tau * jvel[:, :tau.shape[1]])
    dist = max(x[w][-1] - x[w][0], 1e-6)
    cot = float(np.trapezoid(power[w].sum(axis=1), t[w]) /
                (total_mass * 9.81 * dist)) if ok else float("nan")
    per_group = {}
    for g in GROUPS:
        cols = [i for i, n in enumerate(d["names"]) if group_of(n) == g]
        a = np.abs(tau[w][:, cols])
        per_group[g] = dict(
            peak=float(a.max()),
            p95=float(np.percentile(a, 95)),
            rms=float(np.sqrt((a ** 2).mean())))
    return dict(speed=speed, success=ok, cot=cot, per_group=per_group,
                final_x=float(x[-1]), min_z=float(z[w].min()),
                total_mass=total_mass)
