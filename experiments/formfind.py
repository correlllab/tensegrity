#!/usr/bin/env python
"""Form-finding for an assembled tensegrity: find a self-stress state.

Giving every cable the same tension is NOT a self-stress state, and a model
built that way is being pulled apart by its own prestress before any
controller runs. The check is cheap and worth doing on any assembly: switch
gravity off, actuate nothing, and see whether it moves.

Here we solve for tensions that hold every ungrounded body in static
equilibrium at the home pose,

    min ||T||^2   s.t.   W(q) T = -m g   for every free body,   T >= 0,

with W the map from cable tension to body wrench. The residual is the useful
output: if no non-negative tension state balances the structure, the geometry
is not a tensegrity and no amount of control will fix it.
"""
import sys
import numpy as np
import mujoco
from scipy.optimize import lsq_linear


def wrench_map(m, d, grounded):
    mujoco.mj_forward(m, d)
    gid = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, g) for g in grounded}
    free = [b for b in range(1, m.nbody) if b not in gid]
    rows = {b: i for i, b in enumerate(free)}
    W = np.zeros((6 * len(free), m.ntendon))
    rhs = np.zeros(6 * len(free))
    for b in free:
        rhs[6 * rows[b]:6 * rows[b] + 3] = -m.body_mass[b] * m.opt.gravity
    for t in range(m.ntendon):
        adr = m.tendon_adr[t]
        sids = [int(m.wrap_objid[adr + k]) for k in range(m.tendon_num[t])
                if m.wrap_type[adr + k] == mujoco.mjtWrap.mjWRAP_SITE]
        if len(sids) < 2:
            continue
        for k, s in enumerate(sids):
            b = m.site_bodyid[s]
            if b not in rows:
                continue
            v = d.site_xpos[sids[1 - k]] - d.site_xpos[s]
            n = np.linalg.norm(v)
            if n < 1e-9:
                continue
            u = v / n
            i = 6 * rows[b]
            W[i:i + 3, t] += u
            W[i + 3:i + 6, t] += np.cross(d.site_xpos[s] - d.xipos[b], u)
    return W, rhs, free, rows


def solve(m, d, grounded, t_min=0.0, t_max=2000.0, reg=2e-3):
    W, rhs, free, rows = wrench_map(m, d, grounded)
    sol = lsq_linear(np.vstack([W, reg * np.eye(m.ntendon)]),
                     np.concatenate([rhs, np.zeros(m.ntendon)]),
                     bounds=(t_min, t_max), max_iter=500, tol=1e-9)
    r = W @ sol.x - rhs
    per_body = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b):
                float(np.linalg.norm(r[6 * rows[b]:6 * rows[b] + 6]))
                for b in free}
    return dict(T=sol.x, residual=float(np.linalg.norm(r)),
                load=float(np.linalg.norm(rhs)),
                rel=float(np.linalg.norm(r) / max(np.linalg.norm(rhs), 1e-9)),
                per_body=per_body)


def selfstress(m, d, grounded, t_ref=1.0):
    """A self-stress state: tensions that balance with NO external load,
    W T = 0 with T > 0. This is the homogeneous solution, and it is what
    "prestress" actually means -- a floor on tension is not one, which is why
    forcing t_min > 0 in solve() destroys the equilibrium instead of stiffening
    the structure.

    The full prestress is then T = T_gravity + alpha * T_selfstress, for any
    alpha >= 0, and alpha is the free design knob.
    """
    from scipy.optimize import linprog
    W, _, _, _ = wrench_map(m, d, grounded)
    n = W.shape[1]
    c = np.zeros(n + 1)
    c[-1] = -1.0                                   # maximise the slack t
    A_ub = np.hstack([-np.eye(n), np.ones((n, 1))])
    r = linprog(c, A_ub=A_ub, b_ub=np.zeros(n),
                A_eq=np.hstack([W, np.zeros((W.shape[0], 1))]),
                b_eq=np.zeros(W.shape[0]),
                bounds=[(0, None)] * n + [(0, t_ref)])
    if not r.success or r.x[-1] < 1e-9:
        return None, 0.0
    T = r.x[:n]
    return T / max(T.max(), 1e-9), float(r.x[-1])


def drift_test(m, seconds=0.5, zero_g=True):
    """Is the prestress balanced? With gravity off and no actuation a
    self-stressed assembly does not move."""
    mm = m
    g = mm.opt.gravity.copy()
    if zero_g:
        mm.opt.gravity[:] = 0
    d = mujoco.MjData(mm)
    mujoco.mj_forward(mm, d)
    p0 = d.xpos.copy()
    for _ in range(int(seconds / mm.opt.timestep)):
        mujoco.mj_step(mm, d)
    out = float(np.abs(d.xpos - p0).max())
    mm.opt.gravity[:] = g
    return out


if __name__ == "__main__":
    path = sys.argv[1]
    grounded = sys.argv[2:] or ["seg0"]
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    r = solve(m, d, grounded)
    print(f"{path}")
    print(f"  self-stress residual {r['residual']:.2f} N of {r['load']:.1f} N "
          f"load  ({100*r['rel']:.1f}%)  -> "
          f"{'form-findable' if r['rel'] < 0.1 else 'NOT A TENSEGRITY at this geometry'}")
    print(f"  tensions {r['T'].min():.0f}-{r['T'].max():.0f} N")
    print(f"  zero-g drift as built: {1e3*drift_test(m):.1f} mm in 0.5 s")
    worst = sorted(r["per_body"].items(), key=lambda kv: -kv[1])[:5]
    print("  worst-balanced bodies: "
          + ", ".join(f"{k} {v:.0f} N" for k, v in worst))
