#!/usr/bin/env python
"""Map logged walking torque traces through the cable network: at subsampled
states solve the bounded tension distribution min ||M(q) x - tau||, x in
[0,1], and record peak required tension and worst residual. A trajectory is
tendon-feasible if the residual stays small with tensions within limits.

Usage: tension_map.py <log.csv> [cable_model.xml]
"""
import sys
import numpy as np
import mujoco
from scipy.optimize import lsq_linear

MJ = "/Users/ncorrell/Downloads/tensegrity/mujoco"


def analyze(csv, cable_xml=f"{MJ}/humanoid_27dof_tensegrity_cable.xml",
            t0=3.0, t1=15.0, stride=10):
    mc = mujoco.MjModel.from_xml_path(cable_xml)
    dc = mujoco.MjData(mc)
    mt = mujoco.MjModel.from_xml_path(f"{MJ}/humanoid_27dof_tensegrity.xml")
    gear = mt.actuator_gear[:, 0]
    fmax = -mc.actuator_gear[:, 0]
    raw = np.genfromtxt(csv, delimiter=",", invalid_raise=False)
    raw = raw[~np.isnan(raw).any(axis=1)]
    nq = mt.nq
    t = raw[:, 0]
    sel = np.where((t >= t0) & (t <= t1))[0][::stride]
    qpos = raw[:, 1:1 + nq]
    ctrl = raw[:, 1 + nq + mt.nv:1 + nq + mt.nv + mt.nu]
    reg = np.sqrt(0.05) * np.eye(mc.nu)
    x0 = np.zeros(mc.nu)
    peak_T = 0.0
    rms_T = []
    worst_resid = 0.0
    for i in sel:
        dc.qpos[:] = qpos[i]
        mujoco.mj_forward(mc, dc)
        mom = np.zeros((mc.nu, mc.nv))
        mujoco.mju_sparse2dense(mom, dc.actuator_moment, dc.moment_rownnz,
                                dc.moment_rowadr, dc.moment_colind)
        M = mom[:, 6:].T
        tau = ctrl[i] * gear
        sol = lsq_linear(np.vstack([M, reg]),
                         np.concatenate([tau, x0]),
                         bounds=(0.0, 1.0), max_iter=30, tol=1e-6)
        T = fmax * sol.x
        peak_T = max(peak_T, float(T.max()))
        rms_T.append(float(np.sqrt((T ** 2).mean())))
        worst_resid = max(worst_resid, float(np.abs(M @ sol.x - tau).max()))
    return dict(peak_tension=peak_T, mean_rms_tension=float(np.mean(rms_T)),
                worst_torque_residual=worst_resid, samples=len(sel))


if __name__ == "__main__":
    r = analyze(sys.argv[1], *(sys.argv[2:3] or
                               [f"{MJ}/humanoid_27dof_tensegrity_cable.xml"]))
    print(r)
