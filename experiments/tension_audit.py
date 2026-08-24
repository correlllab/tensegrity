#!/usr/bin/env python
"""Is the planned gait realisable by the cable network? (corrected audit)

The first version of this analysis reported a peak tension of 1500 N and read
it as the gait "touching" the Dyneema working limit. It was not touching it:
1500 N is the upper bound of the box constraint, so the solver was pinned
against it, and the meaningful quantity -- how much torque went undelivered --
was reported only as a maximum. It also reported an RMS taken ACROSS the 114
cables at each instant, which says nothing about fatigue in the few cables
that actually work.

This version reports, over states sampled from logged walks:

  sat_frac      fraction of samples at which at least one cable is at Fmax
  worst_residual  largest undelivered joint torque (N m)
  p95_residual    95th percentile of the same
  busiest_rms   RMS tension of the single busiest cable (the fatigue number)
  fmax_needed   peak tension required if the 1500 N bound were lifted, i.e.
                the cable rating this gait actually asks for
"""
import glob
import json
import sys
import numpy as np
import mujoco
from scipy.optimize import lsq_linear

MJ = "/Users/ncorrell/Downloads/tensegrity/mujoco"
RES = "/Users/ncorrell/Downloads/tensegrity/experiments/results"
OUT = f"{RES}/tension_audit.json"


def audit(logs, cable_xml=f"{MJ}/humanoid_27dof_tensegrity_cable.xml",
          t0=3.0, t1=15.0, stride=12, payload=0.0):
    mc = mujoco.MjModel.from_xml_path(cable_xml)
    dc = mujoco.MjData(mc)
    mt = mujoco.MjModel.from_xml_path(f"{MJ}/humanoid_27dof_tensegrity.xml")
    gear = mt.actuator_gear[:, 0]
    fmax = -mc.actuator_gear[:, 0]
    cn = [mujoco.mj_id2name(mc, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
          for i in range(mc.nu)]
    reg = np.sqrt(0.05) * np.eye(mc.nu)
    zero = np.zeros(mc.nu)

    T_hist, resid, sat, need, T_free = [], [], [], [], []
    for csv in logs:
        raw = np.genfromtxt(csv, delimiter=",", invalid_raise=False)
        raw = raw[~np.isnan(raw).any(axis=1)]
        t = raw[:, 0]
        sel = np.where((t >= t0) & (t <= t1))[0][::stride]
        qpos = raw[:, 1:1 + mt.nq]
        ctrl = raw[:, 1 + mt.nq + mt.nv:1 + mt.nq + mt.nv + mt.nu]
        for i in sel:
            dc.qpos[:] = qpos[i]
            mujoco.mj_forward(mc, dc)
            mom = np.zeros((mc.nu, mc.nv))
            mujoco.mju_sparse2dense(mom, dc.actuator_moment, dc.moment_rownnz,
                                    dc.moment_rowadr, dc.moment_colind)
            M = mom[:, 6:].T
            tau = ctrl[i] * gear
            sol = lsq_linear(np.vstack([M, reg]), np.concatenate([tau, zero]),
                             bounds=(0.0, 1.0), max_iter=30, tol=1e-6)
            T = fmax * sol.x
            T_hist.append(T)
            resid.append(float(np.abs(M @ sol.x - tau).max()))
            sat.append(bool((sol.x > 0.999).any()))
            # what rating would this state need? lift the upper bound
            free = lsq_linear(np.vstack([M, reg]),
                              np.concatenate([tau, zero]),
                              bounds=(0.0, np.inf), max_iter=30, tol=1e-6)
            T_free.append(fmax * free.x)
            need.append(float((fmax * free.x).max()))
    T_hist = np.array(T_hist)
    Tf = np.array(T_free)
    rms = np.sqrt((T_hist ** 2).mean(axis=0))
    busiest = int(np.argmax(rms))
    # per-cable demand from the UNBOUNDED solve: what the hardware must carry
    # to execute this gait, as opposed to what the 1.5 kN bound lets it carry
    # p99, not max: at poses where a moment arm collapses the unbounded solve
    # asks for unbounded tension, which is a statement about the geometry, not
    # a rope requirement
    demand_peak = np.percentile(Tf, 99, axis=0)
    demand_max = Tf.max(axis=0)
    demand_rms = np.sqrt((Tf ** 2).mean(axis=0))
    return dict(samples=len(T_hist), payload=payload,
                peak_tension=float(T_hist.max()),
                sat_frac=float(np.mean(sat)),
                worst_residual=float(np.max(resid)),
                p95_residual=float(np.percentile(resid, 95)),
                median_residual=float(np.median(resid)),
                busiest_cable=cn[busiest],
                busiest_rms=float(rms[busiest]),
                busiest_peak=float(T_hist[:, busiest].max()),
                mean_rms_across_cables=float(rms.mean()),
                fmax_needed=float(np.percentile(need, 99)),
                fmax_needed_max=float(np.max(need)),
                cables=cn,
                demand_peak=demand_peak.tolist(),
                demand_max=demand_max.tolist(),
                demand_rms=demand_rms.tolist())


def main():
    base = sorted(glob.glob(f"{RES}/log_baseline_t*.csv"))[:3]
    r = audit(base)
    print("baseline gait, tendon realisability")
    for k, v in r.items():
        print(f"  {k:26s} {v}")
    rows = {"0": r}
    for m in (4, 8, 12, 16, 20):
        logs = sorted(glob.glob(f"{RES}/log_cpayload{m:02d}_t*.csv"))[:2]
        if logs:
            rows[str(m)] = audit(logs, f"{RES}/cable_cpayload_{m}.xml",
                                 payload=m)
            print(f"\n{m} kg payload: peak {rows[str(m)]['peak_tension']:.0f} N, "
                  f"sat {100*rows[str(m)]['sat_frac']:.0f}%, "
                  f"resid {rows[str(m)]['worst_residual']:.0f} Nm, "
                  f"needs {rows[str(m)]['fmax_needed']:.0f} N")
    out = dict(r)
    out["by_payload"] = rows
    json.dump(out, open(OUT, "w"), indent=1)
    print("\naudit ->", OUT)


if __name__ == "__main__":
    main()
