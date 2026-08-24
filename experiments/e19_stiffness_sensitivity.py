#!/usr/bin/env python
"""E19: does the hinge-free joint's impact attenuation survive stiffer cables?

The review's concern: the headline attenuation numbers were measured at the
40 kN/m series-elastic cable stiffness, and hardware rope is far stiffer.
Repeat the E10 drop protocol (cable joint vs. mass-matched pin joint) at the
baseline stiffness and at 10x, and report the attenuation at each height.
"""
import json
import sys
import numpy as np
import mujoco

sys.path.insert(0, "/Users/ncorrell/Downloads/tensegrity/experiments")
import e10_tensegrity_joint as e10

RES = "/Users/ncorrell/Downloads/tensegrity/experiments/results"
OVERLAP = 0.2 * e10.H
HEIGHTS = [0.05, 0.10, 0.20]
STIFF = {"base_40kN": 40e3, "x10_400kN": 400e3}


def drop(h, pin, k):
    xml, _ = e10.build_drop(OVERLAP, pin, prestress_N=100.0, k_cable=k)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    for _ in range(int(0.8 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    if not np.all(np.isfinite(d.qpos)):
        return float("nan")
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            d.qpos[m.jnt_qposadr[j] + 2] += h
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    peak, f6 = 0.0, np.zeros(6)
    for _ in range(int(0.35 / m.opt.timestep)):
        mujoco.mj_step(m, d)
        if not np.all(np.isfinite(d.qpos)):
            return float("nan")
        for i in range(d.ncon):
            c = d.contact[i]
            if floor in (c.geom[0], c.geom[1]):
                mujoco.mj_contactForce(m, d, i, f6)
                peak = max(peak, abs(float(f6[0])))
    return peak


def main():
    out = {}
    for tag, k in STIFF.items():
        rows = []
        for h in HEIGHTS:
            cable = drop(h, pin=False, k=k)
            pin = drop(h, pin=True, k=k)
            delta = 100.0 * (cable - pin) / pin
            rows.append(dict(h=h, cable_N=cable, pin_N=pin, delta_pct=delta))
            print(f"{tag} h={h:.2f}: cable {cable:6.0f} N  pin {pin:6.0f} N "
                  f" delta {delta:+.1f}%", flush=True)
        out[tag] = rows
    json.dump(out, open(f"{RES}/e19_stiffness_sensitivity.json", "w"),
              indent=1)
    print("E19 ->", f"{RES}/e19_stiffness_sensitivity.json")


if __name__ == "__main__":
    main()
