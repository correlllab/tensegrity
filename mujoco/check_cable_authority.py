#!/usr/bin/env python
"""Per-DoF torque capacity of the cable network, at the home pose.

With tension-only actuation, tau = M t with 0 <= t <= Fmax, so the achievable
torque about DoF j is bounded by

    tau_j^max = sum_c max(0, M[j,c]) * Fmax_c
    tau_j^min = sum_c min(0, M[j,c]) * Fmax_c

If either bound is near zero the joint is uncontrollable in that direction no
matter what the planner does — the geometry is wrong, not the controller. This
is the check that catches bad cable routing (too-small moment arms, or cables
with no lever about a yaw axis) before it shows up as a saturated actuator in
a falling robot.

Compared against the torque limits the joint-torque model uses, so the two
models stay consistent.
"""
import numpy as np
import mujoco

CABLE_MODEL = "humanoid_27dof_tensegrity_cable.xml"
TORQUE_MODEL = "humanoid_27dof_tensegrity.xml"


def main():
    mc = mujoco.MjModel.from_xml_path(CABLE_MODEL)
    dc = mujoco.MjData(mc)
    mujoco.mj_resetDataKeyframe(mc, dc, 0)
    mujoco.mj_forward(mc, dc)

    moment = np.zeros((mc.nu, mc.nv))
    mujoco.mju_sparse2dense(moment, dc.actuator_moment, dc.moment_rownnz,
                            dc.moment_rowadr, dc.moment_colind)
    M = moment[:, 6:].T                      # (ndof, ncable), gear included

    mt = mujoco.MjModel.from_xml_path(TORQUE_MODEL)
    want = {mujoco.mj_id2name(mt, mujoco.mjtObj.mjOBJ_ACTUATOR, i):
            float(mt.actuator_gear[i, 0]) for i in range(mt.nu)}

    hinge = [j for j in range(mc.njnt)
             if mc.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    names = [mujoco.mj_id2name(mc, mujoco.mjtObj.mjOBJ_JOINT, j) for j in hinge]

    print(f"{'joint':18s} {'tau- (Nm)':>10s} {'tau+ (Nm)':>10s} "
          f"{'needed':>8s} {'margin':>8s}")
    worst = []
    for k, nm in enumerate(names):
        row = M[k]
        hi = float(np.maximum(row, 0).sum())     # gear already = -Fmax
        lo = float(np.minimum(row, 0).sum())
        need = want[nm]
        margin = min(hi, -lo) / need
        worst.append((margin, nm, lo, hi, need))
        print(f"{nm:18s} {lo:10.1f} {hi:10.1f} {need:8.0f} {margin:7.2f}x")

    worst.sort()
    print()
    bad = [w for w in worst if w[0] < 1.0]
    if bad:
        print("UNDER-POWERED DoFs (cable capacity < joint-torque model limit):")
        for margin, nm, lo, hi, need in bad:
            print(f"  {nm:18s} {margin:.2f}x  (has {min(hi, -lo):.0f} Nm, "
                  f"wants {need:.0f} Nm)")
    else:
        print("all 27 DoFs have cable authority in both directions "
              f"(worst margin {worst[0][0]:.2f}x on {worst[0][1]})")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
