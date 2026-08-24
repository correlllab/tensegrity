#!/usr/bin/env python
"""How many MOTORS does the cable network need? (not how many cables)

A cable is Dyneema plus two anchors — grams — and having many is what makes
the structure a tensegrity. A motor is 0.3-0.9 kg of QDD actuator. The 102
figure is the size of the tension network; it is not a bill of materials.

Three things collapse the motor count:

1. A LOOP DRIVE. One motor with a double-wound spool pays out one cable while
   hauling in its antagonist, so a single motor gives bidirectional torque
   about a DoF. Standard tendon-driven practice. => 1 motor per DoF, not 2.
2. PASSIVE PRESTRESS. Cables that are not driven still do structural work as
   elastic prestress elements (spring + turnbuckle, tuned at assembly).
3. CO-CONTRACTION only where wanted. A second motor per DoF buys variable
   stiffness; it is only worth its mass at the joints where that is the
   research point (spine + shoulders, per codesign_plan.md section 6).

The binding constraint is then per-DoF: one driven loop must deliver the whole
joint torque, so it needs

    Fmax * r_eff >= tau_need

This reports the actual r_eff available at each DoF from the best cable in the
network, and flags DoFs where the geometry is too tight for a single loop —
those are the joints that need a bigger cage or a lever arm (see
design/motor_dimensioning.md, which independently recommends levers at the
ankle).
"""
import numpy as np
import mujoco

CABLE_MODEL = "humanoid_27dof_tensegrity_cable.xml"
TORQUE_MODEL = "humanoid_27dof_tensegrity.xml"

# per-DoF motor choice and mass (kg), from design/motor_dimensioning.md
MOTOR = (("hip_yaw", "AK70-10", 0.52), ("hip_roll", "AK70-10", 0.52),
         ("hip_pitch", "AK70-10", 0.52), ("knee", "AK70-10", 0.52),
         ("ankle", "AK10-9", 0.96), ("waist", "AK70-10", 0.52),
         ("shoulder", "AK60-6", 0.31), ("elbow", "AK60-6", 0.31),
         ("wrist", "XM540", 0.17), ("neck", "XM540", 0.17))
# joints where a second (antagonist) motor buys variable stiffness
COCONTRACT = ("waist", "shoulder")


def spec(name):
    for pre, mot, kg in MOTOR:
        if name.startswith(pre):
            return mot, kg
    raise ValueError(name)


def main():
    mc = mujoco.MjModel.from_xml_path(CABLE_MODEL)
    dc = mujoco.MjData(mc)
    mujoco.mj_resetDataKeyframe(mc, dc, 0)
    mujoco.mj_forward(mc, dc)
    moment = np.zeros((mc.nu, mc.nv))
    mujoco.mju_sparse2dense(moment, dc.actuator_moment, dc.moment_rownnz,
                            dc.moment_rowadr, dc.moment_colind)
    M = moment[:, 6:].T                     # Nm per unit ctrl (gear folded in)
    fmax = -mc.actuator_gear[:, 0]          # N

    mt = mujoco.MjModel.from_xml_path(TORQUE_MODEL)
    jn = [mujoco.mj_id2name(mt, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(mt.nu)]
    need = np.array([abs(float(mt.actuator_gear[i, 0])) for i in range(mt.nu)])

    print(f"tension network: {mc.nu} cables   (cables are cheap; motors are not)\n")
    print(f"{'DoF':18s} {'need Nm':>8s} {'best r_eff mm':>14s} "
          f"{'1 loop gives':>13s} {'verdict':>16s}")
    tight = []
    for k, nm in enumerate(jn):
        # best single cable: |moment| per newton of tension = effective arm
        arms = np.abs(M[k]) / fmax
        c = int(np.argmax(arms))
        r_eff = float(arms[c])
        cap = r_eff * float(fmax[c])
        ok = cap >= need[k]
        if not ok:
            tight.append((nm, need[k], cap, need[k] / max(cap, 1e-9)))
        print(f"{nm:18s} {need[k]:8.0f} {r_eff * 1000:14.1f} {cap:12.0f}  "
              f"{'ok' if ok else 'needs ' + f'{need[k] / max(cap, 1e-9):.1f} loops':>16s}")

    n_dof = len(jn)
    n_extra = sum(1 for nm in jn if nm.startswith(COCONTRACT))
    kg = sum(spec(nm)[1] for nm in jn) + sum(spec(nm)[1] for nm in jn
                                             if nm.startswith(COCONTRACT))
    print(f"\nMOTOR COUNT")
    print(f"  1 loop-drive motor per DoF                 {n_dof}")
    print(f"  + antagonist motor at {'/'.join(COCONTRACT)} (variable stiffness)  "
          f"+{n_extra}")
    print(f"  = {n_dof + n_extra} motors, {kg:.1f} kg "
          f"(budget in codesign_plan.md section 3: 16.2 kg)")
    print(f"  passive prestress cables (elastic + turnbuckle): "
          f"{mc.nu - (n_dof + n_extra)} of {mc.nu}")

    if tight:
        print(f"\nGEOMETRY TOO TIGHT for a single loop at {len(tight)} DoFs — these need a"
              f"\nbigger cage or a lever arm, otherwise they cost extra motors:")
        for nm, nd, cap, ratio in sorted(tight, key=lambda x: -x[3]):
            print(f"  {nm:18s} wants {nd:.0f} Nm, one loop gives {cap:.0f} Nm "
                  f"({ratio:.1f}x short)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
