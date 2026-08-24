#!/usr/bin/env python
"""E16: size the drives from the joint moments the robot actually needs.

The motor selection so far was inherited from the HINGED bill of materials,
where a cable pulled on a 37.5 mm effective moment arm and needed kilonewtons
to make 80 N m. A hinge-free joint acts at the cage radius -- 90 mm and up --
so the same moment costs far less tension, and the drives should shrink
accordingly.

Required moment per joint, for a robot of mass M standing at hip height h:

  ankle   the moment to hold the centre of mass off the ankle, plus what a
          push of F at the pelvis demands: M g d_cop + F h, shared over two
          feet
  knee    gravity moment of everything above it at the worst stance offset
  hip     same, plus the push

Tension follows from the joint's moment authority a (N m per N of tension,
measured by E15), and motor torque from the spool radius: tau = (Mreq/a) r.
"""
import json
import sys
import numpy as np

sys.path.insert(0, "/Users/ncorrell/Downloads/tensegrity/experiments")
from e15_joint_design import (FAMILIES, geometry, wrench_matrix,  # noqa: E402
                              moment_authority)

RES = "/Users/ncorrell/Downloads/tensegrity/experiments/results"
OUT = f"{RES}/e16_drive_sizing.json"
G = 9.81
SPOOL_R = 0.008            # m, from the transmission study
SF = 1.5                   # design safety factor on moment
CONT_FRAC = 0.35           # continuous torque as a fraction of the peak
                           # rating. A standing robot holds its joint moment
                           # indefinitely, so the drive must be picked on
                           # continuous duty, not on stall.

# name, peak torque (N m), mass (kg), mesh
CATALOGUE = [
    ("XM430", 4.1, 0.082, "ak60"),
    ("XM540", 10.6, 0.165, "ak60"),
    ("AK60-6", 9.0, 0.310, "ak60"),
    ("AK70-10", 24.8, 0.520, "ak70"),
    ("AK10-9", 48.0, 0.960, "ak70"),
]


def pick(tau_req):
    for name, tau, mass, mesh in CATALOGUE:
        if tau * CONT_FRAC >= tau_req:
            return name, tau, mass, mesh
    return CATALOGUE[-1]


def main():
    M = 27.2                    # robot mass, kg
    h_hip = 0.69                # pelvis height, m
    d_cop = 0.09                # CoP excursion available on the foot, m
    F_push = 60.0               # design push at the pelvis, N

    coax = moment_authority(wrench_matrix(list("ABCD"), geometry()))
    # hip: eccentric geometry, measured in E15
    hip_geo = geometry(Rp=0.20, Rd=0.09, ecc=0.10)
    ecc = moment_authority(wrench_matrix(list("ABCD"), hip_geo))

    jobs = [
        ("ankle", (M * G * d_cop + F_push * h_hip) / 2.0, coax, 3),
        ("knee", (M * G * d_cop * 0.6 + F_push * h_hip * 0.5) / 2.0, coax, 3),
        ("hip", (M * G * d_cop + F_push * h_hip) / 2.0, ecc, 3),
    ]
    print(f"moment authority: coaxial {coax:.3f} N m/N, "
          f"eccentric (hip) {ecc:.3f} N m/N")
    print(f"spool radius {1e3*SPOOL_R:.0f} mm, safety factor {SF}\n")
    print(f"{'joint':7s} {'M req':>8s} {'tension':>9s} {'tau cont':>9s} "
          f"{'pick':>9s} {'peak':>7s} {'mass ea':>8s} {'n':>3s} {'total':>7s}")
    rows, total = [], 0.0
    for name, Mreq, a, n_per_joint in jobs:
        Mreq *= SF
        T = Mreq / max(a, 1e-9)
        tau = T * SPOOL_R
        pick_name, pick_tau, pick_mass, mesh = pick(tau)
        n = n_per_joint * 2 if name != "hip" else n_per_joint * 2
        tot = n * pick_mass
        total += tot
        print(f"{name:7s} {Mreq:7.1f}N·m {T:8.0f}N {tau:8.2f}N·m "
              f"{pick_name:>9s} {pick_tau:6.1f}N·m {pick_mass:7.2f}kg "
              f"{n:3d} {tot:6.2f}kg")
        rows.append(dict(joint=name, moment=Mreq, tension=T, tau=tau,
                         motor=pick_name, mass=pick_mass, mesh=mesh, n=n,
                         total=tot))
    print(f"{'total':7s} {'':8s} {'':9s} {'':9s} {'':9s} {'':7s} {'':8s} "
          f"{sum(r['n'] for r in rows):3d} {total:6.2f}kg")
    print(f"\nfor comparison, the hinged BOM used AK70/AK10 class throughout: "
          f"{6*0.96 + 6*0.52 + 6*0.52:.1f} kg for the same 18 drives")
    print("The hinge-free joint acts at the cage radius rather than a 37.5 mm")
    print("effective arm, so the tension -- and the motor -- shrink with it.")
    json.dump(dict(rows=rows, total=total, coax=coax, ecc=ecc,
                   spool_r=SPOOL_R, sf=SF), open(OUT, "w"), indent=1)
    print("\nE16 ->", OUT)


if __name__ == "__main__":
    main()
