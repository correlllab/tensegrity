#!/usr/bin/env python
"""E15: designing a coaxial joint that can actually be controlled.

The ankle and knee interfaces of the hinge-free lower body have force closure
(t3 = 1) but no wrench closure (t6 = 0). Three things follow, and they are why
no balance controller can work there:

  * no self-stress state at the joint, so no co-contraction bias;
  * tension-only actuators can then only ADD load to an equilibrium, so from
    the home pose they can push one way and never pull back;
  * no bidirectional moment authority, at exactly the joints that have to
    generate the balancing moment.

The hip does have t6 = 1, but only because the thigh ring sits eccentrically
inside a much wider pelvis ring. Radius ratio alone does not do it: swept
coaxially from 1.0x to 2.9x, t6 stays 0.

So this searches the cable TOPOLOGY instead. Each family is a rule for
connecting ring i to ring j with an index offset; the search is over subsets of
families, scored by t6 and by how much moment authority they buy per cable.
The wrench matrix is built analytically, so a full search costs seconds.
"""
import itertools
import json
import numpy as np
from scipy.optimize import linprog

RES = "/Users/ncorrell/Downloads/tensegrity/experiments/results"
OUT = f"{RES}/e15_joint_design.json"
TWIST = np.deg2rad(29.8)

# a family connects a proximal ring to a distal ring with an index offset
#   (proximal ring, distal ring, offset)   rings: pn/pf = proximal near/far,
#                                                 dn/df = distal near/far
FAMILIES = {
    "A": ("pn", "dn", 0), "B": ("pn", "df", 0), "C": ("pn", "dn", +1),
    "D": ("pn", "dn", -1), "E": ("pf", "dn", 0), "F": ("pf", "dn", +1),
    "G": ("pf", "dn", -1), "H": ("pn", "df", +1), "I": ("pn", "df", -1),
    "J": ("pf", "df", 0),
}
BASELINE = ["A", "B", "C", "E"]          # what the current design uses


def ring(radius, z, phase):
    return [np.array([radius * np.cos(phase + 2 * np.pi * i / 3),
                      radius * np.sin(phase + 2 * np.pi * i / 3), z])
            for i in range(3)]


def geometry(Rp=0.09, Rd=0.09, H=0.30, ov=0.09, dphase=np.pi / 3, ecc=0.0):
    """ecc offsets the distal cage sideways, which is what the hip does."""
    e = np.array([0.0, ecc, 0.0])
    return {
        "pf": ring(Rp, 0.0, 0.0),
        "pn": ring(Rp, H, TWIST),
        "dn": [p + e for p in ring(Rd, H - ov, dphase)],
        "df": [p + e for p in ring(Rd, 2 * H - ov, dphase + TWIST)],
    }


def wrench_matrix(fams, geo):
    """Unit-tension wrench each cable applies to the distal cage, about the
    midpoint of the two cages."""
    o = 0.5 * (np.mean(geo["pf"] + geo["pn"], axis=0)
               + np.mean(geo["dn"] + geo["df"], axis=0))
    W = []
    for f in fams:
        pr, dr, off = FAMILIES[f]
        for i in range(3):
            q = geo[pr][i]
            s = geo[dr][(i + off) % 3]
            v = q - s
            n = np.linalg.norm(v)
            if n < 1e-9:
                continue
            u = v / n
            W.append(np.concatenate([u, np.cross(s - o, u)]))
    return np.array(W).T if W else np.zeros((6, 0))


def closure(A):
    """max t s.t. A lam = 0, lam >= t, t <= 1. Positive iff the positive span
    of the columns is all of R^rows."""
    if A.shape[1] == 0 or np.linalg.matrix_rank(A) < A.shape[0]:
        return 0.0
    n = A.shape[1]
    c = np.zeros(n + 1)
    c[-1] = -1.0
    r = linprog(c, A_ub=np.hstack([-np.eye(n), np.ones((n, 1))]),
                b_ub=np.zeros(n),
                A_eq=np.hstack([A, np.zeros((A.shape[0], 1))]),
                b_eq=np.zeros(A.shape[0]), bounds=[(0, None)] * n + [(0, 1)])
    return float(r.x[-1]) if r.success else 0.0


def moment_authority(W):
    """Smallest moment the joint can generate about the worst axis, per unit
    of the largest cable tension -- the quantity a balance controller spends."""
    if W.shape[1] == 0:
        return 0.0
    worst = np.inf
    for ax in np.eye(3):
        for sgn in (+1, -1):
            # best achievable moment along +/- ax with tensions in [0,1]
            val = float(np.maximum(sgn * (ax @ W[3:]), 0).sum())
            worst = min(worst, val)
    return worst


def score(fams, geo):
    W = wrench_matrix(fams, geo)
    return dict(fams="".join(fams), n=W.shape[1], t3=closure(W[:3]),
                t6=closure(W), moment=moment_authority(W))


def main():
    geo = geometry()
    base = score(BASELINE, geo)
    print("current ankle/knee interface (coaxial, 4 families):")
    print(f"  {base['fams']}  {base['n']} cables  t3={base['t3']:.3f}  "
          f"t6={base['t6']:.3f}  worst-axis moment {base['moment']:.3f} N m/N")

    print("\nsearching cable topologies for wrench closure (coaxial, no "
          "eccentricity)")
    keys = list(FAMILIES)
    found = []
    for k in range(3, 7):
        for combo in itertools.combinations(keys, k):
            s = score(list(combo), geo)
            if s["t6"] > 1e-6:
                found.append(s)
        if found:
            break
    if not found:
        print("  none up to 6 families")
    else:
        found.sort(key=lambda s: (s["n"], -s["moment"]))
        print(f"  {len(found)} topologies close the wrench set; best by "
              f"cable count then moment authority:")
        print(f"  {'families':>10s} {'cables':>7s} {'t3':>6s} {'t6':>6s} "
              f"{'moment':>8s}")
        for s in found[:8]:
            print(f"  {s['fams']:>10s} {s['n']:7d} {s['t3']:6.3f} "
                  f"{s['t6']:6.3f} {s['moment']:8.3f}")

    best = found[0] if found else None
    if best:
        print(f"\nchosen: {best['fams']}  "
              f"({'+'.join(f'{f}:{FAMILIES[f][0]}->{FAMILIES[f][1]}'
                           f'{FAMILIES[f][2]:+d}' for f in best['fams'])})")
        print("  robustness over the joint's working range:")
        print(f"  {'overlap/H':>10s} {'dphase':>7s} {'t3':>6s} {'t6':>6s}")
        for ovf in (0.2, 0.3, 0.4):
            for dp in (0.0, np.pi / 3, 2 * np.pi / 3):
                g = geometry(ov=ovf * 0.30, dphase=dp)
                s = score(list(best["fams"]), g)
                print(f"  {ovf:10.2f} {np.rad2deg(dp):7.0f} {s['t3']:6.3f} "
                      f"{s['t6']:6.3f}")
    json.dump(dict(baseline=base, found=found[:20],
                   chosen=best), open(OUT, "w"), indent=1)
    print("\nE15 ->", OUT)


if __name__ == "__main__":
    main()
