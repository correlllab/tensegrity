#!/usr/bin/env python
"""E26: the counter-wound (ABCD) joint as built, measured.

E15 shows analytically that adding the counter-wound mirror family D to the
single-handed set closes the wrench set. Three claims about that joint were
previously asserted without a backing computation; this measures them:

  closure         t3/t6 of the built ABCD interface (12 cables), from the
                  compiled model geometry -- the same LP as E9/E10
  co-contraction  the largest tension floor t such that a self-stress state
                  exists with every interface cable between t and the 1.5 kN
                  rope rating: max t s.t. W lam = 0, t <= lam_c <= 1500.
                  This is the joint's co-contraction authority in newtons;
                  it is exactly zero for the single-handed set (no
                  self-stress state at all).
  chain stand     a passive vertical chain of three ABCD joints (rigid
                  cages, as in the hybrid), base cage fixed, 1 kg tip
                  payload: does it stand, and how much does it sag?

Geometry, stiffness (40 kN/m series-elastic) and prestress (100 N) follow
the E10 testbed; families follow E15 / the hybrid generator
(A: pn->dn+0, B: pn->df+0, C: pn->dn+1, D: pn->dn-1).
"""
import json
import numpy as np
import mujoco
from scipy.optimize import linprog

RES = "/Users/ncorrell/Downloads/tensegrity/experiments/results"
OUT = f"{RES}/e26_counterwound_chain.json"
R, H, TWIST = 0.09, 0.30, np.deg2rad(29.8)
OVERLAP = 0.06                       # 0.2 H, the E10 operating point
K, T0, FMAX = 40e3, 100.0, 1500.0
TIP_MASS = 1.0


def ring(radius, z, phase):
    return [np.array([radius * np.cos(phase + 2 * np.pi * i / 3),
                      radius * np.sin(phase + 2 * np.pi * i / 3), z])
            for i in range(3)]


def cage_rings(k):
    """Bottom and top rings of stacked cage k (k = 0 fixed base)."""
    z0 = k * (H - OVERLAP)
    ph = k * (TWIST + np.pi / 3)
    return ring(R, z0, ph), ring(R, z0 + H, ph + TWIST)


def interface_cables(k):
    """ABCD families joining cage k (proximal) to cage k+1 (distal)."""
    _, pn = cage_rings(k)                       # proximal near = its top ring
    dn, df = cage_rings(k + 1)                  # distal near = its bottom
    cabs = []
    for fam, tgt, off in (("A", dn, 0), ("B", df, 0),
                          ("C", dn, +1), ("D", dn, -1)):
        for i in range(3):
            cabs.append((f"if{k}_{fam}{i}", pn[i], tgt[(i + off) % 3]))
    return cabs


def chain_xml(n_joints, pre_scale=1.0):
    """n_joints ABCD interfaces stacking n_joints+1 rigid cages; cage 0 is
    the world. Rest lengths are set from the built geometry so prestress is
    a tension, exactly as in E10."""
    body_xml, sites_w, cabs = "", "", []
    for k in range(n_joints + 1):
        bot, top = cage_rings(k)
        mstr = 'mass="0.15" ' if k else ''
        struts = "".join(
            f'\n      <geom type="capsule" size="0.009" '
            f'{mstr}contype="0" conaffinity="0" '
            f'rgba="0.25 0.28 0.33 1" '
            f'fromto="{bot[i][0]:.5f} {bot[i][1]:.5f} {bot[i][2]:.5f} '
            f'{top[(i+1)%3][0]:.5f} {top[(i+1)%3][1]:.5f} '
            f'{top[(i+1)%3][2]:.5f}"/>' for i in range(3))
        sites = "".join(
            f'\n      <site name="c{k}_{tag}{i}" pos="{p[0]:.5f} '
            f'{p[1]:.5f} {p[2]:.5f}" size="0.004"/>'
            for tag, ringp in (("b", bot), ("t", top))
            for i, p in enumerate(ringp))
        if k == 0:
            sites_w += struts + sites
        else:
            tip = (f'\n      <geom type="sphere" size="0.02" '
                   f'mass="{TIP_MASS}" contype="0" conaffinity="0" '
                   f'pos="0 0 {k*(H-OVERLAP)+H:.5f}"/>'
                   if k == n_joints else "")
            body_xml += (f'\n    <body name="cage{k}">\n      <freejoint/>'
                         f'{struts}{sites}{tip}\n    </body>')
    for k in range(n_joints):
        _, pn = cage_rings(k)
        dn, df = cage_rings(k + 1)
        for fam, tgt, off, tag in (("A", dn, 0, "b"), ("B", df, 0, "t"),
                                   ("C", dn, +1, "b"), ("D", dn, -1, "b")):
            for i in range(3):
                cabs.append((f"if{k}_{fam}{i}", f"c{k}_t{i}",
                             f"c{k+1}_{tag}{(i+off)%3}"))

    def make(rests):
        ten = "\n    ".join(
            f'<spatial name="{n}" width="0.003" stiffness="{K}" '
            f'damping="{0.02*np.sqrt(K):.2f}" '
            f'springlength="0 {rests.get(n, 0.001):.6f}">'
            f'\n      <site site="{a}"/>\n      <site site="{b}"/>'
            f'\n    </spatial>' for n, a, b in cabs)
        return f'''<mujoco model="counterwound_chain">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="5e-5" integrator="implicitfast" gravity="0 0 -9.81"/>
  <worldbody>{sites_w}{body_xml}
  </worldbody>
  <tendon>
    {ten}
  </tendon>
</mujoco>'''
    m0 = mujoco.MjModel.from_xml_string(make({}))
    d0 = mujoco.MjData(m0)
    mujoco.mj_forward(m0, d0)
    rests = {n: max(1e-4, float(d0.ten_length[i]) - pre_scale * T0 / K)
             for i, (n, a, b) in enumerate(cabs)}
    return make(rests)


def wrench_matrix(k=0):
    """Unit-tension wrenches of interface k's cables on the distal cage,
    about the midpoint between the cages (the E15 convention)."""
    cabs = interface_cables(k)
    b0, t0_ = cage_rings(k)
    b1, t1 = cage_rings(k + 1)
    o = 0.5 * (np.mean(b0 + t0_, axis=0) + np.mean(b1 + t1, axis=0))
    W = []
    for _, q, s in cabs:
        v = q - s
        u = v / np.linalg.norm(v)
        W.append(np.concatenate([u, np.cross(s - o, u)]))
    return np.array(W).T


def closure(A):
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


def cocontraction_authority(W, fmax=FMAX):
    """max t s.t. W lam = 0, t <= lam <= fmax (LP in [lam; t])."""
    n = W.shape[1]
    c = np.zeros(n + 1)
    c[-1] = -1.0
    A_ub = np.hstack([-np.eye(n), np.ones((n, 1))])      # t - lam <= 0
    r = linprog(c, A_ub=A_ub, b_ub=np.zeros(n),
                A_eq=np.hstack([W, np.zeros((6, 1))]), b_eq=np.zeros(6),
                bounds=[(0, fmax)] * n + [(0, fmax)])
    return float(r.x[-1]) if r.success else 0.0


def main():
    res = {}
    W = wrench_matrix()
    resA = {"t3": closure(W[:3]), "t6": closure(W),
            "cocontraction_N": cocontraction_authority(W)}
    print(f"ABCD interface: t3={resA['t3']:.3f} t6={resA['t6']:.3f} "
          f"co-contraction authority {resA['cocontraction_N']:.0f} N "
          f"(cap {FMAX:.0f})")
    # the single-handed set, for contrast
    Ws = wrench_matrix()[:, :9]                 # A, B, C only
    print(f"ABC (single-handed): t6={closure(Ws):.3f} "
          f"co-contraction {cocontraction_authority(Ws):.0f} N")
    res["interface"] = resA
    res["single_handed"] = {"t6": closure(Ws),
                            "cocontraction_N": cocontraction_authority(Ws)}

    print("\npassive chain stand (base cage fixed, 1 kg tip):")
    res["chain"] = []
    for nj in (1, 2, 3):
        xml = chain_xml(nj)
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        top = m.nbody - 1
        z0 = float(d.xpos[top][2])
        m.tendon_damping[:] = m.tendon_damping + 40.0   # static probe damping
        for _ in range(int(3.0 / m.opt.timestep)):
            mujoco.mj_step(m, d)
            if not np.all(np.isfinite(d.qpos)):
                break
        okf = bool(np.all(np.isfinite(d.qpos)))
        sag = 1e3 * (z0 - float(d.xpos[top][2])) if okf else float("nan")
        stands = okf and abs(sag) < 50 * nj
        res["chain"].append(dict(n_joints=nj, sag_mm=sag, stands=stands))
        print(f"  {nj} joint(s): sag {sag:7.1f} mm  "
              f"{'stands' if stands else 'COLLAPSES'}", flush=True)

    json.dump(res, open(OUT, "w"), indent=1)
    print("\nE26 ->", OUT)


if __name__ == "__main__":
    main()
