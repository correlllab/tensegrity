#!/usr/bin/env python
"""Physical specification of the compression and tension members.

(Design analysis, not one of the E-numbered experiments: it produces the
bill of materials for Sec. VI-D rather than testing a claim.)

The mass budget carries a single 5.9 kg line for "struts, cables, nodes,
anchors, textile shell", which is an allowance rather than a bill of
materials. This script turns the simulated loads into an actual specification
-- tube outside diameter and wall, rope diameter and construction -- and adds
the masses back up, so the allowance can be checked rather than asserted.

Struts. Sized against Euler buckling under the governing axial load, which is
NOT the cable tension (E7: peak 51 N) but the ground reaction entering a cage
through its hinge, shared over the three struts of that cage. Walls are then
rounded UP to roll-wrapped CFRP stock.

Tendons. Sized from the per-cable tension the gait actually demands, taken
from the UNBOUNDED tension distribution in tension_audit.py -- i.e. what the
hardware must carry, not what the 1.5 kN box constraint permitted. Breaking
load and linear density of 12-strand UHMWPE (Dyneema SK75/78 class) follow the
manufacturer fits BL = 0.8 d^2 kN and mu = 0.55 d^2 g/m with d in mm, which
reproduce published 1.5-5 mm rope data to within about 10%.

The script also compares the rope's axial stiffness EA/L against the spring
constant the model actually uses, which is the largest single modelling gap in
the paper.
"""
import json
import re
import numpy as np
import mujoco

MJ = "/Users/ncorrell/Downloads/tensegrity/mujoco"
RES = "/Users/ncorrell/Downloads/tensegrity/experiments/results"
OUT = f"{RES}/member_specs.json"
TORQUE = f"{MJ}/humanoid_27dof_tensegrity.xml"
CABLE = f"{MJ}/humanoid_27dof_tensegrity_cable.xml"

# --- CFRP tube (roll-wrapped, quasi-isotropic-ish layup)
E_CFRP, RHO_CFRP = 70e9, 1600.0
SF_BUCKLE, K_EFF = 2.0, 1.0
WALL_STOCK = (0.5e-3, 0.75e-3, 1.0e-3, 1.5e-3, 2.0e-3)
END_FITTING_G = 3.0            # bonded alloy insert per strut end

# --- 12-strand UHMWPE rope
BL_PER_MM2 = 0.8e3             # N of breaking load per mm^2 of d^2
MU_PER_MM2 = 0.55e-3           # kg/m per mm^2 of d^2
E_ROPE = 50e9                  # N/m^2 on nominal area, bedded-in
SF_ROPE = 3.0                  # on breaking load
SPLICE_EFF = 0.85              # buried splice / thimble termination
D_STOCK = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
ROUTE_ALLOW = 1.4              # spool wraps, terminations, run to the motor
TERMINATION_G = 4.0            # thimble + swage or splice, per cable end

NODE_G = 9.0                   # machined node cluster where 3 struts meet

# --- joint bearings. The declared load path is strut -> node -> HINGE -> node
# -> strut, so each rotational DoF needs a bearing pair, clevis and axle sized
# for the joint reaction (the ground-reaction path, not the cable tension).
# A 15 mm-bore angular-contact pair plus an alloy clevis and axle is about
# 70 g per DoF at this load; the estimate is quoted per DoF because a 3-DoF
# hip gimbal is three such assemblies in series.
BEARING_G_PER_DOF = 70.0


def group_of(name):
    for g in ("hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist",
              "neck"):
        if name.startswith(g):
            return g
    return "cell"                                      # pw_* / rt_* prism cables


def strut_spec():
    m = mujoco.MjModel.from_xml_path(TORQUE)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    e7 = json.load(open(f"{RES}/e7_strut_sizing.json"))
    # governing axial load per strut: hinge path (peak GRF / 3) dominates the
    # cable-induced load by an order of magnitude
    hinge = max(c["per_strut"] for c in e7["load_cases"]
                if c["case"] != "drop 1.0 m")
    rows = []
    for s in e7["struts"]:
        L, ro = s["L"], s["r_o"]
        P = SF_BUCKLE * max(hinge, s["P_peak"])
        I_req = P * (K_EFF * L) ** 2 / (np.pi ** 2 * E_CFRP)
        r4 = ro ** 4 - 4 * I_req / np.pi
        ri = r4 ** 0.25 if r4 > 0 else 0.0
        t_req = ro - ri
        t = next((w for w in WALL_STOCK if w >= t_req), WALL_STOCK[-1])
        ri = max(ro - t, 0.0)
        P_cr = np.pi ** 3 * E_CFRP * (ro ** 4 - ri ** 4) / 4 / (K_EFF * L) ** 2
        rows.append(dict(body=s["body"], L=L, od=2 * ro, wall=t,
                         mass=RHO_CFRP * np.pi * (ro ** 2 - ri ** 2) * L
                         + 2 * END_FITTING_G * 1e-3,
                         P_gov=max(hinge, s["P_peak"]), P_cr=float(P_cr),
                         margin=float(P_cr / max(hinge, s["P_peak"]))))
    return rows, hinge


def tendon_spec():
    mc = mujoco.MjModel.from_xml_path(CABLE)
    dc = mujoco.MjData(mc)
    mujoco.mj_resetDataKeyframe(mc, dc, 0)
    mujoco.mj_forward(mc, dc)
    ta = json.load(open(f"{RES}/tension_audit.json"))
    byp = ta["by_payload"]
    cn = ta["cables"]
    # worst case over the payload range, p99 over states
    peak = np.max([byp[k]["demand_peak"] for k in byp], axis=0)
    rms = np.max([byp[k]["demand_rms"] for k in byp], axis=0)

    # second, pose-independent basis: the tension a single loop must pull to
    # reach the joint's torque limit through the moment arm it actually has
    a = json.load(open(f"{RES}/e5_authority.json"))
    spec_T = {}
    for j, dof in enumerate(a["dofs"]):
        g = group_of(a["best_cable"][j])
        spec_T[g] = max(spec_T.get(g, 0.0), a["need"][j] / a["r_have"][j])

    # actuated (joint-crossing) cables carry the demand; the prism-cell cables
    # carry prestress only, sized from their spring force at 3% prestress
    cable_len = {}
    ten_names = [mujoco.mj_id2name(mc, mujoco.mjtObj.mjOBJ_TENDON, i)
                 for i in range(mc.ntendon)]
    for n, L, k, rest in zip(ten_names, dc.ten_length, mc.tendon_stiffness,
                             mc.tendon_lengthspring[:, 1]):
        cable_len[n] = (float(L), float(k), float(k * max(0.0, L - rest)))

    rows = {}
    for i, n in enumerate(cn):
        g = group_of(n)
        L, k_model, _ = cable_len[n]
        r = rows.setdefault(g, dict(n=0, L=0.0, peak=0.0, rms=0.0, k_model=k_model))
        r["n"] += 1
        r["L"] += L
        r["peak"] = max(r["peak"], float(peak[i]))
        r["gait_peak"] = max(r.get("gait_peak", 0.0), float(peak[i]))
        r["rms"] = max(r["rms"], float(rms[i]))
    for n in ten_names:                                 # passive cell cables
        if n in cn:
            continue
        g = "cell"
        L, k_model, T0 = cable_len[n]
        r = rows.setdefault(g, dict(n=0, L=0.0, peak=0.0, rms=0.0, k_model=k_model))
        r["n"] += 1
        r["L"] += L
        r["peak"] = max(r["peak"], T0)
        r["rms"] = max(r["rms"], T0)

    for g, r in rows.items():
        r["torque_T"] = spec_T.get(g, 0.0)
        r["peak"] = max(r["peak"], r["torque_T"])
        bl_req = SF_ROPE * r["peak"] / SPLICE_EFF
        d_req = np.sqrt(bl_req / BL_PER_MM2)
        d = next((x for x in D_STOCK if x >= d_req), D_STOCK[-1])
        mu = MU_PER_MM2 * d ** 2
        length = r["L"] * ROUTE_ALLOW
        r.update(bl_req=float(bl_req), d_req=float(d_req), d=d, mu=mu,
                 bl=BL_PER_MM2 * d ** 2, length=length,
                 mass=mu * length + 2 * r["n"] * TERMINATION_G * 1e-3,
                 k_rope=E_ROPE * np.pi * (1e-3 * d) ** 2 / 4
                 / (r["L"] / max(r["n"], 1)),
                 util_rms=float(r["rms"] / (BL_PER_MM2 * d ** 2)))
        # prestress: the model sets rest length to (1-p)L with p = 3-4%. With
        # the specified rope that strain is far past break, so the hardware
        # needs either micron-resolution take-up or a deliberate series spring.
        L_avg = r["L"] / max(r["n"], 1)
        r["k_series_needed"] = r["peak"] * 0.25 / (0.04 * L_avg)
        r["prestress_strain_for_T"] = (0.25 * r["peak"]) / (r["k_rope"] * L_avg)
        r["takeup_um"] = 1e6 * (0.25 * r["peak"]) / r["k_rope"]
    return rows


def count_nodes():
    """Distinct strut-tip clusters: a node is where several strut ends and the
    cables that terminate on them meet, and it is one machined part."""
    m = mujoco.MjModel.from_xml_path(TORQUE)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    pts = []
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_CAPSULE:
            continue
        if not np.allclose(m.geom_rgba[g][:3], (0.22, 0.24, 0.28), atol=1e-3):
            continue
        half = float(m.geom_size[g, 1])
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.geom_quat[g])
        ax = R.reshape(3, 3)[:, 2]
        b = int(m.geom_bodyid[g])
        Rb = d.xmat[b].reshape(3, 3)
        for e in (-1, 1):
            pts.append(d.xpos[b] + Rb @ (m.geom_pos[g] + e * half * ax))
    pts = np.array(pts)
    used = np.zeros(len(pts), bool)
    n = 0
    for i in range(len(pts)):
        if used[i]:
            continue
        used[np.linalg.norm(pts - pts[i], axis=1) < 5e-3] = True
        n += 1
    return n


def main():
    struts, hinge = strut_spec()
    tendons = tendon_spec()

    print(f"STRUTS  (governing axial load {hinge:.0f} N/strut, "
          f"CFRP E={E_CFRP/1e9:.0f} GPa, SF={SF_BUCKLE})")
    fam = {}
    for r in struts:
        key = (re.sub(r"_[lr]$", "", r["body"]), round(1e3 * r["od"]),
               round(1e3 * r["wall"], 2))
        f = fam.setdefault(key, dict(n=0, Ls=[], mass=0.0, margin=1e9))
        f["n"] += 1
        f["Ls"].append(r["L"])
        f["mass"] += r["mass"]
        f["margin"] = min(f["margin"], r["margin"])
    print(f"{'family':12s} {'n':>3s} {'L (mm)':>12s} {'OD x wall':>12s} "
          f"{'margin':>7s} {'kg':>6s}")
    m_strut = 0.0
    for (body, od, wall), f in sorted(fam.items()):
        print(f"{body:12s} {f['n']:3d} "
              f"{1e3*min(f['Ls']):5.0f}-{1e3*max(f['Ls']):<6.0f} "
              f"{od:5.0f} x {wall:<4.2f} {f['margin']:6.1f}x {f['mass']:6.3f}")
        m_strut += f["mass"]
    print(f"{'':12s} {len(struts):3d} {'':12s} {'':12s} {'total':>7s} "
          f"{m_strut:6.3f}")

    print(f"\nTENDONS  (12-strand UHMWPE, SF={SF_ROPE} on break, "
          f"splice efficiency {SPLICE_EFF})")
    print(f"{'group':8s} {'n':>3s} {'L tot':>7s} {'gait N':>7s} {'tau/r N':>8s} "
          f"{'d':>5s} {'BL kN':>6s} {'RMS/BL':>7s} {'k rope':>9s} "
          f"{'k model':>8s} {'take-up':>8s} {'kg':>6s}")
    m_ten = 0.0
    for g in sorted(tendons, key=lambda x: -tendons[x]["peak"]):
        r = tendons[g]
        print(f"{g:8s} {r['n']:3d} {r['length']:6.1f}m "
              f"{r.get('gait_peak', 0):7.0f} {r['torque_T']:8.0f} "
              f"{r['d']:5.1f} {r['bl']/1e3:6.1f} "
              f"{100*r['util_rms']:6.1f}% {r['k_rope']/1e3:8.0f}k "
              f"{r['k_model']:7.0f} {r['takeup_um']:6.0f}um {r['mass']:6.3f}")
        m_ten += r["mass"]
    print(f"{'total':8s} {sum(r['n'] for r in tendons.values()):3d} "
          f"{'':7s} {'':8s} {'':5s} {'':6s} {'':7s} {'':9s} {'':8s} {'':8s} "
          f"{m_ten:6.3f}")

    n_nodes = count_nodes()
    mt_j = mujoco.MjModel.from_xml_path(TORQUE)
    n_hinge_dof = int(sum(1 for t in mt_j.jnt_type
                          if t == mujoco.mjtJoint.mjJNT_HINGE))
    m_hinge = n_hinge_dof * BEARING_G_PER_DOF * 1e-3
    m_node = n_nodes * NODE_G * 1e-3
    stiff_ratio = np.median([r["k_rope"] / r["k_model"] for r in tendons.values()])

    total = m_strut + m_ten + m_node + m_hinge

    # --- articulation-hardware safety factors against the measured stumble
    # reaction (the hybrid's knee: alloy-steel clevis pin D12x1.5 in a
    # 61802 deep-groove bearing pair).  The pin is checked in double shear
    # (4130 normalised, tau_allow ~ 250 MPa on the tube section); the
    # bearing pair against its catalogue static rating C0 (SKF 61802:
    # 1.66 kN each), which is the governing element.
    grf_stumble = 3.0 * hinge                      # per-strut -> total GRF
    pin_area = np.pi / 4 * (0.012**2 - 0.009**2)   # D12 x 1.5 wall
    pin_capacity = 2.0 * pin_area * 250e6          # double shear
    C0_61802 = 1.66e3
    sf_pin = pin_capacity / grf_stumble
    sf_bearing = 2.0 * C0_61802 / grf_stumble
    print(f"\narticulation hardware vs {grf_stumble:.0f} N stumble reaction:"
          f" pin SF {sf_pin:.0f}, 61802 pair SF {sf_bearing:.1f} (governs)")

    print(f"\nnodes: {n_nodes} clusters x {NODE_G:.0f} g = {m_node:.2f} kg")
    print(f"joint bearings: {n_hinge_dof} hinge DoF x "
          f"{BEARING_G_PER_DOF:.0f} g = {m_hinge:.2f} kg")
    print(f"MEMBERS TOTAL: struts {m_strut:.2f} + tendons {m_ten:.2f} + "
          f"nodes {m_node:.2f} + bearings {m_hinge:.2f} = {total:.2f} kg")
    print(f"budget line (struts, cables, nodes, shell): 5.90 kg "
          f"-> {5.90 - total:.2f} kg left for shell, channels and fasteners")
    print(f"\nmodelled cable stiffness is {stiff_ratio:.0f}x SOFTER than the "
          f"specified rope (median over groups)")

    json.dump(dict(struts=struts, strut_families={f"{k}": v for k, v in
                                                  fam.items()},
                   tendons=tendons, hinge_load=hinge,
                   m_strut=m_strut, m_tendon=m_ten, m_node=m_node,
                   n_nodes=n_nodes, total=total,
                   n_hinge_dof=n_hinge_dof, m_hinge=m_hinge,
                   bearing_g_per_dof=BEARING_G_PER_DOF,
                   grf_stumble=float(grf_stumble),
                   knee_pin_SF=float(sf_pin),
                   knee_bearing_SF=float(sf_bearing),
                   stiffness_ratio=float(stiff_ratio)),
              open(OUT, "w"), indent=1, default=str)
    print("specs ->", OUT)


if __name__ == "__main__":
    main()
