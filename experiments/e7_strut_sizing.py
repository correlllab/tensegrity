#!/usr/bin/env python
"""E7: what does the compression network actually have to weigh?

Advantage 1 of the paper rests on axial loading: cables cannot buckle and
struts carry pure compression, so the structure should be light. That argument
was asserted, not computed -- and it is exactly the argument that can fail,
because a compression member's mass is set by BUCKLING, not by strength, and
buckling mass grows with the square of the member length.

This script closes it:

  1. reads every strut in the model (capsule geoms of the `strut` class) and
     its two endpoints,
  2. replays logged walking states and the E1 landing transients, computing at
     each state every cable tension from the deadband spring law and resolving,
     at each strut endpoint, the component of the attached cable tensions along
     the strut axis -- the strut's axial load,
  3. sizes each strut as a CFRP tube against Euler buckling with a safety
     factor, and reports the resulting structural mass against the 5.9 kg
     quoted in the mass budget.

Sign convention: positive = compression.
"""
import glob
import json
import re
import numpy as np
import mujoco

MJ = "/Users/ncorrell/Downloads/tensegrity/mujoco"
RES = "/Users/ncorrell/Downloads/tensegrity/experiments/results"
OUT = f"{RES}/e7_strut_sizing.json"
MODEL = f"{MJ}/humanoid_27dof_tensegrity.xml"

E_CFRP = 70e9           # Pa, filament-wound CF tube (conservative vs UD)
RHO_CFRP = 1600.0       # kg/m3
SF_BUCKLE = 2.0         # design safety factor on the critical load
K_EFF = 1.0             # pinned-pinned: struts terminate in cable nodes
T_MIN = 0.4e-3          # mm, minimum manufacturable wall
STRUT_RGBA = (0.22, 0.24, 0.28)


def strut_table(m, xml_path):
    """(body, endpoint A, endpoint B) in body-local coordinates, per strut.

    MjModel does not expose per-geom mass (it is folded into body inertia at
    compile time), so the as-modelled strut masses are read back from the XML
    in geom order."""
    xml_mass = [float(x) for x in re.findall(
        r'<geom class="strut"[^>]*mass="([-\d.eE+]+)"', open(xml_path).read())]
    out = []
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_CAPSULE:
            continue
        if not np.allclose(m.geom_rgba[g][:3], STRUT_RGBA, atol=1e-3):
            continue
        half = float(m.geom_size[g, 1])
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.geom_quat[g])
        axis = R.reshape(3, 3)[:, 2]
        out.append(dict(geom=g, body=int(m.geom_bodyid[g]),
                        xml_mass=xml_mass[len(out)] if len(out) < len(xml_mass)
                        else 0.0,
                        a=m.geom_pos[g] - half * axis,
                        b=m.geom_pos[g] + half * axis,
                        radius=float(m.geom_size[g, 0]),
                        length=2 * half))
    return out


def tendon_sites(m):
    """Site ids per tendon (spatial tendons in this model are site-to-site)."""
    out = []
    for t in range(m.ntendon):
        adr, num = m.tendon_adr[t], m.tendon_num[t]
        ids = [int(m.wrap_objid[adr + i]) for i in range(num)
               if m.wrap_type[adr + i] == mujoco.mjtWrap.mjWRAP_SITE]
        out.append(ids)
    return out


def axial_loads(m, d, struts, tsites, node_of):
    """Compression carried by each strut at the current state."""
    T = m.tendon_stiffness * np.maximum(0.0, d.ten_length -
                                        m.tendon_lengthspring[:, 1])
    load = np.zeros(len(struts))
    for t, ids in enumerate(tsites):
        if T[t] <= 0 or len(ids) < 2:
            continue
        for k, s in enumerate(ids):
            nb = ids[k + 1] if k + 1 < len(ids) else ids[k - 1]
            v = d.site_xpos[nb] - d.site_xpos[s]
            n = np.linalg.norm(v)
            if n < 1e-9:
                continue
            f = T[t] * v / n                       # pull on this endpoint
            for si, end in node_of.get(s, ()):
                ax = struts[si]["axis_w"]
                # +1 at end b, -1 at end a: a pull away from the strut
                # compresses it
                load[si] += -end * float(f @ ax)
    return load


def main():
    m = mujoco.MjModel.from_xml_path(MODEL)
    d = mujoco.MjData(m)
    struts = strut_table(m, MODEL)
    tsites = tendon_sites(m)

    # map site -> (strut index, which end) by coincidence with a strut tip
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    for s in struts:
        bid = s["body"]
        R = d.xmat[bid].reshape(3, 3)
        s["a_w"] = d.xpos[bid] + R @ s["a"]
        s["b_w"] = d.xpos[bid] + R @ s["b"]
    node_of = {}
    for sid in range(m.nsite):
        p = d.site_xpos[sid]
        for si, s in enumerate(struts):
            for end, key in ((-1, "a_w"), (+1, "b_w")):
                if np.linalg.norm(p - s[key]) < 2e-3:
                    node_of.setdefault(sid, []).append((si, end))
    tend_sites = sorted({s for ids in tsites for s in ids})
    on_tip = sum(1 for s in tend_sites if s in node_of)
    print(f"{len(struts)} struts, {m.ntendon} cables, "
          f"{len(tend_sites)} distinct cable-endpoint sites, "
          f"{on_tip} of them on a strut tip")

    def update_axes():
        for s in struts:
            R = d.xmat[s["body"]].reshape(3, 3)
            aw, bw = d.xpos[s["body"]] + R @ s["a"], d.xpos[s["body"]] + R @ s["b"]
            v = bw - aw
            s["axis_w"] = v / max(np.linalg.norm(v), 1e-9)

    peak = np.zeros(len(struts))
    states = 0

    # --- walking states
    for f in sorted(glob.glob(f"{RES}/log_baseline_t*.csv"))[:3]:
        raw = np.genfromtxt(f, delimiter=",", invalid_raise=False)
        raw = raw[~np.isnan(raw).any(axis=1)]
        t = raw[:, 0]
        sel = np.where((t >= 3.0) & (t <= 15.0))[0][::25]
        for i in sel:
            d.qpos[:] = raw[i, 1:1 + m.nq]
            mujoco.mj_forward(m, d)
            update_axes()
            peak = np.maximum(peak, axial_loads(m, d, struts, tsites, node_of))
            states += 1

    # --- a 0.1 m stumble landing (the design impact case from E1)
    mujoco.mj_resetDataKeyframe(m, d,
                                mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY,
                                                  "walk_ready"))
    d.qpos[2] += 0.10
    for _ in range(int(1.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
        update_axes()
        peak = np.maximum(peak, axial_loads(m, d, struts, tsites, node_of))
        states += 1

    # --- buckling sizing
    rows, mass_now, mass_sized = [], 0.0, 0.0
    for si, s in enumerate(struts):
        L, ro = s["length"], s["radius"]
        P = max(peak[si], 1.0)
        I_req = SF_BUCKLE * P * (K_EFF * L) ** 2 / (np.pi ** 2 * E_CFRP)
        ri = max(0.0, (ro ** 4 - 4 * I_req / np.pi) ** 0.25) \
            if ro ** 4 > 4 * I_req / np.pi else 0.0
        t_wall = max(ro - ri, T_MIN)
        ri = ro - t_wall
        area = np.pi * (ro ** 2 - ri ** 2)
        mm = RHO_CFRP * area * L
        P_cr = np.pi ** 2 * E_CFRP * np.pi * (ro ** 4 - ri ** 4) / 4 / (K_EFF * L) ** 2
        rows.append(dict(body=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                                                s["body"]),
                         L=L, r_o=ro, P_peak=float(peak[si]),
                         t_wall_mm=1e3 * t_wall, mass=mm,
                         P_cr=float(P_cr), model_mass=s["xml_mass"]))
        mass_now += s["xml_mass"]
        mass_sized += mm

    order = np.argsort([-r["P_peak"] for r in rows])
    print(f"\nstates sampled: {states}\n")
    print(f"{'body':10s} {'L mm':>6s} {'r_o mm':>7s} {'P_pk N':>8s} "
          f"{'t_wall mm':>10s} {'P_cr N':>9s} {'m g':>6s}")
    for i in order[:12]:
        r = rows[i]
        print(f"{r['body']:10s} {1e3*r['L']:6.0f} {1e3*r['r_o']:7.1f} "
              f"{r['P_peak']:8.0f} {r['t_wall_mm']:10.2f} {r['P_cr']:9.0f} "
              f"{1e3*r['mass']:6.1f}")
    print(f"\nstrut mass: model {mass_now:.2f} kg -> buckling-sized "
          f"{mass_sized:.2f} kg  (CFRP tube, E={E_CFRP/1e9:.0f} GPa, "
          f"SF={SF_BUCKLE}, min wall {1e3*T_MIN:.1f} mm)")
    heaviest = rows[int(np.argmax([r['mass'] for r in rows]))]
    print(f"heaviest strut: {heaviest['body']} {1e3*heaviest['L']:.0f} mm, "
          f"{1e3*heaviest['mass']:.0f} g, wall {heaviest['t_wall_mm']:.2f} mm")
    # --- the load path the cable tensions do NOT include: ground reaction
    # entering a segment through the hinge. Bound it by sharing the measured
    # peak GRF of E1 over the three struts of one cage.
    e1 = json.load(open(f"{RES}/e1_drop.json"))

    def grf(mode, h):
        return [r for r in e1 if r["mode"] == mode and r["h"] == h][0]["peak_grf"]

    cases = [("stand", 27.6 * 9.81), ("stumble 0.1 m",
                                      grf("tensegrity-passive", 0.10)),
             ("drop 1.0 m", grf("tensegrity-passive", 1.00))]
    worst = min(rows, key=lambda r: r["P_cr"])
    print(f"\nhinge load path (GRF shared over the 3 struts of a cage), "
          f"against the weakest strut ({worst['body']}, "
          f"{1e3*worst['L']:.0f} mm, P_cr {worst['P_cr']:.0f} N):")
    load_cases = []
    for name, F in cases:
        per = F / 3.0
        load_cases.append(dict(case=name, grf=float(F), per_strut=float(per),
                               margin=float(worst["P_cr"] / per)))
        print(f"  {name:14s} GRF {F:6.0f} N -> {per:5.0f} N/strut, "
              f"margin {worst['P_cr'] / per:5.1f}x")

    json.dump(dict(struts=rows, mass_model=mass_now, mass_sized=mass_sized,
                   E=E_CFRP, rho=RHO_CFRP, sf=SF_BUCKLE, states=states,
                   peak_cable_axial=float(max(r["P_peak"] for r in rows)),
                   weakest=worst, load_cases=load_cases),
              open(OUT, "w"), indent=1)
    print("E7 ->", OUT)


if __name__ == "__main__":
    main()
