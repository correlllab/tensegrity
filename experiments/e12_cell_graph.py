#!/usr/bin/env python
"""E12: can the whole humanoid be a graph of tensegrity cells, with no hinges?

E10 shows a single hinge-free joint works if the two cages overlap. This asks
whether that composes: a chain of N cells for a limb, and a branching junction
for the pelvis, all struts floating, no bearing anywhere.

Everything is generated from one description: a list of cells, each a Snelson
3-strut prism given by (bottom ring, top ring, phase), and a list of interfaces
between them. Struts are free bodies; the only things connecting anything are
cables. The base cell is anchored to the world so the structure has something
to stand on.

Reported per configuration:
  closure   force-closure margin of every interface (the E9/E10 linear program)
  stands    settles under gravity plus a payload without collapsing
  droop     how far the tip falls while settling
"""
import json
import sys
import numpy as np
import mujoco
from scipy.optimize import linprog

RES = "/Users/ncorrell/Downloads/tensegrity/experiments/results"
MJ = "/Users/ncorrell/Downloads/tensegrity/mujoco"
OUT = f"{RES}/e12_cell_graph.json"

TWIST = np.deg2rad(29.8)
RJ = 0.10                       # junction ring radius
STRUT_R, STRUT_DENS = 0.009, 1600.0
K_CABLE, T_PRE = 40e3, 120.0
DT = 5e-5


def ring(radius, centre, phase, normal=(0, 0, 1)):
    n = np.array(normal, float)
    n /= np.linalg.norm(n)
    a = np.array([1.0, 0, 0])
    if abs(n @ a) > 0.9:
        a = np.array([0, 1.0, 0])
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return [np.array(centre) + radius * (np.cos(phase + 2 * np.pi * i / 3) * u
                                         + np.sin(phase + 2 * np.pi * i / 3) * v)
            for i in range(3)]


class Graph:
    """A tensegrity built only from cells and cables."""

    def __init__(self):
        self.nodes = {}          # name -> (position, owner)  owner: strut name or 'world'
        self.struts = []         # (name, nodeA, nodeB)
        self.cables = []         # (nodeA, nodeB)
        self.clusters = []       # (name, [(nodeA,nodeB)...], anchored)
        self.cluster_payload = {}   # rigid junction -> extra carried mass (kg)
        self.contact_struts = []    # struts given a colliding foot pad
        self.masses = {}

    def cell(self, tag, c0, c1, radius, phase, axis, anchored=False):
        """One prism cell: bottom ring at c0, top ring at c1, struts i -> i+1."""
        bot = ring(radius, c0, phase, axis)
        top = ring(radius, c1, phase + TWIST, axis)
        for i in range(3):
            self.nodes[f"{tag}_b{i}"] = [bot[i], None]
            self.nodes[f"{tag}_t{i}"] = [top[i], None]
        for i in range(3):
            s = f"{tag}_s{i}"
            a, b = f"{tag}_b{i}", f"{tag}_t{(i+1)%3}"
            self.struts.append((s, a, b, anchored))
            self.nodes[a][1] = "world" if anchored else s
            self.nodes[b][1] = "world" if anchored else s
        for i in range(3):                       # cell cables
            self.cables += [(f"{tag}_b{i}", f"{tag}_b{(i+1)%3}"),
                            (f"{tag}_t{i}", f"{tag}_t{(i+1)%3}"),
                            (f"{tag}_b{i}", f"{tag}_t{i}")]
        return tag

    def interface(self, prox, dist):
        """Saddle interface between two prism cells, proximal top to distal
        bottom."""
        self.ring_interface([f"{prox}_t{i}" for i in range(3)],
                            [f"{prox}_b{i}" for i in range(3)],
                            [f"{dist}_b{i}" for i in range(3)],
                            [f"{dist}_t{i}" for i in range(3)])

    def ring_interface(self, p_near, p_far, d_near, d_far):
        """The general form. p_near is the proximal ring the distal cell
        overlaps into; p_far is the proximal cell's other ring, which supplies
        the family that resists the two cells being pushed together; d_far
        supplies the family that resists their being pulled apart."""
        for i in range(3):
            self.cables += [(p_near[i], d_near[i]),
                            (p_near[i], d_far[i]),
                            (p_near[i], d_near[(i + 1) % 3]),
                            (p_far[i], d_near[i])]

    def girdle(self, tag, rings, anchored=False):
        """A junction: several node rings tied by struts, each ring free to
        host an interface. This is what a chain of prism cells cannot do --- a
        3-strut prism has two rings and therefore at most two interfaces, while
        a pelvis needs three and a torso four."""
        names = {}
        for rname, (centre, phase, axis) in rings.items():
            pts = ring(RJ, centre, phase, axis)
            names[rname] = [f"{tag}_{rname}{i}" for i in range(3)]
            for i in range(3):
                self.nodes[names[rname][i]] = [pts[i], None]
        # A junction has three or more struts meeting at each node, so it is a
        # class-k tensegrity element rather than a floating cell: the strut
        # cluster is one rigid part. Limb cells remain fully floating; only the
        # trunk junctions are rigid.
        keys = list(rings)
        members = []
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                for i in range(3):
                    members.append((names[keys[a]][i],
                                    names[keys[b]][(i + 1) % 3]))
        self.clusters.append((tag, members, anchored))
        for rname in keys:
            for i in range(3):
                self.nodes[names[rname][i]][1] = "world" if anchored else tag
        for rname in keys:                      # perimeter cables per ring
            for i in range(3):
                self.cables.append((names[rname][i], names[rname][(i + 1) % 3]))
        return names

    def xml(self, payload=None, rests=None):
        world, bodies = [], []
        for s, a, b, anchored in self.struts:
            pa, pb = self.nodes[a][0], self.nodes[b][0]
            L = float(np.linalg.norm(pb - pa))
            mass = STRUT_DENS * np.pi * (STRUT_R ** 2 - (STRUT_R - 5e-4) ** 2) * L
            self.masses[s] = mass
            if anchored:
                world.append(
                    f'<geom type="capsule" size="{STRUT_R}" contype="0" '
                    f'conaffinity="0" rgba="0.22 0.24 0.28 1" '
                    f'fromto="{pa[0]:.5f} {pa[1]:.5f} {pa[2]:.5f} '
                    f'{pb[0]:.5f} {pb[1]:.5f} {pb[2]:.5f}"/>')
            else:
                c = 0.5 * (pa + pb)
                bodies.append(f'''
    <body name="{s}" pos="{c[0]:.5f} {c[1]:.5f} {c[2]:.5f}">
      <freejoint/>
      <geom type="capsule" size="{STRUT_R}" mass="{max(mass,0.02):.4f}"
            contype="0" conaffinity="0" rgba="0.30 0.33 0.38 1"
            fromto="{pa[0]-c[0]:.5f} {pa[1]-c[1]:.5f} {pa[2]-c[2]:.5f}
                    {pb[0]-c[0]:.5f} {pb[1]-c[1]:.5f} {pb[2]-c[2]:.5f}"/>''')
                for nd in (a, b):
                    p = self.nodes[nd][0]
                    bodies.append(f'      <site name="{nd}" pos="{p[0]-c[0]:.5f} '
                                  f'{p[1]-c[1]:.5f} {p[2]-c[2]:.5f}" size="0.005"/>')
                if s in self.contact_struts:
                    lo = pa if pa[2] < pb[2] else pb
                    bodies.append(
                        f'      <geom name="pad_{s}" type="sphere" size="0.022" '
                        f'mass="0.05" contype="1" conaffinity="1" '
                        f'friction="1.0 0.005 0.0001" '
                        f'pos="{lo[0]-c[0]:.5f} {lo[1]-c[1]:.5f} '
                        f'{lo[2]-c[2]:.5f}"/>')
                bodies.append("    </body>")
        for tag, members, anchored in self.clusters:
            pts = [self.nodes[n][0] for mm in members for n in mm]
            c = np.mean(pts, axis=0)
            seg = []
            for a, b in members:
                pa, pb = self.nodes[a][0], self.nodes[b][0]
                L = float(np.linalg.norm(pb - pa))
                mm = STRUT_DENS * np.pi * (STRUT_R ** 2
                                           - (STRUT_R - 5e-4) ** 2) * L
                seg.append(f'<geom type="capsule" size="{STRUT_R}" '
                           f'mass="{max(mm,0.02):.4f}" contype="0" '
                           f'conaffinity="0" rgba="0.22 0.24 0.28 1" '
                           f'fromto="{pa[0]-c[0]:.5f} {pa[1]-c[1]:.5f} '
                           f'{pa[2]-c[2]:.5f} {pb[0]-c[0]:.5f} '
                           f'{pb[1]-c[1]:.5f} {pb[2]-c[2]:.5f}"/>')
            sites = []
            for nd, (p, owner) in self.nodes.items():
                if owner == tag:
                    sites.append(f'<site name="{nd}" pos="{p[0]-c[0]:.5f} '
                                 f'{p[1]-c[1]:.5f} {p[2]-c[2]:.5f}" '
                                 f'size="0.005"/>')
            if anchored:
                for a, b in members:
                    pa, pb = self.nodes[a][0], self.nodes[b][0]
                    world.append(f'<geom type="capsule" size="{STRUT_R}" '
                                 f'contype="0" conaffinity="0" '
                                 f'rgba="0.22 0.24 0.28 1" '
                                 f'fromto="{pa[0]:.5f} {pa[1]:.5f} {pa[2]:.5f} '
                                 f'{pb[0]:.5f} {pb[1]:.5f} {pb[2]:.5f}"/>')
            else:
                extra = ""
                if tag in self.cluster_payload:
                    extra = (f'<geom type="sphere" size="0.05" '
                             f'mass="{self.cluster_payload[tag]}" contype="0" '
                             f'conaffinity="0" rgba="0.8 0.5 0.2 1" '
                             f'pos="0 0 0"/>')
                bodies.append(f'\n    <body name="{tag}" pos="{c[0]:.5f} '
                              f'{c[1]:.5f} {c[2]:.5f}">\n      <freejoint/>\n      '
                              + "\n      ".join(seg + sites + ([extra] if extra
                                                                else []))
                              + "\n    </body>")
        for nd, (p, owner) in self.nodes.items():
            if owner == "world":
                world.append(f'<site name="{nd}" pos="{p[0]:.5f} {p[1]:.5f} '
                             f'{p[2]:.5f}" size="0.005"/>')
        pay = ""
        if payload:
            nd, m = payload
            p = self.nodes[nd][0]
            pay = (f'\n    <body name="payload" pos="{p[0]:.5f} {p[1]:.5f} '
                   f'{p[2]:.5f}">\n      <freejoint/>\n'
                   f'      <geom type="sphere" size="0.02" mass="{m}" '
                   f'contype="0" conaffinity="0" rgba="0.8 0.5 0.2 1"/>\n'
                   f'      <site name="payload" pos="0 0 0" size="0.008"/>\n'
                   f'    </body>')
            for i in range(3):
                pass

        ten = []
        for i, (a, b) in enumerate(self.cables):
            r = rests[i] if rests is not None else 0.001
            ten.append(f'<spatial name="c{i}" width="0.003" stiffness="{K_CABLE}"'
                       f' damping="{0.02*np.sqrt(K_CABLE):.2f}"'
                       f' springlength="0 {r:.6f}">\n      <site site="{a}"/>'
                       f'\n      <site site="{b}"/>\n    </spatial>')
        return f'''<mujoco model="cell_graph">
  <compiler angle="radian" autolimits="true"/>
  <visual>
    <global offwidth="1600" offheight="1200"/>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.45 0.45 0.45"
               specular="0.1 0.1 0.1"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1"
             width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.88 0.89 0.90"
             rgb2="0.96 0.96 0.97" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="10 10" reflectance="0.05"/>
  </asset>
  <option timestep="{DT}" integrator="implicitfast" gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="1.2 -1.2 2.6" dir="-0.4 0.4 -1" directional="true"
           diffuse="0.85 0.85 0.85"/>
    <light pos="-1.5 1.0 2.0" dir="0.5 -0.35 -1" directional="true"
           diffuse="0.35 0.35 0.35"/>
    <geom name="floor" type="plane" size="4 4 0.1" material="grid"
          contype="1" conaffinity="1" friction="1.0 0.005 0.0001"/>
    {(chr(10) + "    ").join(world)}
    {"".join(bodies)}{pay}
  </worldbody>
  <tendon>
    {(chr(10) + "    ").join(ten)}
  </tendon>
</mujoco>'''

    def build(self, payload=None, t_pre=None):
        t = T_PRE if t_pre is None else t_pre
        m0 = mujoco.MjModel.from_xml_string(self.xml(payload))
        d0 = mujoco.MjData(m0)
        mujoco.mj_forward(m0, d0)
        rests = [max(1e-4, float(d0.ten_length[i]) - t / K_CABLE)
                 for i in range(m0.ntendon)]
        return mujoco.MjModel.from_xml_string(self.xml(payload, rests))


def closure(m, d, distal_bodies):
    mujoco.mj_forward(m, d)
    ids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
           for b in distal_bodies}
    origin = np.mean([d.xpos[i] for i in ids], axis=0)
    W = []
    for t in range(m.ntendon):
        adr = m.tendon_adr[t]
        sids = [int(m.wrap_objid[adr + i]) for i in range(m.tendon_num[t])
                if m.wrap_type[adr + i] == mujoco.mjtWrap.mjWRAP_SITE]
        if len(sids) < 2:
            continue
        side = [m.site_bodyid[s] in ids for s in sids]
        if all(side) or not any(side):
            continue
        for k, s in enumerate(sids):
            if not side[k]:
                continue
            v = d.site_xpos[sids[1 - k]] - d.site_xpos[s]
            n = np.linalg.norm(v)
            if n < 1e-9:
                continue
            u = v / n
            W.append(np.concatenate([u, np.cross(d.site_xpos[s] - origin, u)]))
    W = np.array(W).T if W else np.zeros((6, 0))
    if W.shape[1] == 0 or np.linalg.matrix_rank(W[:3]) < 3:
        return 0.0
    A = W[:3]
    nc = A.shape[1]
    c = np.zeros(nc + 1)
    c[-1] = -1.0
    r = linprog(c, A_ub=np.hstack([-np.eye(nc), np.ones((nc, 1))]),
                b_ub=np.zeros(nc),
                A_eq=np.hstack([A, np.zeros((3, 1))]), b_eq=np.zeros(3),
                bounds=[(0, None)] * nc + [(0, 1)])
    return float(r.x[-1]) if r.success else 0.0


def chain(n_cells, H=0.30, R=0.09, ov_frac=0.3, payload_kg=1.0):
    g = Graph()
    ov = ov_frac * H
    tags = []
    z = 0.0
    for k in range(n_cells):
        t = g.cell(f"c{k}", (0, 0, z), (0, 0, z + H), R, k * np.pi / 3,
                   (0, 0, 1), anchored=(k == 0))
        tags.append(t)
        if k:
            g.interface(tags[k - 1], t)
        z += H - ov
    return g, tags


def lower_body(H=0.30, R=0.09, ov_frac=0.3):
    """Pelvis girdle with a waist ring and one socket per hip, plus two
    three-cell legs. The waist ring is anchored, so the legs hang: the branch
    carries the whole load path."""
    g = Graph()
    ov = ov_frac * H
    zp = 0.0
    rings = {"waist": ((0, 0, zp + 0.10), 0.0, (0, 0, 1)),
             "hipl": ((0, +0.09, zp), np.pi / 3, (0, 0, 1)),
             "hipr": ((0, -0.09, zp), np.pi / 3, (0, 0, 1))}
    names = g.girdle("pel", rings, anchored=True)
    tips = []
    for side, sgn in (("l", +1), ("r", -1)):
        prev_near = names[f"hip{side}"]
        prev_far = names["waist"]
        z = zp
        tags = []
        for k in range(3):                      # thigh, shank, foot
            t = g.cell(f"{side}{k}", (0, sgn * 0.09, z - H + ov),
                       (0, sgn * 0.09, z + ov), R, k * np.pi / 3, (0, 0, 1))
            # the distal cell's TOP ring overlaps up into the proximal socket
            g.ring_interface(prev_near, prev_far,
                             [f"{t}_t{i}" for i in range(3)],
                             [f"{t}_b{i}" for i in range(3)])
            prev_near = [f"{t}_b{i}" for i in range(3)]
            prev_far = [f"{t}_t{i}" for i in range(3)]
            z -= H - ov
            tags.append(t)
        tips.append(tags)
    return g, tips


def full_body(H=0.30, R=0.09, ov_frac=0.3, torso_kg=0.0, standing=False):
    """The whole humanoid as a cell graph: no hinges, no bearings, nothing
    touching. Limb segments are prism cells; the pelvis and torso are girdles,
    because a 3-strut prism has two rings and so can host only two interfaces,
    while a pelvis needs three and a torso four. The feet are anchored."""
    g = Graph()
    ov = ov_frac * H
    step = H - ov
    layout = []

    def limb(side, y, z0, updir, n, hname):
        """n cells stacked from z0 in direction updir, returns ring names of
        the last cell's far end."""
        tags = []
        z = z0
        prev = None
        for k in range(n):
            zb, zt = (z, z + H) if updir > 0 else (z - H, z)
            t = g.cell(f"{hname}{side}{k}", (0, y, zb), (0, y, zt), R,
                       k * np.pi / 3, (0, 0, 1))
            if prev is not None:
                if updir > 0:
                    g.ring_interface([f"{prev}_t{i}" for i in range(3)],
                                     [f"{prev}_b{i}" for i in range(3)],
                                     [f"{t}_b{i}" for i in range(3)],
                                     [f"{t}_t{i}" for i in range(3)])
                else:
                    g.ring_interface([f"{prev}_b{i}" for i in range(3)],
                                     [f"{prev}_t{i}" for i in range(3)],
                                     [f"{t}_t{i}" for i in range(3)],
                                     [f"{t}_b{i}" for i in range(3)])
            prev = t
            tags.append(t)
            z += updir * step
        return tags

    # --- legs, feet anchored to the ground
    leg_tops = {}
    for side, y in (("l", +0.09), ("r", -0.09)):
        tags = limb(side, y, 0.0, +1, 3, "leg")
        if not standing:                      # bolt the foot cell to ground
            for idx, (sn, a, b, anc) in enumerate(g.struts):
                if sn.startswith(f"leg{side}0"):
                    g.struts[idx] = (sn, a, b, True)
                    g.nodes[a][1] = "world"
                    g.nodes[b][1] = "world"
        else:
            g.contact_struts += [sn for sn, a, b, anc in g.struts
                                 if sn.startswith(f"leg{side}0")]
        leg_tops[side] = tags[-1]
    z_hip = 2 * step + H

    # --- pelvis girdle
    pel = g.girdle("pel", {
        "hipl": ((0, +0.09, z_hip - ov), np.pi / 3, (0, 0, 1)),
        "hipr": ((0, -0.09, z_hip - ov), np.pi / 3, (0, 0, 1)),
        "waist": ((0, 0, z_hip + 0.10), 0.0, (0, 0, 1))})
    for side in ("l", "r"):
        t = leg_tops[side]
        g.ring_interface(pel[f"hip{side}"], pel["waist"],
                         [f"{t}_t{i}" for i in range(3)],
                         [f"{t}_b{i}" for i in range(3)])

    # --- torso girdle, overlapping the pelvis waist ring
    z_sh = z_hip + 0.55
    tor = g.girdle("tor", {
        "waist": ((0, 0, z_hip + 0.04), np.pi / 3, (0, 0, 1)),
        "shl": ((0, +0.16, z_sh), 0.0, (0, 0, 1)),
        "shr": ((0, -0.16, z_sh), 0.0, (0, 0, 1)),
        "neck": ((0, 0, z_sh + 0.10), np.pi / 3, (0, 0, 1))})
    g.ring_interface(pel["waist"], pel["hipl"], tor["waist"], tor["shl"])

    # --- arms hanging from the shoulder rings
    for side, y in (("l", +0.16), ("r", -0.16)):
        tags = limb(side, y, z_sh, -1, 3, "arm")
        t = tags[0]
        g.ring_interface(tor[f"sh{side}"], tor["neck"],
                         [f"{t}_t{i}" for i in range(3)],
                         [f"{t}_b{i}" for i in range(3)])

    # --- head
    hd = g.cell("head", (0, 0, z_sh + 0.06), (0, 0, z_sh + 0.30), R * 0.9,
                0.0, (0, 0, 1))
    g.ring_interface(tor["neck"], tor["waist"],
                     [f"{hd}_b{i}" for i in range(3)],
                     [f"{hd}_t{i}" for i in range(3)])
    if torso_kg:
        g.cluster_payload["tor"] = torso_kg
    return g, dict(pelvis=pel, torso=tor, head=hd, z_sh=z_sh)


def run(g, tags, payload_kg, seconds=1.5):
    tip = tags[-1]
    m = g.build()
    d = mujoco.MjData(m)
    distal = [s for s, a, b, anc in g.struts if s.startswith(tags[-1])]
    t3 = closure(m, d, distal)
    top_sites = [f"{tip}_t{i}" for i in range(3)]
    sid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s) for s in top_sites]
    mujoco.mj_forward(m, d)
    z0 = float(np.mean([d.site_xpos[i][2] for i in sid]))
    for _ in range(int(seconds / DT)):
        mujoco.mj_step(m, d)
        if not np.all(np.isfinite(d.qpos)):
            return dict(t3=t3, stands=False, droop=float("nan"),
                        reason="diverged")
    z1 = float(np.mean([d.site_xpos[i][2] for i in sid]))
    return dict(t3=t3, stands=bool(z0 - z1 < 0.15 * z0),
                droop=1e3 * (z0 - z1), z0=z0, z1=z1)


def main():
    res = {"chain": []}
    print("LIMB AS A CHAIN OF CELLS (no hinges anywhere)")
    print(f"{'cells':>6s} {'struts':>7s} {'cables':>7s} {'DoF':>5s} "
          f"{'force closure':>14s} {'droop (mm)':>11s}  verdict")
    for n in (2, 3, 4, 5):
        g, tags = chain(n)
        r = run(g, tags, 1.0)
        nfree = sum(1 for s, a, b, anc in g.struts if not anc)
        print(f"{n:6d} {len(g.struts):7d} {len(g.cables):7d} {6*nfree:5d} "
              f"{r['t3']:14.4f} {r['droop']:11.1f}  "
              f"{'stands' if r['stands'] else 'COLLAPSES'}")
        res["chain"].append(dict(n_cells=n, n_struts=len(g.struts),
                                 n_cables=len(g.cables), dof=6 * nfree, **r))
    print("\nBRANCHING JUNCTION: pelvis girdle + two 3-cell legs, hanging")
    g, tips = lower_body()
    m = g.build()
    d = mujoco.MjData(m)
    nfree = sum(1 for s, a, b, anc in g.struts if not anc)
    print(f"  {len(g.struts)} struts ({nfree} free), {len(g.cables)} cables, "
          f"{6*nfree} DoF")
    for side, tags in zip(("left", "right"), tips):
        distal = [s for s, a, b, anc in g.struts if s.startswith(tags[-1])]
        print(f"  {side} foot cell force closure: "
              f"{closure(m, d, distal):.4f}")
    sid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{tips[0][-1]}_b{i}")
           for i in range(3)]
    mujoco.mj_forward(m, d)
    z0 = float(np.mean([d.site_xpos[i][2] for i in sid]))
    ok = True
    for _ in range(int(1.5 / DT)):
        mujoco.mj_step(m, d)
        if not np.all(np.isfinite(d.qpos)):
            ok = False
            break
    z1 = float(np.mean([d.site_xpos[i][2] for i in sid])) if ok else float("nan")
    print(f"  foot tip {z0:.3f} -> {z1:.3f} m, droop {1e3*(z0-z1):.1f} mm  "
          f"{'HOLDS' if ok and abs(z0-z1) < 0.10 else 'COLLAPSES'}")
    res["lower_body"] = dict(n_struts=len(g.struts), n_free=nfree,
                             n_cables=len(g.cables), dof=6 * nfree,
                             z0=z0, z1=z1, droop_mm=1e3 * (z0 - z1),
                             holds=bool(ok and abs(z0 - z1) < 0.10))

    print("\nFULL BODY AS A CELL GRAPH (feet anchored, nothing else)")
    print("  carried mass on the torso vest, where the motors and batteries go")
    print(f"  {'torso kg':>9s} {'crown (m)':>10s} {'settle (mm)':>12s}  verdict")
    body_rows = []
    for tk in (0.0, 5.0, 10.0, 20.0, 30.0):
        g, info = full_body(torso_kg=tk)
        m = g.build()
        d = mujoco.MjData(m)
        hd = info["head"]
        sid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{hd}_t{i}")
               for i in range(3)]
        mujoco.mj_forward(m, d)
        z0 = float(np.mean([d.site_xpos[i][2] for i in sid]))
        ok = True
        for _ in range(int(1.5 / DT)):
            mujoco.mj_step(m, d)
            if not np.all(np.isfinite(d.qpos)):
                ok = False
                break
        z1 = float(np.mean([d.site_xpos[i][2] for i in sid])) if ok else np.nan
        v = "stands" if ok and abs(z0 - z1) < 0.10 else "COLLAPSES"
        print(f"  {tk:9.0f} {z1:10.3f} {1e3*(z0-z1):12.0f}  {v}")
        body_rows.append(dict(torso_kg=tk, z0=z0, z1=z1,
                              settle_mm=1e3 * (z0 - z1),
                              stands=bool(v == "stands")))
    res["full_body_load"] = body_rows

    print("\n  the same structure vs prestress, carrying 22 kg "
          "(motors + batteries + compute)")
    print(f"  {'prestress N':>12s} {'settle (mm)':>12s}  verdict")
    pre_rows = []
    for tp in (120.0, 300.0, 600.0, 1000.0, 1500.0, 2500.0):
        g, info = full_body(torso_kg=22.0)
        m = g.build(t_pre=tp)
        d = mujoco.MjData(m)
        hd = info["head"]
        sid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{hd}_t{i}")
               for i in range(3)]
        mujoco.mj_forward(m, d)
        z0 = float(np.mean([d.site_xpos[i][2] for i in sid]))
        ok = True
        for _ in range(int(1.5 / DT)):
            mujoco.mj_step(m, d)
            if not np.all(np.isfinite(d.qpos)):
                ok = False
                break
        z1 = float(np.mean([d.site_xpos[i][2] for i in sid])) if ok else np.nan
        v = "stands" if ok and abs(z0 - z1) < 0.10 else "COLLAPSES"
        print(f"  {tp:12.0f} {1e3*(z0-z1):12.0f}  {v}")
        pre_rows.append(dict(t_pre=tp, settle_mm=1e3 * (z0 - z1),
                             stands=bool(v == "stands")))
    res["full_body_prestress"] = pre_rows

    # what does the working prestress do to the compression members?
    g, info = full_body(torso_kg=22.0)
    m = g.build(t_pre=1000.0)
    d = mujoco.MjData(m)
    for _ in range(int(1.0 / DT)):
        mujoco.mj_step(m, d)
    T = m.tendon_stiffness * np.maximum(0.0, d.ten_length
                                        - m.tendon_lengthspring[:, 1])
    axial = {}
    for t in range(m.ntendon):
        adr = m.tendon_adr[t]
        sids = [int(m.wrap_objid[adr + i]) for i in range(m.tendon_num[t])
                if m.wrap_type[adr + i] == mujoco.mjtWrap.mjWRAP_SITE]
        if len(sids) < 2 or T[t] <= 0:
            continue
        for k, sd in enumerate(sids):
            b = m.site_bodyid[sd]
            v = d.site_xpos[sids[1 - k]] - d.site_xpos[sd]
            n = np.linalg.norm(v)
            if n < 1e-9:
                continue
            axial.setdefault(b, []).append(T[t] * v / n)
    peak = 0.0
    for b, fs in axial.items():
        peak = max(peak, float(np.linalg.norm(np.sum(fs, axis=0))))
    print(f"\n  at 1 kN prestress the cables put up to {peak:.0f} N of "
          f"resultant load into a strut cluster,")
    print(f"  against {peak/2358:.1f}x the critical load of the weakest "
          f"0.5 mm-wall CFRP tube specified in Table VII.")
    res["prestress_strut_load"] = dict(t_pre=1000.0, peak_node_load=peak)

    g, info = full_body()
    m = g.build()
    d = mujoco.MjData(m)
    n_cell = len(g.struts)
    n_clu = sum(len(mem) for _, mem, _ in g.clusters)
    nfree = sum(1 for s, a, b, anc in g.struts if not anc)
    n_clu_free = sum(1 for _, _, anc in g.clusters if not anc)
    print(f"\n  {n_cell + n_clu} struts total: {n_cell} in floating cells "
          f"({nfree} free) + {n_clu} in {len(g.clusters)} rigid junction "
          f"clusters")
    print(f"  {len(g.cables)} cables, {6*nfree + 6*n_clu_free} DoF")
    hd = info["head"]
    sid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{hd}_t{i}")
           for i in range(3)]
    mujoco.mj_forward(m, d)
    z0 = float(np.mean([d.site_xpos[i][2] for i in sid]))
    distal = [s for s, a, b, anc in g.struts if s.startswith(hd)]
    t3 = closure(m, d, distal)
    ok = True
    for _ in range(int(1.5 / DT)):
        mujoco.mj_step(m, d)
        if not np.all(np.isfinite(d.qpos)):
            ok = False
            break
    z1 = float(np.mean([d.site_xpos[i][2] for i in sid])) if ok else float("nan")
    verdict = "STANDS" if ok and abs(z0 - z1) < 0.10 else "COLLAPSES"
    print(f"  head force closure {t3:.4f}; crown {z0:.3f} -> {z1:.3f} m, "
          f"settle {1e3*(z0-z1):.0f} mm  {verdict}")
    res["full_body"] = dict(n_struts=n_cell + n_clu, n_cell_struts=n_cell,
                            n_cluster_struts=n_clu, n_free=nfree,
                            n_cables=len(g.cables),
                            dof=6 * nfree + 6 * n_clu_free,
                            head_closure=t3, z0=z0, z1=z1,
                            settle_mm=1e3 * (z0 - z1),
                            stands=bool(verdict == "STANDS"))

    # --- export the models so they can be opened in the viewer
    exports = []
    for tag, (gg, tp) in (("cellgraph", (full_body(torso_kg=22.0)[0], 1000.0)),
                          ("cellgraph_empty", (full_body()[0], T_PRE))):
        mm = gg.build(t_pre=tp)
        path = f"{MJ}/humanoid_{tag}.xml"
        open(path, "w").write(gg.xml(None, [
            max(1e-4, float(l) - tp / K_CABLE)
            for l in mujoco.MjData(mm).ten_length] if False else
            [float(mm.tendon_lengthspring[i, 1]) for i in range(mm.ntendon)]))
        exports.append(path)
    g2, t2 = chain(4)
    m2 = g2.build()
    p2 = f"{MJ}/tensegrity_chain4.xml"
    open(p2, "w").write(g2.xml(None, [float(m2.tendon_lengthspring[i, 1])
                                      for i in range(m2.ntendon)]))
    exports.append(p2)
    print("\nmodels written (open with: .venv/bin/python -m mujoco.viewer "
          "--mjcf=PATH):")
    for e in exports:
        print("  ", e)

    json.dump(res, open(OUT, "w"), indent=1)
    print("\nE12 ->", OUT)


if __name__ == "__main__":
    main()
