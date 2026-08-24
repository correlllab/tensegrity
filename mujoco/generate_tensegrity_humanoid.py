#!/usr/bin/env python
"""Generate the tensegrity humanoid models (M2 node-graph structure).

STRUCTURAL MODEL — read this before changing geometry.

Every segment is a Snelson 3-strut prism cage: strut i runs from bottom node i
to TOP node i+1, so the three struts cross the interior and never touch each
other. The twist between the node triangles is 29.8 deg, which is not a guess —
formfind_prism.py releases a真 floating 3-strut cell from four different start
angles and it settles there every time (theory: 90 - 180/n = 30 deg).

Every cable begins and ends on a NODE, and every node is the physical endpoint
of a strut. Nothing terminates in mid-air. Two cable families:

  * prism cables (class "weave"): the 9-cable network of each segment's own
    cell — bottom triangle, top triangle, and the three verticals joining
    corresponding nodes. In THIS model a segment is one rigid body, so these
    carry no load; they are drawn because they are the cell that the segment
    abstracts, and they are what a floating-strut build would have to tension.
  * joint cables (class "wire"): parent ring nodes -> child ring nodes across
    each joint. These are real: they carry load, provide passive prestress
    stiffness, and (in the cable model) are the actuators.

LOAD PATH, stated plainly: compression runs strut -> node -> hinge -> node ->
strut. The hinge at each joint carries the compressive/shear reaction; the
cable network carries tension only and applies the joint moment. That makes
this a tensegrity-INSPIRED limb, not a pure tensegrity: in a pure tensegrity
nothing but cable tension holds the struts apart, which is what
formfind_prism.py demonstrates at cell scale. Keeping a hinge is what makes
the 27-DoF system controllable by MJPC.

Cables are tension-only: springlength is a DEADBAND "0 L_rest", so force is
zero below L_rest and pulls above it. A single springlength value would make
them bidirectional springs that can PUSH — which silently turns every cable
into a strut and makes the structure fake.

Outputs:
- humanoid_27dof_tensegrity.xml       27 joint-torque motors, ctrl [-1,1]
- humanoid_27dof_tensegrity_cable.xml 66 tension-only cable motors, ctrl [0,1]

Run:  ../.venv/bin/python generate_tensegrity_humanoid.py
"""
import math
from collections import defaultdict

OUT = "humanoid_27dof_tensegrity.xml"
OUT_CABLE = "humanoid_27dof_tensegrity_cable.xml"

TWIST = 29.8            # deg, measured by formfind_prism.py (theory 30)
PHASE = 90.0            # deg, rotation of the node triangles about +z

# max cable tension (N) by joint prefix (3 mm Dyneema working load for legs)
CABLE_FMAX = (("hip", 1500), ("knee", 1500), ("ankle", 1500), ("waist", 1500),
              ("shoulder", 700), ("elbow", 500), ("wrist", 300), ("neck", 250))

JOINT_GEAR = (("hip_yaw", 40), ("hip_roll", 60), ("hip_pitch", 80), ("knee", 80),
              ("ankle_pitch", 70), ("ankle_roll", 40), ("waist", 60), ("neck", 10),
              ("shoulder", 30), ("elbow", 20), ("wrist", 10))

# cable groups: stiffness (N/m), damping (N s/m), prestress fraction
K_LEG, C_LEG, PRE_LEG = 2500.0, 8.0, 0.04
K_ARM, C_ARM, PRE_ARM = 1200.0, 4.0, 0.04
K_PRISM, C_PRISM, PRE_PRISM = 600.0, 3.0, 0.03    # each segment's own cell

# --------------------------------------------------------------- batteries
# Swappable packs on the posterior pelvis (lumbar belt zone, below the waist
# joint so the waist actuators never carry them). Zero-joint bodies welded to
# the pelvis, so a pack can be pulled at runtime by zeroing its mass
# (verify_humanoid27.py --remove-battery). Two packs is the minimum for true
# hot-swap; symmetric placement keeps the single-pack-out CoM offset small.
BATTERY_MASS = 2.5
BATTERY_WH = 450.0
BATTERY_HALF = (0.045, 0.05, 0.055)
BATTERY_POS = ((-0.075, 0.058, 0.005), (-0.075, -0.058, 0.005))

# world origin of every body at the home pose (no rotated frames at qpos=0, so
# cable rest lengths are exact point-to-point distances)
W = {
    "pelvis": (0, 0, 0.936),
    "thigh_l": (0, 0.09, 0.876), "shank_l": (0, 0.09, 0.476), "foot_l": (0, 0.09, 0.076),
    "thigh_r": (0, -0.09, 0.876), "shank_r": (0, -0.09, 0.476), "foot_r": (0, -0.09, 0.076),
    "torso": (0, 0, 1.036), "head": (0, 0, 1.536),
    "upper_arm_l": (0, 0.24, 1.436), "forearm_l": (0, 0.24, 1.156), "hand_l": (0, 0.24, 0.916),
    "upper_arm_r": (0, -0.24, 1.436), "forearm_r": (0, -0.24, 1.156), "hand_r": (0, -0.24, 0.916),
}

RINGS = {}                          # (body, ring) -> [3 local xyz]
DRUMS = {}                          # body -> [(name, radius, halflen, z)]
STRUTS = defaultdict(list)          # body -> [xml strings]
sites = {b: [] for b in W}          # body -> [(site name, local xyz)]
tendons = []
wires = []                          # actuatable joint cables, in order


def fmt(v):
    return " ".join(f"{x:.4g}" for x in v)


def add3(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def ring_pt(r, deg):
    a = math.radians(deg)
    return (r * math.cos(a), r * math.sin(a), 0.0)


def define_ring(body, name, center, r, twist=0.0):
    """A triangle of 3 nodes. Nodes are where struts end and cables attach."""
    pts = [add3(center, ring_pt(r, PHASE + 120 * i + twist)) for i in range(3)]
    RINGS[(body, name)] = pts
    return pts


def define_nodes(body, name, pts):
    """An explicitly placed node triangle.

    Used where a symmetric ring would give bad moment arms. The ankle is the
    case that forces this: cables from a small circular ring around the ankle
    are nearly parallel to the joint axis offset, so plantarflexion torque
    needs absurd tension. A real foot solves it by inserting the Achilles well
    BEHIND the joint and the anterior tendon well in front — so the foot's
    nodes go at the heel and the two forefoot corners.
    """
    RINGS[(body, name)] = list(pts)
    return RINGS[(body, name)]


def strut(body, p1, p2, radius, mass):
    STRUTS[body].append(f'<geom class="strut" size="{radius}" '
                        f'fromto="{fmt(p1)} {fmt(p2)}" mass="{mass:.5g}"/>')


def cable(name, b1, p1, b2, p2, k, c, pre, cls="wire", wrap=None):
    """wrap = (drum geom name, sidesite name) inserts a wrap object mid-path."""
    sites[b1].append((f"{name}_a", p1))
    sites[b2].append((f"{name}_b", p2))
    L0 = math.dist(add3(W[b1], p1), add3(W[b2], p2))
    path = [f'      <site site="{name}_a"/>']
    if wrap:
        path.append(f'      <geom geom="{wrap[0]}" sidesite="{wrap[1]}"/>')
    path.append(f'      <site site="{name}_b"/>')
    tendons.append(dict(name=name, cls=cls, k=k, c=c, pre=pre, L0=L0,
                        path="\n".join(path), wrapped=bool(wrap)))
    if cls == "wire":
        wires.append(name)


def render_tendons(lengths=None, wrapped=True):
    """lengths: {name: true length at home} from a first compile. Wrapped
    tendons are longer than the straight-line distance, so prestress is only
    exact once the real length is known."""
    out = []
    for t in tendons:
        if t["wrapped"] and not wrapped:
            continue
        L = lengths.get(t["name"], t["L0"]) if lengths else t["L0"]
        # deadband "0 L_rest" => tension only, never pushes
        out.append(f'    <spatial name="{t["name"]}" class="{t["cls"]}" '
                   f'stiffness="{t["k"]:.4g}" damping="{t["c"]:.4g}" '
                   f'springlength="0 {(1 - t["pre"]) * L:.4f}">\n{t["path"]}\n'
                   f'    </spatial>')
    return "\n".join(out)


def node_cable(name, b1, ring1, i1, b2, ring2, i2, k, c, pre, cls="wire"):
    """Cable between two NODES, addressed by (body, ring, index)."""
    cable(name, b1, RINGS[(b1, ring1)][i1], b2, RINGS[(b2, ring2)][i2], k, c, pre, cls)


N_DRUM_CABLE = 6        # see drum_drive()
DRUM_FMAX = {}


def drum_drive(prefix, parent, ring_p, child, radius, z, fmax, k, c, pre):
    """Capstan drum on a YAW axis.

    A cable helixed around a limb has almost no moment arm about that limb's
    OWN long axis (16-25 mm here), so yaw DoFs were 1.6-2.5x short of their
    torque targets. A drum coaxial with the yaw axis fixes it structurally:
    the cable wraps the drum, so the moment arm IS the drum radius, set by the
    drum and not by how fat the limb cage is.

    In hardware this is ONE cable of several turns on ONE motor spool. MuJoCo
    can only wrap a geom once, and a single wrapped tendon loses contact at
    some joint angles (moment arm collapses toward the straight-line value),
    so the winding is modelled as N_DRUM_CABLE tendons spaced evenly around
    the drum: at every angle in the +/-1.5 rad range at least one is engaged
    each way, giving a guaranteed ~0.73 * radius of arm in BOTH directions.
    """
    drum = f"{prefix}_drum"
    DRUMS.setdefault(child, []).append((drum, radius, z))
    for i in range(N_DRUM_CABLE):
        j = i % 3                       # which parent node feeds this cable
        s_dir = 1 if i % 2 == 0 else -1  # winding direction -> sign of torque
        # angular position of the anchor about the DRUM AXIS, measured from
        # real geometry. Deriving it from PHASE alone is wrong: the parent ring
        # carries a twist and its centre can be offset, so the "opposite" point
        # computed that way is not opposite at all and the wrap never engages.
        _an = RINGS[(parent, ring_p)][j]
        _aw = add3(W[parent], _an)
        th = math.degrees(math.atan2(_aw[1] - W[child][1], _aw[0] - W[child][0]))
        # terminate DIAMETRICALLY OPPOSITE the anchor so the straight chord
        # always crosses the drum: that is what forces MuJoCo to wrap. A
        # termination near the anchor leaves the path straight (wrapnum 2) and
        # the moment arm collapses to a few mm.
        ta = math.radians(th + 180.0)
        term = (radius * math.cos(ta), radius * math.sin(ta), z)
        # sidesite on the side matching the desired winding -> opposite signs
        sa = math.radians(th + 90.0 * s_dir)
        sname = f"{prefix}_side{i}"
        sites[child].append((sname, (radius * 1.8 * math.cos(sa),
                                     radius * 1.8 * math.sin(sa), z)))
        cable(f"{prefix}{i}", parent, RINGS[(parent, ring_p)][j],
              child, term, k, c, pre, wrap=(drum, sname))
        DRUM_FMAX[f"{prefix}{i}"] = fmax


def prism_segment(body, z_top, z_bot, r_top, r_bot, strut_mass, radius=0.009):
    """Snelson cage: struts bot[i] -> top[i+1], plus the cell's 9 cables."""
    top = define_ring(body, "top", (0, 0, z_top), r_top, twist=TWIST)
    bot = define_ring(body, "bot", (0, 0, z_bot), r_bot)
    for i in range(3):
        strut(body, bot[i], top[(i + 1) % 3], radius, strut_mass)
    prism_cables(body, "bot", "top")


def prism_cables(body, ring_lo, ring_hi):
    tag = f"{body}_{ring_lo}{ring_hi}"
    for i in range(3):
        j = (i + 1) % 3
        node_cable(f"pw_{tag}_b{i}", body, ring_lo, i, body, ring_lo, j,
                   K_PRISM, C_PRISM, PRE_PRISM, cls="weave")
        node_cable(f"pw_{tag}_t{i}", body, ring_hi, i, body, ring_hi, j,
                   K_PRISM, C_PRISM, PRE_PRISM, cls="weave")
        node_cable(f"pw_{tag}_v{i}", body, ring_lo, i, body, ring_hi, i,
                   K_PRISM, C_PRISM, PRE_PRISM, cls="weave")


def ring_triangle(body, ring):
    for i in range(3):
        node_cable(f"rt_{body}_{ring}_{i}", body, ring, i, body, ring, (i + 1) % 3,
                   K_PRISM, C_PRISM, PRE_PRISM, cls="weave")


# Cable families across a joint. "axial" joins corresponding nodes (moment
# arms for pitch/roll, but ZERO about the long axis). "cw"/"ccw" wind opposite
# ways around the joint, which is the only way to get yaw torque in BOTH
# directions — a single winding direction gives yaw authority one way and
# literally none the other (check_cable_authority.py reports 0.00x for it).
FAMILIES = {"axial": lambda i: i, "cw": lambda i: (i + 1) % 3,
            "ccw": lambda i: (i - 1) % 3}


def joint_cables(prefix, b1, ring1, b2, ring2, k, c, pre, families=("axial", "cw")):
    n = 0
    for fam in families:
        pick = FAMILIES[fam]
        for i in range(3):
            node_cable(f"{prefix}{n}", b1, ring1, i, b2, ring2, pick(i), k, c, pre)
            n += 1


# ================================================================ structure
# ---- pelvis: a girdle of 3 node triangles (waist + one socket per hip)
p_waist = define_ring("pelvis", "waist", (0, 0, 0.06), 0.055, twist=TWIST)
p_hip_l = define_ring("pelvis", "hip_l", (0, 0.09, -0.055), 0.062)
p_hip_r = define_ring("pelvis", "hip_r", (0, -0.09, -0.055), 0.062)
for i in range(3):
    strut("pelvis", p_hip_l[i], p_waist[(i + 1) % 3], 0.010, 0.24444)
    strut("pelvis", p_hip_r[i], p_waist[(i + 1) % 3], 0.010, 0.24444)
    strut("pelvis", p_hip_l[i], p_hip_r[i], 0.010, 0.24444)   # pelvic girdle
for r in ("waist", "hip_l", "hip_r"):
    ring_triangle("pelvis", r)

# ---- torso: waist ring, two shoulder sockets, neck ring; clavicle crossbars
t_waist = define_ring("torso", "waist", (0, 0, -0.015), 0.055)
t_sh_l = define_ring("torso", "sh_l", (0, 0.24, 0.465), 0.052, twist=TWIST)
t_sh_r = define_ring("torso", "sh_r", (0, -0.24, 0.465), 0.052, twist=TWIST)
t_neck = define_ring("torso", "neck", (0, 0, 0.455), 0.04, twist=TWIST)
for i in range(3):
    strut("torso", t_waist[i], t_sh_l[(i + 1) % 3], 0.010, 0.375)
    strut("torso", t_waist[i], t_sh_r[(i + 1) % 3], 0.010, 0.375)
    strut("torso", t_sh_l[i], t_sh_r[i], 0.010, 0.375)        # clavicle
    strut("torso", t_waist[i], t_neck[(i + 1) % 3], 0.009, 0.375)
for r in ("waist", "sh_l", "sh_r", "neck"):
    ring_triangle("torso", r)

# ---- limb segments (true prism cages)
for side in ("l", "r"):
    prism_segment(f"thigh_{side}", -0.025, -0.375, 0.065, 0.060, 0.11667, 0.009)
    prism_segment(f"shank_{side}", -0.025, -0.330, 0.060, 0.052, 0.10, 0.008)
    prism_segment(f"upper_arm_{side}", -0.04, -0.255, 0.052, 0.046, 0.06667, 0.008)
    prism_segment(f"forearm_{side}", -0.025, -0.215, 0.044, 0.038, 0.06, 0.007)

# ---- end effectors: a node triangle on a short yoke into the rigid part
for side in ("l", "r"):
    # heel + two forefoot corners: real lever arms for pitch AND roll
    f_top = define_nodes(f"foot_{side}", "top",
                         [(-0.075, 0.0, 0.012),
                          (0.110, 0.048, 0.012),
                          (0.110, -0.048, 0.012)])
    for p in f_top:
        strut(f"foot_{side}", p, (p[0] * 0.85, p[1] * 0.8, -0.018), 0.007, 0.0)
    ring_triangle(f"foot_{side}", "top")

    h_top = define_ring(f"hand_{side}", "top", (0, 0, -0.005), 0.034)
    for i in range(3):
        strut(f"hand_{side}", h_top[i], (0, 0, -0.05), 0.006, 0.0)
    ring_triangle(f"hand_{side}", "top")

hd_top = define_ring("head", "top", (0, 0, -0.02), 0.038)
for i in range(3):
    strut("head", hd_top[i], (0, 0, 0.045), 0.007, 0.0)
ring_triangle("head", "top")

# ================================================================ joints
for side in ("l", "r"):
    joint_cables(f"hip_{side}", "pelvis", f"hip_{side}", f"thigh_{side}", "top",
                 K_LEG, C_LEG, PRE_LEG, families=("axial", "cw", "ccw"))
    joint_cables(f"knee_{side}", f"thigh_{side}", "bot", f"shank_{side}", "top",
                 K_LEG, C_LEG, PRE_LEG, families=("axial", "cw"))
    joint_cables(f"ankle_{side}", f"shank_{side}", "bot", f"foot_{side}", "top",
                 K_LEG, C_LEG, PRE_LEG, families=("axial", "cw"))
    joint_cables(f"shoulder_{side}", "torso", f"sh_{side}", f"upper_arm_{side}", "top",
                 K_ARM, C_ARM, PRE_ARM, families=("axial", "cw"))
    drum_drive(f"shoulder_yawdrum_{side}", "torso", f"sh_{side}",
               f"upper_arm_{side}", 0.045, -0.035, 1000, K_ARM, C_ARM, PRE_ARM)
    joint_cables(f"elbow_{side}", f"upper_arm_{side}", "bot", f"forearm_{side}", "top",
                 K_ARM, C_ARM, PRE_ARM, families=("axial", "cw"))
    drum_drive(f"wrist_yawdrum_{side}", f"forearm_{side}", "bot",
               f"hand_{side}", 0.035, -0.008, 500, K_ARM, C_ARM, PRE_ARM)
joint_cables("waist", "pelvis", "waist", "torso", "waist", K_LEG, C_LEG, PRE_LEG,
             families=("axial", "cw"))
drum_drive("waist_yawdrum", "pelvis", "waist", "torso", 0.055, 0.02, 1500,
           K_LEG, C_LEG, PRE_LEG)
joint_cables("neck", "torso", "neck", "head", "top", K_ARM, C_ARM, PRE_ARM,
             families=("axial", "cw"))
drum_drive("neck_yawdrum", "torso", "neck", "head", 0.035, -0.005, 450,
           K_ARM, C_ARM, PRE_ARM)


# ================================================================ emit
def site_xml(body, indent):
    pad = " " * indent
    return "".join(f'{pad}<site name="{n}" pos="{fmt(p)}"/>\n' for n, p in sites[body])


def struts_xml(body, indent):
    pad = " " * indent
    out = "".join(f"{pad}{g}\n" for g in STRUTS[body])
    for nm, r, z in DRUMS.get(body, []):
        out += (f'{pad}<geom name="{nm}" class="drum" type="cylinder" '
                f'size="{r} 0.009" pos="0 0 {z}"/>\n')
    return out


def battery_bodies(indent=6):
    pad = " " * indent
    sx, sy, sz = BATTERY_HALF
    out = []
    for p in BATTERY_POS:
        side = "l" if p[1] > 0 else "r"
        out.append(
            f'{pad}<body name="battery_{side}" pos="{fmt(p)}">\n'
            f'{pad}  <geom name="battery_{side}_cells" type="box" size="{sx} {sy} {sz}" '
            f'mass="{BATTERY_MASS}" rgba="0.17 0.21 0.30 1"/>\n'
            f'{pad}  <geom name="battery_{side}_rail_t" type="box" size="{sx * 0.9} 0.004 0.006" '
            f'pos="0 0 {sz + 0.005:.4g}" mass="0" rgba="0.45 0.47 0.5 1"/>\n'
            f'{pad}  <geom name="battery_{side}_rail_b" type="box" size="{sx * 0.9} 0.004 0.006" '
            f'pos="0 0 {-sz - 0.005:.4g}" mass="0" rgba="0.45 0.47 0.5 1"/>\n'
            f'{pad}  <geom name="battery_{side}_latch" type="cylinder" size="0.008 0.006" '
            f'pos="{-sx - 0.004:.4g} 0 0" euler="0 1.5708 0" mass="0" rgba="0.9 0.55 0.1 1"/>\n'
            f'{pad}</body>\n')
    return "".join(out)


def leg(side, s):
    hip_roll = "-0.4 0.8" if s > 0 else "-0.8 0.4"
    return f"""
      <body name="thigh_{side}" pos="0 {s * 0.09} -0.06">
        <joint name="hip_yaw_{side}"   axis="0 0 1" range="-0.8 0.8"/>
        <joint name="hip_roll_{side}"  axis="1 0 0" range="{hip_roll}"/>
        <joint name="hip_pitch_{side}" axis="0 1 0" range="-2.0 0.7"/>
{struts_xml(f"thigh_{side}", 8)}        <geom name="knee_motor_{side}" class="motormesh" mesh="ak70" pos="0.055 0 -0.09" euler="1.5708 0 0" mass="0.7"/>
{site_xml(f"thigh_{side}", 8)}        <body name="shank_{side}" pos="0 0 -0.40">
          <joint name="knee_{side}" axis="0 1 0" range="-0.05 2.4"/>
{struts_xml(f"shank_{side}", 10)}          <geom name="ankle_motor_{side}_a" class="motormesh" mesh="ak70" pos="0.05 0 -0.055" euler="1.5708 0 0" mass="0.7"/>
          <geom name="ankle_motor_{side}_b" class="motormesh" mesh="ak70" pos="0.05 0 -0.15" euler="1.5708 0 0" mass="0.7"/>
{site_xml(f"shank_{side}", 10)}          <body name="foot_{side}" pos="0 0 -0.40">
            <joint name="ankle_pitch_{side}" axis="0 1 0" range="-0.9 0.9"/>
            <joint name="ankle_roll_{side}"  axis="1 0 0" range="-0.4 0.4"/>
            <geom name="foot_{side}_geom" type="box" pos="0.04 0 -0.045" size="0.12 0.05 0.03" mass="0.40"/>
{struts_xml(f"foot_{side}", 12)}            <site name="foot_{side}_site" pos="0.04 0 -0.075" size="0.01"/>
            <site name="sp{0 if s < 0 else 2}" pos="0.16 0 -0.075" size="0.008"/>
            <site name="sp{1 if s < 0 else 3}" pos="-0.08 0 -0.075" size="0.008"/>
{site_xml(f"foot_{side}", 12)}          </body>
        </body>
      </body>"""


def arm(side, s):
    sh_roll = "-0.5 1.8" if s > 0 else "-1.8 0.5"
    return f"""
        <body name="upper_arm_{side}" pos="0 {s * 0.24} 0.40">
          <joint name="shoulder_pitch_{side}" axis="0 1 0" range="-3.0 1.0"/>
          <joint name="shoulder_roll_{side}"  axis="1 0 0" range="{sh_roll}"/>
          <joint name="shoulder_yaw_{side}"   axis="0 0 1" range="-1.5 1.5"/>
{struts_xml(f"upper_arm_{side}", 10)}          <geom name="arm_motor_{side}_a" class="motormesh" mesh="ak60" pos="0.045 0 -0.05" euler="1.5708 0 0" mass="0.35"/>
          <geom name="arm_motor_{side}_b" class="motormesh" mesh="ak60" pos="0.045 0 -0.14" euler="1.5708 0 0" mass="0.35"/>
{site_xml(f"upper_arm_{side}", 10)}          <body name="forearm_{side}" pos="0 0 -0.28">
            <joint name="elbow_{side}" axis="0 1 0" range="-2.4 0.05"/>
{struts_xml(f"forearm_{side}", 12)}{site_xml(f"forearm_{side}", 12)}            <body name="hand_{side}" pos="0 0 -0.24">
              <joint name="wrist_{side}" axis="0 0 1" range="-1.4 1.4"/>
              <geom name="hand_{side}_geom" type="capsule" fromto="0 0 -0.03 0 0 -0.08" size="0.026" mass="0.07"/>
{struts_xml(f"hand_{side}", 14)}              <site name="ee_{side}" pos="0 0 -0.08" size="0.01"/>
{site_xml(f"hand_{side}", 14)}            </body>
          </body>
        </body>"""


xml = f"""<mujoco model="humanoid27_tensegrity">
  <!-- GENERATED by generate_tensegrity_humanoid.py — do not edit by hand.
       Snelson 3-strut prism cages (twist {TWIST} deg, form-found by
       formfind_prism.py). Every cable terminates on a strut endpoint. Cables
       are tension-only (springlength deadband "0 L_rest").
       Load path: strut -> node -> HINGE -> node -> strut carries compression;
       the cable network carries tension and applies joint moments.
       No <sensor> block: MJPC requires user (cost) sensors first, so the task
       XMLs own all sensors. -->
  <compiler angle="radian" autolimits="true" meshdir="assets"/>
  <option timestep="0.004" integrator="implicitfast" solver="Newton" iterations="100"/>

  <default>
    <joint damping="2.0" armature="0.03" limited="true"/>
    <geom friction="1.0 0.005 0.0001" condim="3" contype="1" conaffinity="0"
          rgba="0.75 0.75 0.78 1"/>
    <site size="0.0045" rgba="0.95 0.65 0.15 0.95"/>
    <motor ctrllimited="true"/>
    <default class="strut">
      <geom type="capsule" rgba="0.22 0.24 0.28 1" contype="0" conaffinity="0"/>
    </default>
    <default class="wire">
      <tendon width="0.0035" rgba="0.95 0.97 1 1"/>
    </default>
    <default class="weave">
      <tendon width="0.0022" rgba="0.40 0.62 0.85 0.85"/>
    </default>
    <default class="drum">
      <geom rgba="0.85 0.62 0.22 1" contype="0" conaffinity="0" mass="0.02"/>
    </default>
    <default class="motormesh">
      <geom type="mesh" contype="0" conaffinity="0" rgba="0.32 0.33 0.36 1"/>
    </default>
  </default>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.25 0.3" rgb2="0.3 0.35 0.4"
             width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.1"/>
    <mesh name="ak70" file="ak70_10.stl"/>
    <mesh name="ak60" file="ak60_6.stl"/>
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="20 20 0.1" material="grid" contype="1" conaffinity="1"/>

    <body name="pelvis" pos="0 0 0.936">
      <freejoint name="root"/>
{struts_xml("pelvis", 6)}      <geom name="hip_motor_l_a" class="motormesh" mesh="ak70" pos="-0.09 0.115 -0.015" euler="1.5708 0 0" mass="0.56667"/>
      <geom name="hip_motor_l_b" class="motormesh" mesh="ak70" pos="0 0.115 -0.015" euler="1.5708 0 0" mass="0.56667"/>
      <geom name="hip_motor_l_c" class="motormesh" mesh="ak70" pos="0.09 0.115 -0.015" euler="1.5708 0 0" mass="0.56667"/>
      <geom name="hip_motor_r_a" class="motormesh" mesh="ak70" pos="-0.09 -0.115 -0.015" euler="1.5708 0 0" mass="0.56667"/>
      <geom name="hip_motor_r_b" class="motormesh" mesh="ak70" pos="0 -0.115 -0.015" euler="1.5708 0 0" mass="0.56667"/>
      <geom name="hip_motor_r_c" class="motormesh" mesh="ak70" pos="0.09 -0.115 -0.015" euler="1.5708 0 0" mass="0.56667"/>
      <site name="pelvis_site" pos="0 0 0" size="0.01"/>
{site_xml("pelvis", 6)}{battery_bodies(6)}{leg("l", 1)}{leg("r", -1)}
      <body name="torso" pos="0 0 0.10">
        <joint name="waist_yaw"   axis="0 0 1" range="-0.7 0.7"/>
        <joint name="waist_pitch" axis="0 1 0" range="-0.5 0.6"/>
        <joint name="waist_roll"  axis="1 0 0" range="-0.4 0.4"/>
{struts_xml("torso", 8)}        <geom name="vest" type="box" pos="-0.07 0 0.24" size="0.03 0.10 0.09" rgba="0.15 0.15 0.18 1" mass="1.44"/>
        <geom name="sh_motor_l_a" class="motormesh" mesh="ak60" pos="0 0.185 0.345" euler="1.5708 0 0" mass="0.31"/>
        <geom name="sh_motor_l_b" class="motormesh" mesh="ak60" pos="0.065 0.185 0.29" euler="1.5708 0 0" mass="0.31"/>
        <geom name="sh_motor_l_c" class="motormesh" mesh="ak60" pos="-0.065 0.185 0.29" euler="1.5708 0 0" mass="0.31"/>
        <geom name="sh_motor_r_a" class="motormesh" mesh="ak60" pos="0 -0.185 0.345" euler="1.5708 0 0" mass="0.31"/>
        <geom name="sh_motor_r_b" class="motormesh" mesh="ak60" pos="0.065 -0.185 0.29" euler="1.5708 0 0" mass="0.31"/>
        <geom name="sh_motor_r_c" class="motormesh" mesh="ak60" pos="-0.065 -0.185 0.29" euler="1.5708 0 0" mass="0.31"/>
        <site name="imu" pos="0 0 0.30" size="0.01"/>
{site_xml("torso", 8)}
        <body name="head" pos="0 0 0.50">
          <joint name="neck_pitch" axis="0 1 0" range="-0.6 0.6"/>
          <joint name="neck_yaw"   axis="0 0 1" range="-1.0 1.0"/>
          <geom name="head_geom" type="sphere" pos="0 0 0.075" size="0.085" mass="0.60"/>
{struts_xml("head", 10)}          <site name="head_site" pos="0 0 0.075" size="0.01"/>
{site_xml("head", 10)}        </body>
{arm("l", 1)}{arm("r", -1)}
      </body>
    </body>
  </worldbody>

  <tendon>
__TENDONS__
  </tendon>

  <actuator>
__ACTUATORS__
  </actuator>

  <keyframe>
    <!-- "home" keeps the legs straight: it is the assembly reference the cable
         rest lengths are built from, and the stand tasks are tuned around it.
         "walk_ready" is the gait posture reference — at home the legs are fully
         extended (pelvis 0.936 = exactly hip-to-ankle reach) so there is no
         knee flexion left to walk with, and a posture cost pulling toward it
         fights the gait. hip -0.25 / knee +0.70 / ankle -0.45 keeps the soles
         flat, drops the pelvis to 0.883, and leaves 40 deg of knee. -->
    <key name="home"
         qpos="0 0 0.936 1 0 0 0
               0 0 0 0 0 0
               0 0 0 0 0 0
               0 0 0
               0 0
               0 0 0 0 0
               0 0 0 0 0"/>
    <key name="walk_ready"
         qpos="0 0 0.883 1 0 0 0
               0 0 -0.25 0.7 -0.45 0
               0 0 -0.25 0.7 -0.45 0
               0 0 0
               0 0
               0 0 0 0 0
               0 0 0 0 0"/>
  </keyframe>
</mujoco>
"""


def gear_for(name, table):
    for prefix, g in table:
        if name.startswith(prefix):
            return g
    raise ValueError(f"no gear for {name}")


JOINT_ORDER = [
    "hip_yaw_l", "hip_roll_l", "hip_pitch_l", "knee_l", "ankle_pitch_l", "ankle_roll_l",
    "hip_yaw_r", "hip_roll_r", "hip_pitch_r", "knee_r", "ankle_pitch_r", "ankle_roll_r",
    "waist_yaw", "waist_pitch", "waist_roll", "neck_pitch", "neck_yaw",
    "shoulder_pitch_l", "shoulder_roll_l", "shoulder_yaw_l", "elbow_l", "wrist_l",
    "shoulder_pitch_r", "shoulder_roll_r", "shoulder_yaw_r", "elbow_r", "wrist_r",
]

joint_act = "\n".join(
    f'    <motor joint="{j}" name="{j}" gear="{gear_for(j, JOINT_GEAR)}" ctrlrange="-1 1"/>'
    for j in JOINT_ORDER)
def cable_fmax(w):
    return DRUM_FMAX[w] if w in DRUM_FMAX else gear_for(w, CABLE_FMAX)


cable_act = "\n".join(
    f'    <motor tendon="{w}" name="{w}" gear="{-cable_fmax(w)}" ctrlrange="0 1"/>'
    for w in wires)

def emit(lengths=None):
    with open(OUT, "w") as f:
        f.write(xml.replace("__TENDONS__", render_tendons(lengths, wrapped=False))
                   .replace("__ACTUATORS__", joint_act))
    with open(OUT_CABLE, "w") as f:
        f.write(xml.replace("__TENDONS__", render_tendons(lengths))
                   .replace("__ACTUATORS__", cable_act)
                    .replace('model="humanoid27_tensegrity"',
                             'model="humanoid27_tensegrity_cable"'))


# Pass 1: straight-line rest lengths. Pass 2: recompute from the TRUE lengths
# MuJoCo reports at the home pose — a wrapped tendon is longer than the
# straight line between its anchors, so without this the drum cables would sit
# at the wrong prestress (and could start slack).
emit()
import mujoco as _mj
_m = _mj.MjModel.from_xml_path(OUT_CABLE)
_d = _mj.MjData(_m)
_mj.mj_resetDataKeyframe(_m, _d, 0)
_mj.mj_forward(_m, _d)
_true = {_mj.mj_id2name(_m, _mj.mjtObj.mjOBJ_TENDON, i): float(_d.ten_length[i])
         for i in range(_m.ntendon)}
emit(_true)

n_weave = len(tendons) - len(wires)
n_struts = sum(len(v) for v in STRUTS.values())
print(f"wrote {OUT} (nu=27 joint torque)")
print(f"wrote {OUT_CABLE} (nu={len(wires)} cable tension)")
print(f"  {n_struts} struts, {len(tendons)} cables "
      f"({len(wires)} joint/actuatable, {n_weave} prism-cell), twist {TWIST} deg")
