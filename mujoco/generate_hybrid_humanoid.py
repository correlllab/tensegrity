#!/usr/bin/env python
"""Generate the HYBRID full-size humanoid: hinges where torque is king,
tensegrity where compliance is king.

Architecture (the split the evidence supports):
  hips (3 DoF) and knees (1 DoF)  mechanical hinges + joint motors -- these
                                  joints need 40-80 N m of precisely phased
                                  torque, which is what bearings are good at
  ankles                          genuine tensegrity joints: the foot carries a
                                  mast whose top ring overlaps into the shank
                                  cage, joined by the counter-wound ABCD cable
                                  families of E15 (force+wrench closure, no
                                  bearing) -- compliance exactly where heel
                                  strike arrives
  shoulders (pitch) + elbows      small hinges for counter-swing
  waist, neck                     welded this iteration

Every segment is a prism cage; hip and knee joints carry decorative-but-real
joint cables (low stiffness) so the load-path story stays visible. Drives are
sized by E16: AK70-10 class at hips and knees (mounted proximally: pelvis
girdle and thigh cuff), XM540-class at the ankles (on the shank bottom ring,
spool axes VERTICAL), XM540 at shoulders/elbows.

The model is built AT the walk-ready crouch, so qpos = 0 is the home pose and
no keyframe gymnastics are needed. Feet are top-level free bodies (MuJoCo
allows free joints only there); everything else hangs off the pelvis tree.
"""
import os
import sys
import numpy as np
from scipy.optimize import lsq_linear

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "humanoid_hybrid.xml")
TASKDIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../mujoco_mpc/mjpc/tasks/tensegrity"))

TWIST = np.deg2rad(29.8)

# ----------------------------------------------------------------- dimensions
THIGH, SHANK = 0.40, 0.40
HIP_Y = 0.10                    # lateral hip offset
HIP_PITCH_BUILD = -0.35         # crouch, walk.cc sign convention
KNEE_BUILD = 0.70
PAD_R = 0.022
PLATE_Z = 0.044                 # foot body origin (plate centre) height
MAST_TOP = 0.165                # mast top ring height
MAST_R = 0.075
OVERLAP = 0.055                 # shank bottom ring sits this far below mast top
WAIST_OV = 0.055                # torso bottom ring nests into the pelvis cage
CAGE_R = 0.075                  # leg cage radius
PELV_R, PELV_H = 0.16, 0.16
TORSO_R, TORSO_H = 0.13, 0.42
ARM_L = 0.26

# ------------------------------------------------------------------- masses
M_STRUCT = dict(pelvis=0.8, torso=1.2, head=0.5, thigh=0.5, shank=0.5,
                foot=0.45, uarm=0.25, farm=0.18, hand=0.08)
M_AK70, M_XM540, M_XM430 = 0.52, 0.165, 0.082
M_BATT, M_COMPUTE = 5.0, 0.5
# articulation hardware, sized from the 1.5 kN stumble reaction (SF >= 2.3):
# knee = alloy-steel clevis pin D12x1.5 + 61802 bearing pair + 7075 lugs and
# housings; hip = 3-axis gimbal (yaw pivot into the pelvis, cross block, roll
# and pitch axles, bearing pair per axis)
M_KNEE_THIGH = 0.060      # clevis lugs + retainers, on the thigh
M_KNEE_SHANK = 0.082      # pin + bearings + housings, on the shank
M_HIP_PELVIS = 0.086      # yaw pivot + half the gimbal cross
M_HIP_THIGH = 0.171       # cross half + roll/pitch axles + bearings
STEEL = "0.68 0.70 0.74 1"
HOUSING = "0.45 0.47 0.52 1"
JOINT_GEAR = dict(hip_yaw=40, hip_roll=60, hip_pitch=80, knee=80,
                  shoulder_pitch=15, elbow=10)
# servo position-hold stiffness for the arm joints the walking planner does
# not command (an XM430/XM540 holding position IS a stiff spring); the
# planner drives only shoulder pitch and elbow (arm swing)
# An XM-class position hold is stiff -- and 6 N m/rad put the spring-held
# arm DoF at ~1.7 Hz, resonant with the gait cadence, which is what wrecked
# reliability. 25 N m/rad moves the mode to ~3.6 Hz, out of the gait band,
# and is closer to a real servo hold.
K_SERVO, C_SERVO = 60.0, 2.5

# ------------------------------------------------------------------- cables
K_ANKLE = 2400.0
K_WAIST = 2400.0
C_WAIST = 48.0                # per cable; soft end of the series-elastic
C_ANKLE = 48.0                  # range so the 20 ms planner rollout is stable
K_DECOR = 300.0                 # hip/knee joint cables (visual + centering)
C_DECOR = 2.0
PRE_DECOR = 0.015
ACTUATED_ANKLE = True
WELD_ANKLE = False   # A/B: rigid feet, no ankle cables, same nq for hinges


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


def leg_fk(hip, hip_pitch, knee):
    """walk.cc conventions: hip_pitch<0 flexes forward, knee>0 flexes,
    angles about +y. Returns knee and ankle positions."""
    t1 = -hip_pitch                       # thigh tilt from straight-down, +fwd
    kx = hip[0] + THIGH * np.sin(t1)
    kz = hip[2] - THIGH * np.cos(t1)
    t2 = t1 - knee
    ax = kx + SHANK * np.sin(t2)
    az = kz - SHANK * np.cos(t2)
    return np.array([kx, hip[1], kz]), np.array([ax, hip[1], az])


class X:
    """Tiny xml builder."""

    def __init__(self):
        self.rows = []

    def add(self, s, ind=0):
        self.rows.append("  " * ind + s)

    def text(self):
        return "\n".join(self.rows)


def cage_geoms(x, name, c0, c1, radius, phase, body_origin, ind,
               rgba="0.28 0.31 0.36 1", strut_r=0.008, site_prefix=None):
    """Prism cage between ring planes at c0 and c1 (world), emitted relative
    to body_origin. Returns the two world rings."""
    axis = np.array(c1) - np.array(c0)
    bot = ring(radius, c0, phase, axis)
    top = ring(radius, c1, phase + TWIST, axis)
    o = np.array(body_origin)
    for i in range(3):
        a, b = bot[i] - o, top[(i + 1) % 3] - o
        x.add(f'<geom type="capsule" size="{strut_r}" mass="0" contype="0" '
              f'conaffinity="0" rgba="{rgba}" '
              f'fromto="{a[0]:.5f} {a[1]:.5f} {a[2]:.5f} '
              f'{b[0]:.5f} {b[1]:.5f} {b[2]:.5f}"/>', ind)
    # the nine cell cables (two ring perimeters + three verticals). The cage
    # is rigid in this model, so these are visual-only tension members ---
    # without them the struts appear to float unsupported.
    for i in range(3):
        for a, b in ((bot[i] - o, bot[(i + 1) % 3] - o),
                     (top[i] - o, top[(i + 1) % 3] - o),
                     (bot[i] - o, top[i] - o)):
            x.add(f'<geom type="capsule" size="0.0016" mass="0" group="2" '
                  f'contype="0" conaffinity="0" rgba="0.72 0.13 0.10 1" '
                  f'fromto="{a[0]:.5f} {a[1]:.5f} {a[2]:.5f} '
                  f'{b[0]:.5f} {b[1]:.5f} {b[2]:.5f}"/>', ind)
    if site_prefix:
        for i in range(3):
            p = bot[i] - o
            x.add(f'<site name="{site_prefix}_b{i}" pos="{p[0]:.5f} '
                  f'{p[1]:.5f} {p[2]:.5f}" size="0.006"/>', ind)
            p = top[i] - o
            x.add(f'<site name="{site_prefix}_t{i}" pos="{p[0]:.5f} '
                  f'{p[1]:.5f} {p[2]:.5f}" size="0.006"/>', ind)
    return bot, top


def inertial(x, mass, com, I, ind):
    x.add(f'<inertial pos="{com[0]:.5f} {com[1]:.5f} {com[2]:.5f}" '
          f'mass="{mass:.4f}" fullinertia="{I[0,0]:.6f} {I[1,1]:.6f} '
          f'{I[2,2]:.6f} {I[0,1]:.6f} {I[0,2]:.6f} {I[1,2]:.6f}"/>', ind)


def cage_inertia(mass, radius, height):
    ixx = mass * (0.5 * radius ** 2 + height ** 2 / 12.0)
    izz = mass * radius ** 2
    return np.diag([ixx, ixx, izz])


def add_point_masses(I0, M0, com0, pts):
    """Fold point masses into (M, com, I about body origin)."""
    M, com = M0, M0 * np.array(com0, float)
    I = I0 + M0 * (float(np.dot(com0, com0)) * np.eye(3)
                   - np.outer(com0, com0))
    for m, p in pts:
        p = np.array(p, float)
        M += m
        com += m * p
        I += m * (float(p @ p) * np.eye(3) - np.outer(p, p))
    com /= M
    I -= M * (float(com @ com) * np.eye(3) - np.outer(com, com))
    return M, com, I


def motor_geom(x, mesh, pos, ind, vertical=True, rgba="0.32 0.33 0.36 1"):
    """AK-series pancake with its axis VERTICAL (disc lying flat)."""
    xy = '0 0 1 0 1 0' if vertical else '1 0 0 0 1 0'
    x.add(f'<geom type="mesh" mesh="{mesh}" mass="0" contype="0" '
          f'conaffinity="0" rgba="{rgba}" pos="{pos[0]:.5f} {pos[1]:.5f} '
          f'{pos[2]:.5f}" xyaxes="{xy}"/>', ind)


def xm540_geom(x, pos, ind):
    """XM540-class drive as a vertical-axis cylinder (33.5x58.5x44 mm)."""
    x.add(f'<geom type="cylinder" size="0.022 0.024" mass="0" contype="0" '
          f'conaffinity="0" rgba="0.20 0.21 0.24 1" '
          f'pos="{pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f}"/>', ind)


def main():
    # ---------------- geometry at the crouch
    ankle_z = 0.5 * (MAST_TOP + (MAST_TOP - OVERLAP))       # interface centre
    hip_z = ankle_z + THIGH * np.cos(HIP_PITCH_BUILD) \
        + SHANK * np.cos(HIP_PITCH_BUILD - (-KNEE_BUILD) * 0)  # placeholder
    # exact: ankle directly below hip by construction of the symmetric crouch
    drop = THIGH * np.cos(-HIP_PITCH_BUILD) + SHANK * np.cos(-HIP_PITCH_BUILD)
    hip_z = ankle_z + drop
    pelv_z = hip_z + 0.05                                   # pelvis origin
    torso_z = pelv_z + 0.30                                 # torso origin
    sh_z = torso_z + TORSO_H / 2                            # shoulder height
    head_z = sh_z + 0.14

    cables = []          # (name, site_a, site_b, k, c, prestress_N or None)
    x = X()
    x.add('<mujoco model="Hybrid Tensegrity Humanoid">')
    x.add('<compiler angle="radian" autolimits="true" meshdir="assets"/>', 1)
    x.add('<option timestep="0.004" integrator="implicitfast" '
          'solver="Newton" iterations="60"/>', 1)
    x.add('<visual><global offwidth="1600" offheight="1200" realtime="0.3"/>'
          '<map force="0.005"/></visual>', 1)
    x.add('<asset>', 1)
    x.add('<mesh name="ak70" file="ak70_10.stl"/>', 2)
    x.add('<mesh name="ak60" file="ak60_6.stl"/>', 2)
    x.add('<texture name="grid" type="2d" builtin="checker" '
          'rgb1="0.88 0.89 0.90" rgb2="0.96 0.96 0.97" '
          'width="512" height="512"/>', 2)
    x.add('<material name="grid" texture="grid" texrepeat="10 10"/>', 2)
    x.add('</asset>', 1)
    x.add('<default>', 1)
    x.add('<joint damping="1.5" armature="0.02" limited="true"/>', 2)
    x.add('<motor ctrllimited="true" ctrlrange="-1 1"/>', 2)
    x.add('<tendon rgba="0.82 0.10 0.08 1" width="0.0036"/>', 2)
    x.add('</default>', 1)

    x.add('<worldbody>', 1)
    x.add('<light pos="1.5 -1.5 3" dir="-0.35 0.35 -1" directional="true"/>', 2)
    x.add('<light pos="-2 1.5 2.5" dir="0.5 -0.4 -1" directional="true" '
          'diffuse="0.35 0.35 0.35"/>', 2)
    x.add('<geom name="floor" type="plane" size="8 8 0.1" material="grid" '
          'contype="1" conaffinity="1" friction="1.2 0.01 0.001"/>', 2)

    # ================= pelvis tree =================
    x.add(f'<body name="pelvis" pos="0 0 {pelv_z:.5f}">', 2)
    x.add('<freejoint name="root"/>', 3)
    hipL = np.array([0, +HIP_Y, hip_z])
    hipR = np.array([0, -HIP_Y, hip_z])
    o = np.array([0, 0, pelv_z])
    cage_geoms(x, "pelvis", (0, 0, pelv_z - PELV_H / 2),
               (0, 0, pelv_z + PELV_H / 2), PELV_R, 0.0, o, 3,
               strut_r=0.009, site_prefix="pelv")
    # hip motors: 6 AK70, vertical, around the pelvis bottom ring
    pm = []
    for k in range(6):
        th = 2 * np.pi * k / 6 + np.pi / 6
        p = np.array([(PELV_R + 0.045) * np.cos(th),
                      (PELV_R + 0.045) * np.sin(th), -PELV_H / 2 + 0.02])
        motor_geom(x, "ak70", p, 3)
        pm.append((M_AK70, p))
    # two swappable 2.5 kg packs on posterior-lateral rails, symmetric about
    # the sagittal plane (hot-swap: pulling one shifts the CoM only laterally)
    for nm, sy in (("battery_l", +0.085), ("battery_r", -0.085)):
        bp = np.array([-0.085, sy, 0.01])
        x.add(f'<body name="{nm}" pos="{bp[0]:.5f} {bp[1]:.5f} {bp[2]:.5f}">', 3)
        x.add(f'<geom name="{nm}_pack" type="box" size="0.045 0.048 0.055" '
              f'mass="{M_BATT / 2}" contype="0" conaffinity="0" '
              f'rgba="0.16 0.18 0.22 1"/>', 4)
        x.add(f'<geom type="box" size="0.047 0.006 0.055" mass="0.05" '
              f'contype="0" conaffinity="0" rgba="0.55 0.57 0.60 1" '
              f'pos="0 {0.052 if sy > 0 else -0.052:.3f} 0"/>', 4)
        x.add('</body>', 3)
    pm.append((M_COMPUTE, np.array([-0.06, 0, 0.10])))
    for side, hip in (("l", hipL), ("r", hipR)):
        p = hip - o
        x.add(f'<site name="hip_anchor_{side}" pos="{p[0]:.5f} {p[1]:.5f} '
              f'{p[2]:.5f}" size="0.007"/>', 3)
        # hip yaw pivot: vertical bearing housing dropping from the girdle
        x.add(f'<geom type="cylinder" size="0.011 0.022" mass="0" contype="0" '
              f'conaffinity="0" rgba="{HOUSING}" pos="{p[0]:.5f} {p[1]:.5f} '
              f'{p[2] + 0.026:.5f}"/>', 3)
        x.add(f'<geom type="cylinder" size="0.006 0.015" mass="0" contype="0" '
              f'conaffinity="0" rgba="{STEEL}" pos="{p[0]:.5f} {p[1]:.5f} '
              f'{p[2] + 0.012:.5f}"/>', 3)
        pm.append((M_HIP_PELVIS, p + np.array([0, 0, 0.02])))

    M, com, I = add_point_masses(cage_inertia(M_STRUCT["pelvis"], PELV_R,
                                              PELV_H), M_STRUCT["pelvis"],
                                 np.zeros(3), pm)
    inertial(x, M, com, I, 3)

    # ---------------- legs: thigh (3-hinge hip) -> shank (knee)
    leg_rings = {}
    for side, hip in (("l", hipL), ("r", hipR)):
        knee_w, ankle_w = leg_fk(hip, HIP_PITCH_BUILD, KNEE_BUILD)
        x.add(f'<body name="thigh_{side}" pos="{(hip - o)[0]:.5f} '
              f'{(hip - o)[1]:.5f} {(hip - o)[2]:.5f}">', 3)
        x.add(f'<joint name="hip_yaw_{side}" type="hinge" axis="0 0 1" '
              f'range="-0.8 0.8"/>', 4)
        x.add(f'<joint name="hip_roll_{side}" type="hinge" axis="1 0 0" '
              f'range="-0.6 0.6"/>', 4)
        x.add(f'<joint name="hip_pitch_{side}" type="hinge" axis="0 -1 0" '
              f'range="-1.8 1.0"/>', 4)
        bot, top = cage_geoms(x, f"thigh{side}", knee_w, hip, CAGE_R,
                              0.0, hip, 4, site_prefix=f"th_{side}")
        # hip gimbal: cross block at the joint centre, roll axle (x) and
        # pitch axle (y) through it -- the three hinge axes made visible
        x.add(f'<geom type="box" size="0.016 0.016 0.016" mass="0" '
              f'contype="0" conaffinity="0" rgba="{HOUSING}" '
              f'pos="0 0 0"/>', 4)
        x.add(f'<geom type="cylinder" size="0.006 0.032" mass="0" '
              f'contype="0" conaffinity="0" rgba="{STEEL}" '
              f'pos="0 0 0" xyaxes="0 0 1 0 1 0"/>', 4)
        x.add(f'<geom type="cylinder" size="0.006 0.032" mass="0" '
              f'contype="0" conaffinity="0" rgba="{STEEL}" '
              f'pos="0 0 0" xyaxes="0 0 1 1 0 0"/>', 4)
        kw_l = knee_w - hip
        # knee clevis lugs on the thigh bottom
        for sy in (+0.048, -0.048):
            x.add(f'<geom type="box" size="0.008 0.006 0.020" mass="0" '
                  f'contype="0" conaffinity="0" rgba="{HOUSING}" '
                  f'pos="{kw_l[0]:.5f} {kw_l[1] + sy:.5f} '
                  f'{kw_l[2] + 0.014:.5f}"/>', 4)
        # knee drive on the thigh bottom (proximal to the knee)
        km = [(M_HIP_THIGH, np.zeros(3)), (M_KNEE_THIGH, kw_l)]
        p = (bot[0] + bot[1]) / 2 - hip + np.array([0, 0, 0.05])
        motor_geom(x, "ak70", p, 4)
        km.append((M_AK70, p))
        M, com, I = add_point_masses(
            cage_inertia(M_STRUCT["thigh"], CAGE_R, THIGH),
            M_STRUCT["thigh"], (knee_w + hip) / 2 - hip, km)
        inertial(x, M, com, I, 4)

        x.add(f'<body name="shank_{side}" pos="{(knee_w - hip)[0]:.5f} '
              f'{(knee_w - hip)[1]:.5f} {(knee_w - hip)[2]:.5f}">', 4)
        x.add(f'<joint name="knee_{side}" type="hinge" axis="0 -1 0" '
              f'range="-0.1 2.2"/>', 5)
        sb, st = cage_geoms(x, f"shank{side}", ankle_w + [0, 0, MAST_TOP
                                                          - OVERLAP
                                                          - ankle_w[2]
                                                          + ankle_w[2] * 0],
                            knee_w, CAGE_R, np.pi / 3, knee_w, 5,
                            site_prefix=f"sh_{side}")
        leg_rings[side] = dict(shank_bot=sb, shank_top=st, knee=knee_w,
                               ankle=ankle_w)
        # knee axle and bearing housings (the shank side of the clevis)
        x.add(f'<geom type="cylinder" size="0.006 0.055" mass="0" '
              f'contype="0" conaffinity="0" rgba="{STEEL}" '
              f'pos="0 0 0" xyaxes="0 0 1 1 0 0"/>', 5)
        for sy in (+0.046, -0.046):
            x.add(f'<geom type="cylinder" size="0.012 0.008" mass="0" '
                  f'contype="0" conaffinity="0" rgba="{HOUSING}" '
                  f'pos="0 {sy:.5f} 0" xyaxes="0 0 1 1 0 0"/>', 5)
        # ankle drives: 3 XM540, vertical, on the shank bottom ring
        am = [(M_KNEE_SHANK, np.zeros(3))]
        for i in range(3):
            p = sb[i] - knee_w + np.array([0, 0, 0.06])
            xm540_geom(x, p, 5)
            am.append((M_XM540, p))
        M, com, I = add_point_masses(
            cage_inertia(M_STRUCT["shank"], CAGE_R, SHANK),
            M_STRUCT["shank"], (ankle_w + knee_w) / 2 - knee_w, am)
        inertial(x, M, com, I, 5)
        x.add('</body>', 4)      # shank
        x.add('</body>', 3)      # thigh
    x.add('</body>', 2)          # pelvis

    # ================= feet =================
    # WELD_ANKLE=True rigidly attaches each foot to its shank (A/B control
    # for isolating the passive-ankle dynamics); otherwise feet are top-level
    # free bodies held by the ankle cable interface.
    for side, hip in (("l", hipL), ("r", hipR)):
        ankle_w = leg_rings[side]["ankle"]
        fo = np.array([ankle_w[0], ankle_w[1], PLATE_Z])
        x.add(f'<body name="foot_{side}" pos="{fo[0]:.5f} {fo[1]:.5f} '
              f'{fo[2]:.5f}">', 2)
        x.add('<freejoint/>', 3)
        # anthropomorphic pad triangle: heel + two toes
        pads = [np.array([-0.09, 0.0, PAD_R - PLATE_Z]),
                np.array([+0.12, +0.05, PAD_R - PLATE_Z]),
                np.array([+0.12, -0.05, PAD_R - PLATE_Z])]
        for k, p in enumerate(pads):
            x.add(f'<geom name="pad_{side}{k}" type="sphere" size="{PAD_R}" '
                  f'mass="0" contype="1" conaffinity="1" '
                  f'friction="1.2 0.01 0.001" rgba="0.75 0.75 0.78 1" '
                  f'pos="{p[0]:.5f} {p[1]:.5f} {p[2]:.5f}"/>', 3)
        for a, b in ((0, 1), (1, 2), (2, 0)):
            pa, pb = pads[a], pads[b]
            x.add(f'<geom type="capsule" size="0.008" mass="0" contype="0" '
                  f'conaffinity="0" rgba="0.28 0.31 0.36 1" '
                  f'fromto="{pa[0]:.5f} {pa[1]:.5f} {pa[2]:.5f} '
                  f'{pb[0]:.5f} {pb[1]:.5f} {pb[2]:.5f}"/>', 3)
        # the ankle mast: three struts from the pads up to the top ring
        mast = ring(MAST_R, ankle_w - fo + [0, 0, MAST_TOP - ankle_w[2]],
                    np.pi / 6)
        mast = ring(MAST_R, np.array([0, 0, MAST_TOP - PLATE_Z]), np.pi / 6)
        for i in range(3):
            pa, pb = pads[i], mast[(i + 1) % 3]
            x.add(f'<geom type="capsule" size="0.007" mass="0" contype="0" '
                  f'conaffinity="0" rgba="0.28 0.31 0.36 1" '
                  f'fromto="{pa[0]:.5f} {pa[1]:.5f} {pa[2]:.5f} '
                  f'{pb[0]:.5f} {pb[1]:.5f} {pb[2]:.5f}"/>', 3)
        for i in range(3):
            p = mast[i]
            x.add(f'<site name="mast_{side}{i}" pos="{p[0]:.5f} {p[1]:.5f} '
                  f'{p[2]:.5f}" size="0.006"/>', 3)
            p = pads[i]
            x.add(f'<site name="plate_{side}{i}" pos="{p[0]:.5f} {p[1]:.5f} '
                  f'{p[2]:.5f}" size="0.005"/>', 3)
        I = cage_inertia(M_STRUCT["foot"], 0.09, MAST_TOP)
        inertial(x, M_STRUCT["foot"], np.array([0.02, 0, 0.03]), I, 3)
        x.add('</body>', 2)
    # ================= torso: free body on the tensegrity waist =========
    # the torso bottom ring nests into the pelvis cage (overlap WAIST_OV);
    # ABCD cables join them, A+B actuated -- same joint as the ankles
    x.add(f'<body name="torso" pos="0 0 {torso_z:.5f}">', 2)
    x.add('<freejoint/>', 3)
    ot = np.array([0, 0, torso_z])
    tor_b, tor_t = cage_geoms(
        x, "torso", (0, 0, pelv_z + PELV_H / 2 - WAIST_OV),
        (0, 0, torso_z + TORSO_H / 2), TORSO_R, np.pi / 3, ot, 3,
        strut_r=0.009, site_prefix="torso")
    tm = []
    for sy in (+1, -1):
        p = np.array([0, sy * (TORSO_R + 0.03), TORSO_H / 2 - 0.03])
        xm540_geom(x, p, 3)
        tm.append((M_XM540, p))
    # (torso inertial emitted after the arm loop so the shoulder drive
    # masses are included -- an explicit inertial overrides geom masses)
    # neck: welded = a rigid strut tripod from the torso top ring to the
    # head bottom ring (without it the head cage appears to float)
    neck_b = ring(TORSO_R, (0, 0, torso_z + TORSO_H / 2), np.pi / 3 + TWIST)
    neck_t = ring(0.07, (0, 0, head_z - 0.09), 0.0)
    for i in range(3):
        a, b = neck_b[i] - ot, neck_t[i] - ot
        x.add(f'<geom type="capsule" size="0.006" mass="0" contype="0" '
              f'conaffinity="0" rgba="0.28 0.31 0.36 1" '
              f'fromto="{a[0]:.5f} {a[1]:.5f} {a[2]:.5f} '
              f'{b[0]:.5f} {b[1]:.5f} {b[2]:.5f}"/>', 3)
    # head, welded
    x.add(f'<body name="head" pos="0 0 {head_z - torso_z:.5f}">', 3)
    cage_geoms(x, "head", (0, 0, head_z - 0.09), (0, 0, head_z + 0.09),
               0.07, 0.0, np.array([0, 0, head_z]), 4, strut_r=0.006)
    inertial(x, M_STRUCT["head"], np.zeros(3),
             cage_inertia(M_STRUCT["head"], 0.07, 0.18), 4)
    x.add('</body>', 3)
    # arms: 7 DoF each -- shoulder gimbal (yaw/roll/pitch), elbow, wrist
    # (yaw/pitch/roll). The walking planner actuates shoulder pitch and elbow
    # (counter-swing); the other five are servo-held (joint stiffness =
    # position-hold of their drives). All seven drives are massed and drawn:
    # 3x XM540 shoulder cluster on the torso, XM430 at elbow, 3x XM430 wrist.
    for side, sy in (("l", +1), ("r", -1)):
        shp = np.array([0, sy * (TORSO_R + 0.06), sh_z])
        # shoulder drive cluster on the torso (proximal)
        for k in range(3):
            p = np.array([0.03 * (k - 1), sy * (TORSO_R + 0.02),
                          sh_z - torso_z + 0.05])
            xm540_geom(x, p, 3)
            tm.append((M_XM540, p))
        # shoulder mount: bracket + diagonal brace from the torso cage to
        # the gimbal (the joint hardware must visibly hang off structure)
        x.add(f'<geom type="capsule" size="0.007" mass="0" contype="0" '
              f'conaffinity="0" rgba="0.28 0.31 0.36 1" '
              f'fromto="0 {sy * (TORSO_R - 0.05):.5f} '
              f'{sh_z - torso_z:.5f} 0 {sy * (TORSO_R + 0.06):.5f} '
              f'{sh_z - torso_z:.5f}"/>', 3)
        x.add(f'<geom type="capsule" size="0.005" mass="0" contype="0" '
              f'conaffinity="0" rgba="0.28 0.31 0.36 1" '
              f'fromto="0 {sy * (TORSO_R - 0.03):.5f} '
              f'{sh_z - torso_z - 0.11:.5f} 0 {sy * (TORSO_R + 0.055):.5f} '
              f'{sh_z - torso_z - 0.01:.5f}"/>', 3)
        x.add(f'<body name="upper_arm_{side}" pos="{shp[0]:.5f} '
              f'{shp[1]:.5f} {sh_z - torso_z:.5f}">', 3)
        x.add(f'<joint name="shoulder_yaw_{side}" type="hinge" axis="0 0 1" '
              f'range="-1.5 1.5" stiffness="{K_SERVO}" '
              f'damping="{C_SERVO}"/>', 4)
        x.add(f'<joint name="shoulder_roll_{side}" type="hinge" '
              f'axis="1 0 0" range="-1.4 1.4" stiffness="{K_SERVO}" '
              f'damping="{C_SERVO}"/>', 4)
        x.add(f'<joint name="shoulder_pitch_{side}" type="hinge" '
              f'axis="0 -1 0" range="-2.5 1.5"/>', 4)
        # shoulder gimbal hardware (like the hip, lighter)
        x.add(f'<geom type="box" size="0.012 0.012 0.012" mass="0" '
              f'contype="0" conaffinity="0" rgba="{HOUSING}" pos="0 0 0"/>', 4)
        x.add(f'<geom type="cylinder" size="0.0045 0.024" mass="0" '
              f'contype="0" conaffinity="0" rgba="{STEEL}" pos="0 0 0" '
              f'xyaxes="0 0 1 0 1 0"/>', 4)
        x.add(f'<geom type="cylinder" size="0.0045 0.024" mass="0" '
              f'contype="0" conaffinity="0" rgba="{STEEL}" pos="0 0 0" '
              f'xyaxes="0 0 1 1 0 0"/>', 4)
        cage_geoms(x, f"ua{side}", shp, shp + [0, 0, -ARM_L], 0.045,
                   0.0, shp, 4, strut_r=0.006, site_prefix=f"ua_{side}")
        M, com, I = add_point_masses(
            cage_inertia(M_STRUCT["uarm"], 0.045, ARM_L),
            M_STRUCT["uarm"], np.array([0, 0, -ARM_L / 2]),
            [(0.060, np.zeros(3))])              # gimbal hardware share
        inertial(x, M, com, I, 4)
        x.add(f'<body name="forearm_{side}" pos="0 0 {-ARM_L:.5f}">', 4)
        x.add(f'<joint name="elbow_{side}" type="hinge" axis="0 -1 0" '
              f'range="-2.4 0.05"/>', 5)
        fo = shp + [0, 0, -ARM_L]
        # elbow drive
        x.add(f'<geom type="cylinder" size="0.018 0.018" mass="0" '
              f'contype="0" conaffinity="0" rgba="0.20 0.21 0.24 1" '
              f'pos="0 0 0.02" xyaxes="0 0 1 1 0 0"/>', 5)
        cage_geoms(x, f"fa{side}", fo, fo + [0, 0, -ARM_L * 0.82], 0.038,
                   np.pi / 3, fo, 5, strut_r=0.005)
        M, com, I = add_point_masses(
            cage_inertia(M_STRUCT["farm"], 0.038, ARM_L * 0.82),
            M_STRUCT["farm"], np.array([0, 0, -ARM_L * 0.41]),
            [(M_XM430, np.array([0, 0, 0.02]))])
        inertial(x, M, com, I, 5)
        # wrist: 3-DoF cluster + hand
        x.add(f'<body name="hand_{side}" pos="0 0 {-ARM_L * 0.82:.5f}">', 5)
        x.add(f'<joint name="wrist_yaw_{side}" type="hinge" axis="0 0 1" '
              f'range="-1.5 1.5" stiffness="{K_SERVO}" '
              f'damping="{C_SERVO}"/>', 6)
        x.add(f'<joint name="wrist_pitch_{side}" type="hinge" '
              f'axis="0 -1 0" range="-1.0 1.0" stiffness="{K_SERVO}" '
              f'damping="{C_SERVO}"/>', 6)
        x.add(f'<joint name="wrist_roll_{side}" type="hinge" axis="1 0 0" '
              f'range="-0.8 0.8" stiffness="{K_SERVO}" '
              f'damping="{C_SERVO}"/>', 6)
        for k in range(3):
            x.add(f'<geom type="cylinder" size="0.014 0.014" mass="0" '
                  f'contype="0" conaffinity="0" rgba="0.20 0.21 0.24 1" '
                  f'pos="0 0 {-0.015 - 0.030 * k:.4f}"/>', 6)
        x.add(f'<geom type="capsule" size="0.012" mass="0" contype="0" '
              f'conaffinity="0" rgba="0.30 0.33 0.38 1" '
              f'fromto="0 0 -0.09 0 0 -0.15"/>', 6)
        M, com, I = add_point_masses(
            np.diag([1e-4, 1e-4, 1e-4]), M_STRUCT["hand"],
            np.array([0, 0, -0.10]),
            [(3 * M_XM430, np.array([0, 0, -0.045]))])
        inertial(x, M, com, I, 6)
        x.add('</body>', 5)      # hand
        x.add('</body>', 4)      # forearm
        x.add('</body>', 3)      # upper arm
    M, com, I = add_point_masses(cage_inertia(M_STRUCT["torso"], TORSO_R,
                                              TORSO_H), M_STRUCT["torso"],
                                 np.zeros(3), tm)
    inertial(x, M, com, I, 3)
    x.add('</body>', 2)          # torso

    x.add('</worldbody>', 1)

    # ================= cables =================
    # ankle interfaces: counter-wound ABCD (E15) between foot and shank
    for side in ("l", "r"):
        for i in range(3):
            cables += [
                (f"ank_{side}_A{i}", f"mast_{side}{i}", f"sh_{side}_b{i}",
                 K_ANKLE, C_ANKLE, None),
                (f"ank_{side}_B{i}", f"mast_{side}{i}", f"sh_{side}_t{i}",
                 K_ANKLE, C_ANKLE, None),
                (f"ank_{side}_C{i}", f"mast_{side}{i}",
                 f"sh_{side}_b{(i + 1) % 3}", K_ANKLE, C_ANKLE, None),
                (f"ank_{side}_D{i}", f"mast_{side}{i}",
                 f"sh_{side}_b{(i - 1) % 3}", K_ANKLE, C_ANKLE, None),
            ]
    # waist: counter-wound ABCD between pelvis rings and torso rings
    for i in range(3):
        cables += [
            (f"waist_A{i}", f"pelv_t{i}", f"torso_b{i}", K_WAIST, C_WAIST,
             None),
            (f"waist_B{i}", f"pelv_t{i}", f"torso_t{i}", K_WAIST, C_WAIST,
             None),
            (f"waist_C{i}", f"pelv_t{i}", f"torso_b{(i + 1) % 3}", K_WAIST,
             C_WAIST, None),
            (f"waist_D{i}", f"pelv_t{i}", f"torso_b{(i - 1) % 3}", K_WAIST,
             C_WAIST, None),
        ]
    # hip / knee joint cables: real but soft (the hinge carries the reaction;
    # these carry the story and a little centering stiffness)
    for side in ("l", "r"):
        for i in range(3):
            cables.append((f"hipc_{side}{i}", f"pelv_b{i}", f"th_{side}_t{i}",
                           K_DECOR, C_DECOR, "decor"))
            cables.append((f"kneec_{side}{i}", f"th_{side}_b{i}",
                           f"sh_{side}_t{i}", K_DECOR, C_DECOR, "decor"))

    x.add('<tendon>', 1)
    for name, a, b, k, c, tag in cables:
        x.add(f'<spatial name="{name}" stiffness="{k}" damping="{c}" '
              f'springlength="0 REST_{name}">', 2)
        x.add(f'<site site="{a}"/>', 3)
        x.add(f'<site site="{b}"/>', 3)
        x.add('</spatial>', 2)
    x.add('</tendon>', 1)

    # ================= actuators =================
    # Joint motors at the hinges; if ACTUATED_ANKLE, tension actuators on the
    # A and B ankle families as well (XM540 at an 8 mm spool ~ 250 N):
    # tensioning A lifts the hanging foot toward the shank (swing clearance),
    # tensioning B pitches the toe up (scuff prevention). Tension-only,
    # ctrl in [0,1], consistent with the paper's actuation convention.
    x.add('<actuator>', 1)
    for side in ("l", "r"):
        for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee"):
            x.add(f'<motor name="{j}_{side}" joint="{j}_{side}" '
                  f'gear="{JOINT_GEAR[j]}"/>', 2)
    for side in ("l", "r"):
        for j in ("shoulder_pitch", "elbow"):
            x.add(f'<motor name="{j}_{side}" joint="{j}_{side}" '
                  f'gear="{JOINT_GEAR[j]}"/>', 2)
    if ACTUATED_ANKLE:
        for side in ("l", "r"):
            for fam in ("A", "B"):
                for i in range(3):
                    x.add(f'<motor name="ankm_{side}{fam}{i}" '
                          f'tendon="ank_{side}_{fam}{i}" gear="-250" '
                          f'ctrlrange="0 1"/>', 2)
        for fam in ("A", "B"):
            for i in range(3):
                x.add(f'<motor name="waistm_{fam}{i}" '
                      f'tendon="waist_{fam}{i}" gear="-250" '
                      f'ctrlrange="0 1"/>', 2)
    x.add('</actuator>', 1)

    x.add('</mujoco>')
    xml = x.text()

    # ---------------- rest lengths: measure, then prestress
    import mujoco
    os.chdir(os.path.dirname(OUT))
    probe = xml
    for name, *_ in [(c[0],) for c in cables]:
        probe = probe.replace(f"REST_{name}", "0.001")
    m = mujoco.MjModel.from_xml_string(probe)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    tname = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TENDON, t): t
             for t in range(m.ntendon)}

    # ankle prestress: per ankle, solve W T = [0,0,F_up,0,0,0] about the
    # ankle centre so the cables carry the leg's share of the body weight
    M_tot = float(m.body_mass.sum())
    F_up = (M_tot - 2 * M_STRUCT["foot"]) * 9.81 / 2.0
    T_of = {}
    for side in ("l", "r"):
        idx = [tname[f"ank_{side}_{f}{i}"] for f in "ABCD" for i in range(3)]
        origin = np.array([leg_rings[side]["ankle"][0],
                           leg_rings[side]["ankle"][1],
                           MAST_TOP - OVERLAP / 2])
        W = np.zeros((6, len(idx)))
        for j, t in enumerate(idx):
            adr = m.tendon_adr[t]
            sids = [int(m.wrap_objid[adr + q]) for q in range(m.tendon_num[t])]
            # wrench ON THE SHANK: pull at the shank site toward the mast site
            s_sh = [s for s in sids if
                    (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, s) or "")
                    .startswith("sh_")][0]
            s_ms = [s for s in sids if s != s_sh][0]
            v = d.site_xpos[s_ms] - d.site_xpos[s_sh]
            u = v / np.linalg.norm(v)
            W[:3, j] = u
            W[3:, j] = np.cross(d.site_xpos[s_sh] - origin, u)
        rhs = np.array([0, 0, F_up, 0, 0, 0])
        sol = lsq_linear(np.vstack([W, 0.02 * np.eye(len(idx))]),
                         np.concatenate([rhs, np.zeros(len(idx))]),
                         bounds=(8.0, 600.0), max_iter=400)
        resid = np.linalg.norm(W @ sol.x - rhs)
        print(f"ankle_{side}: prestress solve residual {resid:.1f} N of "
              f"{F_up:.0f} N, T in [{sol.x.min():.0f},{sol.x.max():.0f}] N")
        for j, t in enumerate(idx):
            T_of[t] = sol.x[j]

    # waist prestress: the cables carry the torso subtree (torso, head, arms,
    # shoulder drives) -- solve the same bounded wrench problem about the
    # waist centre, wrench ON THE TORSO
    M_upper = (M_STRUCT["torso"] + M_STRUCT["head"]
               + 2 * (M_STRUCT["uarm"] + M_STRUCT["farm"]) + 2 * M_XM540)
    F_up_w = M_upper * 9.81
    idx = [tname[f"waist_{f}{i}"] for f in "ABCD" for i in range(3)]
    origin = np.array([0.0, 0.0, pelv_z + PELV_H / 2 - WAIST_OV / 2])
    W = np.zeros((6, len(idx)))
    for j, t in enumerate(idx):
        adr = m.tendon_adr[t]
        sids = [int(m.wrap_objid[adr + q]) for q in range(m.tendon_num[t])]
        names_s = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, q) or ""
                   for q in sids]
        s_to = sids[[k2 for k2, nn in enumerate(names_s)
                     if nn.startswith("torso")][0]]
        s_pe = sids[[k2 for k2, nn in enumerate(names_s)
                     if nn.startswith("pelv")][0]]
        v = d.site_xpos[s_pe] - d.site_xpos[s_to]
        u = v / np.linalg.norm(v)
        W[:3, j] = u
        W[3:, j] = np.cross(d.site_xpos[s_to] - origin, u)
    rhs = np.array([0, 0, F_up_w, 0, 0, 0])
    sol = lsq_linear(np.vstack([W, 0.02 * np.eye(len(idx))]),
                     np.concatenate([rhs, np.zeros(len(idx))]),
                     bounds=(8.0, 600.0), max_iter=400)
    print(f"waist: prestress solve residual "
          f"{np.linalg.norm(W @ sol.x - rhs):.1f} N of {F_up_w:.0f} N, "
          f"T in [{sol.x.min():.0f},{sol.x.max():.0f}] N")
    for j, t in enumerate(idx):
        T_of[t] = sol.x[j]

    for name, a, b, k, c, tag in cables:
        t = tname[name]
        L = float(d.ten_length[t])
        if tag == "decor":
            rest = L * (1 - PRE_DECOR)
        else:
            rest = max(1e-4, L - T_of[t] / k)
        xml = xml.replace(f"REST_{name}", f"{rest:.6f}")

    open(OUT, "w").write(xml)
    m2 = mujoco.MjModel.from_xml_string(xml)
    print(f"\n{OUT}")
    print(f"  bodies {m2.nbody-1}  nq {m2.nq}  nv {m2.nv}  nu {m2.nu}  "
          f"tendons {m2.ntendon}  mass {m2.body_mass.sum():.1f} kg")
    print(f"  hinge qpos 7..{7 + 12 - 1}, feet qpos {7+12}..{m2.nq-1}")
    print(f"  pelvis z {pelv_z:.3f}  torso z {torso_z:.3f}  "
          f"head top {head_z + 0.09:.3f} m")

    # copy into the MJPC task tree
    task_copy = os.path.join(TASKDIR, "humanoid_hybrid.xml")
    open(task_copy, "w").write(xml)
    print(f"  copied -> {task_copy}")
    return xml


if __name__ == "__main__":
    main()
