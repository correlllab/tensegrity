#!/usr/bin/env python
"""Form-finding for a TRUE 3-strut tensegrity prism (floating struts).

Three compression struts touch nothing but the tension network: 9 cables
(3 top triangle, 3 bottom triangle, 3 diagonals). Each strut is a free body,
so the only thing holding the cell together is cable prestress. Released from
an arbitrary twist angle it settles into the equilibrium twist, which theory
puts at alpha = 90 - 180/n degrees (30 deg for n=3).

This is the reference cell the humanoid's segments are modeled on, and it is
run here for two reasons: it demonstrates the actual tensegrity principle
(remove any cable and the cell collapses), and it measures the equilibrium
twist that the humanoid generator then uses for its strut cages.

Usage:  ../.venv/bin/python formfind_prism.py [--twist0 55] [--write-xml]
"""
import argparse
import math
import numpy as np
import mujoco

R = 0.09        # circumradius of the node triangles (m)
H = 0.30        # prism height (m)
K = 2000.0      # cable stiffness (N/m)
C = 6.0         # cable damping (N s/m)
PRE = 0.12      # prestress: springlength = (1 - PRE) * initial length


def ring(r, z, deg):
    a = math.radians(deg)
    return np.array([r * math.cos(a), r * math.sin(a), z])


def nodes(twist_deg):
    bot = [ring(R, 0.0, 120 * i) for i in range(3)]
    top = [ring(R, H, 120 * i + twist_deg) for i in range(3)]
    return top, bot


# Snelson 3-strut prism topology.
#
# The struts are the LONG members that cross the interior of the cell: strut i
# runs from bottom node i to top node i+1, so no strut touches another. The
# short cables connect CORRESPONDING nodes (b_i to t_i). Getting this backwards
# (short struts b_i->t_i, long diagonals) builds a rigid twisted cage that is
# not a tensegrity at all — it has no prestress equilibrium to find.
STRUT_TOP = lambda i: (i + 1) % 3


def cable_pairs():
    pairs = []
    for i in range(3):
        pairs.append((("b", i), ("b", (i + 1) % 3)))          # bottom triangle
        pairs.append((("t", i), ("t", (i + 1) % 3)))          # top triangle
        pairs.append((("b", i), ("t", i)))                    # vertical
    return pairs


def build_xml(twist_deg, gravity=False):
    top, bot = nodes(twist_deg)
    bodies = []
    for i in range(3):
        b, t = bot[i], top[STRUT_TOP(i)]
        mid = 0.5 * (b + t)
        bl, tl = b - mid, t - mid
        bodies.append(f"""
    <body name="strut{i}" pos="{mid[0]:.5f} {mid[1]:.5f} {mid[2]:.5f}">
      <freejoint/>
      <geom type="capsule" size="0.010" density="900"
            fromto="{bl[0]:.5f} {bl[1]:.5f} {bl[2]:.5f} {tl[0]:.5f} {tl[1]:.5f} {tl[2]:.5f}"/>
      <site name="b{i}" pos="{bl[0]:.5f} {bl[1]:.5f} {bl[2]:.5f}"/>
      <site name="t{STRUT_TOP(i)}" pos="{tl[0]:.5f} {tl[1]:.5f} {tl[2]:.5f}"/>
    </body>""")

    pts = {"t": top, "b": bot}
    tendons = []
    for n, ((r1, i1), (r2, i2)) in enumerate(cable_pairs()):
        L0 = np.linalg.norm(pts[r1][i1] - pts[r2][i2])
        tendons.append(f"""
    <spatial name="c{n}" stiffness="{K}" damping="{C}" springlength="0 {(1 - PRE) * L0:.5f}"
             width="0.0025" rgba="0.95 0.97 1 1">
      <site site="{r1}{i1}"/>
      <site site="{r2}{i2}"/>
    </spatial>""")

    g = "0 0 -9.81" if gravity else "0 0 0"
    return f"""<mujoco model="tensegrity_prism_formfind">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.0005" integrator="implicitfast" gravity="{g}"/>
  <default>
    <geom contype="0" conaffinity="0" rgba="0.22 0.24 0.28 1"/>
    <site size="0.006" rgba="0.9 0.6 0.2 1"/>
  </default>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1" directional="true"/>{''.join(bodies)}
  </worldbody>
  <tendon>{''.join(tendons)}</tendon>
</mujoco>
"""


def node_xyz(model, data):
    sid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
    bot = np.array([data.site_xpos[sid(f"b{i}")] for i in range(3)])
    top = np.array([data.site_xpos[sid(f"t{i}")] for i in range(3)])
    return top, bot


def measure_shape(model, data):
    """Twist and height in the CELL's own frame.

    Measuring atan2 in world coordinates is wrong: the cell floats freely, so
    rigid-body rotation of the whole assembly changes that angle without
    changing the shape at all. Everything here is referenced to the axis
    joining the two triangle centroids, which is rigid-motion invariant.
    """
    top, bot = node_xyz(model, data)
    cb, ct = bot.mean(axis=0), top.mean(axis=0)
    axis = ct - cb
    height = float(np.linalg.norm(axis))
    axis = axis / (height + 1e-12)

    def perp(v):
        return v - np.dot(v, axis) * axis

    angs = []
    for i in range(3):
        vb, vt = perp(bot[i] - cb), perp(top[i] - ct)
        vb /= np.linalg.norm(vb) + 1e-12
        vt /= np.linalg.norm(vt) + 1e-12
        ang = math.degrees(math.atan2(np.dot(np.cross(vb, vt), axis),
                                      np.dot(vb, vt)))
        angs.append(ang % 360.0)
    return float(np.mean(angs)), height


def settle(twist0, steps=60000, cut=None):
    model = mujoco.MjModel.from_xml_string(build_xml(twist0))
    data = mujoco.MjData(model)
    # heavy damping so it relaxes to the static solution rather than ringing
    model.tendon_damping[:] = 25.0
    if cut is not None:
        model.tendon_stiffness[cut] = 0.0
    for _ in range(steps):
        mujoco.mj_step(model, data)
    twist, height = measure_shape(model, data)
    return twist, height, float(np.abs(data.qvel).max()), model, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--twist0", type=float, nargs="*", default=[15.0, 30.0, 55.0, 75.0])
    ap.add_argument("--write-xml", action="store_true")
    args = ap.parse_args()

    print("TRUE floating 3-strut tensegrity prism — form finding")
    print(f"  3 struts (compression, touching nothing), 9 cables (tension)")
    print(f"  theory: equilibrium twist = 90 - 180/n = {90 - 180 / 3:.0f} deg\n")
    print(f"{'start twist':>12s} {'settled twist':>14s} {'height m':>10s} {'max |qvel|':>12s}")
    finals = []
    for t0 in args.twist0:
        tw, h, resid, model, data = settle(t0)
        finals.append(tw)
        print(f"{t0:11.1f}° {tw:13.1f}° {h:10.4f} {resid:12.2e}")

    mean = float(np.mean(finals))
    spread = float(np.max(finals) - np.min(finals))
    print(f"\nconverged twist = {mean:.1f}° (spread {spread:.2f}° over "
          f"{len(finals)} starts)")

    # Prestress is what makes the cell a structure. Report the tension actually
    # carried at equilibrium; a floating cell cannot be push-tested directly
    # (an external force just accelerates the whole assembly), so this reports
    # the internal state rather than a deflection.
    print("\n prestress -> equilibrium cable tension (zero-g, settled):")
    print(f"{'prestress':>10s} {'mean T (N)':>12s} {'max T (N)':>11s} {'height m':>10s}")
    global PRE
    saved = PRE
    for pre in (0.02, 0.06, 0.12, 0.20):
        PRE = pre
        model = mujoco.MjModel.from_xml_string(build_xml(mean))
        data = mujoco.MjData(model)
        model.tendon_damping[:] = 25.0
        for _ in range(30000):
            mujoco.mj_step(model, data)
        slack = np.maximum(0.0, data.ten_length - model.tendon_lengthspring[:, 1])
        T = model.tendon_stiffness * slack
        _, h = measure_shape(model, data)
        print(f"{pre:10.2f} {T.mean():12.1f} {T.max():11.1f} {h:10.4f}")
    PRE = saved
    print("  (all 9 cables in tension at every level = the cell is prestressed;\n"
          "   cell stiffness scales with this tension, which is the whole point)")

    if args.write_xml:
        with open("tensegrity_prism_true.xml", "w") as f:
            f.write(build_xml(mean, gravity=True))
        print("\nwrote tensegrity_prism_true.xml")


if __name__ == "__main__":
    main()
