#!/usr/bin/env python
"""Offscreen still of any model in the repo, settled first.

    .venv/bin/python experiments/snapshot.py mujoco/humanoid_cellgraph.xml out.png

Useful for looking at a structure without launching the interactive viewer,
and for checking that a hinge-free assembly is standing rather than lying on
the floor.
"""
import sys
import numpy as np
import mujoco

W, H = 1100, 900


def shot(path, out, seconds=1.0, azimuth=125.0, elevation=-12.0,
         distance=None, lookat=None):
    import os
    os.chdir(os.path.dirname(os.path.abspath(path)) or ".")
    # white background for paper figures: inject a flat white skybox
    xml = open(path).read()
    if "<asset>" in xml:
        xml = xml.replace("<asset>", '<asset><texture type="skybox" '
            'builtin="flat" rgb1="1 1 1" rgb2="1 1 1" width="32" '
            'height="32"/>', 1)
    else:
        xml = xml.replace("</mujoco>", '<asset><texture type="skybox" '
            'builtin="flat" rgb1="1 1 1" rgb2="1 1 1" width="32" '
            'height="32"/></asset></mujoco>')
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
    if not np.all(np.isfinite(d.qpos)):
        print("model diverged before the snapshot")
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    pts = d.xpos[1:] if m.nbody > 1 else d.xpos
    centre = 0.5 * (pts.max(axis=0) + pts.min(axis=0))
    span = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    cam.lookat[:] = lookat if lookat is not None else centre
    cam.distance = distance if distance is not None else max(1.9 * span, 0.9)
    cam.azimuth, cam.elevation = azimuth, elevation
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = True
    with mujoco.Renderer(m, height=H, width=W) as r:
        r.update_scene(d, camera=cam, scene_option=opt)
        img = r.render()
    try:
        from PIL import Image
        Image.fromarray(img).save(out)
    except ImportError:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.imsave(out, img)
    print(f"{out}  ({m.nbody-1} bodies, {m.ntendon} cables, {m.nv} DoF, "
          f"settled {seconds:.1f}s)")


if __name__ == "__main__":
    a = sys.argv[1:]
    shot(a[0], a[1] if len(a) > 1 else "snapshot.png",
         float(a[2]) if len(a) > 2 else 1.0)
