#!/usr/bin/env python
"""Render a film strip from a testspeed log.

    .venv/bin/python experiments/filmstrip.py results/hybrid/log_cXXX_t0.csv \
        out.png [model.xml] [t0] [t1] [nframes]

Replays logged qpos through the model offscreen and tiles frames left to
right. Works for any model whose log matches its nq.
"""
import os
import sys
import numpy as np
import mujoco

ROOT = "/Users/ncorrell/Downloads/tensegrity"
DEFAULT_MODEL = f"{ROOT}/mujoco/humanoid_hybrid.xml"
W, H = 480, 640


def main():
    csv, out = sys.argv[1], sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_MODEL
    t0 = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0
    t1 = float(sys.argv[5]) if len(sys.argv) > 5 else 11.0
    n = int(sys.argv[6]) if len(sys.argv) > 6 else 6

    model = os.path.abspath(model)
    csv = os.path.abspath(csv)
    out = os.path.abspath(out)
    os.chdir(os.path.dirname(model))
    # paper figures render on a white background: inject a flat white skybox
    xml = open(model).read().replace("<asset>", '<asset><texture type='
        '"skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1" width="32" '
        'height="32"/>', 1)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    raw = np.genfromtxt(csv, delimiter=",", invalid_raise=False)
    raw = raw[~np.isnan(raw).any(axis=1)]
    t = raw[:, 0]
    qpos = raw[:, 1:1 + m.nq]

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = True

    times = np.linspace(t0, t1, n)
    tiles = []
    with mujoco.Renderer(m, height=H, width=W) as r:
        for tt in times:
            i = int(np.argmin(np.abs(t - tt)))
            d.qpos[:] = qpos[i]
            mujoco.mj_forward(m, d)
            cam.lookat[:] = [float(d.qpos[0]), float(d.qpos[1]), 0.75]
            cam.distance, cam.azimuth, cam.elevation = 2.6, 115, -12
            r.update_scene(d, camera=cam, scene_option=opt)
            tiles.append(r.render().copy())
    strip = np.concatenate(tiles, axis=1)
    try:
        from PIL import Image
        Image.fromarray(strip).save(out)
    except ImportError:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.imsave(out, strip)
    print(f"{out}: {n} frames {t0}-{t1} s, x {qpos[0, 0]:.2f} -> "
          f"{qpos[int(np.argmin(np.abs(t - t1))), 0]:.2f} m")


if __name__ == "__main__":
    main()
