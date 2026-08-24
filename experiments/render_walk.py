#!/usr/bin/env python
"""Render a walking log to frames (and an animated GIF via PIL).

    .venv/bin/python experiments/render_walk.py <log.csv> <out.gif> [model]

No ffmpeg needed: PIL writes the GIF. ~12 fps keeps the file small.
"""
import os
import sys
import numpy as np
import mujoco
from PIL import Image

ROOT = "/Users/ncorrell/Downloads/tensegrity"


def main():
    csv = os.path.abspath(sys.argv[1])
    out = os.path.abspath(sys.argv[2])
    model = os.path.abspath(sys.argv[3] if len(sys.argv) > 3
                            else f"{ROOT}/mujoco/humanoid_hybrid.xml")
    t0 = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
    t1 = float(sys.argv[5]) if len(sys.argv) > 5 else 14.0
    fps = 12

    os.chdir(os.path.dirname(model))
    m = mujoco.MjModel.from_xml_path(model)
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

    frames = []
    with mujoco.Renderer(m, height=480, width=520) as r:
        for tt in np.arange(t0, t1, 1.0 / fps):
            i = int(np.argmin(np.abs(t - tt)))
            d.qpos[:] = qpos[i]
            mujoco.mj_forward(m, d)
            cam.lookat[:] = [float(d.qpos[0]), float(d.qpos[1]), 0.8]
            cam.distance, cam.azimuth, cam.elevation = 2.8, 118, -10
            r.update_scene(d, camera=cam, scene_option=opt)
            frames.append(Image.fromarray(r.render().copy()))
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0, optimize=True)
    print(f"{out}: {len(frames)} frames, {os.path.getsize(out)//1024} kB")


if __name__ == "__main__":
    main()
