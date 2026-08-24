#!/usr/bin/env python
"""Render a walking log to an MP4 (ICRA video attachment).

    .venv/bin/python experiments/render_video.py <log.csv> <out.mp4> [t0 t1]

Replays logged qpos through the current model at 25 fps, tracking camera,
white background, tendons on.
"""
import os
import sys
import numpy as np
import mujoco
import imageio

ROOT = "/Users/ncorrell/Downloads/tensegrity"
W, H, FPS = 1280, 720, 25


def main():
    csv = os.path.abspath(sys.argv[1])
    out = os.path.abspath(sys.argv[2])
    t0 = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    t1 = float(sys.argv[4]) if len(sys.argv) > 4 else 15.5
    os.chdir(f"{ROOT}/mujoco")
    xml = open("humanoid_hybrid.xml").read().replace(
        "<asset>", '<asset><texture type="skybox" builtin="flat" '
        'rgb1="1 1 1" rgb2="1 1 1" width="32" height="32"/>', 1)
    xml = xml.replace('offwidth="1600" offheight="1200"',
                      f'offwidth="{W}" offheight="{H}"')
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    raw = np.genfromtxt(csv, delimiter=",", invalid_raise=False)
    raw = raw[~np.isnan(raw).any(axis=1)]
    t, qpos = raw[:, 0], raw[:, 1:1 + m.nq]

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = True

    wr = imageio.get_writer(out, fps=FPS, codec="libx264", quality=8,
                            macro_block_size=8)
    with mujoco.Renderer(m, height=H, width=W) as r:
        for tt in np.arange(t0, t1, 1.0 / FPS):
            i = int(np.argmin(np.abs(t - tt)))
            d.qpos[:] = qpos[i]
            mujoco.mj_forward(m, d)
            cam.lookat[:] = [float(d.qpos[0]) + 0.1, float(d.qpos[1]), 0.8]
            cam.distance, cam.azimuth, cam.elevation = 2.5, 120, -12
            r.update_scene(d, camera=cam, scene_option=opt)
            wr.append_data(r.render())
    wr.close()
    print(f"{out}: {(t1-t0):.1f} s at {FPS} fps")


if __name__ == "__main__":
    main()
