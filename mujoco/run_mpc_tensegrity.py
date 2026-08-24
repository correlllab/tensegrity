#!/usr/bin/env python3
"""Basic MPC tracker for a true tensegrity prism MuJoCo model.

Tracks one top node (s_t0) as an end-effector proxy by actuating three cross
cables in a 3-strut tensegrity prism.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

try:
    import mujoco.viewer as mj_viewer
except Exception:  # pragma: no cover
    mj_viewer = None


@dataclass
class MpcConfig:
    horizon: int = 16
    samples: int = 140
    sigma: float = 18.0
    replan_every: int = 3
    w_pos: float = 130.0
    w_ctrl: float = 0.0008
    w_vel: float = 0.002


def _ensure_mjpython_on_macos(argv: list[str]) -> None:
    if sys.platform != "darwin":
        return
    if "--no-viewer" in argv:
        return
    if os.environ.get("MUJOCO_MJ_RELAUNCHED") == "1":
        return
    if Path(sys.executable).name == "mjpython":
        return

    mjpython = Path(sys.executable).with_name("mjpython")
    if not mjpython.exists():
        return

    env = os.environ.copy()
    env["MUJOCO_MJ_RELAUNCHED"] = "1"
    os.execve(str(mjpython), [str(mjpython), *argv], env)


def _enable_tendon_visualization(viewer) -> None:
    rnd = getattr(mujoco, "mjtRndFlag", None)
    if rnd is not None and hasattr(rnd, "mjRND_TENDON"):
        viewer.user_scn.flags[rnd.mjRND_TENDON] = 1
    vis = getattr(mujoco, "mjtVisFlag", None)
    if vis is not None and hasattr(vis, "mjVIS_TENDON"):
        viewer.opt.flags[vis.mjVIS_TENDON] = 1


def copy_state(model: mujoco.MjModel, src: mujoco.MjData, dst: mujoco.MjData) -> None:
    dst.time = src.time
    dst.qpos[:] = src.qpos
    dst.qvel[:] = src.qvel
    if model.na > 0:
        dst.act[:] = src.act
    dst.ctrl[:] = src.ctrl
    mujoco.mj_forward(model, dst)


def clamp_ctrl(u: np.ndarray, ctrlrange: np.ndarray) -> np.ndarray:
    lo = ctrlrange[:, 0]
    hi = ctrlrange[:, 1]
    return np.clip(u, lo, hi)


def target_position(t: float, center: np.ndarray, radius: float, period: float, moving: bool) -> np.ndarray:
    if not moving:
        return center
    omega = 2.0 * math.pi / period
    return center + np.array(
        [radius * math.sin(omega * t), radius * math.cos(omega * t), 0.0],
        dtype=float,
    )


def rollout_cost(
    model: mujoco.MjModel,
    seed_data: mujoco.MjData,
    ee_site_id: int,
    u: np.ndarray,
    ctrlrange: np.ndarray,
    cfg: MpcConfig,
    target: np.ndarray,
) -> float:
    d = mujoco.MjData(model)
    copy_state(model, seed_data, d)

    ctrl = clamp_ctrl(u, ctrlrange)
    cost = 0.0
    for _ in range(cfg.horizon):
        d.ctrl[:] = ctrl
        mujoco.mj_step(model, d)
        ee = d.site_xpos[ee_site_id]
        e = ee - target
        cost += cfg.w_pos * float(np.dot(e, e))
        cost += cfg.w_ctrl * float(np.dot(ctrl, ctrl))
        cost += cfg.w_vel * float(np.dot(d.qvel, d.qvel))
    return cost


def choose_mpc_action(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ee_site_id: int,
    u_prev: np.ndarray,
    ctrlrange: np.ndarray,
    cfg: MpcConfig,
    target: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    candidates = [u_prev.copy()]
    noise = rng.normal(0.0, cfg.sigma, size=(cfg.samples - 1, model.nu))
    for i in range(cfg.samples - 1):
        candidates.append(clamp_ctrl(u_prev + noise[i], ctrlrange))

    best_u = candidates[0]
    best_cost = rollout_cost(model, data, ee_site_id, best_u, ctrlrange, cfg, target)
    for u in candidates[1:]:
        c = rollout_cost(model, data, ee_site_id, u, ctrlrange, cfg, target)
        if c < best_cost:
            best_u = u
            best_cost = c
    return best_u


def run_sim(args: argparse.Namespace) -> None:
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)

    ee_site_name = "s_t0"
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
    if ee_site_id < 0:
        raise ValueError(f"Site '{ee_site_name}' not found.")

    cfg = MpcConfig(
        horizon=args.horizon,
        samples=args.samples,
        sigma=args.sigma,
        replan_every=args.replan_every,
        w_pos=args.w_pos,
        w_ctrl=args.w_ctrl,
        w_vel=args.w_vel,
    )

    ctrlrange = model.actuator_ctrlrange.copy()
    rng = np.random.default_rng(args.seed)
    u_cmd = np.zeros(model.nu, dtype=float)

    center = np.array([args.target_x, args.target_y, args.target_z], dtype=float)
    steps = int(args.duration / model.opt.timestep)
    errs = []

    viewer = None
    if args.viewer:
        if mj_viewer is None:
            raise RuntimeError("mujoco.viewer is unavailable. Run with --no-viewer.")
        try:
            viewer = mj_viewer.launch_passive(model, data)
            _enable_tendon_visualization(viewer)
        except RuntimeError as exc:
            raise RuntimeError(
                "Viewer launch failed. On macOS, passive viewer requires mjpython. "
                "Try: ../.venv/bin/mjpython run_mpc_tensegrity.py --viewer"
            ) from exc

    for k in range(steps):
        t = data.time
        tgt = target_position(t, center, args.target_radius, args.target_period, args.moving_target)

        if (k % cfg.replan_every) == 0:
            u_cmd = choose_mpc_action(model, data, ee_site_id, u_cmd, ctrlrange, cfg, tgt, rng)

        data.ctrl[:] = clamp_ctrl(u_cmd, ctrlrange)
        mujoco.mj_step(model, data)

        ee = data.site_xpos[ee_site_id]
        errs.append(float(np.linalg.norm(ee - tgt)))

        if viewer is not None:
            viewer.sync()

    if viewer is not None:
        viewer.close()

    print(
        f"Finished. Mean EE error: {np.mean(errs):.4f} m, max EE error: {np.max(errs):.4f} m"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MPC control for tensegrity prism end-effector proxy")
    p.add_argument("--model", default="tensegrity_prism.xml")
    p.add_argument("--duration", type=float, default=12.0)
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--no-viewer", dest="viewer", action="store_false")
    p.set_defaults(viewer=True)

    p.add_argument("--moving-target", action="store_true")
    p.add_argument("--target-x", type=float, default=0.18)
    p.add_argument("--target-y", type=float, default=0.0)
    p.add_argument("--target-z", type=float, default=1.0)
    p.add_argument("--target-radius", type=float, default=0.035)
    p.add_argument("--target-period", type=float, default=4.5)

    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--samples", type=int, default=140)
    p.add_argument("--sigma", type=float, default=18.0)
    p.add_argument("--replan-every", type=int, default=3)
    p.add_argument("--w-pos", type=float, default=130.0)
    p.add_argument("--w-ctrl", type=float, default=0.0008)
    p.add_argument("--w-vel", type=float, default=0.002)
    p.add_argument("--seed", type=int, default=11)
    return p.parse_args()


if __name__ == "__main__":
    _ensure_mjpython_on_macos(sys.argv)
    run_sim(parse_args())
