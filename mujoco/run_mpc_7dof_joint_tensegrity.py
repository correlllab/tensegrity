#!/usr/bin/env python3
"""MPC end-effector tracker for a 7-DoF arm with joint-local tensegrity networks.

Model assumptions:
- rigid upper/lower links,
- tensegrity-style passive cable networks only at shoulder/elbow/wrist joints,
- antagonistic tendon actuation pairs as proxy for proximal motors + cable routing.
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


JOINTS = [
    "shoulder_yaw",
    "shoulder_pitch",
    "shoulder_roll",
    "elbow_flex",
    "forearm_roll",
    "wrist_pitch",
    "wrist_yaw",
]


@dataclass
class MpcConfig:
    horizon: int = 9
    samples: int = 95
    sigma: float = 0.16
    replan_every: int = 2
    cocontraction: float = 0.12
    w_pos: float = 140.0
    w_ctrl: float = 0.30
    w_vel: float = 0.05


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
    # MuJoCo Python enums vary across versions; try available knobs.
    rnd = getattr(mujoco, "mjtRndFlag", None)
    if rnd is not None and hasattr(rnd, "mjRND_TENDON"):
        viewer.user_scn.flags[rnd.mjRND_TENDON] = 1
    vis = getattr(mujoco, "mjtVisFlag", None)
    if vis is not None and hasattr(vis, "mjVIS_TENDON"):
        viewer.opt.flags[vis.mjVIS_TENDON] = 1


def clamp_ctrl(x: np.ndarray) -> np.ndarray:
    return np.clip(x, -1.0, 1.0)


def virtual_to_actuator(u: np.ndarray, cocontraction: float) -> np.ndarray:
    # Map 7 virtual joint commands to 14 antagonistic actuators.
    ctrl = np.zeros(14, dtype=float)
    for i in range(7):
        ctrl[2 * i] = cocontraction + u[i]
        ctrl[2 * i + 1] = cocontraction - u[i]
    return clamp_ctrl(ctrl)


def copy_state(model: mujoco.MjModel, src: mujoco.MjData, dst: mujoco.MjData) -> None:
    dst.time = src.time
    dst.qpos[:] = src.qpos
    dst.qvel[:] = src.qvel
    if model.na > 0:
        dst.act[:] = src.act
    dst.ctrl[:] = src.ctrl
    mujoco.mj_forward(model, dst)


def target_position(t: float, center: np.ndarray, radius: float, period: float, moving: bool) -> np.ndarray:
    if not moving:
        return center
    omega = 2.0 * math.pi / period
    return center + np.array([radius * math.sin(omega * t), 0.0, radius * math.cos(omega * t)], dtype=float)


def rollout_cost(
    model: mujoco.MjModel,
    seed_data: mujoco.MjData,
    ee_site_id: int,
    u: np.ndarray,
    cfg: MpcConfig,
    target: np.ndarray,
) -> float:
    d = mujoco.MjData(model)
    copy_state(model, seed_data, d)

    ctrl = virtual_to_actuator(u, cfg.cocontraction)
    cost = 0.0
    for _ in range(cfg.horizon):
        d.ctrl[:] = ctrl
        mujoco.mj_step(model, d)
        ee = d.site_xpos[ee_site_id]
        e = ee - target
        cost += cfg.w_pos * float(np.dot(e, e))
        cost += cfg.w_ctrl * float(np.dot(u, u))
        cost += cfg.w_vel * float(np.dot(d.qvel, d.qvel))
    return cost


def choose_mpc_action(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ee_site_id: int,
    u_prev: np.ndarray,
    cfg: MpcConfig,
    target: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    candidates = [u_prev.copy()]
    noise = rng.normal(0.0, cfg.sigma, size=(cfg.samples - 1, 7))
    for i in range(cfg.samples - 1):
        candidates.append(clamp_ctrl(u_prev + noise[i]))

    best_u = candidates[0]
    best_c = rollout_cost(model, data, ee_site_id, best_u, cfg, target)
    for u in candidates[1:]:
        c = rollout_cost(model, data, ee_site_id, u, cfg, target)
        if c < best_c:
            best_u = u
            best_c = c
    return best_u


def run_sim(args: argparse.Namespace) -> None:
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)

    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee")
    if ee_site_id < 0:
        raise ValueError("Site 'ee' not found in model.")
    if model.nu != 14:
        raise ValueError(f"Expected 14 actuators for 7 antagonistic pairs, got {model.nu}.")

    cfg = MpcConfig(
        horizon=args.horizon,
        samples=args.samples,
        sigma=args.sigma,
        replan_every=args.replan_every,
        cocontraction=args.cocontraction,
        w_pos=args.w_pos,
        w_ctrl=args.w_ctrl,
        w_vel=args.w_vel,
    )

    center = np.array([args.target_x, 0.0, args.target_z], dtype=float)
    rng = np.random.default_rng(args.seed)
    u_cmd = np.zeros(7, dtype=float)

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
                "Try: ../.venv/bin/mjpython run_mpc_7dof_joint_tensegrity.py --viewer"
            ) from exc

    for k in range(steps):
        t = data.time
        tgt = target_position(t, center, args.target_radius, args.target_period, args.moving_target)

        if (k % cfg.replan_every) == 0:
            u_cmd = choose_mpc_action(model, data, ee_site_id, u_cmd, cfg, tgt, rng)

        data.ctrl[:] = virtual_to_actuator(u_cmd, cfg.cocontraction)
        mujoco.mj_step(model, data)

        ee = data.site_xpos[ee_site_id]
        errs.append(float(np.linalg.norm(ee - tgt)))

        if viewer is not None:
            viewer.sync()

    if viewer is not None:
        viewer.close()

    print(f"Finished. Mean EE error: {np.mean(errs):.4f} m, max EE error: {np.max(errs):.4f} m")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MPC EE tracker for 7-DoF joint-local tensegrity arm")
    p.add_argument("--model", default="arm_7dof_joint_tensegrity.xml")
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--no-viewer", dest="viewer", action="store_false")
    p.set_defaults(viewer=True)

    p.add_argument("--moving-target", action="store_true")
    p.add_argument("--target-x", type=float, default=0.22)
    p.add_argument("--target-z", type=float, default=0.80)
    p.add_argument("--target-radius", type=float, default=0.05)
    p.add_argument("--target-period", type=float, default=5.0)

    p.add_argument("--horizon", type=int, default=9)
    p.add_argument("--samples", type=int, default=95)
    p.add_argument("--sigma", type=float, default=0.16)
    p.add_argument("--replan-every", type=int, default=2)
    p.add_argument("--cocontraction", type=float, default=0.12)
    p.add_argument("--w-pos", type=float, default=140.0)
    p.add_argument("--w-ctrl", type=float, default=0.30)
    p.add_argument("--w-vel", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    _ensure_mjpython_on_macos(sys.argv)
    run_sim(parse_args())
