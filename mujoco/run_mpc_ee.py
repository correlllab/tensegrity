#!/usr/bin/env python3
"""Basic end-effector MPC controller for the tendon-driven arm module.

This script uses random-shooting receding-horizon MPC directly on top of the
MuJoCo Python API. It avoids external MPC dependencies while preserving the
core MPC loop: plan -> apply -> replan.
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
    horizon: int = 22
    samples: int = 180
    sigma: float = 0.18
    replan_every: int = 4
    cocontraction: float = 0.18
    w_pos: float = 90.0
    w_ctrl: float = 0.2
    w_vel: float = 0.08


def _ensure_mjpython_on_macos(argv: list[str]) -> None:
    """Relaunch under mjpython when viewer mode is requested on macOS.

    MuJoCo's launch_passive viewer backend requires mjpython on macOS. This
    helper makes `python run_mpc_ee.py --viewer` work transparently.
    """
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


def clamp_ctrl(x: np.ndarray) -> np.ndarray:
    return np.clip(x, -1.0, 1.0)


def virtual_to_actuator(u_sh: float, u_el: float, cocontraction: float) -> np.ndarray:
    # Antagonistic mapping for each hinge: [flex, ext].
    return clamp_ctrl(
        np.array(
            [
                cocontraction + u_sh,
                cocontraction - u_sh,
                cocontraction + u_el,
                cocontraction - u_el,
            ],
            dtype=float,
        )
    )


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
    sim_data = mujoco.MjData(model)
    copy_state(model, seed_data, sim_data)

    ctrl = virtual_to_actuator(float(u[0]), float(u[1]), cfg.cocontraction)
    cost = 0.0
    for _ in range(cfg.horizon):
        sim_data.ctrl[:] = ctrl
        mujoco.mj_step(model, sim_data)
        ee = sim_data.site_xpos[ee_site_id]
        pos_err = ee - target
        cost += cfg.w_pos * float(np.dot(pos_err, pos_err))
        cost += cfg.w_ctrl * float(np.dot(u, u))
        cost += cfg.w_vel * float(np.dot(sim_data.qvel, sim_data.qvel))
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
    noise = rng.normal(0.0, cfg.sigma, size=(cfg.samples - 1, 2))
    for i in range(cfg.samples - 1):
        candidates.append(clamp_ctrl(u_prev + noise[i]))

    best_u = candidates[0]
    best_cost = rollout_cost(model, data, ee_site_id, best_u, cfg, target)
    for u in candidates[1:]:
        c = rollout_cost(model, data, ee_site_id, u, cfg, target)
        if c < best_cost:
            best_u = u
            best_cost = c
    return best_u


def run_sim(args: argparse.Namespace) -> None:
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)

    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee")
    if ee_site_id < 0:
        raise ValueError("Site 'ee' not found in model.")

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
    u_cmd = np.zeros(2, dtype=float)

    steps = int(args.duration / model.opt.timestep)
    errors = []

    viewer = None
    if args.viewer:
        if mj_viewer is None:
            raise RuntimeError("mujoco.viewer is unavailable. Run with --no-viewer.")
        try:
            viewer = mj_viewer.launch_passive(model, data)
        except RuntimeError as exc:
            raise RuntimeError(
                "Viewer launch failed. On macOS, passive viewer requires mjpython. "
                "Try: ../.venv/bin/mjpython run_mpc_ee.py --viewer"
            ) from exc

    for k in range(steps):
        t = data.time
        target = target_position(t, center, args.target_radius, args.target_period, args.moving_target)

        if (k % cfg.replan_every) == 0:
            u_cmd = choose_mpc_action(model, data, ee_site_id, u_cmd, cfg, target, rng)

        data.ctrl[:] = virtual_to_actuator(float(u_cmd[0]), float(u_cmd[1]), cfg.cocontraction)
        mujoco.mj_step(model, data)

        ee = data.site_xpos[ee_site_id]
        err = float(np.linalg.norm(ee - target))
        errors.append(err)

        if viewer is not None:
            viewer.sync()

    if viewer is not None:
        viewer.close()

    mean_err = float(np.mean(errors)) if errors else float("nan")
    max_err = float(np.max(errors)) if errors else float("nan")
    print(f"Finished. Mean EE error: {mean_err:.4f} m, max EE error: {max_err:.4f} m")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Basic MPC end-effector controller for the arm module")
    p.add_argument("--model", default="arm_module.xml", help="Path to MuJoCo XML model")
    p.add_argument("--duration", type=float, default=12.0, help="Simulation duration in seconds")
    p.add_argument("--viewer", action="store_true", help="Launch interactive MuJoCo viewer")
    p.add_argument("--no-viewer", dest="viewer", action="store_false", help="Run headless")
    p.set_defaults(viewer=True)

    p.add_argument("--moving-target", action="store_true", help="Track circular moving target")
    p.add_argument("--target-x", type=float, default=0.12, help="Target center x position")
    p.add_argument("--target-z", type=float, default=0.52, help="Target center z position")
    p.add_argument("--target-radius", type=float, default=0.04, help="Moving target radius")
    p.add_argument("--target-period", type=float, default=5.0, help="Moving target period in seconds")

    p.add_argument("--horizon", type=int, default=22)
    p.add_argument("--samples", type=int, default=180)
    p.add_argument("--sigma", type=float, default=0.18)
    p.add_argument("--replan-every", type=int, default=4)
    p.add_argument("--cocontraction", type=float, default=0.18)
    p.add_argument("--w-pos", type=float, default=90.0)
    p.add_argument("--w-ctrl", type=float, default=0.2)
    p.add_argument("--w-vel", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


if __name__ == "__main__":
    _ensure_mjpython_on_macos(sys.argv)
    run_sim(parse_args())
