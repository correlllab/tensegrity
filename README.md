# Tensegrity Humanoid

Apparel-mounted tensegrity humanoid: MuJoCo models, MJPC walking, and
hardware co-design experiments.

## Layout

- `mujoco/` — MuJoCo models and model-generation scripts
- `experiments/` — experiment scripts (E3–E14, sweeps, plotting).
  Generated outputs land in `experiments/results/` (gitignored, regenerable).
- `paper/` — co-design paper and notes (LaTeX)
- `design/` — design assets
- `references/` — reference material
- `HYBRID_WALKER.md` — hybrid walker notes

## Rebuilding mujoco_mpc

The `mujoco_mpc/` directory (a clone of
[google-deepmind/mujoco_mpc](https://github.com/google-deepmind/mujoco_mpc)
with a custom tensegrity task) is not committed. To reconstruct it:

```sh
git clone https://github.com/google-deepmind/mujoco_mpc
cd mujoco_mpc
git checkout ff572a21e7c2bf9fda62e1862a758da7e9a8719b
git apply ../mujoco_mpc_tensegrity.patch
```

The patch adds `mjpc/tasks/tensegrity/` (task code, XML models, STL meshes)
and registers the task in `mjpc/tasks/tasks.cc`, `mjpc/agent.cc`,
`mjpc/app.cc`, `mjpc/testspeed.cc`, and `mjpc/CMakeLists.txt`.
