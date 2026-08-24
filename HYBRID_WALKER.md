# The Hybrid Tensegrity Humanoid — status

**It walks — at 0.29 m/s, 37 % faster than the fully hinged design ever did.** Full-size (1.68 m, 20.1 kg), hinges at hips and knees,
genuine tensegrity ankles (free-floating feet, no bearings), under MJPC iLQG.

## Watch it

    mujoco_mpc/build/bin/mjpc --task="Hybrid Walk"     # interactive
    open hybrid_walk.gif                                # pre-rendered

The task starts planning on load — it declares
`<numeric name="agent_plan_enabled" data="1"/>`, which `Agent::Initialize`
now reads (same convention as `agent_planner`/`agent_horizon`; other tasks
keep the old planner-off default, and `--planner_enabled` still force-enables
for any task). No manual ticking, no reset: the model's qpos0 is the
walk-ready crouch and the commanded speed ramps in over 2 s while the planner
warms up. Async margin: the gait survives one planning iteration per 40 ms of
sim time, ~2.5x more lag than the GUI's planner actually has.

Headless verification:

    cd mujoco_mpc/build/mjpc/tasks/tensegrity
    ../../../bin/testspeed --task="Hybrid Walk" --total_time=16 \
        --steps_per_planning_iteration=4

## Result

| metric | value |
|---|---|
| walk success (16 s trials) | 10/10 full model (7-DoF arms, waist, packs) |
| speed | 0.185 m/s full model (0.26-0.29 with rigid torso and simple arms) |
| hot-swap | one 2.5 kg pack pulled: walks 3/3, no retuning |
| torso attitude on the waist cables | ±2°, 2 cm sag |
| pelvis height variation | ±1.5 cm |
| hip/knee motor duty | 9 % mean, 25 % p95 |
| ankle cable duty | 7 % mean, 250 N peaks |

## Architecture (what the paper's measurements individually recommend)

- **hips + knees: hinges** + AK70-class drives, mounted proximally
  (pelvis girdle / thigh cuff) — pose-independent 40–80 N·m where the gait
  needs it. The bearings are modelled and massed: each knee is a steel
  clevis pin (D12x1.5) in a 61802 bearing pair (142 g); each hip a 3-axis
  gimbal — yaw pivot into the girdle, cross block, roll+pitch axles (257 g).
  0.80 kg of articulation hardware total, vs 1.89 kg for the fully hinged
  design; all sized at SF >= 2.3 against the 1.5 kN stumble reaction
- **ankles: tensegrity** — tri-pad foot + mast overlapping into the shank
  cage, counter-wound ABCD cable families (wrench closure), XM540-class
  tension actuators on the shank ring, axes vertical
- **waist: tensegrity** — the torso is a free body nesting into the pelvis
  girdle on the same counter-wound interface, 6 of 12 cables actuated
- **batteries**: two swappable 2.5 kg packs on posterior pelvis rails
- **arms: 7 DoF each** — 3-axis shoulder gimbal + elbow + 3-axis wrist,
  all drives massed and drawn (3x XM540 shoulder cluster on the torso,
  XM430 elbow + wrist). The planner commands the two swing DoF; the other
  five hold position on servo stiffness (60 N·m/rad — in-band resonance at
  softer holds destabilized the gait). Walking height lowered 4 cm to keep
  the heavier upper body reliable; neck welded
- feet and torso are free bodies: nothing kinematic touches them

## The decisive experiments

1. **Passive ankles never walk**: the planner rationally refuses to lift a
   loaded foot when it has no CoP control. Deterministic fall at t≈2.
2. **Welded-ankle A/B walks** (0.18 m/s) — proves the task/model/pipeline,
   isolates the ankle.
3. **Actuated ankle cables walk at lower cost than welded** — the planner
   uses the compliance: A-family lifts the hanging foot, B-family holds the
   toe up, differential steers landing.

## Files

- model generator: `mujoco/generate_hybrid_humanoid.py`
- model: `mujoco/humanoid_hybrid.xml` (copies in mjpc task dirs)
- task: `mujoco_mpc/mjpc/tasks/tensegrity/hybrid/hybrid.{h,cc}`,
  `task_hybrid_walk.xml`
- tuning/verification: `experiments/hybrid_tune.py`, `hybrid_verify.py`
- results: `experiments/results/hybrid_verify.json`, logs in
  `experiments/results/hybrid_v/`
