# MuJoCo models: tendon-driven / tensegrity humanoid

## 27-DoF full humanoid — current focus

Target: full 27-DoF apparel-mounted tensegrity humanoid with MJPC balance and
walking. See `../design/codesign_plan.md` for the DoF budget, mass/torque
sizing, motor catalog, and staging.

- humanoid_27dof_tensegrity.xml: **joint-torque model** — 27 torque motors,
  MJPC-normalized (ctrl [-1,1], gear = torque limit in Nm).
- humanoid_27dof_tensegrity_cable.xml: **cable-driven model** — no joint
  motors; 114 tension-only cable motors (= 36 hardware motors, see codesign_plan 6b) (ctrl [0,1], gear = -F_max).
- Structure (both models): every segment is a Snelson 3-strut prism cage at a
  form-found 29.8 deg twist, and **every one of the 210 cables begins and ends
  on a strut endpoint** — nothing terminates in mid air (checked mechanically:
  all 348 cable sites coincide with a strut tip to 0 mm). Blue cables are each
  segment's own prism cell; white cables cross the joints and are the
  actuators. Load path, compression: strut -> node -> HINGE -> node -> strut;
  the cable network carries tension only and applies the joint moments.
  Full explanation, including three routing bugs this exposed:
  `../design/structure.md`. Renders: `../design/tensegrity_m1_render.png`,
  `../design/tensegrity_leg_detail.png`.
- **Yaw drums**: capstan drums (brass, `*_yawdrum_drum`) coaxial with the waist,
  neck, shoulder and wrist yaw axes. A helixed cable has almost no moment arm
  about a limb's own long axis; a drum's arm IS its radius. Modelled as 6
  wrapped tendons per drum (one multi-turn cable on one motor in hardware).
  The joint-torque model carries the drum GEOMS but not the drum CABLES —
  wrap engagement is discontinuous and destroys iLQG's derivatives.
  See `../design/yaw_drums.png`.
- Cables are TENSION-ONLY via a springlength deadband `"0 L_rest"`. A single
  springlength value is a bidirectional spring that can PUSH, which silently
  turns every cable into a strut.
- formfind_prism.py: form-finds a TRUE floating 3-strut tensegrity (struts
  touch nothing, held only by 9 cables). Converges to 29.8 deg from four start
  angles vs theory 30; also reports prestress -> cable tension. This is the
  reference cell the humanoid's cages are built from.
- check_cable_authority.py: per-DoF torque capacity of the cable network.
  Catches routing that leaves a DoF uncontrollable (a straight cable has ZERO
  moment arm about a yaw axis; one winding direction gives yaw one way only).
  Run it after ANY geometry change.
- generate_tensegrity_humanoid.py: parametric generator for BOTH models. Edit
  + re-run; never edit the XMLs by hand.
- Swappable batteries: two 2.5 kg / ~450 Wh packs (`battery_l`, `battery_r`)
  in dovetail rails on the posterior pelvis, below the waist joint. Modeled as
  zero-joint bodies welded to the pelvis so a pack can be pulled at runtime
  with `--remove-battery battery_l` — both models balance on one pack (lateral
  CoM shift only 5.8 mm), i.e. hot-swap is a LIVE swap. Total mass 27.6 kg
  (25.1 kg one-pack). See `../design/codesign_plan.md` 3b and
  `../design/battery_packs.png`.
- verify_humanoid27.py: joint-torque smoke test — PD + CoM-ankle-strategy
  stand, forward/lateral push recovery, torque utilization report.
- verify_cable_stand.py: cable-driven balance proof — the same PD law's joint
  torques distributed to 114 cable tensions by bounded least squares (tension
  only, co-contraction floor) at the full 250 Hz physics rate. A 62.5 Hz
  tension loop is unstable against the PD's ~15 Hz stiffness — a real
  bandwidth requirement for the spool drives.

```bash
../.venv/bin/python verify_humanoid27.py --model humanoid_27dof_tensegrity.xml
../.venv/bin/python verify_cable_stand.py
# hot-swap case (one pack pulled):
../.venv/bin/python verify_humanoid27.py --model humanoid_27dof_tensegrity.xml --remove-battery battery_l
../.venv/bin/python verify_cable_stand.py --remove-battery battery_l
```

NB when editing masses at runtime: `body_subtreemass` is a compile-time
derived constant that `subtree_com` divides by, so `mj_setConst` MUST be
called after changing `body_mass` (remove_battery does this). Without it every
CoM readout — the balance feedback here and MJPC's balance residuals — is
silently wrong (~85 mm low) while the mass matrix stays correct.

MJPC (official `google-deepmind/mujoco_mpc`) is built in `../mujoco_mpc/` with
three custom tasks (`mujoco_mpc/mjpc/tasks/tensegrity/`):

```bash
# --planner_enabled is REQUIRED: it defaults to false, and without it the
# robot gets zero control and collapses (or toggle Plan in the Agent panel,
# then press Backspace to reset).
# Force arrows are on by default; toggle with the Task > Visualize checkbox.
../mujoco_mpc/build/bin/mjpc --task="Tensegrity Stand" --planner_enabled        # 27 torques, iLQG
../mujoco_mpc/build/bin/mjpc --task="Tensegrity Walk" --planner_enabled         # 27 torques, iLQG
../mujoco_mpc/build/bin/mjpc --task="Tensegrity Cable Stand" --planner_enabled  # 114 tensions, sampling
```

Headless regression (no GUI; synchronous planning, prints cost + height):

```bash
../mujoco_mpc/build/bin/testspeed --task="Tensegrity Stand" --total_time=6 --steps_per_planning_iteration=8
```

### Stance shaping (Stand + Cable Stand)

Five cost terms shape the stance, all measured in the PELVIS frame so they keep
their meaning when the robot yaws (world x/y would silently reward a splayed
stance after a turn):

| term | residual | default norm |
|---|---|---|
| Stance Width | `target - lateral separation` | quadratic = hold the target |
| Stance Offset | fore/aft separation along heading | quadratic |
| Feet Parallel | vertical component of foot-x cross product | quadratic |
| Feet Flat | each foot's up-axis z, minus 1 | quadratic |
| Feet Square | foot forward dotted with pelvis lateral (toe-out) | quadratic |

`Stance Width` is a slider in the Task pane (`residual_Stance Width`, default
0.20 m, range 0.08-0.45). The SAME residual serves either intent — just change
the norm id in task_stand.xml:
- **`0` (quadratic, default)** holds the width AT the target. Measured 0.1997 m.
- **`8` (rectify, softplus)** makes it a MINIMUM and permits anything wider.
  Measured 0.36 m: nothing penalises splay, so the planner spreads out.

Measured effect on the joint-torque task (6 s, testspeed): foot gap 0.166 m
with the terms off (feet nearly touching) vs 0.200 m with them on, feet
parallel to 0.99999.

### Walk task — WALKING (straight line)

`walk.cc` tracks an analytic walking reference generated inside the residual:

- **Stateless phase clock** from `data->time * cadence` — identical in the
  plant and in every MJPC rollout.
- **Lateral pelvis sway reference** (LIPM-style weight shift), peaking over
  each stance foot just before the other lifts. The earlier cost-shaping
  version had this exactly backwards: a straight-line term PINNED the pelvis
  to y = 0, which forbids the weight shift walking requires; and the
  capture-point term forever centred the CoM between the feet. Both conflicts
  are gone.
- **Treadmill footstep targets** (duty 0.65, sine clearance arcs) fed through
  **closed-form flat-foot leg IK** (verified < 0.1 mm against model FK) into a
  27-joint posture reference, plus counter-swinging arms.
- **Capture-point (Raibert) landing bias** on the CoM velocity error, applied
  to the swing foot only, ramped in over the swing.
- Sliders: Torso height, Speed, Cadence, Step Height, Stance Width, Sway.

Measured (testspeed, iLQG, one planning iteration per 4 steps): **6.0 m in
29 s** at steady pelvis height 0.78 and feet parallel to 0.9999; three 14 s
runs gave x = 2.71 / 2.40 / 2.49 m (the cost-shaping version scored
-0.32 / +0.11 / +0.77). Still walks at one iteration per 8 steps (GUI-like
density: 1.96 m in 13 s).

Tuning notes (measured): commanded speed 0.30 walks at ~0.21 m/s actual;
0.45 falls. Sway 0.05 is the sweet spot — 0.04 falls (not enough weight
shift to unload the swing foot), 0.07 walks slightly worse. The `walk_ready`
keyframe (hip -0.25 / knee 0.70 / ankle -0.45) exists because at `home` the
legs are fully extended (pelvis 0.936 = exact hip-to-ankle reach), leaving no
knee flexion to walk with; do NOT make the crouch the global home — that broke
the cable-stand task once already.

Planner-robustness findings (testspeed): the stand task uses iLQG because it
keeps standing down to one planning iteration per 64 ms, while the sampling
planner needs one per ~12 ms and loses balance during the GUI's planner
warm-up. The cable task keeps the sampling planner, but at nu=102 it needs a planning
iteration EVERY physics step (250 Hz) to stay up — at 125 Hz it falls. That
runs 1.08x realtime, i.e. no margin. The fix is architectural, not tuning:
plan in the 27-DoF joint space and map to cable tensions underneath (the
least-squares distribution in verify_cable_stand.py already is that mapping).
"Tensegrity Walk" currently stands in place — the gait cost needs tuning
before it locomotes.

- motor_count.py / select_actuated_cables.py: how many MOTORS the network
  needs (not cables). Loop drive + passive prestress + selective
  co-contraction => 36 motors, 16.1 kg. Driving all cables would be 44.7 kg.
- motor_dimensioning.py: extracts per-joint torque/speed and per-cable
  tension/velocity/slew/stroke requirements from the balance sims and runs
  the spool-vs-lever transmission study. Results + recommendation:
  `../design/motor_dimensioning.md`.
- Force visualization: all three MJPC tasks draw live force arrows
  (Task::ModifyScene): cable tensions green->red at both anchors of every
  cable above 25 N (length 0.5 mm/N, red at 500 N), per-foot net ground
  reaction at the center of pressure (orange, 1.2 mm/N), and joint-actuator
  torques along hinge axes (cyan, torque model only). MuJoCo's built-in
  per-contact arrows are also available via the Rendering pane ("Contact
  Force"). render_forces.py mirrors the same overlay offscreen for stills.

The canonical models live here (generated); after regenerating, sync into the
task dir and rebuild (the build re-copies task assets):

```bash
../.venv/bin/python generate_tensegrity_humanoid.py
cp humanoid_27dof_tensegrity*.xml ../mujoco_mpc/mjpc/tasks/tensegrity/
cp assets/*.stl ../mujoco_mpc/mjpc/tasks/tensegrity/assets/
PATH="$PWD/../.venv/bin:$PATH" cmake --build ../mujoco_mpc/build --target mjpc -j 12
```

macOS build notes (already applied in `../mujoco_mpc/`): configure with
`-DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_OSX_ARCHITECTURES=arm64
-DCMAKE_CXX_FLAGS=-Wno-deprecated-builtins`, and the fetched abseil needs its
x86 HWAES flag list emptied in
`build/_deps/abseil-cpp-src/absl/copts/GENERATED_AbseilCopts.cmake`
(cmake de-duplicates the `-Xarch_x86_64` guards, leaking `-msse4.1` into the
arm64 compile).

# Earlier starter models: tendon-driven shoulder-elbow module

This directory contains a preliminary simulation model aligned with the proposal direction:
- tensegrity-inspired compliance concept,
- tendon-driven antagonistic actuation,
- textile-shell-compatible proximal actuation layout.

## File
- arm_module.xml: 2-DOF arm module with 4 tendon actuators (antagonistic pair per DOF).
- run_mpc_ee.py: basic receding-horizon MPC controller for end-effector tracking.
- tensegrity_prism.xml: true 3-strut tensegrity prism with prestressed passive tendons and 3 active cross tendons.
- run_mpc_tensegrity.py: MPC controller for the tensegrity prism end-effector proxy site.
- arm_7dof_joint_tensegrity.xml: 7-DoF arm with rigid upper/lower links and tensegrity-style cable networks localized at shoulder, elbow, and wrist joints.
- run_mpc_7dof_joint_tensegrity.py: MPC end-effector tracker for the 7-DoF joint-local tensegrity arm.
- arm_minicell_tensegrity.xml: shoulder + elbow mini-cell tensegrity arm (struts + wires) with minimal rigid coupler links.
- run_mpc_minicell_tensegrity.py: MPC end-effector tracker for the mini-cell tensegrity arm.

## Recommended (shoulder/elbow mini-cell tensegrity)

```bash
../.venv/bin/python run_mpc_minicell_tensegrity.py --viewer
```

Moving target:

```bash
../.venv/bin/python run_mpc_minicell_tensegrity.py --viewer --moving-target
```

## Recommended (7-DoF arm with joint-local tensegrity)

```bash
../.venv/bin/python run_mpc_7dof_joint_tensegrity.py --viewer
```

Moving target:

```bash
../.venv/bin/python run_mpc_7dof_joint_tensegrity.py --viewer --moving-target
```

## Recommended (tensegrity) run

```bash
../.venv/bin/python run_mpc_tensegrity.py --viewer
```

Moving target:

```bash
../.venv/bin/python run_mpc_tensegrity.py --viewer --moving-target
```

## Run it
From this directory, run:

```bash
../.venv/bin/python run_mpc_ee.py --viewer
```

On macOS, the script now auto-relaunches itself with `mjpython` when needed.
If your shell/environment blocks relaunch, run explicitly:

```bash
../.venv/bin/mjpython run_mpc_ee.py --viewer
```

Headless test run:

```bash
../.venv/bin/python run_mpc_ee.py --no-viewer --duration 6
```

Moving target tracking:

```bash
../.venv/bin/python run_mpc_ee.py --viewer --moving-target
```

## About MuJoCo MPC
- This environment did not provide an installable official MuJoCo MPC package.
- The controller in run_mpc_ee.py implements a lightweight MPC loop (random-shooting receding horizon) directly with MuJoCo Python, so you can still test end-effector MPC behavior now.
- The same lightweight MPC structure is used in run_mpc_tensegrity.py for the true tensegrity model.

## What this model is for
- Fast feasibility checks.
- Co-contraction and apparent stiffness experiments.
- Early controller prototyping before hardware build.

## Modeling assumptions in this first version
- Joint-level fixed tendons approximate routed tendon behavior.
- Friction/hysteresis in Bowden channels is not yet explicitly modeled.
- Textile deformation is represented implicitly via damping and actuator tuning.

## Suggested first experiments
1. Range-of-motion sweep under symmetric low control bounds.
2. Disturbance rejection test with increasing co-contraction.
3. Joint step-response characterization for shoulder and elbow.
4. Parameter sweep over actuator gear and joint damping.

## Immediate extensions
1. Replace fixed tendons with routed spatial tendons and wrap geometries.
2. Add friction and dead-zone elements in tendon transmission.
3. Add contact-rich tasks for endpoint interaction testing.
4. Add identified parameters from bench tests.
5. Swap the lightweight MPC with official MuJoCo MPC once the package/toolchain is available.

## Proposal-facing outputs to extract
- Effective joint stiffness vs co-contraction.
- Settling time and overshoot for joint tracking.
- Endpoint displacement under standardized load.
- Parameter set selected for hardware prototype A0.
