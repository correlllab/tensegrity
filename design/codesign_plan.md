# 27-DoF Apparel-Mounted Tensegrity Humanoid — Hardware/Control Co-Design Plan

Goal: a 27-DoF humanoid whose load-bearing structure is tensegrity cells (struts + prestressed
tendons) indexed and held in place by pockets/channels of standard apparel, simulated in MuJoCo,
controlled by MJPC (predictive sampling / iLQG) for standing balance and walking, with actuator
choices grounded in real motor mass/size/torque so the simulation drives hardware selection.

## 1. Scale

Set by the apparel constraint: the robot must fit standard garment sizes.

| Parameter | Value | Rationale |
|---|---|---|
| Height | ~1.65 m (sole → head top) | adult S/M apparel |
| Total mass target | ~25 kg | motor + battery budget below; light enough for safe HRI |
| Thigh / shank length | 0.40 / 0.40 m | Winter anthropometry at 1.65 m |
| Upper arm / forearm | 0.28 / 0.24 m | " |
| Foot | 0.24 × 0.10 m | standard shoe size ~ EU 38 |

## 2. DoF budget (27)

| Group | DoF | Joints |
|---|---|---|
| Leg ×2 | 6 each (12) | hip yaw, hip roll, hip pitch, knee pitch, ankle pitch, ankle roll |
| Spine/waist | 3 | yaw, pitch, roll |
| Arm ×2 | 5 each (10) | shoulder pitch, shoulder roll, shoulder yaw, elbow pitch, wrist pronation |
| Neck | 2 | pitch, yaw |

This is the minimum set for 3D walking with arm swing and manipulation reach. No toe joints in
rev A (foot rocker geometry substitutes).

## 3. Mass budget (~27.6 kg)

Key architectural advantage: tendon transmission lets every motor mount **proximally** (pelvis
belt, torso vest, proximal thigh/shank/upper-arm cuffs), so distal segments are nearly massless —
low leg swing inertia, low reflected inertia at contact.

| Item | Mass | Location (apparel anchor) |
|---|---|---|
| Hip motors 6× | 4.8 kg | pelvis belt / hip girdle |
| Knee motors 2× | 1.4 kg | proximal thigh cuff |
| Ankle motors 4× (2/side) | 2.8 kg | proximal shank cuff (Bowden to ankle) |
| Spine motors 3× + shoulder motors 6× + neck 2× | 5.8 kg | torso vest |
| Elbow + wrist motors (2/side) | 1.4 kg | proximal upper-arm cuff |
| **Swappable battery packs 2 × 2.5 kg** (450 Wh each) | **5.0 kg** | posterior pelvis / lumbar belt |
| Compute + power electronics | 0.5 kg | torso |
| Tensegrity structure (struts, tendons, textile shell, inserts) | 5.9 kg | distributed |
| **Total** | **27.6 kg** | (25.1 kg with one pack out) |

The per-motor allowance (~0.8 kg for leg DoF, ~0.35–0.5 kg for arm DoF) is deliberately
conservative — it leaves room for antagonistic pairs at joints where we want active stiffness
modulation (see §6).

## 3b. Swappable battery packs

Two 2.5 kg packs (~450 Wh each at ~180 Wh/kg *pack* level — cell-level 240
Wh/kg less packaging, BMS, and case) in dovetail rails on the posterior
pelvis. Simulated as `battery_l` / `battery_r` zero-joint bodies welded to the
pelvis; pull one at runtime with `--remove-battery battery_l`.

Why two, why there:
- **Two is the minimum for true hot-swap** — one keeps the bus live while the
  other is out. A single larger pack means a full shutdown to change it.
- **Below the waist joint**, so the waist actuators never carry pack mass, and
  the CoM stays low.
- **Lumbar belt zone** — the classic backpack load-transfer point, and the one
  place on a garment already engineered to carry ~5 kg against the pelvis.
- **Symmetric about the sagittal plane**, so a single-pack-out condition is a
  small lateral CoM offset rather than a large one.

Verified in simulation (both models, stand + 60 N forward / 40 N lateral push):

| condition | mass | CoM shift vs. both packs | balance |
|---|---|---|---|
| both packs | 27.60 kg | — | PASS |
| one pack pulled | 25.10 kg | Δx +6.9 mm, Δy −5.8 mm, Δz +2.4 mm | PASS |

The lateral offset is 5.8 mm — 12% of a foot half-width — and the balance
controller absorbs it without re-tuning. **The robot can operate on one pack,
so hot-swap is a live-swap, not a shutdown.**

Runtime (450 Wh/pack; power figures are *estimates* pending bench measurement
of drivetrain efficiency and standby draw):

| condition | energy | quiet standing ~45 W | walking ~130 W |
|---|---|---|---|
| both packs | 900 Wh | ~20 h | ~7 h |
| one pack (swapping) | 450 Wh | ~10 h | ~3.5 h |

Open items: hot-swap electronics (ideal-diode ORing so packs share the bus
without circulating current, per-pack BMS, inrush limiting on insertion), and
whether the swap is user-facing (garment pocket zip) or service-only.

## 4. Torque sizing

Peak joint moments in human walking, normalized (Winter gait data), × 25 kg, × safety factor ≈2
for push recovery / stairs / uneven ground:

| Joint | Human peak (Nm/kg) | @25 kg | Design limit (Nm) |
|---|---|---|---|
| Hip pitch | 1.1 | 28 | 80 |
| Hip roll | 0.9 | 23 | 60 |
| Hip yaw | 0.3 | 8 | 40 |
| Knee pitch | 1.0 | 25 | 80 |
| Ankle pitch (push-off) | 1.5 | 38 | 70 |
| Ankle roll | 0.4 | 10 | 40 |
| Waist (each) | — | — | 60 |
| Shoulder (each) | 2 kg payload @ 0.55 m | — | 30 |
| Elbow | 2 kg payload @ 0.25 m | — | 20 |
| Wrist, neck | — | — | 10 |

These limits are encoded as `ctrlrange` in the M0 model, so MJPC can never command torque the
hardware couldn't deliver. **The co-design loop tightens these numbers**: MJPC torque traces
during walking/push-recovery tell us actual peak and RMS demand per joint → downsize motors where
utilization is low.

## 5. Motor catalog (candidates)

Specs are approximate, from public datasheets as of my knowledge date — **re-verify each before
purchase**. All are quasi-direct-drive (QDD) BLDC + planetary unless noted.

| Motor | Mass (kg) | Peak torque (Nm) | Rated (Nm) | ~Free speed | Notes |
|---|---|---|---|---|---|
| CubeMars/T-Motor AK60-6 | ~0.31 | ~9 | ~3 | ~420 rpm | arms, neck |
| CubeMars AK70-10 | ~0.52 | ~25 | ~8 | ~475 rpm | **default leg motor** |
| CubeMars AK80-9 | ~0.49 | ~18 | ~9 | ~390 rpm | alt. leg motor, faster |
| CubeMars AK10-9 | ~0.96 | ~48 | ~18 | ~340 rpm | hips if direct-heavy |
| CubeMars AK80-64 | ~0.85 | ~120 | ~48 | ~55 rpm | high-torque/slow option |
| MyActuator RMD-X6 | ~0.49 | ~20 | ~6 | — | integrated driver |
| MyActuator RMD-X8 | ~0.82 | ~32 | ~9 | — | integrated driver |
| Unitree GO-M8010-6 | ~0.53 | ~24 | ~8 | ~600 rpm | good torque density |
| Dynamixel XM540-W270 | ~0.17 | ~10.6 stall | — | slow | wrist/neck, integrated servo |

### Transmission: motor → spool → Bowden/textile channel → joint

The spool gives a *free extra reduction stage*: joint torque = motor torque × (joint moment arm r_j /
spool radius r_s), joint speed = motor speed × (r_s / r_j).

Example, AK70-10 with r_s = 12 mm, r_j = 40 mm (ratio 3.3):
- tendon force at peak motor torque: 25/0.012 ≈ 2000 N → cap at ~1500 N (3 mm braided Dyneema,
  breaking ~4000 N, SF 2.7) → **joint torque ≈ 60–80 Nm** ✓ covers every leg joint
- joint speed ≈ 50 rad/s × 0.3 ≈ 15 rad/s ✓ (walking needs < 8 rad/s at knee swing)

So a single 0.52 kg motor class covers all 12 leg DoF; arms run AK60-6 class at ~0.31 kg.

## 6. Antagonism strategy (key trade)

Full antagonistic pairs on all 27 DoF = 54 motors ≈ +10 kg. Not worth it. Recommendation:

- **Legs (12 DoF)**: single motor + spring-return / pretensioned elastic antagonist. Walking
  needs torque and speed, not fast stiffness modulation; the tensegrity prestress network already
  provides passive compliance and impact tolerance.
- **Spine + shoulders (9 DoF)**: true antagonistic pairs — this is where variable stiffness
  (co-contraction) pays off for manipulation and disturbance rejection, and it's the research
  contribution (hypotheses H1–H3 in the notes).
- **Elbow/wrist/neck (6 DoF)**: single motor + elastic return.

Motor count: 27 + 9 = 36 motors. Revisit after MJPC torque/stiffness traces from WP-C below.

### 6b. 102 cables ≠ 102 motors (verified)

The cable-driven simulation model has **102 actuated cables**, which is the size of the
*tension network*, not a bill of materials. A cable is Dyneema plus two anchors — grams —
and having many is what makes the structure a tensegrity. A motor is 0.3–0.96 kg. Driving
every cable would be **44.7 kg of motors on a 27.6 kg robot**: not an option.

`mujoco/motor_count.py` computes the real number. Three things collapse it:

1. **Loop drive.** One motor with a double-wound spool pays out one cable while hauling in
   its antagonist, so a single motor gives bidirectional torque about a DoF. → 1 motor per
   DoF, not 2.
2. **Passive prestress.** Undriven cables still do structural work as elastic prestress
   elements (spring + turnbuckle, tuned at assembly). 66 of the 102 are passive.
3. **Co-contraction only where it pays** — the second motor per DoF at spine + shoulders.

Result: **36 motors, 16.1 kg** — against the 16.2 kg motor line in the §3 budget, so the
architecture above is confirmed rather than revised. The simulation's 102 controls are a
*modeling* choice (it lets MJPC command each cable independently); hardware drives 36 of
them and pretensions the rest.

**Yaw drums (resolved).** For a single loop to serve a DoF it needs
`F_max × r_eff ≥ τ_need`. The yaw DoFs failed this badly — a cable helixed around a limb
has an inherently small moment arm about that limb's *own* long axis (16–25 mm), leaving
shoulder yaw 2.5× and wrist 2.1× short. Fixed structurally with **capstan drums** coaxial
with each yaw axis, where the moment arm *is* the drum radius and no longer depends on how
fat the limb cage is:

| joint | drum R | cable F_max | before | after |
|---|---|---|---|---|
| waist yaw | 55 mm | 1500 N | 1.0× | **6.7×** |
| neck yaw | 35 mm | 450 N | 1.6× short | **4.7×** |
| shoulder yaw | 45 mm | 1000 N | 2.5× short | **2.5×** |
| wrist | 35 mm | 500 N | 2.1× short | **1.7×** |

DoFs still tight for one loop dropped from 15 to 7, and none of those are yaw — they are
hip pitch (1.4×), knee/elbow (1.1×) and waist roll (1.0×), all fixable with a modestly
larger cage or higher cable tension rather than a new mechanism. Motor count is unchanged
at **36 / 16.1 kg**.

Two implementation notes that cost real debugging time:
- In hardware a drum is ONE cable of several turns on ONE motor spool. MuJoCo can only
  wrap a geom once and a single wrapped tendon loses contact at some joint angles, so the
  winding is modelled as 6 tendons spaced around the drum. That is why the cable model
  reports 114 actuators: still 36 motors.
- Tendon wrap engagement is **discontinuous**, which destroys the finite-difference
  derivatives iLQG needs. The joint-torque model therefore carries the drum *geometry* but
  not the drum *cables* (it drives its joints directly, so they do nothing for it). With
  them included, MJPC threw the robot across the floor at every planning rate.

**Packaging note:** the shoulder drum currently sits inside the AK60 motor cluster on the
upper arm — see `yaw_drums.png`. Real hardware needs that resolved.

## 7. Apparel integration

- Struts (carbon/glass rod, ~8–12 mm) live in indexed sleeve pockets; tendon channels are sewn
  Bowden-like sleeves with low-friction liner (PTFE tape) along principal load paths.
- Motor packs: pelvis belt, torso vest, limb cuffs — all standard garment anchor zones that
  already carry load in exosuit literature (Wyss soft exosuit load paths).
- Prestress is applied at donning time via cam/ratchet tensioners at cell boundaries; garment
  seams are load-rated along the reinforcement tape directions only.

## 8. Simulation staging

- **M0 (now): mass-true rigid skeleton.** 27 hinge DoF + free root, correct segment lengths,
  motor masses lumped at their *mounting* sites (not at joints!), torque limits from §4,
  armature ≈ reflected motor+spool inertia, joint damping ≈ tendon/channel losses.
  File: `mujoco/humanoid_27dof.xml`. Purpose: MJPC gait feasibility + actuator sizing.
- **M1: tendon transmission layer.** Replace joint torque actuators with MuJoCo spatial tendons
  over wrap geoms + spool actuators; joint-local tensegrity cells at shoulder/hip/spine
  (scale up the existing `arm_minicell_tensegrity.xml` pattern); antagonist pairs per §6.
- **M2: identified compliance.** Add Bowden friction/dead-zone, textile stretch as tendon
  elasticity, parameters from bench tests (WP5 of the proposal work packages).

## 9. MJPC integration

Official `google-deepmind/mujoco_mpc` (C++ GUI + predictive sampling / iLQG / gradient planners):

1. Build the GUI app (`mjpc` target) — in progress in `mujoco_mpc/`.
2. Add a custom task `TensegrityHumanoid`: copy the `humanoid` task (Stand/Walk residuals:
   torso height, balance/capture-point, CoM velocity tracking, gait cycle, control effort),
   point its task XML at `humanoid_27dof.xml`, adjust residual dimensions to 27 actuators.
3. For co-design sweeps (headless, scripted): rebuild with `-DMJPC_BUILD_GRPC_SERVICE=ON` and
   use the `mujoco_mpc` Python bindings (agent server) to run batches over motor/gear/spool
   parameters.

### Co-design loop metrics (per candidate hardware config)

- peak / RMS torque per joint vs. motor limit (thermal proxy) → motor selection
- cost of transport at target speed → battery sizing
- push-recovery envelope (max survivable impulse at pelvis) → antagonism/stiffness value
- distal mass sensitivity: re-run with ankle motors moved thigh→shank→ankle to price the
  Bowden-length vs swing-inertia trade

## 10. Risks

1. Random-shooting/sampling MPC degrades with tendon-coupled DoF (M1) — mitigate with
   joint-space abstraction retained as MJPC's internal model while the plant is tendon-driven.
2. Bowden friction at the knee-crossing ankle tendons (if ankle motors move to thigh) —
   keep ankle motors at shank in rev A.
3. Apparel anchor migration under cyclic load — bench fatigue protocol before M2 parameters.
4. Motor specs in §5 are approximate — verify datasheets before freezing the BOM.
