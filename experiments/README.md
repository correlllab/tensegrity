# Experiment suite

All experiments run against the *rigid-equivalent baseline* (same model,
tension network zeroed) or variants generated on the fly. Results land in
`results/` as JSON/CSV; `make_plots.py` renders `paper/figures/*.pdf` and
`make_numbers.py` emits `paper/numbers.tex`, a set of LaTeX macros the paper
`\input`s, so text and data cannot drift.

| script | experiment | headline result |
|---|---|---|
| e1_drop.py | drops 0.05-1.0 m, {cables,none}x{held,limp}, 3x prestress | **matched** comparison: cables *raise* peak GRF at every height; neutral when both are servo-held |
| e2_stiffness.py | endpoint stiffness vs prestress & co-contraction, 0.2 N differential probe | assembly prestress does not modulate stiffness (12-17 N/m); co-contraction does (14 -> 115 N/m) |
| e3_degradation.py | sever k cables (random, leg-only, adversarial), stand + push x4 directions | all random k<=8 survive; adversarial survives k=4, fails k=8 |
| e5_authority.py | Eq. (5) over gait poses and over the BOM actuation set | 10/27 DoFs short over the gait with all cables; 13/27 with the 36-motor BOM |
| e7_strut_sizing.py | axial loads from cable tensions; CFRP tube buckling sizing | struts not buckling-critical in gait; 0.54 kg total |
| run_batch_extended.py | baseline/distal/speed/payload sweeps, n=10 per point | baseline walks 7/10 at 0.21 m/s; no monotone payload limit to 20 kg |
| e6_tendon_limited_walk.py | re-plan with joint limits capped at the tendon-achievable envelope | does a gait exist inside what the cables can deliver? |
| member_specs.py | strut + rope specification from the simulated loads | 60 CFRP tubes 0.5 mm wall (1.03 kg), 1-4 mm UHMWPE (1.88 kg, of which 1.78 kg is terminations), 93 nodes (0.84 kg) |
| e9_hinge_audit.py | kinematic audit + wrench-closure LP + hinge-free variant | 27 hinge DoF, all 3 translations constrained per joint; 0/14 joints have cable force closure; hinge-free robot collapses |
| e10_tensegrity_joint.py | overlapping floating-strut joint testbed | force closure from 0.2H overlap; hinge-free joint attenuates impact 27-32%; prestress-stiffness set by geometric/elastic ratio, not topology |
| e11_physical_rerun.py | E1/E2 at series-elastic cable stiffness | conclusions hold and strengthen: network raises impact 110-197%; prestress beyond slack take-up moves stiffness -2% |
| tension_audit.py | logged torques -> bounded tension distribution | saturates 11% of the stride at 0 kg, 75% at 20 kg; asks for ~2.4 kN cable |
| payload_carry.py | static single-arm box carry | 5 kg/arm, motor-limited (cable at 44%) |
| e15_joint_design.py | cable-topology search for wrench closure | counter-wound cross pair (ABCD) closes coaxial and eccentric joints at the same cable count |
| e16_drive_sizing.py | drive sizing from joint moments at cage-radius arms | XM540/XM430 class suffices: 2.5 kg vs 12 kg of AK-class drives |
| hybrid_tune.py / hybrid_verify.py | MJPC Hybrid Walk gait search (early model iterations) | superseded by hybrid_baseline_verify.py for every paper number |
| hybrid_baseline_verify.py | full-model baseline n=10 + welded-ankle, passive-ankle and one-battery controls, with realtime factor, planner cost, duty and torso attitude | the paper's headline hybrid numbers, all from logged runs |
| e3b_random_multipush.py | random k=8 cuts under the adversary's four-push protocol | de-confounds E3's adversarial result |
| e21fix_horizon.py | reruns E21's horizon rows (case-insensitive log-name collision had reused the step-height logs) | |
| e25_longrun.py | 30 s continuous runs of the hinged walker, n=3 | the paper's "longest run" figure |
| e26_counterwound_chain.py | built ABCD interface: closure, co-contraction authority LP, passive 3-joint chain stand | backs the counter-wound joint claims |
| make_plots.py, make_numbers.py, walklog.py, paperview.py, view.py, snapshot.py, filmstrip.py, formfind.py | tooling | |

The hybrid walker (hinged hips/knees, tensegrity cable ankles):
```
.venv/bin/python mujoco/generate_hybrid_humanoid.py     # model -> mujoco/ + mjpc tasks
cmake --build mujoco_mpc/build --target mjpc testspeed  # once
mujoco_mpc/build/bin/mjpc --task="Hybrid Walk"          # watch it walk
```

Reproduce a full refresh:
```
.venv/bin/python experiments/e1_drop.py
.venv/bin/python experiments/e2_stiffness.py
.venv/bin/python experiments/e3_degradation.py
.venv/bin/python experiments/e5_authority.py
.venv/bin/python experiments/e7_strut_sizing.py
.venv/bin/python experiments/run_batch_extended.py
.venv/bin/python experiments/e6_tendon_limited_walk.py
.venv/bin/python experiments/payload_carry.py
.venv/bin/python experiments/tension_audit.py
.venv/bin/python experiments/member_specs.py
.venv/bin/python experiments/e9_hinge_audit.py
.venv/bin/python experiments/e10_tensegrity_joint.py
.venv/bin/python experiments/e11_physical_rerun.py
.venv/bin/python experiments/make_plots.py
.venv/bin/python experiments/make_numbers.py && (cd paper && latexmk -pdf codesign_paper.tex)
```

Notes.
* `testspeed` logs per-step CSV when `TESTSPEED_LOG` is set (time, qpos, qvel,
  ctrl).
* Walk success = min pelvis height > 0.55 m **and** > 1.2 m travelled in the
  3-15 s window. The height test alone passes a robot marching in place, which
  is the failure mode the analytic gait reference was introduced to fix.
* n = 10 trials per sweep point. The iLQG success boundary is stochastic:
  at n = 2-4 the Wilson interval spans most of the unit interval and no
  capability claim is separable from planner noise. Success rates are reported
  with intervals throughout.
* `member_specs.py` reports the largest modelling gap in the project: the
  cables are modelled at 600-2500 N/m but a rope of the specified rating is
  0.3-7.1 MN/m, and even the series-elastic element the transmission study
  assumes is ~46x stiffer than the model's springs.
* The design intent was a hinge-free machine whose prism cells are the
  joints. `e9_hinge_audit.py` shows the implementation is not that, and that
  the cable topology could not replace the bearings; `e10_tensegrity_joint.py`
  shows what geometry would (cages overlapped by >= 0.2 H) and what it buys.
* `e5_authority.py` is the load-bearing correction: Eq. (5) is optimistic
  unless it is evaluated over the poses the robot actually visits and over the
  cables a motor can actually reach.
