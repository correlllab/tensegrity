# Motor Dimensioning & Transmission Study (spool vs lever arm)

Source: `mujoco/motor_dimensioning.py` — requirements extracted from the two
verified balance simulations (stand + 60 N forward / 40 N lateral pelvis push,
8 s, 250 Hz tension control). Numbers below are from the 2026-08-08 run at
**27.6 kg** (two 2.5 kg swappable battery packs; see codesign_plan.md 3b).
An earlier 25.1 kg run gave ~5-10% lower figures — re-run after any mass
change, the script reads the model.
**Scope caveat:** these are *balance* requirements; walking (especially ankle
push-off, human ~1.5 Nm/kg → ~500–750 N heel-cable tension) will raise the
ankle numbers — re-run this study on MJPC walking traces once gait works.

## 1. Requirements from simulation

Per cable group (peak over stand + both push recoveries):

| group | F_pk (N) | F_rms (N) | v_pk (m/s) | dF/dt (kN/s) | stroke (mm) |
|---|---|---|---|---|---|
| hip | 155 | 32 | 0.009 | 3.7 | **177** |
| knee | 255 | 35 | 0.020 | 9.6 | 122 |
| ankle | **453** | 70 | 0.096 | **64.4** | 94 |
| waist | 269 | 44 | 0.014 | 6.8 | 87 |
| shoulder | 28 | 20 | 0.004 | 0.3 | 228 |
| elbow/wrist/neck | ≤27 | ≤20 | ≤0.003 | ≤0.1 | ≤98 |

Leg design point (safety factor 1.5, swing-speed headroom): **F ≥ 680 N,
v ≥ 0.40 m/s, stroke ≥ 177 mm (hip) / 94 mm (ankle)**.

Two structural findings from the balance work feed in here:
- The tension loop needed the full **250 Hz** update rate in sim — a 62.5 Hz
  zero-order hold went unstable against the ~15 Hz joint-stiffness dynamics.
  The digital rate is trivial for FOC drives; the binding constraint is the
  *mechanical* force bandwidth below.
- The **ankle is the bandwidth-critical joint**: 64 kN/s tension slew at
  ~225 N amplitude ≈ **46 Hz** force-modulation requirement. Hips/knees need
  only ~5–10 Hz.

## 2. The spool-vs-lever trade

Force control happens through the series compliance k_ser of the tendon run
(cable + textile channel + anchors). The open-loop force bandwidth is the
resonance f = (1/2π)·√(k_ser/m_eff), where m_eff = I_rotor·G²/r² is the
motor's reflected mass at the cable. Closed-loop force feedback buys ~2–3×
beyond it, not more. Small radius (spool) ⇒ huge m_eff ⇒ low bandwidth;
large radius (lever) ⇒ bandwidth, but stroke ≤ R·(±60°) and force ∝ 1/R.

k_ser cases: **soft** = 40 kN/m (textile-channel-dominated run), **stiff** =
300 kN/m (short 3 mm Dyneema run, rigid anchors).

| motor | type | r (mm) | F_pk (N) | v (m/s) | stroke (mm) | m_eff (kg) | f_bw soft/stiff (Hz) | meets F/v/stroke |
|---|---|---|---|---|---|---|---|---|
| AK70-10 | spool | 12 | 2067 | 0.60 | ∞ | 83 | 3.5 / 9.5 | **yes** |
| AK70-10 | lever | 40 | 620 | 1.99 | 84 | 7.5 | 11.6 / 31.8 | no (stroke) |
| AK70-10 | lever | 50 | 496 | 2.49 | 105 | 4.8 | 14.5 / 39.8 | no (F, marginal) |
| AK60-6 | spool | 12 | 750 | 0.53 | ∞ | 15 | 8.2 / 22.5 | yes (arms) |
| AK60-6 | lever | 50 | 180 | 2.20 | 105 | 0.9 | 34.2 / 93.8 | arms only |

(Full grid in the script output. Rotor inertias are ~estimates — AK70-10
1.2e-4 kg·m², AK60-6 6e-5 kg·m² motor-side — **verify against datasheets**.)

## 3. Recommendation

- **Hips, knees, waist — spool (r = 12 mm) + AK70-10.** Stroke kills the
  lever here (177 mm needed vs ~105 mm max), and these joints only need
  5–10 Hz force modulation: a stiff, short tendon run (9.5 Hz open-loop,
  ~20–30 Hz closed-loop) covers it. Keep the runs short and anchor-stiff;
  the soft textile-channel case (3.5 Hz) does NOT cover it — channel
  stiffness is a first-class design requirement, not a detail.
- **Ankles — lever arm (R ≈ 50 mm), bigger motor.** The 46 Hz slew
  requirement is out of reach for any spool option even closed-loop; the
  lever reaches ~40 Hz open / ~100 Hz closed with a stiff run, and the 94 mm
  ankle stroke just fits R = 50 mm. AK70-10 is marginal on force (496 N vs
  453 N peak — SF 1.10); use an **AK10-9-class motor (~48 Nm peak, ~0.96 kg)
  → ~960 N** at R = 50 mm. Mass cost: +0.26 kg/motor × 4 = **+1.0 kg** over
  the AK70 baseline (update the mass budget when frozen).
- **Arms/neck — spool (r = 10–12 mm) + AK60-6** everywhere; requirements are
  tiny (≤42 N with SF) and stroke rules (shoulder 228 mm) favor spools.
- **Lean on the prestress network for stiffness.** The passive cable network
  supplies baseline joint stiffness; motors only modulate force on top. This
  is what makes the modest closed-loop bandwidths above sufficient — a pure
  active-impedance design would need several times more.

## 4. Open items

1. Re-run on MJPC walking traces (ankle push-off F and v; swing-phase cable
   speeds) before freezing the BOM.
2. Verify rotor inertia + friction of AK70-10 / AK60-6 / AK10-9 on the bench;
   the f_bw column scales with 1/√I.
3. Measure textile-channel stiffness (soft k_ser) on the bench rig — it is
   the difference between the spool working and not working at the hips.
4. Ankle lever detail: ±60° crank usable range assumed; check linkage
   geometry over the full ankle envelope (94 mm stroke at R = 50 mm leaves
   ~10% margin).
