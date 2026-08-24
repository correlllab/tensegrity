# Structure: how the tensegrity actually works

Written because an earlier version of the model had cables ending in mid air,
which made the structural concept impossible to read. It was not only a
drawing problem — the cable anchors were computed from parametric rings that
had nothing to do with where the struts were, so the "tensegrity" was
decorative. This documents what the geometry is now and, more importantly,
what carries what.

## 1. The reference cell (a real tensegrity)

`mujoco/formfind_prism.py` builds a genuine Snelson 3-strut prism: three
compression struts that touch nothing at all, held only by nine cables (top
triangle, bottom triangle, three verticals). Each strut is a free body.

Topology matters and is easy to get backwards. The **struts are the long
members crossing the interior** — strut *i* runs from bottom node *i* to top
node *i+1* — and the **short cables join corresponding nodes**. Building it the
other way (short struts, long diagonals) produces a rigid twisted cage with no
prestress equilibrium.

Released from start twists of 15°, 30°, 55° and 75°, the cell settles to
**29.8°** every time, against a theoretical equilibrium of 90 − 180/n = 30° for
n = 3. That measured angle is what the humanoid's cages use.

Prestress is the whole mechanism — at equilibrium every one of the nine cables
carries tension, and the cell's stiffness scales with it:

| prestress | mean cable tension | max | cell height |
|---|---|---|---|
| 2% | 7 N | 14 N | 0.3006 m |
| 6% | 21 N | 41 N | 0.3021 m |
| 12% | 42 N | 81 N | 0.3046 m |
| 20% | 70 N | 136 N | 0.3080 m |

**Cables must be tension-only.** In MuJoCo a single `springlength` value is a
*bidirectional* spring that can push, which silently turns every cable into a
strut. The model now uses the deadband form `springlength="0 L_rest"`: zero
force below the rest length, tension above. This was a real bug in the earlier
humanoid — all 69 of its cables could push.

## 2. The humanoid's segments

Every limb segment is a Snelson cage at the twist above: three struts, six
nodes, and the cell's nine cables drawn in blue. Every cable in the whole
model — 210 of them — begins and ends on a **node**, and every node is the
physical endpoint of a strut. This is checked mechanically, not by eye: the
verification walks all 348 cable-endpoint sites and confirms each coincides
with a strut tip to within 0 mm.

The trunk is the same idea without the prism symmetry: the pelvis is a girdle
of three node triangles (waist plus one socket per hip) joined by nine struts;
the torso carries a waist ring, two shoulder sockets and a neck ring, tied by
twelve struts including three clavicle crossbars.

## 3. Load path — the honest version

    compression:  strut → node → HINGE → node → strut
    tension:      the cable network, which also applies every joint moment

Each joint keeps a **mechanical hinge**, and that hinge carries the compressive
and shear reaction across the joint. The cables carry tension only and generate
the joint torque. So this is a tensegrity-**inspired** limb, not a pure
tensegrity: in a pure tensegrity nothing but cable tension holds the struts
apart, which is exactly what the reference cell in §1 demonstrates and what a
floating-strut build would require.

That is a deliberate trade, and it is the same one the concept notes already
make ("tensegrity-inspired internal load-bearing network... not necessarily
pure tensegrity at whole-body scale"). Keeping the hinge is what makes a 27-DoF
system tractable for MJPC; floating struts would add 6 DoF per strut, need
form-finding at every pose, and lose the joint abstraction the planner uses.

Within a segment the three struts are welded into one rigid body, so that
segment's own nine cables carry no load in this model. They are drawn because
they are the cell the segment abstracts, and because they are precisely what a
floating-strut build would have to tension.

## 4. Cable authority — the check that catches bad routing

With tension-only actuation, τ = M t with 0 ≤ t ≤ F_max, so the achievable
torque about each DoF is bounded by Σ max(0, M) F_max and Σ min(0, M) F_max.
If either bound is near zero the joint is uncontrollable in that direction no
matter how good the planner is. `mujoco/check_cable_authority.py` reports this
per DoF, and it caught three genuine design errors:

1. **Hip yaw and wrist had exactly zero authority.** All cable families wound
   the same way around the joint, which gives yaw torque in one direction and
   none in the other. Fixed by routing `cw` and `ccw` families with opposite
   handedness — a straight cable has no moment arm about the long axis at all.
2. **Shoulder roll had 0.15×.** The shoulder socket sat inboard of the joint
   axis, so every cable rolled the arm the same way. Fixed by centering the
   socket on the joint axis and offsetting it vertically, as the hip already
   did.
3. **Ankle cables ran nearly horizontal** (only 15 mm of vertical offset
   between the shank ring and the foot), so plantarflexion needed absurd
   tension and saturated at 1500 N. Fixed by ending the shank cage higher and
   placing the foot's nodes anatomically — heel behind the joint, two forefoot
   corners ahead — instead of on a small symmetric ring.

All 27 DoFs now have authority in both directions, worst margin 1.01× (knee).

## 5. What this cost, and the open question

The cable model now has **102 actuated cables** (up from 51). Both models still
balance: the joint-torque model passes stand + push recovery, and the
tension-distribution controller balances the cable model on tension alone.

But MJPC's sampling planner now needs a planning iteration **every physics
step** (250 Hz) to hold the cable model up; at 125 Hz it falls. That runs at
1.08× realtime — feasible, with no margin. The joint-torque model by contrast
tolerates one iteration per 64 ms.

This points at the architecture flagged as a risk early on: **plan in the
27-DoF joint space and map to cable tensions underneath**, rather than planning
directly in 102-dimensional tension space. The least-squares tension
distribution in `verify_cable_stand.py` is exactly that mapping and already
works. Wiring it under MJPC is the next structural step.
