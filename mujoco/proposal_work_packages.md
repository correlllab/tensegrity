# NSF FRR next-step work packages (design + MuJoCo)

## WP1: Preliminary design freeze (1-2 weeks)
- Define A0 geometry, DOF, tendon count, actuator locations.
- Define measurable requirements: ROM, force, compliance, cycle life.
- Output: parameter table for simulation and prototype constraints.

## WP2: Baseline MuJoCo model (1 week)
- Implement 2-DOF shoulder-elbow module.
- Implement antagonistic tendon actuation.
- Output: baseline XML and reproducible run configuration.

## WP3: Control and stiffness study (1-2 weeks)
- Joint PD control and co-contraction policy.
- Quantify apparent stiffness and disturbance rejection.
- Output: performance plots and candidate operating envelope.

## WP4: Sensitivity and risk reduction (1 week)
- Sweep damping, gear, and tendon gain parameters.
- Identify fragile regions and robust settings.
- Output: robust parameter subset for prototype A0.

## WP5: Hardware handoff package (1 week)
- Map simulation parameters to physical components.
- Define bench tests to identify friction/hysteresis not in baseline model.
- Output: build-ready assumptions sheet and test protocol.

## Suggested primary metrics
- ROM achieved at each joint (deg).
- Endpoint error under external load (mm).
- Settling time and overshoot (%).
- Effective stiffness slope around operating point.
- Drift after repeated cycle batches.
