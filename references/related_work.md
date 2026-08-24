# Related Work: Tensegrity + Tendon-Driven + Textile Robotics

This note collects references relevant to a humanoid robot concept that combines:
- tensegrity-inspired structure,
- tendon transmission with electric actuation,
- and apparel-like fabrication for the outer shell and load paths.

## 1) Core tensegrity robotics

1. System design and locomotion of SUPERball, an untethered tensegrity robot (ICRA 2015)
- DOI: https://doi.org/10.1109/ICRA.2015.7139590
- Why relevant: establishes practical design/control tradeoffs for cable-actuated tensegrity hardware.

2. Design and control of compliant tensegrity robots through simulation and hardware validation (J. R. Soc. Interface, 2014)
- DOI: https://doi.org/10.1098/rsif.2014.0520
- Why relevant: simulation-to-hardware methodology, rolling tensegrity validation, and control architecture insights.

3. A bio-inspired tensegrity manipulator with multi-DOF, structurally compliant joints (IROS 2016)
- DOI: https://doi.org/10.1109/IROS.2016.7759811
- arXiv: https://arxiv.org/abs/1604.08667
- Why relevant: direct precedent for compliant, multi-DOF, tendon-driven tensegrity joints.

4. Bio-inspired Tensegrity Soft Modular Robots (Living Machines 2017)
- DOI: https://doi.org/10.1007/978-3-319-63537-8_42
- arXiv: https://arxiv.org/abs/1703.10139
- Why relevant: modularity and planar manufacturing ideas that map well to soft-goods assembly.

5. Variable-stiffness tensegrity spine (Smart Materials and Structures, 2020)
- DOI: https://doi.org/10.1088/1361-665X/ab87e0
- Why relevant: explicit variable stiffness modes (soft/global stiff/directional stiff) for tasks requiring both compliance and load support.

6. Cellular morphogenesis of three-dimensional tensegrity structures (CMAME, 2019)
- DOI: https://doi.org/10.1016/j.cma.2018.10.048
- arXiv: https://arxiv.org/abs/1902.09953
- Why relevant: principled topology/form-finding framework for generating new tensegrity geometries.

## 2) Textile and soft wearable actuation (manufacturing analogs)

1. Assistance magnitude versus metabolic cost reductions for a tethered multiarticular soft exosuit (Science Robotics, 2017)
- DOI: https://doi.org/10.1126/scirobotics.aah4416
- Why relevant: mature textile load-path design and human-compatible tendon force transmission.

2. Wyss soft exosuits for back support (technology page)
- URL: https://wyss.harvard.edu/technology/soft-exosuits/
- Why relevant: practical integration details for textile interfaces, sensing, and motorized assistance.

3. Design, fabrication and control of soft robots (Nature, 2015)
- DOI: https://doi.org/10.1038/nature14543
- Why relevant: broad design principles for compliant robots and fabrication constraints.

## 3) Anthropomimetic / tendon-driven humanoid precedents

1. Roboy overview (project history and specs)
- URL: https://en.wikipedia.org/wiki/Roboy
- Why relevant: tendon-driven anthropomimetic architecture, many cable routes, distributed BLDC actuation.
- Caution: this is a secondary source; verify claims against primary publications.

2. Devanthro Robody platform (company/product direction)
- URL: https://www.devanthro.com/
- Why relevant: modern commercialization direction for humanoid service platforms with human-in-the-loop teleoperation.

## 4) NASA context and mission framing

1. Super Ball Bot overview (NASA)
- URL: https://www.nasa.gov/content/super-ball-bot
- Why relevant: high-level rationale for tensegrity in impact tolerance and terrain robustness.

## Observed design patterns across sources

1. Compliance is useful only if it is controllable.
- Passive compliance alone is insufficient for precision tasks.
- Systems that can switch stiffness modes are more versatile.

2. Tendon routing quality determines repeatability.
- Friction, hysteresis, and cable stretch dominate error budget in tendon-driven systems.

3. Modular prototypes outperform monolithic first builds.
- Spine/shoulder/elbow modules with independent validation de-risk full humanoid integration.

4. Textile interfaces can carry meaningful loads when load paths are explicit.
- Reinforcement direction, seam construction, and anchor hardware are first-class design variables.

## Suggested next paper-search directions

1. Bowden cable friction compensation and tendon hysteresis models for robotics.
2. Seam/laminate fatigue in repeated high-load textile channels.
3. Distributed proprioception in soft/compliant manipulators.
4. Safety standards and contact-force envelopes for home/assistive humanoids.
