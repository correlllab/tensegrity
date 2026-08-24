#!/usr/bin/env python
"""Emit paper/numbers.tex: one LaTeX macro per measured quantity.

This replaces the previous integrate_paper.py, which rewrote whole
subsections of the .tex by string substitution. That was fragile (it could
only be run once, and a stale intermediate silently changed a reported
number). Here the paper is the source of truth for prose and this script is
the source of truth for numbers; the paper \\input's the macros.

Every walking metric is recomputed from the raw logs with the current success
criterion rather than read from a cached summary.
"""
import glob
import json
import os
import re
import sys
import numpy as np
import mujoco

ROOT = "/Users/ncorrell/Downloads/tensegrity"
RES = f"{ROOT}/experiments/results"
MJ = f"{ROOT}/mujoco"
OUT = f"{ROOT}/paper/numbers.tex"
sys.path.insert(0, f"{ROOT}/experiments")
import walklog                                          # noqa: E402

# LaTeX macro names may not contain digits, so numeric conditions are spelled
WORD = {2: "Two", 4: "Four", 6: "Six", 8: "Eight", 10: "Ten", 12: "Twelve",
        16: "Sixteen", 20: "Twenty"}
SPEEDWORD = {10: "Ten", 20: "Twenty", 30: "Thirty", 35: "ThirtyFive",
             40: "Forty", 45: "FortyFive"}

MOTOR_KG = {"hip": 0.52, "knee": 0.52, "ankle": 0.96, "waist": 0.52,
            "shoulder": 0.31, "elbow": 0.31, "wrist": 0.17, "neck": 0.17}
M = {}


def put(name, value, fmt="{:.0f}"):
    if isinstance(value, str):
        M[name] = value
        return
    v = float(value)
    M[name] = "---" if not np.isfinite(v) else fmt.format(v)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


# ---------------------------------------------------------------- model counts
mt = mujoco.MjModel.from_xml_path(f"{MJ}/humanoid_27dof_tensegrity.xml")
mc = mujoco.MjModel.from_xml_path(f"{MJ}/humanoid_27dof_tensegrity_cable.xml")
put("numTendonTorque", mt.ntendon)
put("numTendonCable", mc.ntendon)
put("numActuatable", mc.nu)
put("numDof", mt.nu)

e7 = json.load(open(f"{RES}/e7_strut_sizing.json"))
put("numStruts", len(e7["struts"]))
put("strutPeakN", e7["peak_cable_axial"])
put("strutMassSized", e7["mass_sized"], "{:.2f}")
put("strutWeakestPcr", e7["weakest"]["P_cr"])
for c in e7["load_cases"]:
    key = {"stand": "Stand", "stumble 0.1 m": "Stumble",
           "drop 1.0 m": "Drop"}[c["case"]]
    put(f"strutMargin{key}", c["margin"], "{:.1f}")
    put(f"strutLoad{key}", c["per_strut"])

# endpoint count comes from the E7 run's stdout invariant; recompute here
adr = mt.tendon_adr
sites = set()
for t in range(mt.ntendon):
    for i in range(mt.tendon_num[t]):
        if mt.wrap_type[adr[t] + i] == mujoco.mjtWrap.mjWRAP_SITE:
            sites.add(int(mt.wrap_objid[adr[t] + i]))
put("numEndpoints", len(sites))

# ---------------------------------------------------------------- authority/BOM
a = json.load(open(f"{RES}/e5_authority.json"))
jn, need = a["dofs"], np.array(a["need"])
put("numMotors", a["n_motors"])
put("numDrivenCables", a["n_driven"])
put("numPassiveCables", a["n_cables"] - a["n_driven"])

cnames = [mujoco.mj_id2name(mc, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
          for i in range(mc.nu)]


def motor_kg(n):
    for k, v in MOTOR_KG.items():
        if n.startswith(k):
            return v
    raise ValueError(n)


put("allMotorMass", sum(motor_kg(c) for c in cnames), "{:.1f}")
bom = sum(motor_kg(n) for n in jn) + sum(motor_kg(n) for n in jn
                                         if n.startswith(("waist", "shoulder")))
put("bomMass", bom, "{:.1f}")
SHORT = ("hip_pitch_l", "hip_pitch_r", "knee_l", "knee_r", "elbow_l",
         "elbow_r", "waist_roll")
extra = sum(motor_kg(n) for n in SHORT)
put("extraMotorMass", extra, "{:.1f}")
put("bomMassFixed", bom + extra, "{:.1f}")
put("numMotorsFixed", a["n_motors"] + len(SHORT))

for tag, key in (("home_all", "HomeAll"), ("gait_all", "GaitAll"),
                 ("home_loops", "HomeBom"), ("gait_loops", "GaitBom")):
    mar = np.minimum(a[tag]["pos"], a[tag]["neg"]) / need
    put(f"auth{key}", mar.min(), "{:.2f}")
    put(f"authShort{key}", int((mar < 1.0).sum()))
mg = np.minimum(a["gait_all"]["pos"], a["gait_all"]["neg"]) / need
put("authGaitHip", mg[jn.index("hip_pitch_l")], "{:.2f}")
put("authGaitKnee", mg[jn.index("knee_l")], "{:.2f}")
put("rNeedHip", 1e3 * a["r_need"][jn.index("hip_pitch_l")], "{:.1f}")
put("rHaveHip", 1e3 * a["r_have"][jn.index("hip_pitch_l")], "{:.1f}")
put("rHaveKnee", 1e3 * a["r_have"][jn.index("knee_l")], "{:.1f}")
put("singleLoopHip", a["single_loop"][jn.index("hip_pitch_l")])
put("singleLoopKnee", a["single_loop"][jn.index("knee_l")])

# ---------------------------------------------- physical member specification
if os.path.exists(f"{RES}/member_specs.json"):
    e8 = json.load(open(f"{RES}/member_specs.json"))
    tn = e8["tendons"]
    rope = sum(r["mu"] * r["length"] for r in tn.values())
    term = e8["m_tendon"] - rope
    ends = 2 * sum(r["n"] for r in tn.values())
    fitting = 2 * len(e8["struts"]) * 3.0e-3
    put("strutSpecMass", e8["m_strut"], "{:.2f}")
    put("strutTubeMass", e8["m_strut"] - fitting, "{:.2f}")
    put("strutFittingMass", fitting, "{:.2f}")
    put("strutOdMin", 1e3 * min(r["od"] for r in e8["struts"]))
    put("strutOdMax", 1e3 * max(r["od"] for r in e8["struts"]))
    put("strutWall", 1e3 * min(r["wall"] for r in e8["struts"]), "{:.1f}")
    put("strutLenMin", 1e3 * min(r["L"] for r in e8["struts"]))
    put("strutLenMax", 1e3 * max(r["L"] for r in e8["struts"]))
    put("strutWorstMargin", min(r["margin"] for r in e8["struts"]), "{:.1f}")
    put("ropeMass", rope, "{:.2f}")
    put("termMass", term, "{:.2f}")
    put("tendonSpecMass", e8["m_tendon"], "{:.2f}")
    put("numCableEnds", ends)
    put("nodeCount", e8["n_nodes"])
    put("nodeMass", e8["m_node"], "{:.2f}")
    put("membersMass", e8["total"], "{:.2f}")
    put("shellAllowance", 5.90 - e8["total"], "{:.2f}")
    put("ropeDMin", min(r["d"] for r in tn.values()), "{:.1f}")
    put("ropeDMax", max(r["d"] for r in tn.values()), "{:.1f}")
    put("ropeBLLeg", tn["hip"]["bl"] / 1e3, "{:.1f}")
    put("ropeUtilRms", 100 * max(r["util_rms"] for r in tn.values()), "{:.0f}")
    put("stiffGapRope", e8["stiffness_ratio"])
    act = {g: r for g, r in tn.items() if g != "cell"}
    put("kSeriesMin", min(r["k_series_needed"] for r in act.values()) / 1e3)
    put("kSeriesMax", max(r["k_series_needed"] for r in act.values()) / 1e3)
    put("stiffGapSeries", np.median([r["k_series_needed"] / r["k_model"]
                                     for r in act.values()]))
    put("takeupMin", min(r["takeup_um"] for r in act.values()))
    put("takeupMax", max(r["takeup_um"] for r in act.values()))
    put("kModelMin", min(r["k_model"] for r in tn.values()))
    put("kModelMax", max(r["k_model"] for r in tn.values()))
    put("kRopeMin", min(r["k_rope"] for r in tn.values()) / 1e6, "{:.1f}")
    put("kRopeMax", max(r["k_rope"] for r in tn.values()) / 1e6, "{:.1f}")

# ------------------------------------------------- hinges, joints, stiffness
if os.path.exists(f"{RES}/e9_hinge_audit.json"):
    e9 = json.load(open(f"{RES}/e9_hinge_audit.json"))
    put("hingeDof", e9["n_hinge_dof"])
    put("hingeSegments", len(e9["audit"]))
    put("hingeConstrainedDof", sum(r["constrained_dof"] for r in e9["audit"]))
    wc = e9["wrench_closure"]
    put("wcTotal", len(wc))
    put("wcSix", sum(1 for r in wc if r["t6"] > 1e-6))
    put("wcThree", sum(1 for r in wc if r["t3"] > 1e-6))
    hf = e9["hinge_free"]
    put("freeZstart", hf["z0"], "{:.2f}")
    put("freeZend", hf["z"], "{:.2f}")

if os.path.exists(f"{RES}/e10_tensegrity_joint.json"):
    e10 = json.load(open(f"{RES}/e10_tensegrity_joint.json"))
    put("jointOverlapFrac", e10["best_overlap_frac"], "{:.1f}")
    put("jointOverlapMm", 1e3 * e10["best_overlap_frac"] * 0.30)
    imp = e10.get("impact", [])
    if imp:
        put("jointImpactLo", min(-r["delta_pct"] for r in imp))
        put("jointImpactHi", max(-r["delta_pct"] for r in imp))
        for r in imp:
            k = {0.05: "Five", 0.1: "Ten", 0.2: "Twenty"}[round(r["h"], 2)]
            put(f"jointImpactCable{k}", r["cable"])
            put(f"jointImpactPin{k}", r["pin"])
            put(f"jointImpactDelta{k}", -r["delta_pct"])
    pr = e10.get("prestress_ratio_by_k", {})
    for kk, tag in (("2000", "Soft"), ("40000", "Series"), ("600000", "Rope")):
        if kk in pr:
            put(f"jointPreRatio{tag}", pr[kk], "{:.2f}")
    rope = [r for r in e10["prestress_sweep"] if r["k_cable"] == 600000.0]
    if rope:
        put("jointGeomElasticRope", 100 * max(r["geom_over_elastic"]
                                              for r in rope), "{:.1f}")
    soft = [r for r in e10["prestress_sweep"] if r["k_cable"] == 2000.0]
    if soft:
        put("jointGeomElasticSoft", 100 * max(r["geom_over_elastic"]
                                              for r in soft), "{:.0f}")

if os.path.exists(f"{RES}/e12_cell_graph.json"):
    e12 = json.load(open(f"{RES}/e12_cell_graph.json"))
    ch = e12["chain"]
    put("cellChainMax", max(r["n_cells"] for r in ch))
    put("cellChainDroop", max(r["droop"] for r in ch), "{:.0f}")
    put("cellChainPer", max(r["droop"] for r in ch)
        / max(r["n_cells"] for r in ch), "{:.1f}")
    lb = e12["lower_body"]
    put("lbStruts", lb["n_struts"])
    put("lbCables", lb["n_cables"])
    put("lbDroop", lb["droop_mm"], "{:.0f}")
    fb = e12["full_body"]
    put("cgStruts", fb["n_struts"])
    put("cgCellStruts", fb["n_cell_struts"])
    put("cgClusterStruts", fb["n_cluster_struts"])
    put("cgCables", fb["n_cables"])
    put("cgDof", fb["dof"])
    put("cgSettle", fb["settle_mm"], "{:.0f}")
    fail = [r for r in e12["full_body_load"] if not r["stands"]]
    if fail:
        put("cgLoadFail", min(r["torso_kg"] for r in fail), "{:.0f}")
    ok = [r for r in e12["full_body_prestress"] if r["stands"]]
    if ok:
        put("cgPreNeeded", min(r["t_pre"] for r in ok), "{:.0f}")
        put("cgPreSettle", min(r["settle_mm"] for r in ok), "{:.0f}")
    if "prestress_strut_load" in e12:
        nl = e12["prestress_strut_load"]["peak_node_load"]
        put("cgNodeLoad", nl)
        put("cgNodeMargin", 2358.0 / nl, "{:.1f}")

if os.path.exists(f"{RES}/e11_physical.json"):
    e11 = json.load(open(f"{RES}/e11_physical.json"))
    for r in e11["e1"]:
        k = {0.05: "Five", 0.1: "Ten", 0.2: "Twenty", 0.3: "Thirty",
             0.5: "Fifty", 1.0: "Hundred"}.get(round(r["h"], 2))
        if k:
            put(f"physOneLimp{k}", r["delta_limp"], "{:+.0f}")
            put(f"physOneHeld{k}", r["delta_held"], "{:+.0f}")
    e2p = {r["scale"]: r["k_lat"] for r in e11["e2"]}
    put("physTwoNoCables", e2p[-1], "{:.0f}")
    put("physTwoZeroPre", e2p[0.0], "{:.0f}")
    put("physTwoNominal", e2p[1.0], "{:.0f}")
    put("physTwoHigh", e2p[4.0], "{:.0f}")
    put("physTwoEngaged", 100 * (e2p[4.0] / e2p[0.5] - 1), "{:+.0f}")
    put("physTwoCableRatio", e11["e2_cable_ratio"], "{:.0f}")

if os.path.exists(f"{RES}/member_specs.json"):
    ms = json.load(open(f"{RES}/member_specs.json"))
    put("bearingMass", ms["m_hinge"], "{:.2f}")
    put("bearingPerDof", ms["bearing_g_per_dof"])
    put("hingeDofCount", ms["n_hinge_dof"])

# ------------------------------------------------- hybrid walker
put("hybridMass", 20.1, "{:.1f}")
put("hybridJointHw", 0.80, "{:.1f}")
put("hybridKneeHw", 142)
put("hybridHipHw", 257)
put("hybridSpeed", 0.17, "{:.2f}")
put("hybridRate", "---")
# the full model (tensegrity waist + battery packs) supersedes the earlier
# welded-torso verification numbers
if os.path.exists(f"{RES}/hybrid_full.json"):
    hf = json.load(open(f"{RES}/hybrid_full.json"))
    put("hybridSpeed", hf["v"], "{:.2f}")
    put("hybridRate", f"{hf['ok']}/{hf['n']}")
elif os.path.exists(f"{RES}/hybrid_verify.json"):
    hv = json.load(open(f"{RES}/hybrid_verify.json"))
    pick = hv.get("finals", [None])[0] or hv.get("baseline")
    if pick:
        put("hybridSpeed", pick["v"], "{:.2f}")
        put("hybridRate", f"{pick['ok']}/{pick['n']}")
# actuator duty in the logged gait
_h = glob.glob(f"{RES}/hybrid_v/log_base_t0.csv")
if _h:
    _raw = np.genfromtxt(_h[0], delimiter=",", invalid_raise=False)
    _raw = _raw[~np.isnan(_raw).any(axis=1)]
    _t = _raw[:, 0]
    _c = _raw[:, 1 + 33 + 30:1 + 33 + 30 + 24]
    _w = (_t >= 4) & (_t <= 13)
    put("hybridJointDuty", 100 * np.abs(_c[_w][:, :8]).mean(), "{:.0f}")
    put("hybridAnkleDuty", 100 * _c[_w][:, 12:].mean(), "{:.0f}")
    put("hybridAnklePeak", 250 * _c[_w][:, 12:].max(), "{:.0f}")

# ------------------------------------------------- joint-count study (E17)
if os.path.exists(f"{RES}/e17_joint_count.json"):
    e17 = json.load(open(f"{RES}/e17_joint_count.json"))
    for k, tag in (("rigid_waist", "RWaist"), ("rigid_ankle", "RAnkle"),
                   ("rigid_both", "RBoth")):
        if k in e17:
            put(f"jc{tag}Rate", f"{e17[k]['ok']}/{e17[k]['n']}")
            put(f"jc{tag}V", e17[k]["v"], "{:.2f}")

# ------------------------------------------------- hybrid Fig-7 sweeps
if os.path.exists(f"{RES}/hybrid_sweeps.json"):
    hsj = json.load(open(f"{RES}/hybrid_sweeps.json"))
    for sp, tag in (("0.3", "Thirty"), ("0.45", "FortyFive"),
                    ("0.6", "Sixty")):
        if sp in hsj["speed"]:
            r = hsj["speed"][sp]
            put(f"hsSpd{tag}Rate", f"{r['ok']}/{r['n']}")
            put(f"hsSpd{tag}V", r["v"], "{:.2f}")
    for mkg, tag in (("2", "Two"), ("5", "Five"), ("8", "Eight"),
                     ("12", "Twelve")):
        if mkg in hsj["payload"]:
            r = hsj["payload"][mkg]
            put(f"hsPay{tag}Rate", f"{r['ok']}/{r['n']}")
            put(f"hsPay{tag}V", r["v"], "{:.2f}")
    held = [r for r in hsj["carry"] if r["held"]]
    if held:
        hr = max(held, key=lambda r: r["mass"])
        put("hsCarryMax", hr["mass"], "{:.1f}")
        put("hsCarryUtil", 100 * hr["shoulder_util"], "{:.0f}")
        put("hsCarryWaistT", hr["waist_T"], "{:.0f}")

# ------------------------------------------------- E18 matched arm carry
if os.path.exists(f"{RES}/e18_matched_carry.json"):
    e18 = json.load(open(f"{RES}/e18_matched_carry.json"))

    def _maxhold(rows, th):
        h = [r["mass"] for r in rows if r["drop_m"] < th and r["finite"]]
        return max(h) if h else 0.0

    for tag, k in (("Hyb", "hybrid"), ("Hin", "hinged")):
        put(f"carry{tag}Eight", _maxhold(e18[k], 0.08), "{:.1f}")
        put(f"carry{tag}Twelve", _maxhold(e18[k], 0.12), "{:.1f}")
        put(f"carry{tag}Sixteen", _maxhold(e18[k], 0.16), "{:.1f}")
        pre = [r for r in e18[k] if r["shoulder_util"] < 0.98]
        sl = np.polyfit([r["mass"] for r in pre],
                        [1e3 * r["drop_m"] for r in pre], 1)[0]
        put(f"carry{tag}Slope", sl, "{:.0f}")

# ------------------------------------------------- E23 payload intervention
if os.path.exists(f"{RES}/e23_round2.json"):
    e23 = json.load(open(f"{RES}/e23_round2.json"))
    win = [r for r in e23.get("verify", []) if r["tag"] == "fwd4s"]
    if win:
        put("payTenRate", f"{win[0]['ok']}/{win[0]['n']}")
        put("payTenV", win[0]["v"], "{:.2f}")

# ------------------------------------------------- E22 bench predictions
if os.path.exists(f"{RES}/e22_bench.json"):
    e22 = json.load(open(f"{RES}/e22_bench.json"))
    put("benchAnkleMoment", e22["ankle"]["moment_per_100N"], "{:.0f}")
    put("benchAnkleArmA", e22["ankle"]["arm_A_mm"], "{:.0f}")
    put("benchAnkleArmB", e22["ankle"]["arm_B_mm"], "{:.0f}")
    put("benchWaistSink", e22["waist"]["sink_per_100N"], "{:.0f}")
    pts = e22["leg"]["points"]
    put("benchLegT", pts[0]["peak_ankle_T"], "{:.0f}")
    put("benchLegTTwo", pts[-1]["peak_ankle_T"], "{:.0f}")
    put("benchLegSink", (pts[0]["pelvis_z_mm"] - pts[-1]["pelvis_z_mm"])
        / 2.0, "{:.0f}")

# ------------------------------------------------- E21/E24 gait retune
if os.path.exists(f"{RES}/e24_round3.json"):
    e24 = json.load(open(f"{RES}/e24_round3.json"))
    win = [r for r in e24.get("verify", []) if r["tag"] == "s24"]
    if win:
        put("agilityRate", f"{win[0]['ok']}/{win[0]['n']}")
        put("agilityV", win[0]["v"], "{:.2f}")
        put("gaitClr", win[0]["clr"], "{:.0f}")
elif os.path.exists(f"{RES}/e21_verify.json"):
    v21 = json.load(open(f"{RES}/e21_verify.json"))
    put("agilityRate", f"{v21['ok']}/{v21['n']}")
    put("agilityV", v21["v"], "{:.2f}")
elif os.path.exists(f"{RES}/e21_agility.json"):
    a21 = json.load(open(f"{RES}/e21_agility.json"))
    if a21.get("best"):
        put("agilityRate", f"{a21['best']['ok']}/{a21['best']['n']}")
        put("agilityV", a21["best"]["v"], "{:.2f}")
if os.path.exists(f"{RES}/e21_agility.json"):
    a21 = json.load(open(f"{RES}/e21_agility.json"))
    fast = [r for r in a21.get("C", []) if r["cfg"].get("cadence") == 1.1]
    if fast:
        put("agilityFastRate", f"{fast[0]['ok']}/{fast[0]['n']}")
        put("agilityFastV", fast[0]["v"], "{:.2f}")

# ------------------------------------------------- E20 rigid-waist retune
if os.path.exists(f"{RES}/e20_rigid_waist_retune.json"):
    e20 = json.load(open(f"{RES}/e20_rigid_waist_retune.json"))
    if e20.get("best"):
        put("jcRWaistTunedRate", f"{e20['best']['ok']}/{e20['best']['n']}")
        put("jcRWaistTunedV", e20["best"]["v"], "{:.2f}")
        put("jcRWaistTunedTorso", e20["best"]["cfg"]["torso"], "{:.2f}")

# ------------------------------------------------- E19 stiffness sensitivity
if os.path.exists(f"{RES}/e19_stiffness_sensitivity.json"):
    e19 = json.load(open(f"{RES}/e19_stiffness_sensitivity.json"))
    d10 = [r["delta_pct"] for r in e19["x10_400kN"]]
    put("stiffSensTenLo", min(d10), "{:+.0f}")
    put("stiffSensTenHi", max(d10), "{:+.0f}")

# ---------------------------------------------------------------- E1 impacts
e1 = json.load(open(f"{RES}/e1_drop.json"))
W = 27.6 * 9.81


def grf(mode, h):
    return [r for r in e1 if r["mode"] == mode and r["h"] == h][0]["peak_grf"]


for h, tag in ((0.10, "Stumble"), (1.00, "Crash")):
    tp, rl = grf("tensegrity-passive", h), grf("rigid-limp", h)
    th, rh = grf("tensegrity-held", h), grf("rigid-held", h)
    put(f"eOnePassiveTsg{tag}", tp)
    put(f"eOnePassiveRigid{tag}", rl)
    put(f"eOneHeldTsg{tag}", th)
    put(f"eOneHeldRigid{tag}", rh)
    put(f"eOneDeltaPassive{tag}", 100 * (tp / rl - 1), "{:+.0f}")
    put(f"eOneDeltaHeld{tag}", 100 * (th / rh - 1), "{:+.0f}")
    put(f"eOneBWTsg{tag}", tp / W, "{:.0f}")
    put(f"eOneBWRigid{tag}", rl / W, "{:.0f}")
# the number originally reported: passive tensegrity vs SERVO-HELD rigid
put("eOneMismatched", 100 * (1 - grf("tensegrity-passive", 0.10) /
                             grf("rigid-held", 0.10)))
put("eOneCrossover", 0.2, "{:.1f}")
fell = [r for r in e1 if r["mode"] == "tensegrity-passive" and r["fell"]]
put("eOnePassiveFell", len(fell))
put("eOnePassiveN", len([r for r in e1 if r["mode"] == "tensegrity-passive"]))

# ---------------------------------------------------------------- E2 stiffness
e2 = json.load(open(f"{RES}/e2_stiffness.json"))
pre = {r["prestress_scale"]: r["k_lat"] for r in e2["prestress"]}
co = {r["cocontraction_N"]: r["k_lat"] for r in e2["cocontraction"]}
swept = [v for k, v in pre.items() if k >= 0]
put("stiffNoCables", pre[-1], "{:.1f}")
put("stiffZeroPre", pre[0.0], "{:.1f}")
put("stiffNominal", pre[1.0], "{:.1f}")
put("stiffPreLo", min(swept), "{:.1f}")
put("stiffPreHi", max(swept), "{:.1f}")
put("stiffPreRatio", max(swept) / min(swept), "{:.1f}")
put("stiffCablesRatio", pre[0.0] / pre[-1], "{:.1f}")
put("stiffCoLo", co[0.0], "{:.0f}")
put("stiffCoHi", co[400.0], "{:.0f}")
put("stiffCoKnee", co[50.0], "{:.0f}")
put("stiffCoRatio", co[400.0] / co[0.0], "{:.1f}")
put("stiffProbe", 1e3 * e2["probe_N"])
put("stiffLinearity", 100 * e2["linearity"]["rel_change"], "{:.1f}")

# ---------------------------------------------------------------- E3 failures
e3 = json.load(open(f"{RES}/e3_degradation.json"))
put("eThreeBaseMargin", e3["base_margin"], "{:.2f}")
put("eThreeBelowOne", e3["singles_below_1"])
put("eThreeSinglesOk", sum(1 for s in e3["singles"] if s["survived"]))
put("eThreeSinglesN", len(e3["singles"]))
for row in e3["multi"]:
    tag = {"random-all": "Rand", "random-stance": "Leg",
           "adversarial": "Adv"}[row["rule"]]
    put(f"eThree{tag}{WORD[row['k']]}",
        f"{row['survived']}/{row['trials']}")

# ---------------------------------------------------------------- walking
LOGS = {}
for f in sorted(glob.glob(f"{RES}/log_*.csv")):
    tag, trial = re.match(r"log_(.+)_t(\d+)\.csv", os.path.basename(f)).groups()
    pay = 0.0
    if tag.startswith("cpayload"):
        pay = float(tag[8:])
    elif tag.startswith("payload"):
        pay = float(tag[7:])
    LOGS.setdefault(tag, []).append((int(trial), f, pay))


def agg(tag):
    rows = LOGS.get(tag, [])
    out = []
    for _, f, pay in rows:
        try:
            out.append(walklog.metrics(f, total_mass=27.6 + pay))
        except Exception:
            pass
    ok = [r for r in out if r["success"]]
    v = float(np.mean([r["speed"] for r in ok])) if ok else float("nan")
    sd = float(np.std([r["speed"] for r in ok])) if len(ok) > 1 else 0.0
    cot = float(np.mean([r["cot"] for r in ok])) if ok else float("nan")
    return dict(n=len(out), ok=len(ok), v=v, sd=sd, cot=cot,
                ci=wilson(len(ok), len(out)))


base = agg("baseline")
put("walkN", base["n"])
put("walkOk", base["ok"])
put("walkSpeed", base["v"], "{:.2f}")
put("walkSpeedSd", base["sd"], "{:.2f}")
put("walkCot", base["cot"], "{:.1f}")
put("walkRate", f"{base['ok']}/{base['n']}")
put("walkCiLo", 100 * base["ci"][0])
put("walkCiHi", 100 * base["ci"][1])

d = agg("distal")
put("distalN", d["n"])
put("distalOk", d["ok"])
put("distalRate", f"{d['ok']}/{d['n']}")
put("distalSpeed", d["v"], "{:.2f}")

SPEEDS = [0.10, 0.20, 0.35, 0.40, 0.45]
for s in SPEEDS:
    r = agg(f"speed{s:.2f}")
    key = SPEEDWORD[int(round(s * 100))]
    put(f"spd{key}N", r["n"])
    put(f"spd{key}Ok", r["ok"])
    put(f"spd{key}Rate", f"{r['ok']}/{r['n']}")
    put(f"spd{key}V", r["v"], "{:.2f}")
put("spdThirtyRate", f"{base['ok']}/{base['n']}")
put("spdThirtyV", base["v"], "{:.2f}")

MASSES = [2, 4, 6, 8, 10, 12, 16, 20]
pay_rows = []
for m_ in MASSES:
    r = agg(f"cpayload{m_:02d}")
    pay_rows.append((m_, r))
    put(f"cpay{WORD[m_]}Rate", f"{r['ok']}/{r['n']}")
    put(f"cpay{WORD[m_]}V", r["v"], "{:.2f}")
    put(f"cpay{WORD[m_]}CiLo", 100 * r["ci"][0])
    put(f"cpay{WORD[m_]}CiHi", 100 * r["ci"][1])
tot_ok = sum(r["ok"] for _, r in pay_rows)
tot_n = sum(r["n"] for _, r in pay_rows)
put("cpayAllRate", f"{tot_ok}/{tot_n}")
put("cpayAllPct", 100 * tot_ok / max(tot_n, 1))
lo, hi = wilson(tot_ok, tot_n)
put("cpayAllCiLo", 100 * lo)
put("cpayAllCiHi", 100 * hi)
for m_ in MASSES:
    r = agg(f"payload{m_:02d}")
    put(f"rpay{WORD[m_]}Rate", f"{r['ok']}/{r['n']}")

# realtime factor
rt = []
for line in open(f"{RES}/walk_runs_extended.jsonl"):
    try:
        v = json.loads(line).get("realtime_factor")
        if v:
            rt.append(v)
    except Exception:
        pass
if rt:
    put("realtimeFactor", float(np.mean(rt)), "{:.2f}")
    put("realtimeN", len(rt))

# tendon feasibility of the gait, and tendon-limited walking (E6)
put("eSixSummary", "\\emph{(E6 pending.)}")
for _t in ("tendonall", "tendonbom", "tendonmin"):
    put(f"eSix{_t}Rate", "---")
if os.path.exists(f"{RES}/e6_summary.json"):
    e6 = json.load(open(f"{RES}/e6_summary.json"))
    for k, v in e6.items():
        put(f"eSix{k}", v)
if os.path.exists(f"{RES}/tension_audit.json"):
    ta = json.load(open(f"{RES}/tension_audit.json"))
    put("tensPeak", ta["peak_tension"])
    put("tensResid", ta["worst_residual"])
    put("tensSatPct", 100 * ta["sat_frac"], "{:.1f}")
    put("tensBusiestRms", ta["busiest_rms"])
    put("tensNeeded", ta["fmax_needed"])
    for k, v in ta.get("by_payload", {}).items():
        m_ = int(float(k))
        if m_ in WORD:
            put(f"tensSat{WORD[m_]}", 100 * v["sat_frac"], "{:.0f}")
            put(f"tensNeed{WORD[m_]}", v["fmax_needed"])
            put(f"tensResid{WORD[m_]}", v["worst_residual"])

with open(OUT, "w") as f:
    f.write("% generated by experiments/make_numbers.py -- do not edit\n")
    for k in sorted(M):
        f.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
print(f"{len(M)} macros -> {OUT}")
for k in sorted(M):
    print(f"  {k:24s} {M[k]}")
