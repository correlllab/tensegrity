#!/usr/bin/env python
"""Offscreen render of the cable model mid push-recovery with force arrows:
cable tensions (green->red at both anchors), per-foot net GRF at the CoP
(orange), mirroring the C++ ModifyScene overlay in the MJPC tasks."""
import numpy as np
import mujoco
from scipy.optimize import lsq_linear
from verify_humanoid27 import gains_for

TENSION_MIN = 25.0        # N, hide quiet cables
TENSION_SCALE = 5e-4      # m per N arrow length
GRF_SCALE = 1.2e-3          # m per N
ARROW_W = 0.009


def cable_tensions(m, d):
    active = np.zeros(m.ntendon)
    for i in range(m.nu):
        if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_TENDON:
            t = m.actuator_trnid[i, 0]
            active[t] += -(m.actuator_gear[i, 0] * d.actuator_force[i])
    passive = (m.tendon_stiffness * (d.ten_length - m.tendon_lengthspring[:, 0])
               + m.tendon_damping * d.ten_velocity)
    return active + np.maximum(0.0, passive)


def add_arrow(scene, p_from, p_to, rgba, width=ARROW_W):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3),
                        np.zeros(3), np.eye(3).flatten(),
                        np.asarray(rgba, dtype=np.float32))
    g.emission = 0.7
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width,
                         np.asarray(p_from), np.asarray(p_to))
    scene.ngeom += 1


def draw_forces(m, d, scene):
    # cable tension arrows at both anchors, pointing along the cable
    T = cable_tensions(m, d)
    for t in range(m.ntendon):
        if T[t] < TENSION_MIN:
            continue
        adr, num = d.ten_wrapadr[t], d.ten_wrapnum[t]
        pts = d.wrap_xpos.reshape(-1, 3)
        p0 = pts[adr].astype(np.float64).copy()
        p1 = pts[adr + num - 1].astype(np.float64).copy()
        u = p1 - p0
        u /= np.linalg.norm(u) + 1e-12
        ln = max(0.04, min(0.30, TENSION_SCALE * T[t]))
        frac = min(1.0, T[t] / 1000.0)
        rgba = (frac, 1.0 - frac, 0.15, 1.0)
        add_arrow(scene, p0, p0 + u * ln, rgba)
        add_arrow(scene, p1, p1 - u * ln, rgba)

    # per-foot net GRF at CoP
    for foot in ("foot_l_geom", "foot_r_geom"):
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, foot)
        F = np.zeros(3)
        cop = np.zeros(3)
        wsum = 0.0
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            if gid not in (c.geom[0], c.geom[1]):
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            fw = c.frame.reshape(3, 3).T @ f6[:3]
            if c.geom[0] == gid:      # force acts on geom2; flip if foot first
                fw = -fw
            F += fw
            w = abs(fw[2]) + 1e-9
            cop += w * c.pos
            wsum += w
        if wsum > 1e-6 and np.linalg.norm(F) > 5.0:
            cop /= wsum
            if F[2] < 0:
                F = -F
            add_arrow(scene, cop, cop + GRF_SCALE * F, (1.0, 0.55, 0.05, 1.0),
                      width=0.016)


def main():
    m = mujoco.MjModel.from_xml_path("humanoid_27dof_tensegrity_cable.xml")
    m.vis.global_.offwidth, m.vis.global_.offheight = 1200, 900
    d = mujoco.MjData(m)

    # run the tension-distribution balance controller to mid-push (t=2.1 s)
    hinge = [j for j in range(m.njnt)
             if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    qadr, vadr = m.jnt_qposadr[hinge], m.jnt_dofadr[hinge]
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in hinge]
    kp = np.array([gains_for(n)[0] for n in names])
    kd = np.array([gains_for(n)[1] for n in names])
    idx = {n: i for i, n in enumerate(names)}
    fmax = -m.actuator_gear[:, 0]
    sfl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "foot_l_site")
    sfr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "foot_r_site")
    pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    com_prev = d.subtree_com[0].copy()
    x0 = 20.0 / fmax
    reg = np.sqrt(0.1) * np.eye(m.nu)
    for step in range(int(2.1 / m.opt.timestep)):
        t = step * m.opt.timestep
        com = d.subtree_com[0]
        com_v = (com - com_prev) / m.opt.timestep
        com_prev = com.copy()
        sup = 0.5 * (d.site_xpos[sfl] + d.site_xpos[sfr])
        tau = -kp * d.qpos[qadr] - kd * d.qvel[vadr]
        fx = 400 * (com[0] - sup[0]) + 150 * com_v[0]
        fy = 400 * (com[1] - sup[1]) + 150 * com_v[1]
        for n in ("ankle_pitch_l", "ankle_pitch_r"):
            tau[idx[n]] += fx
        for n in ("ankle_roll_l", "ankle_roll_r"):
            tau[idx[n]] -= fy
        mom = np.zeros((m.nu, m.nv))
        mujoco.mju_sparse2dense(mom, d.actuator_moment, d.moment_rownnz,
                                d.moment_rowadr, d.moment_colind)
        sol = lsq_linear(np.vstack([mom[:, 6:].T, reg]),
                         np.concatenate([tau, np.sqrt(0.1) * x0]),
                         bounds=(0.0, 1.0), max_iter=30, tol=1e-6)
        d.ctrl[:] = sol.x
        d.xfrc_applied[pelvis, 0] = 60.0 if 2.0 <= t else 0.0
        mujoco.mj_step(m, d)

    r = mujoco.Renderer(m, 900, 1200)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, 0.8]
    cam.distance = 2.2
    cam.elevation = -12
    cam.azimuth = 125
    r.update_scene(d, cam)
    draw_forces(m, d, r.scene)
    from PIL import Image
    out = ("/private/tmp/claude-502/-Users-ncorrell-Downloads-tensegrity/"
           "6f5ff018-aab3-44d6-b857-b5ad178376ee/scratchpad/forces.png")
    Image.fromarray(r.render()).save(out)
    T = cable_tensions(m, d)
    print(f"rendered; {int((T > TENSION_MIN).sum())} cables above "
          f"{TENSION_MIN:.0f} N, max tension {T.max():.0f} N")


if __name__ == "__main__":
    main()
