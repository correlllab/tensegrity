#!/usr/bin/env python
"""E14: a standing controller for the hinge-free humanoid.

The first attempt (E13) asked a bounded tension distribution to realise a
generalised force on all 78 degrees of freedom, and it fell over. The reason is
structural rather than a matter of gains: cable tensions are INTERNAL forces.
They cannot change the total linear or angular momentum of the robot, so six of
those 78 components are unreachable no matter what the least-squares solver
returns, and asking for them corrupts the whole solution.

This controller is built the other way round.

1. It works in RELATIVE WRENCH space. The cables crossing one interface apply
   +W to the segment above it and -W to the segment below, which is internal by
   construction. Each interface is solved independently -- the wrench matrix is
   block diagonal by cable -- so the whole thing is a set of small bounded
   least-squares problems rather than one large ill-posed one.

2. Each interface gets a gravity FEEDFORWARD: the static wrench needed to
   support everything it carries. Without this the PD term has to generate
   support from position error, which means sagging until the error is large
   enough, and with unilateral actuators that sag does not recover.

3. Posture is held by PD on the relative pose across each interface.

4. Balance is an ANKLE STRATEGY expressed as a virtual force: the foot-shank
   interfaces apply the moment that a desired restoring force at the centre of
   mass would produce, which gets the signs right without hand-tuning them.

5. The passive prestress already generates part of the required wrench, so it
   is subtracted before the actuators are asked for the remainder.
"""
import json
import sys
import numpy as np
import mujoco
from scipy.optimize import lsq_linear

sys.path.insert(0, "/Users/ncorrell/Downloads/tensegrity/experiments")

MJ = "/Users/ncorrell/Downloads/tensegrity/mujoco"
RES = "/Users/ncorrell/Downloads/tensegrity/experiments/results"
OUT = f"{RES}/e14_stand.json"
G = 9.81

# load path: which segment each interface supports, and its share of the body
UPPER = ["pelvis", "torso", "head", "uarm_l", "farm_l", "uarm_r", "farm_r"]
TREE = [
    # (lower, upper, carried bodies, share)
    ("foot_l", "shank_l", ["shank_l", "thigh_l"], 1.0, UPPER, 0.5),
    ("foot_r", "shank_r", ["shank_r", "thigh_r"], 1.0, UPPER, 0.5),
    ("shank_l", "thigh_l", ["thigh_l"], 1.0, UPPER, 0.5),
    ("shank_r", "thigh_r", ["thigh_r"], 1.0, UPPER, 0.5),
    ("thigh_l", "pelvis", [], 0.0, UPPER, 0.5),
    ("thigh_r", "pelvis", [], 0.0, UPPER, 0.5),
    ("pelvis", "torso", ["torso", "head", "uarm_l", "farm_l",
                         "uarm_r", "farm_r"], 1.0, [], 0.0),
    ("torso", "head", ["head"], 1.0, [], 0.0),
    ("torso", "uarm_l", ["uarm_l", "farm_l"], 1.0, [], 0.0),
    ("torso", "uarm_r", ["uarm_r", "farm_r"], 1.0, [], 0.0),
    ("uarm_l", "farm_l", ["farm_l"], 1.0, [], 0.0),
    ("uarm_r", "farm_r", ["farm_r"], 1.0, [], 0.0),
]
ANKLES = {("foot_l", "shank_l"), ("foot_r", "shank_r")}


class StandController:
    def __init__(self, m, d, kp=(900.0, 260.0), kd=(90.0, 26.0),
                 com_kp=900.0, com_kd=260.0, rate_hz=200.0, reg=0.03,
                 tree=None, grounded=None, feedforward=False):
        """`tree` is a list of (lower, upper) body-name pairs and `grounded`
        the bodies standing on the floor. Given those, the set each interface
        must carry, and hence its gravity feedforward, is derived by walking
        the graph outward from the ground -- so the controller is not specific
        to one robot."""
        self.m, self.d = m, d
        self.kp_lin, self.kp_ang = kp
        self.kd_lin, self.kd_ang = kd
        self.com_kp, self.com_kd = com_kp, com_kd
        # With a form-found prestress the structure already carries gravity at
        # the home pose, so the gravity feedforward would double-count it and
        # the controller would fight its own equilibrium. Leave it off unless
        # the model was built with an arbitrary prestress.
        self.feedforward = feedforward
        self.reg = reg
        self.decim = max(1, int(round(1.0 / (rate_hz * m.opt.timestep))))
        self.step_i = 0
        self.bid = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b): b
                    for b in range(m.nbody)}
        self.fmax = -m.actuator_gear[:, 0]

        # tendon -> the body pair it crosses
        self.ten_sites = []
        for t in range(m.ntendon):
            adr = m.tendon_adr[t]
            sids = [int(m.wrap_objid[adr + i]) for i in range(m.tendon_num[t])
                    if m.wrap_type[adr + i] == mujoco.mjtWrap.mjWRAP_SITE]
            self.ten_sites.append(sids)
        self.act_of_ten = {int(m.actuator_trnid[a, 0]): a for a in range(m.nu)}

        # derive the load tree: what does each interface have to hold up?
        if tree is None:
            tree = [(a, b) for a, b, *_ in TREE]
            grounded = grounded or ["foot_l", "foot_r"]
        grounded = grounded or []
        adj = {}
        for a, b in tree:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)

        def beyond(lower, upper):
            """bodies on the far side of the (lower, upper) interface"""
            seen, stack = {lower, upper}, [upper]
            out = [upper]
            while stack:
                x = stack.pop()
                for y in adj.get(x, ()):
                    if y not in seen:
                        seen.add(y)
                        out.append(y)
                        stack.append(y)
            return out

        n_ground = max(1, len(grounded))
        spec = []
        for a, b in tree:
            far = beyond(a, b)
            # a body reachable from more than one grounded limb is shared
            shares = []
            for nm in far:
                paths = sum(1 for g in grounded
                            if nm in beyond(g, g) or nm not in grounded)
                shares.append(1.0)
            legs = [g for g in grounded if b in beyond(g, g)]
            share = 1.0 / n_ground if all(g not in far for g in grounded) and \
                len([g for g in grounded if a in beyond(g, g)]) else 1.0
            spec.append((a, b, far, share))
        self.ankles = {(a, b) for a, b in tree if a in grounded}

        # group cables by interface
        self.iface = []
        for lower, upper, carried, cshare in spec:
            if lower not in self.bid or upper not in self.bid:
                continue
            extra, eshare = [], 0.0
            bl, bu = self.bid[lower], self.bid[upper]
            cabs = []
            for t, sids in enumerate(self.ten_sites):
                if len(sids) < 2:
                    continue
                sb = [m.site_bodyid[s] for s in sids]
                if set(sb) == {bl, bu}:
                    cabs.append(t)
            if not cabs:
                continue
            mass = sum(m.body_mass[self.bid[b]] for b in carried if b in
                       self.bid) * cshare
            mass += sum(m.body_mass[self.bid[b]] for b in extra if b in
                        self.bid) * eshare
            self.iface.append(dict(lower=bl, upper=bu, name=f"{lower}->{upper}",
                                   cables=cabs, carried_mass=mass,
                                   carried=[self.bid[b] for b in carried
                                            if b in self.bid],
                                   cshare=cshare,
                                   extra=[self.bid[b] for b in extra
                                          if b in self.bid], eshare=eshare,
                                   ankle=(lower, upper) in self.ankles,
                                   x=np.zeros(len(cabs))))
        mujoco.mj_forward(m, d)
        self.home = {}
        for it in self.iface:
            self.home[it["name"]] = self._rel(it)

        # Co-contraction bias. A tension-only actuator on top of a form-found
        # prestress can only ADD load: from equilibrium it can push one way but
        # never pull back. Biasing each interface to a positive point in the
        # NULL SPACE of its own wrench matrix -- a self-stress of that joint --
        # costs no net wrench but lets the controller modulate in both
        # directions. This is co-contraction, and without it the controller can
        # only ever disturb an equilibrium it is trying to hold.
        from scipy.optimize import linprog
        for it in self.iface:
            origin = 0.5 * (d.xpos[it["lower"]] + d.xpos[it["upper"]])
            B, _ = self._wrench_matrix(it, origin)
            n = B.shape[1]
            c = np.zeros(n + 1)
            c[-1] = -1.0
            r = linprog(c, A_ub=np.hstack([-np.eye(n), np.ones((n, 1))]),
                        b_ub=np.zeros(n),
                        A_eq=np.hstack([B, np.zeros((6, 1))]),
                        b_eq=np.zeros(6),
                        bounds=[(0, 1)] * n + [(0, 0.25)])
            it["x0"] = (r.x[:n] if r.success and r.x[-1] > 1e-9
                        else np.zeros(n))
            it["x"] = it["x0"].copy()
        # support reference: the bodies actually standing on the floor
        self.feet = [self.bid[b] for b in (grounded or []) if b in self.bid]
        if not self.feet:
            self.feet = [self.bid[b] for b in ("foot_l", "foot_r")
                         if b in self.bid]
        self.floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    # ---------------------------------------------------------------- helpers
    def _rel(self, it):
        """Pose of the upper segment expressed in the lower segment's frame."""
        d = self.d
        Rl = d.xmat[it["lower"]].reshape(3, 3)
        p = Rl.T @ (d.xpos[it["upper"]] - d.xpos[it["lower"]])
        Rrel = Rl.T @ d.xmat[it["upper"]].reshape(3, 3)
        return p, Rrel

    def _wrench_matrix(self, it, origin):
        """Unit-control wrench each cable of this interface applies to the
        upper segment, about `origin`, plus the wrench the passive prestress
        already contributes."""
        m, d = self.m, self.d
        B = np.zeros((6, len(it["cables"])))
        w_pas = np.zeros(6)
        for j, t in enumerate(it["cables"]):
            sids = self.ten_sites[t]
            up = [s for s in sids if m.site_bodyid[s] == it["upper"]]
            lo = [s for s in sids if m.site_bodyid[s] == it["lower"]]
            if not up or not lo:
                continue
            su, sl = up[0], lo[0]
            v = d.site_xpos[sl] - d.site_xpos[su]
            n = np.linalg.norm(v)
            if n < 1e-9:
                continue
            u = v / n                                  # pull on the upper site
            r = d.site_xpos[su] - origin
            w = np.concatenate([u, np.cross(r, u)])
            a = self.act_of_ten.get(t)
            if a is not None:
                B[:, j] = w * self.fmax[a]
            T = m.tendon_stiffness[t] * max(
                0.0, float(d.ten_length[t]) - m.tendon_lengthspring[t, 1])
            w_pas += w * T
        return B, w_pas

    # ------------------------------------------------------------------- step
    def step(self):
        if self.step_i % self.decim:
            self.step_i += 1
            return
        self.step_i += 1
        m, d = self.m, self.d
        com = d.subtree_com[0].copy()
        comv = np.zeros(3)
        # centroid of the live contact points is the true support reference;
        # fall back to the grounded bodies before contact is established
        pts = [d.contact[i].pos for i in range(d.ncon)
               if self.floor in (d.contact[i].geom[0], d.contact[i].geom[1])]
        if pts:
            sup = np.mean(pts, axis=0)
        elif self.feet:
            sup = np.mean([d.xpos[b] for b in self.feet], axis=0)
        else:
            sup = com
        if not hasattr(self, "_com_prev"):
            self._com_prev = com.copy()
        comv = (com - self._com_prev) / (m.opt.timestep * self.decim)
        self._com_prev = com.copy()
        # virtual restoring force at the centre of mass
        f_com = np.array([-self.com_kp * (com[0] - sup[0]) - self.com_kd * comv[0],
                          -self.com_kp * (com[1] - sup[1]) - self.com_kd * comv[1],
                          0.0])

        ctrl = np.zeros(m.nu)
        for it in self.iface:
            origin = 0.5 * (d.xpos[it["lower"]] + d.xpos[it["upper"]])

            # 1. gravity feedforward: hold up everything above this interface
            mass = it["carried_mass"]
            bodies = [(b, it["cshare"]) for b in it["carried"]]
            bodies += [(b, it["eshare"]) for b in it["extra"]]
            if self.feedforward:
                F = np.array([0.0, 0.0, mass * G])
                M = np.zeros(3)
                for b, sh in bodies:
                    M += np.cross(d.xpos[b] - origin,
                                  np.array([0, 0, sh * m.body_mass[b] * G]))
                w_des = np.concatenate([F, M])
            else:
                w_des = np.zeros(6)

            # 2. posture PD on the relative pose
            p, R = self._rel(it)
            p0, R0 = self.home[it["name"]]
            Rl = d.xmat[it["lower"]].reshape(3, 3)
            e_lin = Rl @ (p - p0)
            dR = R0.T @ R
            ang = np.zeros(3)
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, dR.flatten())
            mujoco.mju_quat2Vel(ang, q, 1.0)
            e_ang = Rl @ ang
            dv = d.cvel[it["upper"]][3:] - d.cvel[it["lower"]][3:]
            dw = d.cvel[it["upper"]][:3] - d.cvel[it["lower"]][:3]
            w_des[:3] += -self.kp_lin * e_lin - self.kd_lin * dv
            w_des[3:] += -self.kp_ang * e_ang - self.kd_ang * dw

            # 3. ankle strategy as the moment of a virtual force at the CoM
            if it["ankle"]:
                w_des[3:] += 0.5 * np.cross(com - origin, f_com)

            # 4. remove what the prestress already supplies, then distribute
            B, w_pas = self._wrench_matrix(it, origin)
            # the actuators add tension on top of the prestress, so only the
            # deviation from equilibrium is theirs to supply
            b = w_des - (w_pas if self.feedforward else np.zeros(6))
            n = B.shape[1]
            # The bias belongs in the REGULARISER, not the target: x0 is a
            # null-space vector at the home pose, so B x0 is only zero there.
            # Adding it to the target makes the bias itself command a wrench
            # that grows with deflection, which pumps the structure.
            A = np.vstack([B, np.sqrt(self.reg) * np.eye(n)])
            y = np.concatenate([b, np.sqrt(self.reg) * it["x0"]])
            sol = lsq_linear(A, y, bounds=(0.0, 1.0), max_iter=12, tol=1e-4)
            it["x"] = sol.x
            it["dbg"] = dict(w_des=w_des.copy(), b=b.copy(),
                             e_lin=e_lin.copy(), e_ang=e_ang.copy(),
                             dv=dv.copy(), dw=dw.copy(),
                             achieved=B @ sol.x)
            for j, t in enumerate(it["cables"]):
                a = self.act_of_ten.get(t)
                if a is not None:
                    ctrl[a] = sol.x[j]
        d.ctrl[:] = ctrl


def evaluate(xml_path, seconds=5.0, push=None, **kw):
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    c = StandController(m, d, **kw)
    head = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "head")
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
    mujoco.mj_forward(m, d)
    z0 = float(d.xpos[head][2])
    zmin, peakT = z0, 0.0
    for _ in range(int(seconds / m.opt.timestep)):
        c.step()
        if push and push[0] <= d.time < push[0] + 0.2:
            d.xfrc_applied[torso, 0] = push[1]
        else:
            d.xfrc_applied[torso, :3] = 0.0
        mujoco.mj_step(m, d)
        if not np.all(np.isfinite(d.qpos)):
            return dict(ok=False, z0=z0, z=float("nan"), reason="diverged")
        zmin = min(zmin, float(d.xpos[head][2]))
        peakT = max(peakT, float((c.fmax * d.ctrl).max()))
    z = float(d.xpos[head][2])
    return dict(ok=bool(z > 0.85 * z0), z0=z0, z=z, zmin=zmin, peak_T=peakT)


def main():
    xml = f"{MJ}/humanoid_hingefree_walker.xml"
    m = mujoco.MjModel.from_xml_path(xml)
    print(f"HINGE-FREE HUMANOID: {m.nv} DoF, {m.ntendon} cables, "
          f"{m.nu} tension-only actuators")
    res = {}
    r = evaluate(xml, seconds=5.0)
    print(f"\n  5 s quiet stand : head {r['z0']:.3f} -> {r['z']:.3f} m "
          f"(min {r.get('zmin', float('nan')):.3f})  "
          f"{'HOLDS' if r['ok'] else 'FALLS'}"
          + (f", peak cable {r['peak_T']:.0f} N" if r["ok"] else ""))
    res["stand"] = r
    if r["ok"]:
        for f in (20.0, 40.0, 60.0):
            rp = evaluate(xml, seconds=5.0, push=(2.0, f))
            print(f"  {f:.0f} N torso push: "
                  f"head -> {rp['z']:.3f} m  "
                  f"{'HOLDS' if rp['ok'] else 'FALLS'}")
            res[f"push{int(f)}"] = rp
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nE14 ->", OUT)


if __name__ == "__main__":
    main()
