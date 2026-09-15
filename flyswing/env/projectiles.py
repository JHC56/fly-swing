"""The world the fly acts in: projectiles, wind, drag, obstacles, hit detection, reset/step.

The fly stands on the floor, web slack. `launch()` fires a sphere of radius r at speed v
toward a target (aim error ~ N(0, sigma), constant wind pushes both bodies through a drag
model); several balls can be in flight at once. `set_obstacles()` places pillars, sliding
pillars, hanging bars and low walls. `features()` always describes the most urgent
approaching thing (largest angular size). Hits: swept-sphere test for balls, MuJoCo
contacts for obstacles.

Control tick = 1 ms (10 physics substeps of 0.1 ms). Policies act between ticks.
"""
import numpy as np
import mujoco
from .rope import make_model, Rope, CEILING_Z, FLY_RADIUS, FLY_HALF, N_BALLS, N_OBST, FLY_STAND_Z, PARK

FLY_HALF_LEN = FLY_HALF + FLY_RADIUS          # ~1.25 mm
CTRL_DT = 1e-3
RHO_AIR = 1.2
MU_AIR = 1.8e-5
# fly drag: Stokes linear + quadratic (sphere of radius ~1.25 mm, Cd ~0.5)
FLY_B_LIN = 6 * np.pi * MU_AIR * FLY_HALF_LEN                       # N s / m
FLY_C_QUAD = 0.5 * RHO_AIR * 0.5 * np.pi * FLY_HALF_LEN ** 2        # N s^2 / m^2
BALL_WIND_K = 1.0     # 1/s : ball's lateral velocity relaxes toward the wind at this rate
AIM_WIND_COMP = 0.8   # fraction of the expected wind drift the shooter compensates for
FLY_HIT_RADIUS = 0.001  # m : fly treated as a 1 mm sphere for the swept hit test
NO_THREAT = dict(theta=0.0, theta_dot=0.0, az=0.0, el=0.0, ttc=2.0, dist=1e3, ddot=1.0)


def visual_features(fly_pos, fly_vel, ball_pos, ball_vel, r):
    """Analytic looming features seen from the fly (degrees, deg/s, seconds).

    Returns dict: theta (full angular size), theta_dot, az, el, ttc, dist, ddot
    """
    rel = ball_pos - fly_pos
    d = float(np.linalg.norm(rel))
    if d <= r + 1e-9:
        return dict(theta=180.0, theta_dot=0.0, az=0.0, el=0.0, ttc=0.0, dist=d, ddot=0.0)
    alpha = np.arcsin(r / d)
    rel_v = ball_vel - fly_vel
    ddot = float(np.dot(rel, rel_v) / d)
    theta_dot = -2.0 * r * ddot / (d * np.sqrt(max(d * d - r * r, 1e-12)))
    az = np.arctan2(rel[1], rel[0])
    el = np.arcsin(np.clip(rel[2] / d, -1, 1))
    ttc = min(d / max(-ddot, 1e-6), 2.0)
    return dict(theta=np.degrees(2 * alpha), theta_dot=np.degrees(theta_dot),
                az=np.degrees(az), el=np.degrees(el), ttc=ttc, dist=d, ddot=ddot)


class FlySwingEnv:
    def __init__(self, scene="lab"):
        self.scene = scene
        self.model = make_model(scene)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.rope = Rope(m, self.data)
        self.fly_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "fly")
        self.fly_q = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "fly_x")]
        self.fly_v = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "fly_x")]
        self.ball_body = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"ball{i}") for i in range(N_BALLS)]
        self.ball_geom = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"ball{i}_geom") for i in range(N_BALLS)]
        self.ball_q = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"ball{i}_x")]
                       for i in range(N_BALLS)]
        self.ball_v = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"ball{i}_x")]
                       for i in range(N_BALLS)]
        self.fly_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fly_geom")
        self.obst_geom = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"obst{i}") for i in range(N_OBST)]
        self.obstacles = []            # [(pos, radius)] static pillars seen as threats (course mode)
        self.obst_hits = 0
        self._in_contact = False
        self.threat_pos = np.zeros(3)
        self.threat_vel = np.zeros(3)
        self.substeps = int(round(CTRL_DT / m.opt.timestep))
        self.ball_mass = float(m.body_mass[self.ball_body[0]])
        self.drag_on = True
        self.t = 0.0
        self.hit = False
        self.hits = 0
        self.wind = np.zeros(3)
        self.active = np.zeros(N_BALLS, bool)
        self.radii = np.full(N_BALLS, 0.02)
        self.launch_dirs = np.tile([1.0, 0, 0], (N_BALLS, 1))
        self.launch_dir = self.launch_dirs[0]
        self.radius = 0.02
        self.speed = 1.0
        self.T = 1.0
        self.rng = np.random.default_rng(0)

    # ---- accessors ----
    @property
    def fly_pos(self):
        return self.data.qpos[self.fly_q:self.fly_q + 3].copy()

    @property
    def fly_vel(self):
        return self.data.qvel[self.fly_v:self.fly_v + 3].copy()

    def ball_pos_i(self, i):
        return self.data.qpos[self.ball_q[i]:self.ball_q[i] + 3].copy()

    def ball_vel_i(self, i):
        return self.data.qvel[self.ball_v[i]:self.ball_v[i] + 3].copy()

    @property
    def ball_pos(self):
        """Position of the current threat (ball or static obstacle)."""
        return self.threat_pos.copy()

    @property
    def ball_vel(self):
        return self.threat_vel.copy()

    def features(self):
        """Features of the most urgent approaching threat (largest angular size):
        active balls and, in course mode, static obstacle pillars."""
        p, v = self.fly_pos, self.fly_vel
        best_f, bpos, bvel = None, None, None
        cands = [(self.ball_pos_i(i), self.ball_vel_i(i), self.radii[i]) for i in np.flatnonzero(self.active)]
        for ob in self.obstacles:            # cylinders: nearest point on the axis, own velocity if sliding
            y = ob["y"] + ob["amp"] * np.sin(ob["omega"] * self.t + ob["phase"])
            vy = ob["amp"] * ob["omega"] * np.cos(ob["omega"] * self.t + ob["phase"])
            z = float(np.clip(p[2], ob["z"] - ob["h"], ob["z"] + ob["h"]))
            cands.append((np.array([ob["x"], y, z]), np.array([0.0, vy, 0.0]), ob["r"]))
        for pos, vel, r in cands:
            f = visual_features(p, v, pos, vel, r)
            if f["ddot"] < 0 and (best_f is None or f["theta"] > best_f["theta"]):
                best_f, bpos, bvel = f, pos, vel
        if best_f is None:
            if cands:
                bpos, bvel, r = cands[0]
                best_f = visual_features(p, v, bpos, bvel, r)
            else:
                best_f, bpos, bvel = dict(NO_THREAT), p + [1.0, 0, 0], np.zeros(3)
        self.threat_pos, self.threat_vel = bpos, bvel
        return best_f

    # ---- course-mode obstacles (cylinders; may slide in y, hang from the ceiling or stand low) ----
    def set_obstacles(self, obstacles):
        """obstacles: list of dicts {x, y, r, z, h, amp, omega, phase, kind}. z/h = centre/half-height,
        amp/omega/phase = sideways oscillation y(t) = y + amp sin(omega t + phase).
        ((x, y), r) tuples are accepted too (fixed full-height pillar)."""
        self.obstacles = []
        for ob in obstacles:
            if not isinstance(ob, dict):
                (x, y), r = ob
                ob = dict(x=x, y=y, r=r, z=CEILING_Z / 2, h=CEILING_Z / 2, amp=0.0, omega=0.0, phase=0.0, kind="pillar")
            self.obstacles.append({k: float(v) if k != "kind" else v for k, v in ob.items()})
        for i in range(N_OBST):
            g = self.obst_geom[i]
            if i < len(self.obstacles):
                ob = self.obstacles[i]
                self.model.geom_size[g, 0] = ob["r"]
                self.model.geom_size[g, 1] = ob["h"]
                self.model.geom_rbound[g] = np.hypot(ob["r"], ob["h"])
            else:
                self.model.geom_size[g, 1] = CEILING_Z / 2
                self.model.geom_pos[g] = PARK + [0, 0.05 * i, 0]
        self.update_obstacles()
        mujoco.mj_forward(self.model, self.data)

    def update_obstacles(self):
        """Move sliding obstacles to their position at the current time."""
        for i, ob in enumerate(self.obstacles):
            y = ob["y"] + ob["amp"] * np.sin(ob["omega"] * self.t + ob["phase"])
            self.model.geom_pos[self.obst_geom[i]] = [ob["x"], y, ob["z"]]

    def _obstacle_contact(self):
        d = self.data
        for k in range(d.ncon):
            c = d.contact[k]
            other = c.geom2 if c.geom1 == self.fly_geom else (c.geom1 if c.geom2 == self.fly_geom else -1)
            if other in self.obst_geom:
                return True
        return False

    # ---- episode ----
    def set_ball_radius(self, r, i=0):
        self.radii[i] = r
        if i == 0:
            self.radius = r
        self.model.geom_size[self.ball_geom[i], 0] = r
        self.model.geom_rbound[self.ball_geom[i]] = r

    def park(self, i):
        self.active[i] = False
        self.data.qpos[self.ball_q[i]:self.ball_q[i] + 3] = PARK + [0, 0.1 * i, 0]
        self.data.qvel[self.ball_v[i]:self.ball_v[i] + 3] = 0

    def launch(self, i, speed, radius, target, target_vel=None, aim_sigma=None, dist=None,
               direction=None):
        """Launch ball i toward `target` (leading a moving target) from `dist` away."""
        rng = self.rng
        self.set_ball_radius(radius, i)
        if aim_sigma is None:
            aim_sigma = 0.5 * (radius + FLY_HALF_LEN)
        if direction is None:
            az = rng.uniform(0, 2 * np.pi)
            el = np.radians(rng.uniform(-30, -5))     # comes in from above, angled down at the fly
            direction = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        u = np.asarray(direction, float)
        u = u / np.linalg.norm(u)
        D = max(0.5 * speed, 8 * radius + 0.05) if dist is None else dist
        tf = D / speed
        aim = np.asarray(target, float) + (0 if target_vel is None else np.asarray(target_vel) * tf)
        w_perp = self.wind - np.dot(self.wind, u) * u
        drift = w_perp * (tf - (1 - np.exp(-BALL_WIND_K * tf)) / BALL_WIND_K)
        aim = aim - AIM_WIND_COMP * drift + rng.normal(0, aim_sigma, 3)
        self.data.qpos[self.ball_q[i]:self.ball_q[i] + 3] = aim - u * D
        self.data.qvel[self.ball_v[i]:self.ball_v[i] + 3] = u * speed
        self.launch_dirs[i] = u
        self.active[i] = True
        return tf

    def reset(self, speed=1.0, radius=0.02, seed=None, rope_len=None, wind_max=0.3,
              aim_sigma=None, launch=True):
        self.rng = np.random.default_rng(seed)
        rng = self.rng
        mujoco.mj_resetData(self.model, self.data)
        self.t, self.hit, self.hits, self.speed = 0.0, False, 0, speed
        self.obst_hits, self._in_contact = 0, False
        self.set_obstacles([])
        for i in range(N_BALLS):
            self.park(i)
        # fly stands on the floor; the web is slack until it is shot
        fly = np.array([0.0, 0.0, FLY_STAND_Z])
        self.data.qpos[self.fly_q:self.fly_q + 3] = fly
        self.rope.reel_speed = 0.0
        self.rope.attach(np.array([0.0, 0.0, CEILING_Z]), taut=False)
        self.rope.release()
        if rope_len is not None:       # hanging start (used by the rope physics test)
            fly = np.array([0.0, 0.0, CEILING_Z - rope_len])
            self.data.qpos[self.fly_q:self.fly_q + 3] = fly
            self.rope.set_length(rope_len)
        # wind: horizontal, constant per episode
        wa = rng.uniform(0, 2 * np.pi)
        self.wind = rng.uniform(0, wind_max) * np.array([np.cos(wa), np.sin(wa), 0.0])
        self.set_ball_radius(radius, 0)
        if launch:
            tf = self.launch(0, speed, radius, fly, aim_sigma=aim_sigma)
            self.T = tf + 0.25
        else:
            self.T = 1.0
        self.launch_dir = self.launch_dirs[0]
        mujoco.mj_forward(self.model, self.data)
        return self.features()

    def jump(self, direction, v_takeoff=0.6):
        """Release the rope and add take-off velocity (GF -> TTMn jump)."""
        direction = np.asarray(direction, float)
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        self.rope.release()
        self.data.qvel[self.fly_v:self.fly_v + 3] += direction * v_takeoff

    def _apply_drag(self):
        d = self.data
        if not self.drag_on:
            return
        vrel = self.fly_vel - self.wind
        s = np.linalg.norm(vrel)
        d.xfrc_applied[self.fly_body, :3] = -(FLY_B_LIN + FLY_C_QUAD * s) * vrel
        # ball: only the component perpendicular to its launch axis relaxes toward the wind
        for i in np.flatnonzero(self.active):
            u = self.launch_dirs[i]
            bv = self.ball_vel_i(i)
            w_perp = self.wind - np.dot(self.wind, u) * u
            v_perp = bv - np.dot(bv, u) * u
            d.xfrc_applied[self.ball_body[i], :3] = self.ball_mass * BALL_WIND_K * (w_perp - v_perp)

    def _swept_hit(self, rel0, rel1, r):
        """Continuous collision check: min distance of the linearly-moving ball centre
        to the fly centre over one tick, against r + FLY_HIT_RADIUS."""
        dr = rel1 - rel0
        den = float(dr @ dr)
        s = 0.0 if den == 0 else float(np.clip(-(rel0 @ dr) / den, 0.0, 1.0))
        return np.linalg.norm(rel0 + s * dr) <= r + FLY_HIT_RADIUS

    def step(self):
        """Advance one control tick (1 ms). Returns (features, done, info)."""
        if self.obstacles:
            self.update_obstacles()
        self._apply_drag()
        self.rope.update(CTRL_DT)
        act = np.flatnonzero(self.active)
        p0 = self.fly_pos
        rel0 = [self.ball_pos_i(i) - p0 for i in act]
        mujoco.mj_step(self.model, self.data, self.substeps)
        p1 = self.fly_pos
        for i, r0 in zip(act, rel0):
            if self._swept_hit(r0, self.ball_pos_i(i) - p1, self.radii[i]):
                self.hit = True
                self.hits += 1
                self.park(i)          # a ball that hit is spent
        if self.obstacles:
            touching = self._obstacle_contact()
            if touching and not self._in_contact:
                self.obst_hits += 1            # count each new collision once
            self._in_contact = touching
        self.t += CTRL_DT
        f = self.features()
        passed = f["ddot"] > 0 and f["dist"] > 4 * self.radius + 0.02
        done = self.hit or self.t >= self.T or passed
        return f, done, {"hit": self.hit, "t": self.t}
