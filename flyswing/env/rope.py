"""MuJoCo world + rope (spatial tendon with a length limit).

Rope semantics
  shoot(dir)   : mj_ray from the fly along dir; the first static surface it hits
                 (ceiling, skyway, building) becomes the new anchor and the rope is made taut.
  reel(speed)  : every control tick the tendon upper range shrinks by speed*dt.
  release()    : upper range -> 10 m (slack).

The fly is a 1 mg capsule on 3 slide joints (no rotation: at this mass/size the
rotational dynamics add nothing but numerical trouble).
"""
import numpy as np
import mujoco

CEILING_Z = 0.15       # anchor surface (lab ceiling / city skyway) height above the floor
FLY_STAND_Z = 0.0011   # fly body-centre height when standing on the floor
FLY_MASS = 1e-6        # kg (1 mg)
FLY_RADIUS = 0.0006    # m
FLY_HALF = 0.00065     # capsule half-length -> total length ~2.5 mm
N_BALLS = 6            # projectile bodies
N_OBST = 10            # static obstacle pillars (course mode); parked far away otherwise
PARK = np.array([3.0, 3.0, -1.0])   # where unused geoms wait
BALL_RGBA = (".85 .85 .85 .95", ".75 .8 .9 .95", ".9 .8 .7 .95", ".8 .9 .8 .95", ".9 .75 .85 .95", ".8 .8 .7 .95")

XML = f"""
<mujoco model="fly-swing">
  <option timestep="1e-4" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <map znear="0.0005" zfar="20"/>
    <headlight ambient=".45 .45 .45" diffuse=".7 .7 .7" specular=".1 .1 .1"/>
  </visual>
  <asset>
    __SCENE_ASSETS__
  </asset>
  <worldbody>
    <light pos="0.2 -0.3 0.6" dir="-0.3 0.4 -1" diffuse=".9 .9 .9"/>
    <light pos="-0.3 0.2 0.5" dir="0.4 -0.3 -1" diffuse=".5 .5 .5"/>
    <!-- floor: the fly stands on it (contact tuned for a 1 mg body); balls pass through (visual only) -->
    <geom name="floor" type="plane" size="3 3 0.01" pos="0 0 0" material="floor" contype="1" conaffinity="1"
          solref="0.002 1" solimp="0.95 0.99 0.0001" friction="1 0.005 0.0001"/>
    __SCENE_GEOMS__
    {''.join(f'<geom name="obst{i}" type="cylinder" size="0.01 {CEILING_Z / 2}" pos="3 {3 + 0.05 * i} -1" material="obst" contype="1" conaffinity="1" solref="0.002 1" solimp="0.95 0.99 0.0001"/>' for i in range(N_OBST))}
    <site name="goal_flag" type="cylinder" size="0.0015 0.02" pos="3 3 -1" rgba=".1 .8 .2 1"/>
    <site name="goal_disc" type="cylinder" size="0.015 0.0003" pos="3 3 -1" rgba=".1 .8 .2 .7"/>
    <site name="start_disc" type="cylinder" size="0.015 0.0003" pos="3 3 -1" rgba=".9 .3 .2 .7"/>
    <body name="anchor" mocap="true" pos="0 0 {CEILING_Z}">
      <site name="anchor_site" size="0.0008" rgba="1 1 1 1"/>
    </body>
    <body name="fly" pos="0 0 0">
      <joint name="fly_x" type="slide" axis="1 0 0"/>
      <joint name="fly_y" type="slide" axis="0 1 0"/>
      <joint name="fly_z" type="slide" axis="0 0 1"/>
      <!-- physical body: one 1 mg capsule along x (head at +x) -->
      <geom name="fly_geom" type="capsule" size="{FLY_RADIUS} {FLY_HALF}" mass="{FLY_MASS}"
            zaxis="1 0 0" pos="0 0 -0.0005" rgba="0 0 0 0" contype="1" conaffinity="1" group="3"
            solref="0.002 1" solimp="0.95 0.99 0.0001" friction="1 0.005 0.0001"/>
      <site name="fly_site" size="0.0002" pos="0.0002 0 0.0005"/>
    </body>
    {''.join(f'''<body name="ball{i}" pos="0 0 0" gravcomp="1">
      <joint name="ball{i}_x" type="slide" axis="1 0 0"/>
      <joint name="ball{i}_y" type="slide" axis="0 1 0"/>
      <joint name="ball{i}_z" type="slide" axis="0 0 1"/>
      <geom name="ball{i}_geom" type="sphere" size="0.02" mass="1e-3" rgba="{BALL_RGBA[i % len(BALL_RGBA)]}" contype="0" conaffinity="0"/>
    </body>''' for i in range(N_BALLS))}
  </worldbody>
  <tendon>
    <spatial name="rope" limited="true" range="0 0.05" width="0.00005" rgba="1 1 1 1"
             solreflimit="0.001 1" solimplimit="0.95 0.99 0.0001">
      <site site="anchor_site"/>
      <site site="fly_site"/>
    </spatial>
  </tendon>
</mujoco>
"""


_CEIL = f'<geom name="ceiling" type="box" size="1.5 0.25 0.002" pos="0 0 {CEILING_Z + 0.002}" material="ceiling" contype="1" conaffinity="1" solref="0.002 1" solimp="0.95 0.99 0.0001"/>'

SCENES = {
    # lab: checker floor + ceiling slab, blue-ish gradient sky
    "lab": dict(
        assets='<texture type="skybox" builtin="gradient" rgb1=".55 .65 .85" rgb2=".95 .95 1" width="32" height="32"/>'
               '<texture name="grid" type="2d" builtin="checker" rgb1=".85 .85 .9" rgb2=".7 .7 .8" width="256" height="256"/>'
               '<material name="floor" texture="grid" texrepeat="60 60" reflectance="0"/>'
               '<material name="ceiling" texture="grid" texrepeat="30 30" reflectance="0"/>'
               '<material name="obst" rgba=".55 .3 .3 1"/>',
        geoms=_CEIL),
    # city: asphalt floor, an overhead skyway (the web anchor surface), building blocks around
    "city": dict(
        assets='<texture type="skybox" builtin="gradient" rgb1=".35 .45 .65" rgb2=".9 .75 .6" width="32" height="32"/>'
               '<texture name="asphalt" type="2d" builtin="checker" rgb1=".2 .2 .22" rgb2=".24 .24 .26" width="64" height="64"/>'
               '<texture name="concrete" type="2d" builtin="checker" rgb1=".5 .5 .52" rgb2=".45 .45 .47" width="64" height="64"/>'
               '<texture name="win" type="2d" builtin="checker" rgb1=".25 .3 .4" rgb2=".75 .8 .9" width="64" height="64"/>'
               '<material name="floor" texture="asphalt" texrepeat="200 200" reflectance="0"/>'
               '<material name="ceiling" texture="concrete" texrepeat="20 20" reflectance="0"/>'
               '<material name="bldg" texture="win" texrepeat="6 12" reflectance="0.1"/>'
               '<material name="lane" rgba=".95 .85 .3 1"/>'
               '<material name="obst" rgba=".4 .4 .45 1"/>',
        geoms=_CEIL +
              ''.join(f'<geom type="box" size="{sx} {sy} {sz}" pos="{x} {y} {sz}" material="bldg" contype="0" conaffinity="0"/>'
                      for x, y, sx, sy, sz in ((0.45, 0.45, 0.12, 0.10, 0.9), (0.05, 0.5, 0.15, 0.12, 0.7),
                                               (-0.5, 0.4, 0.10, 0.14, 1.1), (0.25, 0.75, 0.30, 0.10, 0.8),
                                               (0.7, 0.8, 0.28, 0.12, 1.0), (1.0, 0.0, 0.10, 0.35, 1.3),
                                               (-0.6, 0.05, 0.10, 0.30, 0.85), (0.9, -0.5, 0.12, 0.12, 0.6))) +
              ''.join(f'<geom type="box" size="0.04 0.004 0.0002" pos="{x} 0 0.0002" material="lane" contype="0" conaffinity="0"/>'
                      for x in np.arange(-1.0, 1.01, 0.16)) +
              ''.join(f'<geom type="cylinder" size="0.004 {CEILING_Z / 2}" pos="{x} {y} {CEILING_Z / 2}" rgba=".3 .3 .3 1" contype="0" conaffinity="0"/>'
                      for x, y in ((0.1, 0.3), (0.4, 0.3), (-0.2, 0.3), (0.7, 0.3)))),
}


def make_model(scene="lab"):
    s = SCENES[scene]
    xml = XML.replace("__SCENE_ASSETS__", s["assets"]).replace("__SCENE_GEOMS__", s["geoms"])
    return mujoco.MjModel.from_xml_string(xml)


class Rope:
    SLACK = 10.0
    MIN_LEN = 0.008

    def __init__(self, model, data):
        self.m, self.d = model, data
        self.tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, "rope")
        self.anchor_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "anchor")
        self.mocap_id = model.body_mocapid[self.anchor_body]
        self.fly_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fly")
        self.ceiling_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ceiling")
        self.floor_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.reel_speed = 0.0
        self._geomid = np.zeros(1, dtype=np.int32)

    # ---- state ----
    @property
    def max_length(self):
        return float(self.m.tendon_range[self.tid, 1])

    @property
    def length(self):
        return float(self.d.ten_length[self.tid])

    @property
    def anchor(self):
        return self.d.mocap_pos[self.mocap_id].copy()

    @property
    def taut(self):
        return self.length >= self.max_length - 1e-5

    def direction(self):
        """Unit vector fly -> anchor."""
        v = self.anchor - self.d.xpos[self.fly_body]
        n = np.linalg.norm(v)
        return v / n if n > 0 else np.array([0, 0, 1.0])

    # ---- actions ----
    def set_length(self, L):
        self.m.tendon_range[self.tid] = (0.0, max(L, self.MIN_LEN))

    def attach(self, anchor_pos, taut=True):
        self.d.mocap_pos[self.mocap_id] = anchor_pos
        mujoco.mj_forward(self.m, self.d)          # refresh ten_length for the new anchor
        if taut:
            self.set_length(self.length)

    def shoot(self, direction, min_rise=0.015, min_len=0.02):
        """Raycast from the fly; the first static surface hit (ceiling, skyway, building,
        pillar) becomes the anchor. Rejected: the floor, anchors less than `min_rise`
        above the fly (a pillar base would just tie the fly down) and ropes shorter
        than `min_len`. Returns True on success."""
        direction = np.asarray(direction, float)
        n = np.linalg.norm(direction)
        if n == 0:
            return False
        direction = direction / n
        origin = self.d.xpos[self.fly_body].copy()
        dist = mujoco.mj_ray(self.m, self.d, origin, direction, None, 1, self.fly_body, self._geomid)
        g = int(self._geomid[0])
        if dist < 0 or g == self.floor_geom or self.m.geom_bodyid[g] != 0:
            return False
        hit = origin + direction * dist
        if hit[2] - origin[2] < min_rise or dist < min_len:
            return False
        self.attach(hit)
        return True

    def release(self):
        self.reel_speed = 0.0
        self.set_length(self.SLACK)

    def reel(self, speed):
        self.reel_speed = float(speed)

    def update(self, dt):
        """Call once per physics step (or control tick with dt = tick)."""
        if self.reel_speed > 0.0:
            L = min(self.max_length, self.length) - self.reel_speed * dt
            self.set_length(L)
