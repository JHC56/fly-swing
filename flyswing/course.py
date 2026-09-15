"""Course mode: start -> goal through random obstacles, scored by the competition rules.

    +5,000,000  reaching the goal          (GOAL_BONUS)
    -100,000    per collision              (HIT_PENALTY)   obstacle pillar or projectile
    +1,000      per web-swing              (SWING_BONUS)   every successful web shot
    -2,000      instead, for the 5th+ web-swing in a row (CHAIN_LIMIT / CHAIN_PENALTY); landing resets the chain
    0 (void)    if the fly ever passes through the floor or the ceiling (both are solid, so this is a physics guard)

The fly starts standing at the start disc, web-swings along +x (Tarzan cycle: release on the
up-swing, new web at the apex), evades looming pillars and projectiles with the GF reflex +
whitetree memory, lands on the goal disc.

    python -m flyswing.course --episodes 20      # score statistics
"""
import argparse, json, os, time
import numpy as np
import mujoco

from flyswing.env import FlySwingEnv
from flyswing.env.rope import N_BALLS, N_OBST, CEILING_Z
from flyswing.circuit import load_circuit, calibrate_gain
from flyswing.memory import ThreatMemory
from flyswing.policies import Warmup, run_episode
from flyswing.swing_policy import SwingPolicy, FORWARD

GOAL_BONUS, HIT_PENALTY, SWING_BONUS = 5_000_000, -100_000, 1_000
CHAIN_LIMIT, CHAIN_PENALTY = 4, -2_000     # a 5th (or later) consecutive web-swing costs -2000 instead of +1000


OBSTACLE_KINDS = ("pillar", "slider", "slider", "hanging", "low")   # draw weights


def make_course(rng, length=0.5, n_obst=None):
    """Random obstacles between start and goal (see FlySwingEnv.set_obstacles):
    pillar  - fixed, floor to ceiling
    slider  - full-height pillar sliding sideways (amp 1.5-4 cm, period 0.5-1.5 s)
    hanging - bar from the ceiling down to 5-9 cm above the floor: swing UNDER it
    low     - wall/post 3-6 cm tall on the floor: swing OVER it"""
    n = int(rng.integers(6, N_OBST + 1)) if n_obst is None else n_obst
    xs = np.sort(rng.uniform(0.08, length - 0.06, n))
    out = []
    for x in xs:
        kind = OBSTACLE_KINDS[int(rng.integers(len(OBSTACLE_KINDS)))]
        ob = dict(x=float(x), y=float(rng.uniform(-0.04, 0.04)), r=float(rng.uniform(0.004, 0.012)),
                  z=CEILING_Z / 2, h=CEILING_Z / 2, amp=0.0, omega=0.0, phase=0.0, kind=kind)
        if kind == "slider":
            ob.update(amp=float(rng.uniform(0.015, 0.04)), omega=float(2 * np.pi / rng.uniform(0.5, 1.5)),
                      phase=float(rng.uniform(0, 2 * np.pi)))
        elif kind == "hanging":
            bottom = float(rng.uniform(0.05, 0.09))
            ob.update(z=(CEILING_Z + bottom) / 2, h=(CEILING_Z - bottom) / 2, y=float(rng.uniform(-0.02, 0.02)),
                      r=float(rng.uniform(0.006, 0.015)))
        elif kind == "low":
            top = float(rng.uniform(0.03, 0.06))
            ob.update(z=top / 2, h=top / 2, y=float(rng.uniform(-0.02, 0.02)), r=float(rng.uniform(0.006, 0.015)))
        out.append(ob)
    return out


class CoursePolicy(SwingPolicy):
    """SwingPolicy that starts swinging at once and lands only at the goal."""
    name = "course"

    def __init__(self, gain, memory, goal_x, circ=None, seed=0):
        super().__init__(gain, memory, circ, seed, explore=0.0)
        self.goal_x = goal_x

    forward_only = True
    lane_gain = 25.0            # 1 cm off the centre line -> 0.25 of sideways web direction back toward it

    def _locomotion(self, env):
        p, v = env.fly_pos, env.fly_vel
        if self.mode == "swing" and env.rope.taut and v[0] < -0.05 and env.t - self.mode_t > 0.05:
            # swinging backwards: let go and re-launch forward instead of pendulum-ing back
            env.rope.release()
            self._set(env, "fly" if p[2] > 0.02 and not self._must_land() else "land")
            return
        if self.mode == "stand":
            if env.t - self.mode_t > 0.05 and p[0] < self.goal_x - 0.03:
                # take off: forward-up web; if that surface is unusable try steeper / sideways, else hop forward
                for d in ((0.5, 0, 1.0), (0.25, 0, 1.0), (0.5, 0.4, 1.0), (0.5, -0.4, 1.0), (0.0, 0, 1.0)):
                    if self._web(env, self._forward(np.array(d), env), reel=0.6):     # first pump from the ground
                        self.swings += 1
                        self._set(env, "swing")
                        return
                env.jump(FORWARD * 0.8 + np.array([0, 0, 0.6]), 0.4)
                self.jumps += 1
                self._set(env, "land")
            return
        if self.mode in ("swing", "fly") and p[0] >= self.goal_x - 0.025 and v[0] > 0:
            env.rope.release()
            self._set(env, "land")
            return
        # never idle-land before the goal: pretend a threat is always recent
        self.last_threat = env.t
        super()._locomotion(env)


class Scorer:
    def __init__(self):
        self.hits = self.swings = self.chain_violations = 0
        self.swing_log = []
        self.goal = False
        self.void = False            # went through floor/ceiling -> attempt voided (0 points)

    def update(self, policy, hits):
        self.swing_log = list(policy.swing_log)
        self.swings = len(self.swing_log)
        self.chain_violations = sum(1 for c in self.swing_log if c > CHAIN_LIMIT)
        self.hits = hits

    @property
    def score(self):
        if self.void:
            return 0
        swing_pts = sum(SWING_BONUS if c <= CHAIN_LIMIT else CHAIN_PENALTY for c in self.swing_log)
        return (GOAL_BONUS if self.goal else 0) + HIT_PENALTY * self.hits + swing_pts


def run_course(env, policy, seed, length=0.5, seconds=10.0, projectiles=True, on_tick=None):
    rng = np.random.default_rng(seed + 777)
    f = env.reset(seed=seed, launch=False, wind_max=0.15)
    env.T = seconds
    obstacles = make_course(rng, length)
    env.set_obstacles(obstacles)
    m = env.model
    for name, xy in (("start_disc", (0.0, 0.0)), ("goal_disc", (length, 0.0)), ("goal_flag", (length, 0.0))):
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)
        m.site_pos[sid, :2] = xy
    policy.goal_x = length
    policy.reset(env)
    sc = Scorer()
    next_launch, launches = 0.6, 0
    while env.t < seconds:
        if projectiles and env.t >= next_launch:
            free = [i for i in range(N_BALLS) if not env.active[i]]
            if free:
                speed = float(rng.choice([0.6, 0.8, 1.0, 1.3]))
                radius = float(rng.choice([0.005, 0.008, 0.012]))
                az = rng.uniform(-np.pi / 2, np.pi / 2)            # mostly from ahead
                el = np.radians(rng.uniform(-30, -5))
                d = -np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), -np.sin(el)])
                env.launch(free[0], speed, radius, env.fly_pos, target_vel=env.fly_vel * 0.7,
                           dist=0.10 + 0.25 * speed, direction=d)
                launches += 1
            next_launch = env.t + rng.uniform(0.3, 0.6)
        policy.act(env, f)
        f, _, info = env.step()
        for i in np.flatnonzero(env.active):
            rel = env.ball_pos_i(i) - env.fly_pos
            if np.dot(rel, env.ball_vel_i(i)) > 0 and np.linalg.norm(rel) > 0.08:
                env.park(i)
        sc.update(policy, env.hits + env.obst_hits)
        p = env.fly_pos
        if p[2] < -0.002 or p[2] > CEILING_Z + 0.003:            # beyond the slab: went through -> void
            sc.void = True
            break
        if p[0] >= length - 0.015 and p[2] < 0.004 and np.linalg.norm(env.fly_vel) < 0.05:
            sc.goal = True
        if on_tick is not None:
            on_tick(env, f, sc, launches)
        if sc.goal:
            break
    policy.end(env, info)
    return dict(score=sc.score, goal=sc.goal, void=sc.void, hits=sc.hits, swings=sc.swings, chain_violations=sc.chain_violations,
                jumps=policy.jumps, landings=policy.landings, t=round(env.t, 3), launches=launches,
                obstacles=len(obstacles), kinds="".join(o["kind"][0] for o in obstacles))


def build_policy(env, circuit_json=None, warm_episodes=150):
    circ = load_circuit()
    gain = json.load(open(circuit_json))["gain"] if circuit_json and os.path.exists(circuit_json) \
        else calibrate_gain(circ, verbose=False)
    mem = ThreatMemory(min_fit=5000)
    warm = Warmup(gain, mem, circ)
    rng = np.random.default_rng(1)
    for i in range(warm_episodes):
        run_episode(env, warm, float(rng.choice([0.5, 1, 1.5, 2])), float(rng.choice([0.005, 0.01, 0.02])),
                    seed=30_000_000 + i)
    pol = CoursePolicy(gain, mem, goal_x=0.5, circ=circ)
    pol.frozen = True
    return pol


def evaluate(episodes=20, length=0.5, circuit_json=None, projectiles=True, seed0=0, log=print):
    env = FlySwingEnv()
    policy = build_policy(env, circuit_json)
    rows = []
    t0 = time.time()
    for ep in range(episodes):
        info = run_course(env, policy, seed0 + ep, length, projectiles=projectiles)
        rows.append(info)
        log(f"  ep {ep:3d}: score {info['score']:>10,}  goal={info['goal']}  hits={info['hits']}  "
            f"swings={info['swings']} (chain>4: {info['chain_violations']})  landings={info['landings']}  "
            f"jumps={info['jumps']}  t={info['t']}s  obstacles={info['obstacles']} [{info['kinds']}]")
    scores = np.array([r["score"] for r in rows])
    log(f"episodes {episodes}: mean score {scores.mean():,.0f}, goal rate {np.mean([r['goal'] for r in rows]):.0%}, "
        f"mean hits {np.mean([r['hits'] for r in rows]):.2f}, mean swings {np.mean([r['swings'] for r in rows]):.1f} "
        f"({time.time() - t0:.0f}s)")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--length", type=float, default=0.5)
    ap.add_argument("--no-projectiles", action="store_true")
    ap.add_argument("--circuit-json", default=None)
    a = ap.parse_args()
    evaluate(a.episodes, a.length, a.circuit_json, not a.no_projectiles, a.seed)
