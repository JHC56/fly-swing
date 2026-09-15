"""Policies that act on FlySwingEnv between 1 ms control ticks.

GFSwing  : GF spike -> 8 ms later (3 ms neural + 5 ms motor) shoot a web perpendicular to the
           approach and reel in.
GFMemory : GFSwing + whitetree threat memory: every 10 ms the memory scores 8 web directions;
           a dangerous "stay" risk triggers the swing early (before the GF), and the direction
           is the one with the lowest remembered hit rate.
Warmup   : random-threshold, random-direction swings used to seed the memory.
"""
import numpy as np
from .circuit import EscapeCircuit
from .memory import ThreatMemory, CANDIDATE_DIRS, state_vector

MOTOR_DELAY = 0.008     # s  (3 ms neural + 5 ms take-off / web launch)
JUMP_V = 0.6            # m/s take-off speed (30 mm jump)
REEL_SPEED = 0.5        # m/s
LIF_MIN_THETA = 3.0     # deg: below this LC drive is negligible, skip the LIF
MEM_TICK = 0.010        # s
MEM_MIN_THETA = 8.0     # deg: memory starts scoring once the object is clearly visible
MEM_STAY_RISK = 0.5     # act early when remembered hit rate of "stay" exceeds this
MEM_MARGIN = 0.1        # ... and some candidate is at least this much safer


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def perpendicular_candidate(env, rng):
    """Index of the rope-direction candidate most perpendicular to the ball's approach."""
    u = _unit(env.ball_vel)
    score = np.linalg.norm(CANDIDATE_DIRS - np.outer(CANDIDATE_DIRS @ u, u), axis=1)
    score = score + rng.normal(0, 1e-3, len(score))
    return int(np.argmax(score))


def jump_direction(env, rng):
    u = _unit(env.ball_vel)
    side = _unit(np.cross(u, [0, 0, 1.0]))
    if np.linalg.norm(side) == 0:
        side = np.array([1.0, 0, 0])
    if rng.random() < 0.5:
        side = -side
    return _unit(side + np.array([0, 0, 0.3]))


def rope_azimuth(env):
    d = env.rope.direction()
    return float(np.degrees(np.arctan2(d[1], d[0]))) if np.hypot(d[0], d[1]) > 1e-4 else 0.0


class Policy:
    name = "base"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self, env):
        # per-episode rng derived from the episode seed -> same seed, same behaviour
        self.rng = np.random.default_rng(int(env.rng.integers(2**31)))

    def act(self, env, f):
        pass

    def end(self, env, info):
        pass



class GFPolicy(Policy):
    """Shared GF machinery: run LIF, schedule a motor action MOTOR_DELAY after the spike."""

    def __init__(self, gain, circ=None, seed=0):
        super().__init__(seed)
        self.circuit = EscapeCircuit(circ, gain=gain, seed=seed)

    def reset(self, env):
        super().reset(env)
        self.circuit.reset()
        self.action_time = None
        self.acted = False
        self.gf_time = None

    def act(self, env, f):
        if not self.acted and self.action_time is None and not self.circuit.gf_spiked \
                and f["theta"] >= LIF_MIN_THETA:
            if self.circuit.step(f["theta"], f["theta_dot"], f["az"], f["el"]):
                self.gf_time = env.t
                self.action_time = env.t + MOTOR_DELAY
        if self.action_time is not None and env.t >= self.action_time and not self.acted:
            self.acted = True
            self.motor(env, f)

    def motor(self, env, f):
        raise NotImplementedError



class GFSwing(GFPolicy):
    name = "gf_swing"

    def swing(self, env, cand_idx):
        if env.rope.shoot(CANDIDATE_DIRS[cand_idx]):
            env.rope.reel(REEL_SPEED)
        else:                       # ray missed the ceiling: fall back to a jump
            env.jump(jump_direction(env, self.rng), JUMP_V)

    def motor(self, env, f):
        self.swing(env, perpendicular_candidate(env, self.rng))


class GFMemory(GFSwing):
    name = "gf_memory"

    def __init__(self, gain, memory: ThreatMemory, circ=None, seed=0, explore=0.1):
        super().__init__(gain, circ, seed)
        self.memory = memory
        self.explore = explore
        self.early = 0

    def reset(self, env):
        super().reset(env)
        self.next_mem = 0.0
        self.chosen = None
        self.last_risks = None

    def _state(self, env, f):
        return state_vector(f, np.linalg.norm(env.wind), np.linalg.norm(env.fly_vel), rope_azimuth(env))

    def act(self, env, f):
        if env.t >= self.next_mem - 1e-9:
            self.next_mem += MEM_TICK
            self.memory.record(env.t, self._state(env, f))
            if not self.acted and self.action_time is None and f["theta"] >= MEM_MIN_THETA \
                    and self.memory.fitted:
                best, stay, cand = self.memory.choose_direction(
                    f, np.linalg.norm(env.wind), np.linalg.norm(env.fly_vel), rope_azimuth(env),
                    explore=self.explore)
                self.chosen = best
                self.last_risks = (stay, cand)          # kept for the neural-activity display
                if stay >= MEM_STAY_RISK and cand[best] <= stay - MEM_MARGIN:
                    self.action_time = env.t + MOTOR_DELAY        # early trigger
                    self.early += 1
        super().act(env, f)

    def motor(self, env, f):
        idx = self.chosen if self.chosen is not None else perpendicular_candidate(env, self.rng)
        self.swing(env, idx)

    def end(self, env, info):
        if getattr(self, "frozen", False):      # evaluation: do not learn
            self.memory._ep = []
            return
        self.memory.end_episode(info["hit"], info["t"] if info["hit"] else None)


class Warmup(GFSwing):
    """Random-threshold, random-direction swings; records states for the memory."""
    name = "warmup"

    def __init__(self, gain, memory, circ=None, seed=0):
        super().__init__(gain, circ, seed)
        self.memory = memory

    def reset(self, env):
        super().reset(env)
        self.next_mem = 0.0
        self.thresh = self.rng.uniform(10, 70)
        self.cand = int(self.rng.integers(8))

    def act(self, env, f):
        if env.t >= self.next_mem - 1e-9:
            self.next_mem += MEM_TICK
            self.memory.record(env.t, state_vector(f, np.linalg.norm(env.wind),
                                                   np.linalg.norm(env.fly_vel), rope_azimuth(env)))
        if not self.acted and self.action_time is None and f["theta"] >= self.thresh:
            self.action_time = env.t + MOTOR_DELAY
        if self.action_time is not None and env.t >= self.action_time and not self.acted:
            self.acted = True
            self.swing(env, self.cand)

    def end(self, env, info):
        self.memory.end_episode(info["hit"], info["t"] if info["hit"] else None)


def run_episode(env, policy, speed, radius, seed, on_tick=None, on_start=None):
    f = env.reset(speed=speed, radius=radius, seed=seed)
    policy.reset(env)
    if on_start is not None:
        on_start(env)
    done = False
    while not done:
        policy.act(env, f)
        f, done, info = env.step()
        if on_tick is not None:
            on_tick(env, f)
    policy.end(env, info)
    return info
