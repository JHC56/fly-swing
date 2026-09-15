"""Locomotion + evasion policy for the course.

The fly stands on the floor until something looms. Evasion is the GF reflex + whitetree
memory (GFMemory). Between threats it web-swings Tarzan-style: release on the up-swing
ahead of the anchor, fly a parabola, shoot the next web at the apex, pump briefly by
reeling. Every MAX_CHAIN webs it drops to the ground first (competition rule), and with
the chain full it jumps instead of shooting a web.
"""
import numpy as np

from flyswing.env.rope import CEILING_Z
from flyswing.memory import CANDIDATE_DIRS
from flyswing.policies import GFMemory, REEL_SPEED, JUMP_V, perpendicular_candidate, jump_direction

FORWARD = np.array([1.0, 0.0, 0.0])


SWING_AFTER_THREAT = 1.0      # keep web-swinging this long after the last threat, then land
MAX_CHAIN = 4                 # competition rule: no 5 web-swings in a row -> land to reset the chain


class SwingPolicy(GFMemory):
    """Ground fly: stands until threatened, evades by web-swing, keeps swinging
    Tarzan-style for a while, then lands and stands again."""
    name = "swing"

    def reset(self, env):
        super().reset(env)
        self.mode = "stand"          # stand | evade | swing | fly | land
        self.mode_t = self.stall_t = 0.0
        self.last_threat = -10.0
        self.swings = self.evasions = self.gf_count = self.landings = self.jumps = 0
        self.chain = 0               # consecutive web shots since the last landing
        self.swing_log = []          # chain index of every web shot (for scoring)
        self.max_chain = MAX_CHAIN

    def _web(self, env, direction, reel=0.0):
        """One web shot = one swing for the score. Returns True if it attached."""
        if not env.rope.shoot(direction):
            return False
        if reel > 0:
            env.rope.reel(reel)
        self.chain += 1
        self.swing_log.append(self.chain)
        return True

    def _must_land(self):
        return self.chain >= self.max_chain

    def _set(self, env, mode):
        self.mode, self.mode_t = mode, env.t
        self.stall_t = env.t

    def _locomotion(self, env):
        p, v = env.fly_pos, env.fly_vel
        if self.mode == "swing":
            a = env.rope.anchor
            if env.rope.reel_speed > 0 and env.t - self.mode_t > 0.06:
                env.rope.reel(0.0)                     # end of the pump
            quiet = env.t - self.last_threat > SWING_AFTER_THREAT
            if (quiet or self._must_land()) and env.rope.taut and abs(v[2]) < 0.2 and p[2] < 0.08:
                env.rope.release()                     # near the bottom, low: drop and land (chain reset)
                self._set(env, "land")
                return
            horiz = np.dot(p - a, FORWARD)
            if env.rope.taut and v[2] > 0 and np.dot(v, FORWARD) > 0 and horiz > 0.35 * env.rope.length:
                env.rope.release()                     # let go on the way up, ahead of the anchor
                self._set(env, "fly")
            elif np.linalg.norm(v) > 0.2:
                self.stall_t = env.t
            elif env.t - self.stall_t > 0.3:           # swing died out: drop, land, take off again
                env.rope.release()
                self._set(env, "land")
        elif self.mode == "fly":
            if (v[2] <= 0 or env.t - self.mode_t > 0.25) and env.t - self.mode_t > 0.03:
                d = self._forward(FORWARD * 0.55 + np.array([0, 0, 1.0]), env)
                if self._must_land():                  # 4 in a row: glide down, touch the ground first
                    self._set(env, "land")
                elif p[2] < CEILING_Z - 0.02 and self._web(env, d, reel=0.35 if np.linalg.norm(v) < 0.6 else 0.0):
                    self.swings += 1                   # pump only while slow (variable-length pendulum)
                    self._set(env, "swing")
                elif p[2] < 0.03:                      # too low to catch a web: land
                    self._set(env, "land")
        elif self.mode == "land":
            if p[2] < 0.0025 and np.linalg.norm(v) < 0.05:
                self.landings += 1
                self.chain = 0                         # ground contact resets the swing chain
                self._set(env, "stand")

    def act(self, env, f):
        if self.mode == "evade":                       # hold the evasive web for 0.18 s
            if env.rope.reel_speed > 0 and (env.rope.length < 0.03 or env.fly_pos[2] > CEILING_Z - 0.025):
                env.rope.reel(0.0)                     # close enough: stop reeling before hitting the ceiling
            if env.t - self.mode_t > 0.18:
                env.rope.reel(0.0)
                self._set(env, "swing")
                self.acted, self.action_time, self.chosen = False, None, None
                self.circuit.reset()
            return
        self._locomotion(env)
        # re-arm the reflex: nothing in view, or the last action is 0.25 s old (obstacle courses keep
        # something in view all the time, so waiting for an empty view would silence the GF for good)
        if (self.acted or self.circuit.gf_spiked) and (
                f["theta"] < 3.0 or (self.action_time is not None and env.t - self.action_time > 0.25)):
            self.acted, self.action_time, self.chosen = False, None, None
            self.circuit.reset()
        if f["theta"] < 3.0:
            return
        self.last_threat = env.t
        super().act(env, f)

    forward_only = False        # course mode: never swing/jump backwards (goal is ahead, +x)

    lane_gain = 0.0             # course mode: steer back toward the centre line y = 0 (thread between obstacles)

    def _forward(self, d, env=None):
        if not self.forward_only:
            return d
        d = np.array(d, float)
        if d[0] < 0.15:                                    # bend backward/sideways choices toward +x
            d[0] = 0.15 + abs(d[0]) * 0.5
        if env is not None and self.lane_gain > 0:         # sideways component pulls back to the lane
            y = env.fly_pos[1]
            d[1] = float(np.clip(d[1] * 0.5 - self.lane_gain * y, -0.6, 0.6))
        return d / np.linalg.norm(d)

    def motor(self, env, f):
        idx = self.chosen if self.chosen is not None else perpendicular_candidate(env, self.rng)
        if self._must_land() or not self._web(env, self._forward(CANDIDATE_DIRS[idx], env), reel=REEL_SPEED):
            env.jump(self._forward(jump_direction(env, self.rng), env), JUMP_V)   # chain full (or no surface): GF -> jump
            self.jumps += 1
            self._set(env, "land")
            return
        self.evasions += 1
        if self.gf_time is not None:
            self.gf_count += 1
        self._set(env, "evade")

    def end(self, env, info):
        self.memory._ep = []
