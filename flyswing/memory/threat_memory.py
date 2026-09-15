"""Threat memory on top of whitetree.MahalanobisIndex.

State (9-D): theta, theta_dot, threat azimuth, threat elevation, time-to-contact,
wind speed, own speed, cos/sin of (rope azimuth - threat azimuth).

Records are labelled after the fact: a state is "dangerous" (1) if the fly was hit
within `delay` s of it. Labels are attached when the episode ends and the records
are streamed into the index (insert_many / delete oldest beyond the window).

Action selection: attach each of the 8 candidate rope directions to the current
state, one batched kneighbors() call, pick the direction whose k neighbours were
hit least often.  This module is an engineering add-on, not fly biology.
"""
import numpy as np
from collections import deque

try:
    from whitetree import MahalanobisIndex
except ImportError as e:  # pragma: no cover
    raise ImportError("pip install whitetree") from e

_az = np.radians(np.arange(8) * 45.0)
_el = np.radians(50.0)
CANDIDATE_DIRS = np.stack([np.cos(_el) * np.cos(_az), np.cos(_el) * np.sin(_az),
                           np.full(8, np.sin(_el))], -1)
CANDIDATE_AZ_DEG = np.degrees(_az)
DIM = 9


def state_vector(f, wind_speed, self_speed, rope_az_deg):
    d = np.radians(rope_az_deg - f["az"])
    return np.array([f["theta"], f["theta_dot"], f["az"], f["el"], f["ttc"],
                     wind_speed, self_speed, np.cos(d), np.sin(d)])


class ThreatMemory:
    def __init__(self, k=10, window=200_000, min_fit=2000, delay=0.5, seed=0):
        self.k, self.window, self.min_fit, self.delay = k, window, min_fit, delay
        self.idx = None
        self.labels = []            # index = point id (ids are sequential from 0)
        self.live = deque()         # ids currently in the window (FIFO)
        self._warm_X, self._warm_y = [], []
        self._ep = []               # (t, state) for the running episode
        self.rng = np.random.default_rng(seed)
        self.n_inserted = self.n_deleted = 0

    # ---- streaming ----
    @property
    def fitted(self):
        return self.idx is not None

    def record(self, t, state):
        self._ep.append((t, np.asarray(state, float)))

    def end_episode(self, hit, t_hit=None):
        if not self._ep:
            return
        T = np.array([t for t, _ in self._ep])
        X = np.stack([s for _, s in self._ep])
        y = np.zeros(len(T))
        if hit and t_hit is not None:
            y[(t_hit - T) <= self.delay] = 1.0
        self._ep = []
        if self.idx is None:
            self._warm_X.append(X)
            self._warm_y.append(y)
            if sum(len(a) for a in self._warm_X) >= self.min_fit:
                Xw, yw = np.vstack(self._warm_X), np.concatenate(self._warm_y)
                self.idx = MahalanobisIndex().fit(Xw)
                self.labels = list(yw)
                self.live.extend(range(len(yw)))
                self.n_inserted += len(yw)
                self._warm_X, self._warm_y = [], []
            return
        ids = self.idx.insert_many(X)
        assert ids[0] == len(self.labels), "ids must stay sequential"
        self.labels.extend(y.tolist())
        self.live.extend(int(i) for i in ids)
        self.n_inserted += len(ids)
        while len(self.live) > self.window:
            self.idx.delete(self.live.popleft())
            self.n_deleted += 1

    # ---- queries ----
    def risk(self, states):
        """Mean hit-label of the k nearest records for each state row. NaN if not fitted."""
        states = np.atleast_2d(states)
        if self.idx is None:
            return np.full(len(states), np.nan)
        k = min(self.k, len(self.idx))
        if k == 0:
            return np.full(len(states), np.nan)
        _, ids = self.idx.kneighbors(states, k=k, workers=1)
        lab = np.asarray(self.labels)
        out = np.empty(len(states))
        for r, row in enumerate(ids):
            row = row[row >= 0]
            out[r] = lab[row].mean() if len(row) else np.nan
        return out

    def choose_direction(self, f, wind_speed, self_speed, current_rope_az, explore=0.1):
        """Returns (best_candidate_index, risk_of_staying, risks[8]) or (None, nan, nan) if unfitted."""
        if self.idx is None:
            return None, np.nan, None
        S = np.stack([state_vector(f, wind_speed, self_speed, current_rope_az)] +
                     [state_vector(f, wind_speed, self_speed, a) for a in CANDIDATE_AZ_DEG])
        r = self.risk(S)
        stay, cand = r[0], r[1:]
        if self.rng.random() < explore:
            return int(self.rng.integers(8)), stay, cand
        return int(np.argmin(cand)), stay, cand

    def drift(self):
        return self.idx.drift() if self.idx is not None else 0.0

    def __len__(self):
        return len(self.idx) if self.idx is not None else 0
