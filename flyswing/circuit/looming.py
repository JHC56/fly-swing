"""Analytic visual features -> input currents for LC4 / LPLC2 populations.

The retina -> lobula pathway is not simulated.
Each LC neuron gets a Gaussian receptive field on the sphere (az, el). Its input
current is the RF weight times a feature of the object:
  LC4   : angular velocity  (theta_dot, deg/s)      [velocity-tuned, Ache 2019]
  LPLC2 : angular size      (theta, deg), expansion only [size-tuned, Klapoetke 2017]
Both are rectified (no response to receding objects).
"""
import numpy as np

FEAT_SCALE = {"LC4": 1.0 / 800.0, "LPLC2": 1.0 / 40.0}   # normalise to ~1 at a strong loom


def _fibonacci_sphere(n, rng):
    i = np.arange(n) + 0.5
    el = np.degrees(np.arcsin(1 - 2 * i / n))
    az = np.degrees((i * 2.399963) % (2 * np.pi)) - 180
    # keep RFs mostly in the visual field (|el| < 70)
    el = np.clip(el, -70, 70) + rng.normal(0, 2, n)
    return az, el


class LoomingEncoder:
    def __init__(self, lc_type, rf_sigma_deg=35.0, seed=0):
        rng = np.random.default_rng(seed)
        self.type = np.asarray(lc_type)
        n = len(self.type)
        az, el = _fibonacci_sphere(n, rng)
        perm = rng.permutation(n)
        self.az, self.el = az[perm], el[perm]
        self.sigma = rf_sigma_deg
        self.scale = np.array([FEAT_SCALE[t] for t in self.type])
        self.is_lc4 = self.type == "LC4"
        self._unit = self._to_unit(self.az, self.el)

    @staticmethod
    def _to_unit(az, el):
        a, e = np.radians(az), np.radians(el)
        return np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], -1)

    def rf_weight(self, az, el):
        u = self._to_unit(np.array([az]), np.array([el]))[0]
        ang = np.degrees(np.arccos(np.clip(self._unit @ u, -1, 1)))
        return np.exp(-0.5 * (ang / self.sigma) ** 2)

    def currents(self, theta, theta_dot, az, el):
        """Input current per LC neuron (dimensionless, ~1 = strong)."""
        if theta_dot <= 0:
            return np.zeros(len(self.type))
        w = self.rf_weight(az, el)
        feat = np.where(self.is_lc4, theta_dot, theta) * self.scale
        return w * feat
