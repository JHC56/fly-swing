"""LIF simulation of LC4 / LPLC2 -> GF at 0.1 ms.

Weights = synapse count x sign x ONE global gain. The gain is the only free
parameter; ``calibrate_gain`` sets it so that GF fires near 40 deg angular size
for a standard r/v = 40 ms loom (von Reyn 2014 / Ache 2019). The value found is
printed at every run.
"""
import numpy as np
from .looming import LoomingEncoder
from .extract import load_circuit

DT = 1e-4                      # 0.1 ms
LC_TAU, LC_GAIN = 0.010, 3.0   # LC membrane
LC_REFRAC = 0.002
GF_TAU, SYN_TAU = 0.004, 0.003
GF_THRESH = 1.0


class EscapeCircuit:
    def __init__(self, circ=None, gain=1.0, seed=0):
        self.circ = circ or load_circuit()
        self.enc = LoomingEncoder(self.circ["type"], seed=seed)
        self.w = self.circ["syn"] * self.circ["sign"]     # synapse count x sign
        self.gain = gain
        self.n = len(self.w)
        self.reset()

    def reset(self):
        self.v_lc = np.zeros(self.n)
        self.refrac = np.zeros(self.n)
        self.g_syn = 0.0
        self.v_gf = 0.0
        self.gf_spiked = False
        self.gf_spike_time = None
        self.t = 0.0
        self.lc_spike_count = 0
        self.last_spikes = np.zeros(self.n, bool)

    def step(self, theta, theta_dot, az, el, duration=1e-3):
        """Advance `duration` seconds with constant visual input. Returns True if GF spiked now."""
        I = LC_GAIN * self.enc.currents(theta, theta_dot, az, el)
        n = int(round(duration / DT))
        a_lc = DT / LC_TAU
        a_gf, a_syn = DT / GF_TAU, DT / SYN_TAU
        wg = self.w * self.gain
        fired_now = False
        v, rf = self.v_lc, self.refrac
        self.last_spikes = np.zeros(self.n, bool)     # which LC neurons spiked during this call (for display)
        for _ in range(n):
            v += a_lc * (I - v)
            rf -= DT
            v[rf > 0] = 0.0
            spk = v >= 1.0
            if spk.any():
                v[spk] = 0.0
                rf[spk] = LC_REFRAC
                self.g_syn += float(wg[spk].sum())
                self.lc_spike_count += int(spk.sum())
                self.last_spikes |= spk
            self.g_syn -= a_syn * self.g_syn
            self.v_gf += a_gf * (self.g_syn - self.v_gf)
            self.t += DT
            if self.v_gf >= GF_THRESH and not self.gf_spiked:
                self.gf_spiked, self.gf_spike_time, fired_now = True, self.t, True
                self.v_gf = 0.0
        return fired_now


def looming_trace(r_over_v=0.040, theta0=5.0, tick=1e-3):
    """Standard frontal loom: theta(t) = 2 atan((r/v) / (tc - t)). Yields (theta, theta_dot)."""
    tc = r_over_v / np.tan(np.radians(theta0 / 2))
    t = 0.0
    while t < tc - 1e-6:
        rem = tc - t
        theta = np.degrees(2 * np.arctan(r_over_v / rem))
        theta_dot = np.degrees(2 * r_over_v / (rem * rem + r_over_v ** 2))
        yield t, theta, theta_dot
        t += tick


def gf_spike_angle(circuit, r_over_v=0.040, az=0.0, el=0.0):
    circuit.reset()
    for t, th, thd in looming_trace(r_over_v):
        if circuit.step(th, thd, az, el):
            return th
    return None


def calibrate_gain(circ=None, target_deg=40.0, r_over_v=0.040, tol=1.0, verbose=True):
    """Bisection on log-gain: larger gain -> earlier (smaller-angle) GF spike."""
    circuit = EscapeCircuit(circ, gain=1.0)
    lo, hi = -6.0, 2.0      # log10 gain
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        circuit.gain = 10 ** mid
        ang = gf_spike_angle(circuit, r_over_v)
        if ang is None or ang > target_deg:
            lo = mid
        else:
            hi = mid
        if ang is not None and abs(ang - target_deg) < tol:
            break
    circuit.gain = 10 ** hi
    if verbose:
        print(f"[calibrate] global gain = {circuit.gain:.3e}  ->  GF spikes at "
              f"{gf_spike_angle(circuit, r_over_v):.1f} deg for r/v={r_over_v*1e3:.0f} ms "
              f"(circuit source: {circuit.circ['source']})")
        for rv in (0.010, 0.020, 0.080):
            a = gf_spike_angle(circuit, rv)
            print(f"            r/v={rv*1e3:3.0f} ms -> GF at {a if a is None else round(a, 1)} deg")
    return circuit.gain


if __name__ == "__main__":
    calibrate_gain()
