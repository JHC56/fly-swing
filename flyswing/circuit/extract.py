"""The LC4 / LPLC2 -> GF (DNp01) sub-circuit.

Source: the Janelia male CNS v1.0 flat connectome, extracted by `malecns.extract_circuit`
and cached in data/malecns/circuit_lc_gf.npz (126 LC4 + 185 LPLC2, real synapse counts).
If the cache is missing, a bundled approximation of the published anatomy is used and
the log says so.

The GF -> TTMn (jump motor neuron) connection is electrical (gap junction) and is not in
the EM synapse table; it is added by hand (`add_motor_output`).
"""
import numpy as np

BUNDLED = {
    "LC4":   dict(n=126, syn_mean=50.0, syn_sd=20.0),
    "LPLC2": dict(n=185, syn_mean=26.0, syn_sd=12.0),
}


def _bundled(seed=0):
    rng = np.random.default_rng(seed)
    lc_type, syn = [], []
    for t, p in BUNDLED.items():
        lc_type += [t] * p["n"]
        syn.append(np.clip(rng.normal(p["syn_mean"], p["syn_sd"], p["n"]), 1, None).round())
    syn = np.concatenate(syn)
    return dict(type=np.array(lc_type), syn=syn, sign=np.ones(len(syn)), source="bundled_approximation")


def load_circuit():
    """dict with arrays over presynaptic LC neurons: type, syn (count), sign, source (+ body_id)."""
    from .malecns import load_cached_circuit
    circ = load_cached_circuit()
    if circ is None:
        print("[extract] data/malecns/circuit_lc_gf.npz not found; using the bundled approximation")
        circ = _bundled()
    add_motor_output(circ)
    return circ


def add_motor_output(circ):
    circ["gf_to_ttmn"] = dict(kind="electrical", note="hand-added; von Reyn et al. 2014")
    return circ


def summary(circ):
    out = [f"source: {circ['source']}"]
    for t in ("LC4", "LPLC2"):
        m = circ["type"] == t
        out.append(f"{t:6s}: {m.sum():3d} neurons, {int(circ['syn'][m].sum()):5d} synapses onto GF")
    if "n_connections" in circ:
        out.append(f"male CNS v1.0: {circ['n_connections']:,} neuron-to-neuron connections")
    return "\n".join(out)


if __name__ == "__main__":
    print(summary(load_circuit()))
