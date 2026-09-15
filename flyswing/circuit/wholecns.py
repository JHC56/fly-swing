"""Whole-CNS leaky integrate-and-fire simulation on the Janelia male CNS v1.0 connectome.

All 165,122 traced neurons, all 25.6 M neuron-to-neuron connections, one LIF per neuron,
signs from the neurotransmitter predictions (acetylcholine excitatory; GABA and glutamate
inhibitory), parameters after Shiu et al. 2024 (Nature, "A Drosophila computational brain
model reveals sensorimotor processing").

  build_network   signed sparse connection matrix over the traced neurons (cached)
  simulate        step the whole network with Poisson input on chosen neurons
  escape_cascade  one loom on LC4 / LPLC2, 150 ms, every spike recorded
"""
import os, time
import numpy as np

from .malecns import DATA_DIR, download, load_annotations, BASE

DT = 1e-4             # s
TAU_M = 0.020         # membrane time constant
TAU_SYN = 0.005       # synaptic current decay
V_REST, V_TH, V_RESET = -52.0, -45.0, -52.0   # mV
REFRAC = 0.0022       # s
W_SYN = 0.275         # mV per synapse (Shiu et al. 2024)
DELAY = 0.0018        # synaptic delay, s (Shiu et al. 2024)
W_SCALE = 0.06        # fraction of W_SYN actually used. At full strength this graph never returns to
                      # rest after the stimulus (runaway recurrent excitation); 0.06 is the largest
                      # value at which a loom still drives GF -> descending -> motor neurons and the
                      # activity then dies out.
INH = {"gaba", "glutamate"}
NT_FILE = "body-neurotransmitters-male-cns-v1.0.feather"


def _nt_path():
    return os.path.join(DATA_DIR, "body-neurotransmitters.feather")


def build_network(verbose=True):
    """Sparse signed connection matrix over traced neurons. Cached in data/malecns/network.npz."""
    import scipy.sparse as sp
    cache = os.path.join(DATA_DIR, "network.npz")
    meta = os.path.join(DATA_DIR, "network_meta.npz")
    if os.path.exists(cache) and os.path.exists(meta):
        m = np.load(meta, allow_pickle=True)
        return sp.load_npz(cache), {k: m[k] for k in m.files}
    import pyarrow as pa, pyarrow.ipc as ipc, pyarrow.compute as pc, pyarrow.feather as f
    ann = load_annotations()
    traced = ann[ann["status"] == "Traced"]
    body = traced["bodyId"].values.astype(np.int64)
    n = len(body)
    lut_keys = pa.array(body)
    # neurotransmitter sign per presynaptic neuron
    if not os.path.exists(_nt_path()):
        import urllib.request
        urllib.request.urlretrieve(BASE + NT_FILE, _nt_path())
    nt = f.read_table(_nt_path(), columns=["body", "consensus_nt"]).to_pandas().set_index("body")["consensus_nt"]
    nt = nt.reindex(body).fillna("acetylcholine").values.astype(str)
    sign = np.where(np.isin(nt, list(INH)), -1.0, 1.0).astype(np.float32)
    wp = download("weights", verbose)
    pre_l, post_l, w_l = [], [], []
    t0 = time.time()
    with pa.memory_map(wp, "r") as src:
        reader = ipc.open_file(src)
        for b in range(reader.num_record_batches):
            rb = reader.get_batch(b)
            pre, post, w = rb.column("body_pre"), rb.column("body_post"), rb.column("weight")
            ip = pc.index_in(pre, value_set=lut_keys)
            ipo = pc.index_in(post, value_set=lut_keys)
            keep = pc.and_(pc.is_valid(ip), pc.is_valid(ipo))
            pre_l.append(pc.filter(ip, keep).to_numpy().astype(np.int32))
            post_l.append(pc.filter(ipo, keep).to_numpy().astype(np.int32))
            w_l.append(pc.filter(w, keep).to_numpy().astype(np.float32))
            if verbose and b % 20 == 0:
                print(f"[wholecns] batch {b}/{reader.num_record_batches}  {time.time() - t0:.0f}s")
    pre, post, w = np.concatenate(pre_l), np.concatenate(post_l), np.concatenate(w_l)
    W = sp.csr_matrix((w * sign[pre] * W_SYN, (pre, post)), shape=(n, n), dtype=np.float32)
    if verbose:
        print(f"[wholecns] {n:,} neurons, {W.nnz:,} connections, {int((sign < 0).sum()):,} inhibitory neurons")
    sp.save_npz(cache, W)
    metad = dict(body=body, type=traced["type"].fillna("").values.astype(str),
                 superclass=traced["superclass"].fillna("").values.astype(str), sign=sign)
    np.savez_compressed(meta, **metad)
    return W, metad


def simulate(W, stim_idx, stim_rate_hz, duration=0.15, stim_until=0.06, seed=0, verbose=True,
             w_scale=1.0, delay=DELAY):
    """LIF over all neurons. `stim_idx` receive Poisson input spikes at `stim_rate_hz` until
    `stim_until`; spikes reach their targets after `delay` s. Returns (spike_neuron, spike_time)."""
    rng = np.random.default_rng(seed)
    n = W.shape[0]
    v = np.full(n, V_REST, np.float32)
    isyn = np.zeros(n, np.float32)
    ref = np.zeros(n, np.float32)
    a_m, a_s = DT / TAU_M, DT / TAU_SYN
    out_n, out_t = [], []
    steps = int(round(duration / DT))
    p_stim = stim_rate_hz * DT
    W = W.tocsr()
    nd = max(1, int(round(delay / DT)))
    queue = [None] * nd                                   # spike ids in flight, one slot per step
    t0 = time.time()
    for k in range(steps):
        t = k * DT
        spk = v >= V_TH
        if t < stim_until:
            ext = stim_idx[rng.random(len(stim_idx)) < p_stim]
            spk[ext] = True
        ids = np.flatnonzero(spk)
        if len(ids):
            out_n.append(ids.astype(np.int32)); out_t.append(np.full(len(ids), t, np.float32))
            v[ids] = V_RESET
            ref[ids] = REFRAC
        arriving = queue[k % nd]
        queue[k % nd] = ids if len(ids) else None
        if arriving is not None and len(arriving):
            isyn += w_scale * np.asarray(W[arriving].sum(axis=0)).ravel()
        # exponential synaptic current: a spike adds w (mV) to isyn, which decays with TAU_SYN;
        # its integrated effect on the membrane is exactly w
        active = ref <= 0
        v[active] += a_m * (V_REST - v[active]) + a_s * isyn[active]
        isyn -= a_s * isyn
        ref -= DT
        if verbose and k % 200 == 0:
            print(f"[wholecns] t={t * 1e3:5.1f} ms  spikes so far {sum(len(x) for x in out_n):,}  {time.time() - t0:.0f}s")
    if not out_n:
        return np.zeros(0, np.int32), np.zeros(0, np.float32)
    return np.concatenate(out_n), np.concatenate(out_t)


def escape_cascade(stim_rate_hz=200.0, w_scale=None, verbose=True, out=None):
    """Loom the fly: drive every LC4 and LPLC2 neuron at `stim_rate_hz` for 60 ms and record
    how activity spreads through the whole CNS. Saves data/malecns/escape_cascade.npz."""
    w_scale = W_SCALE if w_scale is None else w_scale
    W, meta = build_network(verbose)
    types, sup = meta["type"], meta["superclass"]
    stim = np.flatnonzero(np.isin(types, ["LC4", "LPLC2"]))
    sn, st = simulate(W, stim, stim_rate_hz, verbose=verbose, w_scale=w_scale)
    gf = np.flatnonzero(types == "DNp01")
    gf_t = st[np.isin(sn, gf)]
    t_gf = float(gf_t.min()) if len(gf_t) else np.nan
    motor = np.isin(sn, np.flatnonzero(sup == "vnc_motor"))
    dn = np.isin(sn, np.flatnonzero(sup == "descending_neuron"))
    if verbose:
        print(f"[wholecns] GF first spike at {t_gf * 1e3:.1f} ms; "
              f"{len(np.unique(sn[dn]))} descending neurons and {len(np.unique(sn[motor]))} VNC motor neurons fired; "
              f"first motor spike {(st[motor].min() - t_gf) * 1e3 if motor.any() else float('nan'):.1f} ms after GF")
    out = out or os.path.join(DATA_DIR, "escape_cascade.npz")
    np.savez_compressed(out, neuron=sn, t=st, t_gf=t_gf, body=meta["body"][sn],
                        superclass=sup[sn], stim_rate_hz=stim_rate_hz, w_scale=w_scale)
    return out


if __name__ == "__main__":
    import sys
    escape_cascade(w_scale=float(sys.argv[1]) if len(sys.argv) > 1 else None,
                   out=sys.argv[2] if len(sys.argv) > 2 else None)
