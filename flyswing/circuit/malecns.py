"""Janelia FlyEM *male CNS* connectome v1.0 (Cell, Sept 2026) - the public flat files.

Source (CC-BY 4.0, no login): https://male-cns.janelia.org/download/
  body-annotations-male-cns-v1.0-minconf-0.5.feather   13 MB   types, classes, soma positions
  connectome-weights-male-cns-v1.0-minconf-0.5.feather 1.1 GB  body -> body synapse counts

Used for the real LC4 / LPLC2 -> DNp01 (giant fiber) synapse table of the LIF circuit
(`extract_circuit`) and, through wholecns.py, the full signed connection matrix.
"""
import os
import numpy as np

BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
FILES = {
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "weights": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
}
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "data", "malecns")
GF_TYPES = ("DNp01",)
LC_TYPES = ("LC4", "LPLC2")


def path(key):
    return os.path.join(DATA_DIR, {"annotations": "body-annotations.feather",
                                   "weights": "connectome-weights.feather"}[key])


def download(key, verbose=True):
    """Fetch a flat file from the public bucket if it is not on disk."""
    p = path(key)
    if os.path.exists(p):
        return p
    import urllib.request
    os.makedirs(DATA_DIR, exist_ok=True)
    url = BASE + FILES[key]
    if verbose:
        print(f"[malecns] downloading {url} -> {p}")
    urllib.request.urlretrieve(url, p + ".part")
    os.replace(p + ".part", p)
    return p


def load_annotations(auto_download=True):
    import pyarrow.feather as f
    p = path("annotations")
    if not os.path.exists(p):
        if not auto_download:
            return None
        download("annotations")
    return f.read_table(p).to_pandas()


def extract_circuit(min_weight=1, verbose=True):
    """LC4 / LPLC2 -> DNp01 synapse counts from the flat connectome. Needs the 1.1 GB weights
    file (downloaded on demand). Returns dict(type, syn, sign, body_id, gf_body, source) or None."""
    import pyarrow.compute as pc
    ann = load_annotations()
    gf = ann[ann["type"].isin(GF_TYPES)]["bodyId"].values.astype(np.int64)
    lc = ann[ann["type"].isin(LC_TYPES)][["bodyId", "type", "somaSide"]]
    if len(gf) == 0 or len(lc) == 0:
        return None
    wp = download("weights", verbose)
    import pyarrow as pa
    import pyarrow.ipc as ipc
    traced = pa.array(ann[ann["status"] == "Traced"]["bodyId"].values.astype(np.int64))
    gf_arr = pa.array(gf.tolist())
    parts, n_conn = [], 0
    with pa.memory_map(wp, "r") as src:               # 152M rows: stream record batches, never load it all
        reader = ipc.open_file(src)
        for b in range(reader.num_record_batches):
            rb = reader.get_batch(b)
            pre, post = rb.column("body_pre"), rb.column("body_post")
            both = pc.and_(pc.is_in(pre, value_set=traced), pc.is_in(post, value_set=traced))
            n_conn += int(pc.sum(both).as_py() or 0)
            sel = pc.is_in(post, value_set=gf_arr)
            if pc.any(sel).as_py():
                parts.append(rb.filter(sel).to_pandas())
    import pandas as pd
    df = pd.concat(parts)
    df = df[df["body_pre"].isin(lc["bodyId"].values)]
    df = df.groupby("body_pre")["weight"].sum()
    df = df[df >= min_weight]
    lc = lc.set_index("bodyId").loc[df.index]
    circ = dict(type=lc["type"].values.astype(str), syn=df.values.astype(float),
                sign=np.ones(len(df)), body_id=df.index.values.astype(np.int64),
                side=lc["somaSide"].fillna("").values.astype(str), gf_body=gf,
                n_connections=n_conn, source="malecns_v1.0")
    if verbose:
        for ty in LC_TYPES:
            m = circ["type"] == ty
            print(f"[malecns] {ty}: {m.sum()} neurons -> DNp01, {int(circ['syn'][m].sum())} synapses")
        print(f"[malecns] flat connectome rows (body->body connections): {n_conn:,}")
    npz = os.path.join(DATA_DIR, "circuit_lc_gf.npz")
    np.savez(npz, **{k: v for k, v in circ.items() if k != "source"}, source=circ["source"])
    return circ


def load_cached_circuit():
    npz = os.path.join(DATA_DIR, "circuit_lc_gf.npz")
    if not os.path.exists(npz):
        return None
    d = np.load(npz, allow_pickle=True)
    return dict(type=d["type"].astype(str), syn=d["syn"].astype(float), sign=d["sign"].astype(float),
                body_id=d["body_id"], side=d["side"].astype(str), gf_body=d["gf_body"],
                n_connections=int(d["n_connections"]), source=str(d["source"]))
