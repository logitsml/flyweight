"""Janelia's male CNS v1.0 (Berg et al., Cell 2026; brain and ventral nerve cord) in Shiu et al.'s
conventions, as a Connectome the executors and the Qwen3.5 compiler accept.

- Neurons: bodies with a superclass, Berg et al.'s definition of a neuron (~166k; fragments,
  glia and orphans have none).
- Signs: from the consensus neurotransmitter, as Shiu signed FlyWire: GABA and glutamate
  inhibitory, histamine inhibitory too (a chloride channel, HisCl, in flies; FlyWire's
  predictions had no histamine class), everything else excitatory.
- Weights: synapse counts between traced bodies, times `scale` (Shiu's w_syn was fitted on
  FlyWire counts; the male CNS detects more synapses per connection, so the count is rescaled,
  by default with Janelia's FlyWire alignment factor 0.581).
- Positions: soma location (8 nm voxels) where the soma is in the CNS; otherwise the
  synapse-weighted centroid of the neuron's partners' somata (sensory afferents, whose somata
  are in the periphery).
- The giant fiber's electrical synapses: each giant fiber (DNp01) drives the jump motor neuron
  (TTMn) and the PSI on its own side through gap junctions that follow it spike for spike (shakB;
  Tanouye and Wyman 1980, Phelan et al. 1996). Electron microscopy does not see gap junctions, so the
  chemical connectome holds only a few synapses there (20 and 70 onto TTMn, 9 and 2 onto PSI) and
  its inhibition of TTMn wins. Each junction is added as GF_COUPLING excitatory synapses: one
  event of ~50 mV, enough that every giant fiber spike fires its target.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .connectome import ROOT, Connectome

DIR = ROOT / "external" / "male-cns"
ANN = DIR / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NT = DIR / "body-neurotransmitters-male-cns-v1.0.feather"
W = DIR / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
CACHE = DIR / "malecns-neurons.npz"
INHIBITORY = {"gaba", "glutamate", "histamine"}
FLYWIRE_SCALE = 0.581
GF_COUPLING = 250  # synapses per giant fiber electrical junction (x 0.275 mV x 0.73: ~50 mV, a peak of ~8 mV over threshold)


def _build_cache():
    a = pd.read_feather(ANN)
    a = a[a.superclass.notna()].drop_duplicates("bodyId").reset_index(drop=True)
    ids = a.bodyId.values.astype(np.int64)
    lut = pd.Series(np.arange(len(ids)), index=ids)
    w = pd.read_feather(W)
    keep = w.body_pre.isin(ids) & w.body_post.isin(ids) & (w.body_pre != w.body_post)
    w = w[keep]
    pre = lut[w.body_pre.values].values
    post = lut[w.body_post.values].values
    cnt = w.weight.values.astype(np.int32)
    nt = pd.read_feather(NT).drop_duplicates("body").set_index("body").reindex(ids)
    ntv = nt.consensus_nt.fillna(nt.predicted_nt).fillna("unclear").astype(str).str.lower().values
    sign = np.where(np.isin(ntv, list(INHIBITORY)), -1, 1).astype(np.int8)
    order = np.lexsort((post, pre))  # by presynaptic neuron, as Brian2 orders synapses
    soma = np.full((len(ids), 3), np.nan, np.float32)
    has = a.somaLocation.notna().values
    soma[has] = np.stack(a.somaLocation[has].values).astype(np.float32)
    np.savez_compressed(CACHE, ids=ids, pre=pre[order], post=post[order], cnt=cnt[order], sign=sign,
                        nt=ntv, soma=soma)


def load(scale=FLYWIRE_SCALE, min_syn=1, kc_kc=True, gf_coupling=True):
    """min_syn: drop connections with fewer synapses. kc_kc=False drops Kenyon cell -> Kenyon cell
    synapses: the male CNS has 284 per KC against FlyWire's 73, and in Shiu's model (no APL-driven
    sparsening, no adaptation) that loop ignites the whole mushroom body. gf_coupling: the giant
    fiber's electrical synapses onto its own side's TTMn and PSI (module notes)."""
    if not CACHE.exists():
        _build_cache()
    z = np.load(CACHE, allow_pickle=True)
    keep = z["cnt"] >= min_syn
    if not kc_kc:
        a = pd.read_feather(ANN, columns=["bodyId", "class"]).drop_duplicates("bodyId").set_index("bodyId")
        kc = (a.reindex(z["ids"])["class"] == "Kenyon_Cell").values
        keep &= ~(kc[z["pre"]] & kc[z["post"]])
    pre, post, cnt = z["pre"][keep].astype(np.int64), z["post"][keep].astype(np.int64), z["cnt"][keep].astype(np.int64)
    count = cnt * z["sign"][pre]
    if gf_coupling:
        a = pd.read_feather(ANN, columns=["bodyId", "type", "somaSide", "rootSide"]).drop_duplicates("bodyId").set_index("bodyId").reindex(z["ids"])
        side = a.somaSide.fillna(a.rootSide).fillna("").values
        one = lambda ty, s: int(np.flatnonzero((a["type"].values == ty) & (side == s))[0])
        for s in ("L", "R"):
            for target in ("TTMn", "PSI"):
                g, t = one("DNp01", s), one(target, s)
                k = np.flatnonzero((pre == g) & (post == t))
                if len(k):
                    count[k[0]] += GF_COUPLING
                else:
                    pre, post, count = np.append(pre, g), np.append(post, t), np.append(count, GF_COUPLING)
        order = np.lexsort((post, pre))  # by presynaptic neuron, as Brian2 orders synapses
        pre, post, count = pre[order], post[order], count[order]
    con = Connectome(flyids=z["ids"], pre=pre, post=post, count=count)
    con.scale = scale
    con.nt = z["nt"]
    return con


@dataclass
class Annot:
    flyids: np.ndarray
    xyz: np.ndarray  # nm
    superclass: np.ndarray
    cls: np.ndarray
    subclass: np.ndarray
    type: np.ndarray
    flywire: np.ndarray
    side: np.ndarray
    neuromere: np.ndarray


def annotations(con):
    a = pd.read_feather(ANN).drop_duplicates("bodyId").set_index("bodyId").reindex(con.flyids)
    z = np.load(CACHE, allow_pickle=True)
    soma = z["soma"].astype(np.float64)
    has = np.isfinite(soma).all(1)
    # neurons without a CNS soma: weighted centroid of partners that have one
    pre, post, cnt = z["pre"], z["post"], z["cnt"].astype(np.float64)
    acc = np.zeros((con.n, 3)); wsum = np.zeros(con.n)
    for src, dst in ((pre, post), (post, pre)):
        ok = has[dst]
        np.add.at(acc, src[ok], soma[dst[ok]] * cnt[ok, None])
        np.add.at(wsum, src[ok], cnt[ok])
    fill = ~has & (wsum > 0)
    soma[fill] = acc[fill] / wsum[fill, None]
    soma[~np.isfinite(soma).all(1)] = np.nanmedian(soma, 0)
    side = a.somaSide.fillna(a.rootSide).fillna("").astype(str).values
    s = lambda c: a[c].fillna("").astype(str).values
    return Annot(flyids=con.flyids, xyz=(soma * 8.0).astype(np.float32), superclass=s("superclass"),
                 cls=s("class"), subclass=s("subclass"), type=s("type"), flywire=s("flywireType"),
                 side=side, neuromere=s("somaNeuromere"))
