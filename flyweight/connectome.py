"""The FlyWire v783 connectome as Shiu et al. (2024) feed it to Brian2.

Units are SI and every constant is computed the way Brian2 computes it (0.275 * mV is
0.275 * 0.001), so a float64 executor can match Brian2's numpy target bit for bit.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SHIU = ROOT / "external" / "Drosophila_brain_model"
VERSIONS = {
    "630": (SHIU / "2023_03_23_completeness_630_final.csv",
            SHIU / "2023_03_23_connectivity_630_final.parquet"),  # what the paper used
    "783": (SHIU / "Completeness_783.csv", SHIU / "Connectivity_783.parquet"),  # public release
}
COMP, CONN = VERSIONS["630"]

mV, ms, Hz = 0.001, 0.001, 1.0

# Shiu et al. default_params, in Brian2's own arithmetic.
P = dict(
    dt=0.0001,  # Brian2 defaultclock.dt = 100 * usecond
    v_0=-52 * mV,
    v_rst=-52 * mV,
    v_th=-45 * mV,
    t_mbr=20 * ms,
    tau=5 * ms,
    t_rfc=2.2 * ms,
    t_dly=1.8 * ms,
    w_syn=0.275 * mV,
    r_poi=150 * Hz,
    f_poi=250,
)


def timestep(x, dt):
    """brian2.core.functions.timestep"""
    return int((x + 1e-3 * dt) / dt)


@dataclass
class Connectome:
    flyids: np.ndarray  # (n,) int64 FlyWire root ids, in Brian2 index order
    pre: np.ndarray  # (m,) int64 presynaptic index, Brian2 synapse order
    post: np.ndarray  # (m,) int64
    count: np.ndarray  # (m,) int64 signed synapse count ('Excitatory x Connectivity')

    @property
    def n(self):
        return len(self.flyids)

    @property
    def m(self):
        return len(self.pre)

    def index(self, flyids):
        lut = {f: i for i, f in enumerate(self.flyids.tolist())}
        return [lut[int(f)] for f in flyids]

    def weights(self, w_syn=P["w_syn"]):
        # Brian2: syn.w = df_con['Excitatory x Connectivity'].values * params['w_syn']
        # (a connectome may carry a count rescaling, as the male CNS does; FlyWire has none)
        scale = getattr(self, "scale", None)
        return self.count.astype(np.float64) * (w_syn if scale is None else w_syn * scale)

    def csr(self, device, silence=(), w_syn=P["w_syn"]):
        """Outgoing edges grouped by presynaptic neuron, keeping Brian2's synapse order
        inside each group (the order its numpy target accumulates in)."""
        w = self.weights(w_syn)
        if len(silence):
            w = w.copy()
            w[np.isin(self.pre, np.asarray(silence))] = 0.0
        order = np.argsort(self.pre, kind="stable")
        indptr = np.zeros(self.n + 1, dtype=np.int64)
        np.add.at(indptr, self.pre + 1, 1)
        indptr = np.cumsum(indptr)
        t = lambda a: torch.as_tensor(a, device=device)
        return t(indptr), t(self.post[order]), t(w[order]), t(order)


def load(version="630") -> Connectome:
    comp, conn = VERSIONS[version]
    df_comp = pd.read_csv(comp, index_col=0)
    df_con = pd.read_parquet(conn, columns=["Presynaptic_Index", "Postsynaptic_Index",
                                            "Excitatory x Connectivity"])
    return Connectome(
        flyids=df_comp.index.values.astype(np.int64),
        pre=df_con["Presynaptic_Index"].values.astype(np.int64),
        post=df_con["Postsynaptic_Index"].values.astype(np.int64),
        count=df_con["Excitatory x Connectivity"].values.astype(np.int64),
    )


# Shiu et al.'s example: the 21 right-side sugar-sensing gustatory receptor neurons, and MN9
# (the proboscis motor neuron their feeding predictions are read from).
SUGAR_R = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]
