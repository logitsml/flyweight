"""Shiu et al.'s Brian2 LIF model, on a GPU, B trials at a time.

One step follows Brian2's schedule exactly:

  groups      not_refractory = (k - lastspike) >= rfc; exact linear update of (v, g) where not
              refractory, with the constants Brian2 folds out of the loop (same operations)
  thresholds  spike = v > v_th and not_refractory; spiking neurons become refractory
  synapses    g += w for spikes from step k - delay, but only into non-refractory targets
              (g is "unless refractory", and Brian2 drops input to it); v += f_poi * w_syn
              for Poisson events
  resets      v = v_rst, g = 0, lastspike = k

`adapt` (the adaptation variant, fly_layers.ADAPT): at the start of each step k the threshold rise theta
becomes a theta + (1 - a) kappa r, r = the neuron's spikes in steps k-22..k-1 (the model's refractory
read), a = exp(-dt / tau); a neuron spikes when v > v_th + theta. Spikes do not reset theta.

`exact=True` accumulates synaptic input in Brian2's order (presynaptic index, then synapse
index, one addition at a time), so float64 results match Brian2's numpy target bit for bit.
`exact=False` uses atomic adds: same float64 math, summation order free.
"""

import numpy as np
import torch

from .connectome import P, Connectome, timestep


def brian_constants(p=P):
    """The loop invariants of Brian2's generated update (`_lio_*`), computed as it computes them:
        _g = _lio_2 * g
        _v = _lio_8 + ((_lio_9 * g) + (_lio_10 * v))"""
    dt, t_mbr, tau, v_0 = p["dt"], p["t_mbr"], p["tau"], p["v_0"]
    exp = np.exp
    lio_1 = (-dt) / tau
    lio_2 = exp(lio_1)
    lio_3 = tau / (t_mbr - tau)
    lio_4 = dt / t_mbr
    lio_5 = dt / tau
    lio_6 = (-dt) / t_mbr
    lio_8 = v_0 - (v_0 * exp(lio_6))
    lio_9 = ((lio_3 * ((-exp(lio_4)) + exp(lio_5))) * exp(lio_6)) * exp(lio_1)
    lio_10 = exp(lio_6)
    return float(lio_2), float(lio_8), float(lio_9), float(lio_10)


class FlyLIF:
    def __init__(self, con: Connectome, batch=1, device="cuda", silence=(), p=P, exact=False, adapt=None):
        self.con, self.B, self.n, self.dev, self.p, self.exact = con, batch, con.n, device, p, exact
        self.adapt = adapt
        self.indptr, self.post, self.w, _ = con.csr(device, silence=silence, w_syn=p["w_syn"])
        self.delay = timestep(p["t_dly"], p["dt"])  # 18
        self.rfc_default = timestep(p["t_rfc"], p["dt"])  # 22
        self.L2, self.L8, self.L9, self.L10 = brian_constants(p)
        self.reset_state()

    def reset_state(self):
        B, n, d = self.B, self.n, self.dev
        self.v = torch.full((B, n), self.p["v_0"], dtype=torch.float64, device=d)
        self.g = torch.zeros((B, n), dtype=torch.float64, device=d)
        self.last = torch.full((B, n), -(10**9), dtype=torch.int64, device=d)
        self.rfc = torch.full((n,), self.rfc_default, dtype=torch.int64, device=d)
        self.queue = [None] * (self.delay + 1)  # flat spike indices by step mod (delay+1)
        self.k = 0
        self.tonic = None  # (B, n) mV added to v at the end of each step unless refractory (drive registers)
        self.theta = torch.zeros((B, n), dtype=torch.float64, device=d)  # adaptation variant: threshold rise, volts
        self.recent = torch.zeros((self.rfc_default, B, n), dtype=torch.bool, device=d)  # spikes of the last 22 steps, by step mod 22

    def set_poisson_targets(self, idx):
        """Shiu: Poisson targets get no refractory period."""
        self.rfc[torch.as_tensor(idx, device=self.dev)] = 0

    # -- synaptic delivery ----------------------------------------------------------------
    def _expand(self, flat):
        """Edges of the spiking (trial, neuron) pairs, in Brian2's order."""
        b, i = flat // self.n, flat % self.n
        start, cnt = self.indptr[i], self.indptr[i + 1] - self.indptr[i]
        tot = int(cnt.sum())
        if tot == 0:
            return None, None
        rep = torch.repeat_interleave(torch.arange(len(i), device=self.dev), cnt)
        csum = torch.cumsum(cnt, 0) - cnt
        e = start[rep] + (torch.arange(tot, device=self.dev) - csum[rep])
        return b[rep] * self.n + self.post[e], self.w[e]

    def _deliver(self, flat, nr):
        tgt, w = self._expand(flat)
        if tgt is None:
            return
        keep = nr.view(-1)[tgt]
        tgt, w = tgt[keep], w[keep]
        if len(tgt) == 0:
            return
        g = self.g.view(-1)
        if not self.exact:
            g.index_add_(0, tgt, w)
            return
        # one addition per target per round, rounds in Brian2's order
        order = torch.argsort(tgt, stable=True)
        tgt, w = tgt[order], w[order]
        first = torch.ones_like(tgt, dtype=torch.bool)
        first[1:] = tgt[1:] != tgt[:-1]
        seg = torch.cumsum(first.long(), 0) - 1
        seg_start = torch.nonzero(first).squeeze(1)
        rank = torch.arange(len(tgt), device=self.dev) - seg_start[seg]
        for r in range(int(rank.max()) + 1):
            s = rank == r
            g[tgt[s]] = g[tgt[s]] + w[s]

    # -- one Brian2 time step ---------------------------------------------------------------
    def step(self, poisson=None):
        """poisson: flat (trial * n + neuron) indices receiving a Poisson event this step."""
        p, k = self.p, self.k
        nr = (k - self.last) >= self.rfc  # groups: not_refractory
        g, v = self.g, self.v
        self.g = torch.where(nr, self.L2 * g, g)
        self.v = torch.where(nr, self.L8 + ((self.L9 * g) + (self.L10 * v)), v)
        if self.adapt:
            a = float(np.exp(-p["dt"] / (self.adapt["tau_ms"] * 1e-3)))
            r = self.recent.sum(0, dtype=torch.float64)  # spikes in steps k-22..k-1
            self.theta = a * self.theta + (1.0 - a) * (self.adapt["kappa_mv"] * 1e-3) * r
            spk = (self.v > p["v_th"] + self.theta) & nr
        else:
            spk = (self.v > p["v_th"]) & nr  # thresholds
        nr = nr & ~spk
        flat = torch.nonzero(spk.view(-1)).squeeze(1)
        self.spk_v = self.v.view(-1)[flat]  # at threshold, for greedy (descending v) order
        old = self.queue[(k - self.delay) % (self.delay + 1)]  # synapses
        if old is not None and len(old):
            self._deliver(old, nr)
        if self.tonic is not None:  # after delivery, before resets: where the model's TICK adds gated input
            self.v = torch.where(nr, self.v + self.tonic, self.v)
        if poisson is not None and len(poisson):
            vv = self.v.view(-1)
            vv[poisson] = vv[poisson] + p["f_poi"] * p["w_syn"]
        vv, gg, ll = self.v.view(-1), self.g.view(-1), self.last.view(-1)  # resets
        vv[flat] = p["v_rst"]
        gg[flat] = 0.0
        ll[flat] = k
        self.queue[k % (self.delay + 1)] = flat
        if self.adapt:
            self.recent[k % self.rfc_default] = spk
        self.k = k + 1
        return flat

    def run(self, steps, poisson_fn=None):
        out = []
        for _ in range(steps):
            out.append(self.step(poisson_fn(self.k) if poisson_fn else None))
        return out
