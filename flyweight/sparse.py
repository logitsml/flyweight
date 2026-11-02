"""Sparse weight storage for stock modules: the fly's weight matrices are almost all zeros, so they
are kept as CSR tensors behind the nn.Linear / nn.Embedding interface the HF modules call."""

import numpy as np
import torch
from torch import nn


class SparseLinear(nn.Module):
    deterministic = False  # every product by columns: the same on every run, and slower (host syncs)
    last_row = None  # inside fly_layers.BlockGraphs: a flag tensor; the connectome's product runs on a block's last row only

    class LastRow:
        """A step's block: only each fly's last token (its TICK) can feed the connectome, so the product
        runs on those rows alone, with no host sync; `flag` counts blocks whose other rows were not zero."""

        def __init__(self, flag):
            self.flag = flag

        def __enter__(self):
            SparseLinear.last_row = self.flag

        def __exit__(self, *exc):
            SparseLinear.last_row = None

    def __init__(self, rows, cols, vals, out_features, in_features, device):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        idx = torch.as_tensor(np.stack([rows, cols]), dtype=torch.int64)
        w = torch.sparse_coo_tensor(idx, torch.as_tensor(vals, dtype=torch.float32), (out_features, in_features)).coalesce()
        w = w.to_sparse_csr()
        if max(out_features, in_features, w._nnz()) < 2 ** 31:  # 32-bit indices: half the index traffic per product
            w = torch.sparse_csr_tensor(w.crow_indices().int(), w.col_indices().int(), w.values(), w.shape)
        self.register_buffer("w", w.to(device), persistent=False)
        self.empty = w._nnz() == 0  # a matrix of zeros: its product is zeros, no kernel needed
        # set on the connectome's product: token rows whose input is exactly zero are skipped (spike
        # tokens never read the ring, so only a block's TICK feeds it); each row gets its own product, so
        # a block, a batch and a lone token give the same numbers. cuSPARSE's matrix-vector kernel splits
        # the connectome's long rows across threads, so its sums vary from run to run in the last bit;
        # with SparseLinear.deterministic a row is multiplied by the columns it touches (the neurons that
        # spiked 19 steps ago and the refractory ones), each target summed by a fixed segmented reduction
        self.skip_zero = False
        self._csc = None

    @property
    def weight(self):  # the GDN reads in_proj dtypes in places
        return self.w

    def _by_columns(self, x):
        """W x for a vector with few nonzeros, from the columns it touches, deterministically."""
        if self._csc is None:
            c = self.w.to_sparse_coo().coalesce()
            r, col = c.indices()
            order = torch.argsort(col * self.out_features + r)
            cnt = torch.bincount(col, minlength=self.in_features)
            self._csc = (torch.cat([cnt.new_zeros(1), torch.cumsum(cnt, 0)]), r[order], c.values()[order])
        ccol, rows, vals = self._csc
        j = torch.nonzero(x).squeeze(1)  # ascending: presynaptic order
        y = x.new_zeros(self.out_features)
        cnt = ccol[j + 1] - ccol[j]
        tot = int(cnt.sum())
        if tot == 0:
            return y
        rep = torch.repeat_interleave(torch.arange(len(j), device=x.device), cnt)
        e = ccol[j][rep] + (torch.arange(tot, device=x.device) - (torch.cumsum(cnt, 0) - cnt)[rep])
        r, v = rows[e], vals[e] * x[j][rep]
        order = torch.sort(r, stable=True).indices  # each target's contributions in presynaptic order
        r, v = r[order], v[order]
        ur, counts = torch.unique_consecutive(r, return_counts=True)
        y[ur] = torch.segment_reduce(v, "sum", lengths=counts)
        return y

    def forward(self, x):
        sh = x.shape
        if self.empty:
            return x.new_zeros(*sh[:-1], self.out_features)
        x2 = x.reshape(-1, sh[-1])
        if self.skip_zero or SparseLinear.deterministic:  # cuSPARSE sums a long row in an order that varies from run to run
            mul = self._by_columns if SparseLinear.deterministic else (lambda v: torch.mv(self.w, v))
            if x2.shape[0] == 1:
                return mul(x2[0]).view(*sh[:-1], self.out_features)
            if SparseLinear.last_row is not None and not SparseLinear.deterministic:
                xb = x.reshape(-1, sh[-2], sh[-1])  # (flies, block, in): each fly's TICK ends its row
                SparseLinear.last_row.add_((xb[:, :-1] != 0).any().long())
                out = torch.zeros(*xb.shape[:2], self.out_features, device=x.device)
                for b in range(xb.shape[0]):
                    out[b, -1] = mul(xb[b, -1])
                return out.view(*sh[:-1], self.out_features)
            out = torch.zeros(x2.shape[0], self.out_features, device=x.device)
            for r in torch.nonzero((x2 != 0).any(1)).squeeze(1).tolist():
                out[r] = mul(x2[r])
            return out.view(*sh[:-1], self.out_features)
        if x2.shape[0] == 1:  # one token: a CSR matrix-vector product, no transposes or copies
            return torch.mv(self.w, x2[0]).view(*sh[:-1], self.out_features)
        if x2.shape[0] <= 16:  # a short block: one matrix-vector product per token beats cuSPARSE's csrmm here
            return torch.stack([torch.mv(self.w, r) for r in x2]).view(*sh[:-1], self.out_features)
        y = torch.sparse.mm(self.w, x2.t().contiguous()).t()
        return y.reshape(*sh[:-1], self.out_features)


class SparseEmbedding(nn.Module):
    def __init__(self, rows, cols, vals, vocab, hidden, device):
        super().__init__()
        self.hidden = hidden
        # every row padded to the longest (padding writes 0.0 into column 0, which is ballast or
        # an always-set dim of every token), so a lookup is one gather and one scatter, no host sync
        order = np.lexsort((cols, rows))
        rows, cols, vals = rows[order], cols[order], vals[order]
        cnt = np.bincount(rows, minlength=vocab)
        K = max(1, int(cnt.max()))
        slot = np.arange(len(rows)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        C = np.zeros((vocab, K), np.int64); V = np.zeros((vocab, K), np.float32); M = np.zeros((vocab, K), bool)
        C[rows, slot] = cols; V[rows, slot] = vals; M[rows, slot] = True
        C[~M] = C[:, :1].repeat(K, 1)[~M]  # pad with the row's first column, re-writing its own value
        V[~M] = V[:, :1].repeat(K, 1)[~M]
        self.register_buffer("C", torch.as_tensor(C, device=device), persistent=False)
        self.register_buffer("V", torch.as_tensor(V, device=device), persistent=False)

    def forward(self, ids):
        flat = ids.reshape(-1)
        out = torch.zeros(len(flat), self.hidden, device=flat.device)
        out.scatter_(1, self.C[flat], self.V[flat])
        return out.reshape(*ids.shape, self.hidden)


class COO:
    def __init__(self, shape):
        self.shape, self.r, self.c, self.v = shape, [], [], []

    def add(self, r, c, v):
        r, c, v = np.broadcast_arrays(np.asarray(r, np.int64), np.asarray(c, np.int64), np.asarray(v, np.float64))
        self.r.append(r.ravel()); self.c.append(c.ravel()); self.v.append(v.ravel())

    def arrays(self):
        if not self.r:
            return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
        r, c, v = np.concatenate(self.r), np.concatenate(self.c), np.concatenate(self.v)
        keep = v != 0
        return r[keep], c[keep], v[keep]
