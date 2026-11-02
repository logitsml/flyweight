"""Sparse storage: a block of token rows gives each row's own numbers, and both equal the dense product."""

import numpy as np
import pytest
import torch

from flyweight.sparse import SparseEmbedding, SparseLinear


def matrix(seed=0, out=50, inp=80, nnz=400):
    rng = np.random.default_rng(seed)
    r, c, v = rng.integers(0, out, nnz), rng.integers(0, inp, nnz), rng.normal(size=nnz)
    dense = torch.zeros(out, inp)
    dense.index_put_((torch.as_tensor(r), torch.as_tensor(c)), torch.as_tensor(v, dtype=torch.float32), accumulate=True)
    return SparseLinear(r, c, v, out, inp, "cpu"), dense


def test_block_equals_its_rows_and_the_dense_product():
    sl, dense = matrix()
    x = torch.randn(1, 12, 80)
    block = sl(x)[0]
    rows = torch.stack([sl(x[:, t:t + 1])[0, 0] for t in range(12)])
    assert torch.equal(block, rows)
    assert torch.allclose(block, x[0] @ dense.T, atol=1e-5)


@pytest.mark.parametrize("deterministic", [False, True])
def test_zero_rows_are_skipped_exactly(deterministic):
    sl, dense = matrix(1)
    sl.skip_zero = True
    SparseLinear.deterministic = deterministic
    try:
        x = torch.randn(1, 6, 80)
        x[0, [1, 4]] = 0.0
        x[0, 2, 40:] = 0.0  # a sparse row, as the connectome's input is
        y = sl(x)[0]
        assert torch.equal(y[[1, 4]], torch.zeros(2, 50))
        assert all(torch.equal(y[t], sl(x[:, t:t + 1])[0, 0]) for t in range(6))
        assert torch.allclose(y, x[0] @ dense.T, atol=1e-5)
    finally:
        SparseLinear.deterministic = False


def test_embedding_is_a_row_lookup():
    rng = np.random.default_rng(2)
    r, c, v = rng.integers(0, 20, 60), rng.integers(0, 30, 60), rng.normal(size=60)
    keep = np.unique(r * 30 + c, return_index=True)[1]
    r, c, v = r[keep], c[keep], v[keep]
    emb = SparseEmbedding(r, c, v, 20, 30, "cpu")
    dense = torch.zeros(20, 30)
    dense[torch.as_tensor(r), torch.as_tensor(c)] = torch.as_tensor(v, dtype=torch.float32)
    ids = torch.tensor([[3, 7, 7, 19]])
    assert torch.equal(emb(ids)[0], dense[ids[0]])
