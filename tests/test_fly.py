"""The fly's layers on the CPU: spike for spike against the float64 executor, and the MLP rows that gate
the drive (the tonic rows, mutual inhibition, inert MOVE tokens)."""

import numpy as np
import torch

from flyweight import fly_layers
from flyweight.connectome import Connectome, P
from flyweight.lif import FlyLIF


def small(seed=0, n=300, m=4000):
    rng = np.random.default_rng(seed)
    pre, post = rng.integers(0, n, m), rng.integers(0, n, m)
    cnt = rng.integers(1, 20, m) * np.where(rng.random(m) < 0.3, -1, 1)
    con = Connectome(flyids=np.arange(n), pre=pre.astype(np.int64), post=post.astype(np.int64), count=cnt.astype(np.int64))
    con.scale = 1.0
    groups = [rng.choice(n, 6, replace=False) for _ in range(4)]
    roles = [rng.choice(n, 10, replace=False) for _ in range(3)]
    return con, groups, roles


def test_fly_matches_the_float64_executor():
    con, groups, roles = small()
    fm = fly_layers.FlyModel(con, groups, roles, device="cpu")
    ex = FlyLIF(con, exact=True, device="cpu")
    sched = {0: [10, 0, 6, 0], 150: [0, 13, 0, 4], 300: [15, 15, 0, 0], 450: [3, 0, 12, 9]}
    bad = spikes = 0
    for s in range(600):
        if s in sched:
            fm.set_levels(sched[s])
            ton = torch.zeros(1, con.n, dtype=torch.float64)
            for g, members in enumerate(groups):
                ton[0, torch.as_tensor(members)] = torch.as_tensor(sched[s][g] * fly_layers.STEP * fly_layers.member_gain(members) * 1e-3)
            ex.tonic = ton
        a, b = sorted(fm.step()[0]), sorted(ex.step().tolist())
        bad += a != b
        spikes += len(b)
    assert spikes > 50 and bad == 0 and fm.mismatch == 0


def components(cls=None):
    con, groups, roles = small()
    W = con.count * P["w_syn"] * con.scale * 1000.0
    txt = None if cls is None else fly_layers.H_FLY - 1
    c = fly_layers.fly_components(con.n, con.post, con.pre, W, groups, roles, device="cpu", txt=txt, cls=cls)
    return c, groups


def run_mlp(mlp, L, **dims):
    x = torch.zeros(1, 1, L.H)
    x[..., L.ONE] = 1.0
    for d, v in dims.items():
        x[..., int(d)] = v
    with torch.no_grad():
        return mlp(x)[0, 0]


def test_tonic_rows_drive_only_at_a_tick_and_never_while_refractory():
    c, groups = components()
    L, mlp = c["L"], c["layers"][1].mlp
    m = int(groups[0][0])
    gain = float(fly_layers.member_gain(m))
    drive = lambda reg, tick, refr: float(run_mlp(mlp, L, **{str(L.reg.start): reg, str(L.is_tick): tick, str(L.refr22.start + m): refr})[L.inp.start + m])
    assert drive(1e5, 0.0, 0.0) == 0.0  # a huge register drives nothing off a TICK
    assert drive(1e5, 1.0, 1.0) == 0.0 and drive(1e12, 1.0, 1.0) == 0.0  # nor a refractory member at a TICK
    assert drive(1e3, 1.0, 0.99999994) == 0.0  # the refractory flag as layer 2 returns it
    assert drive(6.0, 1.0, 0.0) == 5 * fly_layers.STEP * gain  # level 5, exactly
    assert drive(41.0, 1.0, 0.0) == 15 * fly_layers.STEP * gain  # clipped at 15


def test_inhibited_registers_drive_by_their_margin():
    cls = dict(k_key=1e4, dim=fly_layers.H_FLY - 2, affine={0: (4.0, 1.0), 1: (4.0, 1.0), 2: (4.0, 1.0)},
               inhibit={1: [(2, 0.0)], 2: [(1, 0.0)], 0: [(1, 1.5), (2, 1.5)]})
    c, groups = components(cls)
    L, l1, l2 = c["L"], c["layers"][0].mlp, c["layers"][1].mlp
    rect = {pc: L.rect.start + k for k, pc in enumerate(sorted({(1, 0.0), (2, 0.0), (1, 1.5), (2, 1.5)}))}
    assert float(run_mlp(l1, L, **{str(L.reg.start + 2): 3.0})[rect[(2, 0.0)]]) == 3.0  # relu of the other register
    assert float(run_mlp(l1, L, **{str(L.reg.start + 2): -2.0})[rect[(2, 0.0)]]) == 0.0
    assert float(run_mlp(l1, L, **{str(L.reg.start + 2): 3.0})[rect[(2, 1.5)]]) == 1.5  # relu(register - offset)
    assert float(run_mlp(l1, L, **{str(L.reg.start + 2): 1.0})[rect[(2, 1.5)]]) == 0.0
    m = int(groups[1][0])
    gain = float(fly_layers.member_gain(m))
    drive = lambda reg, other: float(run_mlp(l2, L, **{str(L.reg.start + 1): reg, str(rect[(2, 0.0)]): other, str(L.is_tick): 1.0})[L.inp.start + m])
    assert drive(2.0, 0.5) == 6 * fly_layers.STEP * gain  # 4 x (2 - 0.5) levels
    assert drive(2.0, 3.0) == 0.0  # the other side is favoured: nothing
    f = int(groups[0][0])
    gain_f = float(fly_layers.member_gain(f))
    walk = lambda reg, a, b: float(run_mlp(l2, L, **{str(L.reg.start + 0): reg, str(rect[(1, 1.5)]): a, str(rect[(2, 1.5)]): b,
                                                    str(L.is_tick): 1.0})[L.inp.start + f])
    assert walk(2.0, 0.0, 0.0) == 8 * fly_layers.STEP * gain_f  # no pull to either side: 4 x 2 levels
    assert walk(2.0, 0.5, 0.25) == 5 * fly_layers.STEP * gain_f  # 4 x (2 - 0.5 - 0.25)


def test_move_tokens_are_inert():
    c, _ = components()
    L = c["L"]
    r, col, v = c["E"].arrays()
    rows = {int(a) for a, b in zip(r, col) if b == L.is_other}
    assert all(L.move + k in rows for k in range(L.R))


def test_fly_token_norm_equals_the_reference():
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm
    H = fly_layers.H_FLY
    norm = Qwen3_5RMSNorm(H, eps=1e-6)
    with torch.no_grad():
        norm.weight.fill_(-1.0); norm.weight[:fly_layers.M_BALLAST] = 2.0 ** -9 - 1.0; norm.weight[fly_layers.M_BALLAST:200000] = 2.0 ** 24 - 1.0
    gen = torch.Generator().manual_seed(0)
    x = torch.zeros(3, 4, H)
    x[..., :fly_layers.M_BALLAST] = fly_layers.B0
    idx = torch.randint(fly_layers.M_BALLAST, H, (50000,), generator=gen)
    x[..., idx] = torch.randn(3, 4, 50000, generator=gen) * 300
    ref = norm(x)
    fly_layers.FlyTokenNorm.install(norm)
    with fly_layers.FlyTokenNorm.Mode():
        fast = norm(x)
    assert torch.equal(fast, ref) and torch.equal(norm(x), ref)
