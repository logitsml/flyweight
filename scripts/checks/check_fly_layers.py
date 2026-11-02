"""Checks for the fly's layers (flyweight/fly_layers.py) against the exact executor (flyweight/lif.py).

    python scripts/checks/check_fly_layers.py [--full] [--ms 300] [--heavy] [--stats] [--adapt]

1. Exact normalisation: torch.rsqrt and Qwen3_5RMSNorm on a fly token with 2^33 ballast.
2. Small random connectome (n=400), random register schedules: spikes vs the executor at every
   step; MOVE rates vs a float64 recomputation from the spikes.
3. --full: the whole male CNS, registers on the commands and senses, same comparisons. In float32
   a membrane within ~1e-6 mV of threshold can fire a step apart from float64, and the runs part
   after it: the check passes when the first difference is such a tie (both within 1e-3 mV of
   threshold). --heavy uses strong schedules.
--adapt: the model and the executor both run the adaptation variant (fly_layers.ADAPT).
4. --stats (with --full): after a threshold tie the two trajectories part, so compare statistics over
   the whole run: per-neuron spike counts (correlation over neurons that fired in either), each
   register group's and role's rate, and how many neurons fire at all.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from flyweight import fly_layers  # noqa: E402
from flyweight.connectome import P  # noqa: E402
from flyweight.lif import FlyLIF  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
DIAG = True


def check_norm():
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm
    H = fly_layers.H_FLY
    r = torch.rsqrt(torch.tensor([2.0 ** 48], device="cuda"))
    x = torch.zeros(1, 1, H, device="cuda")
    x[..., :fly_layers.M_BALLAST] = fly_layers.B0
    gen = torch.Generator(device="cuda").manual_seed(0)
    idx = torch.randint(fly_layers.M_BALLAST, H, (200000,), device="cuda", generator=gen)
    x[0, 0, idx] = torch.randn(200000, device="cuda", generator=gen) * 300
    norm = Qwen3_5RMSNorm(H, eps=1e-6).cuda()
    norm.weight.data.fill_(2.0 ** 24 - 1.0); norm.weight.data[:fly_layers.M_BALLAST] = 2.0 ** -9 - 1.0
    y = norm(x)
    ok_rest = bool((y[0, 0, fly_layers.M_BALLAST:] == x[0, 0, fly_layers.M_BALLAST:]).all())
    ok_ball = bool((y[0, 0, :fly_layers.M_BALLAST] == 1.0).all())
    print(f"1. rsqrt(2^48) == 2^-24: {float(r) == 2.0 ** -24}; RMSNorm returns every fly dim exactly: {ok_rest}; ballast -> 1: {ok_ball}")
    return float(r) == 2.0 ** -24 and ok_rest and ok_ball


def ema_rates(spikes, roles, steps, a_r):
    """MOVE at TICK_t: X_t = a X_{t-1} + (1 - a) c_t, c_t = members that spiked in steps t-22..t-1;
    Hz per member = X / 22 / dt / members."""
    out = np.zeros((steps + 1, len(roles)))
    x = np.zeros(len(roles))
    member = [set(r.tolist()) for r in roles]
    per = [np.array([sum(1 for i in spikes[t] if i in m) for m in member], np.float64) for t in range(steps)]
    for t in range(steps):
        c = sum(per[u] for u in range(max(0, t + 1 - fly_layers.RFC), t + 1))
        x = a_r * x + (1 - a_r) * c
        out[t + 1] = x / fly_layers.RFC / P["dt"] / np.maximum(1, [len(r) for r in roles])
    return out


def stats(model_sp, ex_sp, n, groups, roles, names, ms):
    cm, ce = np.zeros(n), np.zeros(n)
    for a in model_sp:
        np.add.at(cm, a, 1)
    for b in ex_sp:
        np.add.at(ce, b, 1)
    act = (cm > 0) | (ce > 0)
    r = float(np.corrcoef(cm[act], ce[act])[0, 1])
    hz = lambda c, idx: float(c[idx].sum() / max(1, len(idx)) / (ms / 1000))
    rows = [f"{nm} {hz(cm, idx):.1f}/{hz(ce, idx):.1f}" for nm, idx in zip(names, list(groups) + list(roles))]
    print(f"   statistics over {ms:.0f} ms: neurons firing {int((cm > 0).sum())} (model) / {int((ce > 0).sum())} (float64); "
          f"per-neuron spike counts r = {r:.4f} over {int(act.sum())} neurons; mean |count diff| {np.abs(cm - ce)[act].mean():.2f} "
          f"(mean count {ce[act].mean():.1f}); rates Hz per member, model/float64: " + ", ".join(rows))
    return r


ADAPT = None  # --adapt: both sides run the adaptation variant (fly_layers.ADAPT)


def compare(con, groups, roles, schedule, steps, label, inputs_note="", stat_names=None):
    t0 = time.time()
    fm = fly_layers.FlyModel(con, groups, roles, adapt=ADAPT)
    build = time.time() - t0
    ex = FlyLIF(con, exact=True, adapt=ADAPT)
    model_sp, ex_sp, moves, bad, gap = [], [], [], [], 0.0
    torch.cuda.synchronize(); t1 = time.time()
    for s in range(steps):
        if s in schedule:  # the same levels reach both at the end of step s
            lv = schedule[s]
            fm.set_levels(lv)
            ton = torch.zeros(1, con.n, dtype=torch.float64, device="cuda")
            for g, members in enumerate(groups):
                ton[0, torch.as_tensor(members, device="cuda")] = torch.as_tensor(
                    lv[g] * fly_layers.STEP * fly_layers.member_gain(members) * 1e-3, device="cuda")  # the executor works in volts
            ex.tonic = ton
        moves.append(fm.move()[0])
        prev_logits = fm.logits.clone()
        m = sorted(fm.step()[0])
        e = sorted(ex.step().tolist())
        model_sp.append(m); ex_sp.append(e)
        if m != e:
            bad.append(s)
            if len(bad) <= 2:  # how far from threshold the differing neurons were, in both
                L = fm.L
                for i in sorted(set(m) ^ set(e))[:4]:
                    vm = float(prev_logits[0, L.tok_spk + i]) / fly_layers.KAPPA
                    ve = float(ex.v[0, i] - P["v_th"]) * 1000 if i not in e else float((ex.spk_v[e.index(i)] - P["v_th"]) * 1000)
                    if len(bad) == 1:
                        gap = max(gap, abs(vm), abs(ve))
                    if DIAG:
                        grp = [g for g, mem in enumerate(groups) if i in set(mem.tolist())]
                        print(f"   step {s}: neuron {i} model {'spiked' if i in m else 'silent'} (v - v_th {vm:+.6f} mV), "
                              f"executor {'spiked' if i in e else 'silent'} (v - v_th after step {ve:+.6f} mV); drive groups {grp}")
    torch.cuda.synchronize(); wall = time.time() - t1
    a_r = math.exp(-P["dt"] / (fly_layers.TAU_RATE * 1e-3))
    ref = ema_rates(ex_sp, roles, steps, a_r)
    first_bad = bad[0] if bad else steps
    mv = np.array(moves[:first_bad]); err = float(np.abs(mv - ref[:first_bad]).max()) if first_bad else 0.0
    if stat_names is not None:
        stats(model_sp, ex_sp, con.n, groups, roles, stat_names, steps / 10)
    print(f"{label}: n={con.n}, built in {build:.0f} s; {steps} steps, model {sum(map(len, model_sp))} spikes, executor "
          f"{sum(map(len, ex_sp))}; steps that differ: {len(bad)}" + (f" (first {bad[0]}, a threshold tie within {gap * 1e6:.0f} nV)" if bad else "")
          + f"; MOVE vs float64 rates: max error {err:.2e} Hz; block check {fm.mismatch} set / {fm.swaps} order; "
          f"{wall / steps * 1e3:.2f} ms per step, {fm.n_forward} forwards {inputs_note}")
    return len(bad), err, gap


def schedule_for(G, steps, rng, hi=15, p_on=0.5, every=(50, 200)):
    sched, s = {}, 0
    while s < steps:
        sched[s] = [int(rng.integers(0, hi + 1)) if rng.random() < p_on else 0 for _ in range(G)]
        s += int(rng.integers(*every))
    return sched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--ms", type=float, default=300)
    ap.add_argument("--heavy", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--adapt", action="store_true")
    a = ap.parse_args()
    global ADAPT
    ADAPT = fly_layers.ADAPT if a.adapt else None
    rng = np.random.default_rng(0)
    ok = check_norm()
    if not a.full:
        n, m = 400, 5000
        pre, post = rng.integers(0, n, m), rng.integers(0, n, m)
        cnt = rng.integers(1, 20, m) * np.where(rng.random(m) < 0.3, -1, 1)
        from flyweight.connectome import Connectome
        con = Connectome(flyids=np.arange(n), pre=pre.astype(np.int64), post=post.astype(np.int64), count=cnt.astype(np.int64))
        con.scale = 1.0
        groups = [rng.choice(n, 6, replace=False) for _ in range(5)]
        roles = [rng.choice(n, 10, replace=False) for _ in range(4)]
        steps = 4000
        bad, err, _ = compare(con, groups, roles, schedule_for(5, steps, rng), steps, "2. random n=400")
        ok = ok and bad == 0 and err < 1e-2
    else:
        from flyweight import malecns
        from flyweight.sets import SCALE, cns_sets
        from flyweight.flyworld import World
        from flyweight.sets import DRIVEN, ROLES, SENSES
        con = malecns.load(kc_kc=False); con.scale = SCALE
        A = malecns.annotations(con)
        sets = cns_sets(con, A)
        ro = np.load(ROOT / "data" / "readout.npz")
        w = World(A, sets, readout={k: ro[k] for k in ro.files})
        groups = [np.asarray(w.group(nm)) for _, nm in DRIVEN] + [np.asarray(w.group(nm)) for nm in SENSES]
        roles = [np.asarray(w.group(nm)) for nm in ROLES]
        steps = int(a.ms * 10)
        if a.heavy:
            sched = schedule_for(len(groups), steps, rng, hi=15, p_on=0.9, every=(100, 400))
        else:
            sched = schedule_for(len(groups), steps, rng, hi=12, p_on=0.35, every=(100, 400))
        names = [nm for _, nm in DRIVEN] + list(SENSES) + list(ROLES) if a.stats else None
        bad, err, gap = compare(con, groups, roles, sched, steps, "3. whole CNS" + (" (heavy)" if a.heavy else ""), stat_names=names)
        ok = ok and (bad == 0 or gap < 1e-3) and err < 1e-2
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
