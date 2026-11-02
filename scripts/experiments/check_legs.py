"""What each command does to the fly's six legs, tested against known biology and against shuffled wiring.

    python scripts/experiments/check_legs.py [--ms 300] [--shuffles 5] [--adapt] [--model]

Each command is held from rest for --ms. Per leg, the rate of all its motor neurons (Hz per member) is
taken over the last two thirds; the leg pools are flyweight/sets.py LEGS. Runs on the float64 executor
(flyweight/lif.py), which the fly's layers match spike for spike; --model runs the fly's layers instead.

Predictions, from experiments and the connectome literature:
  1. P9 (DNp09) walks with all six legs (Bidaye et al. 2020): under forward every leg fires above rest.
  2. DNa02 steers through its own side's legs (Rayshubskiy et al. 2025; Yang et al. 2024; Cheong et al.
     2024): walking and turning left, the left legs fire less than the right, and the reverse for right.
     Asymmetry index (R - L) / (R + L): above 0.05 for left, below -0.05 for right.
  3. MDN drives backward walking through the hind legs (Feng et al. 2020): the hind legs' share of leg
     activity under back exceeds their share under forward.
  4. The antennal grooming descending neuron (aDN, DNg12_e) works the front legs (Hampel et al. 2015):
     the front legs carry more than half of the leg activity under groom.
Each prediction also runs on wiring shuffles: every synapse keeps its presynaptic neuron, sign and
count, and its postsynaptic partner comes from a permutation of all targets, so every neuron keeps
its in- and out-degree. A prediction is supported by the wiring when it holds on the connectome and
on no shuffle. A report, not a gate: it says which of the predictions Shiu's model on the male CNS
reproduces. Rates are saved to results/legs.npz (results/legs_adapt.npz with --adapt).
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch

DEV = "cuda"  # --device cpu runs the executor on the CPU

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from flyweight import fly_layers, setup  # noqa: E402
from flyweight.lif import FlyLIF  # noqa: E402
from flyweight.sets import LEGS, REG  # noqa: E402

CONDS = [("rest", {}), ("forward", {"forward": 12}), ("forward + left", {"forward": 9, "turn_left": 12}),
         ("forward + right", {"forward": 9, "turn_right": 12}), ("left", {"turn_left": 12}), ("right", {"turn_right": 12}),
         ("back", {"back": 12}), ("groom", {"groom": 12})]


def shuffled(con, seed):
    c = copy.copy(con)
    c.post = np.random.default_rng(seed).permutation(np.asarray(con.post))
    return c


def leg_rates(con, groups, roles, legs, ms, adapt, model):
    """Per condition, per leg: Hz per member over the last two thirds of ms."""
    steps, B = int(ms * 10), len(CONDS)
    counts = np.zeros((B, con.n))
    if model:
        out = np.zeros((B, len(legs)))
        for c0 in range(0, B, 4):
            batch = CONDS[c0:c0 + 4]
            fm = fly_layers.FlyModel(con, groups, roles, batch=len(batch), adapt=adapt)
            for b, (_, lv) in enumerate(batch):
                fm.set_levels({REG[k]: v for k, v in lv.items()}, b)
            for s in range(steps):
                sp = fm.step()
                if s >= steps // 3:
                    for b, idx in enumerate(sp):
                        np.add.at(counts[c0 + b], idx, 1)
            del fm
            torch.cuda.empty_cache()
    else:
        ex = FlyLIF(con, batch=B, adapt=adapt, device=DEV)
        ton = torch.zeros(B, con.n, dtype=torch.float64, device=DEV)
        for b, (_, lv) in enumerate(CONDS):
            for k, v in lv.items():
                g = REG[k]
                ton[b, torch.as_tensor(groups[g], device=DEV)] = torch.as_tensor(v * fly_layers.STEP * fly_layers.member_gain(groups[g]) * 1e-3, device=DEV)
        ex.tonic = ton
        acc = torch.zeros(B * con.n, dtype=torch.float64, device=DEV)
        for s in range(steps):
            flat = ex.step()
            if s >= steps // 3 and len(flat):
                acc.index_add_(0, flat, torch.ones(len(flat), dtype=torch.float64, device=DEV))
        counts = acc.view(B, con.n).cpu().numpy()
    secs = (steps - steps // 3) * 1e-4
    return np.array([[counts[b, leg].sum() / max(1, len(leg)) / secs for leg in legs] for b in range(B)])


def verdicts(R):
    """R: conditions x legs (fL fR mL mR hL hR). -> {prediction: (holds, value)}."""
    c = {nm: R[i] for i, (nm, _) in enumerate(CONDS)}
    side = lambda r: (r[[1, 3, 5]].sum() - r[[0, 2, 4]].sum()) / max(1e-9, r.sum())  # (R - L) / (R + L)
    share = lambda r, k: r[[2 * k, 2 * k + 1]].sum() / max(1e-9, r.sum())  # k: 0 front, 1 middle, 2 hind
    return {"1. P9 drives all six legs": (bool((c["forward"] > c["rest"] + 2.0).all()), float((c["forward"] - c["rest"]).min())),
            "2a. DNa02 left: left legs quieter": (side(c["forward + left"]) > 0.05, side(c["forward + left"])),
            "2b. DNa02 right: right legs quieter": (side(c["forward + right"]) < -0.05, side(c["forward + right"])),
            "3. MDN: hind legs' share up": (share(c["back"], 2) > share(c["forward"], 2), share(c["back"], 2) - share(c["forward"], 2)),
            "4. aDN: front legs carry most": (share(c["groom"], 0) > 0.5, share(c["groom"], 0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", type=float, default=300)
    ap.add_argument("--shuffles", type=int, default=5)
    ap.add_argument("--adapt", action="store_true")
    ap.add_argument("--model", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    global DEV
    DEV = a.device
    torch.set_num_threads(8)
    adapt = fly_layers.ADAPT if a.adapt else None
    con, A, world, groups, roles = setup.world_and_groups()
    legs = [world.group(nm) for nm in LEGS]
    print(f"leg pools: {', '.join(f'{nm} {len(l)}' for nm, l in zip(LEGS, legs))} motor neurons; "
          f"{'the fly layers' if a.model else 'float64 executor'}{', adaptation variant' if adapt else ''}", flush=True)
    R = leg_rates(con, groups, roles, legs, a.ms, adapt, a.model)
    print("rates (Hz per member):  " + "  ".join(f"{nm[4:]:>5s}" for nm in LEGS))
    for (nm, _), r in zip(CONDS, R):
        print(f"  {nm:16s}      " + "  ".join(f"{x:5.1f}" for x in r), flush=True)
    real = verdicts(R)
    nulls = []
    for sd in range(a.shuffles):
        Rs = leg_rates(shuffled(con, sd), groups, roles, legs, a.ms, adapt, False)
        nulls.append(verdicts(Rs))
        print(f"  shuffle {sd}: forward {Rs[1].round(1).tolist()}, groom {Rs[7].round(1).tolist()}", flush=True)
    supported = 0
    for k, (holds, val) in real.items():
        null_holds = sum(n[k][0] for n in nulls)
        wiring = holds and null_holds == 0
        supported += wiring
        print(f"  {'yes ' if wiring else 'no  '}  {k}: {val:+.3f} on the connectome ({'holds' if holds else 'does not hold'}); "
              f"holds on {null_holds} of {len(nulls)} shuffles ({', '.join(f'{n[k][1]:+.2f}' for n in nulls)})", flush=True)
    (ROOT / "results").mkdir(exist_ok=True)
    np.savez(ROOT / "results" / ("legs_adapt.npz" if adapt else "legs.npz"), rates=R, conds=[c for c, _ in CONDS], legs=LEGS)
    print(f"predictions supported by the wiring: {supported} of {len(real)}")


if __name__ == "__main__":
    main()
