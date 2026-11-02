"""Does the connectome make a walking rhythm by itself?

    python scripts/experiments/probe_rhythm.py [--ms 1000] [--adapt] [--shuffles 3]

DNg100 (BDN2), a descending neuron that starts forward walking (Bidaye et al. 2020), is held at levels
6, 9, 12 and 15 from rest, beside P9 (DNp09) at 12, for --ms on the float64 executor (flyweight/lif.py,
which the fly's layers match spike for spike). Read are:
- the core that Pugliese et al. (Nature 2026) found oscillating in a rate model of the nerve cord:
  IN17A001 -> INXXX466 -> IN16B036 (three of each per side);
- each leg's coxa promotor and remotor motor neurons, which swing the leg forward and back.
A rhythm needs all three:
- a peak between 2 and 30 Hz in the population rate (1 ms bins, after the first 100 ms) at least
  PEAK x the median of the spectrum there;
- the legs' promotors and remotors in antiphase (correlation of their 5 ms-smoothed rates below -0.2);
- the peak absent on wiring shuffles (scripts/experiments/check_legs.py's degree-preserving permutation).
Shiu's model is uniform (one LIF type, no adaptation), and Pugliese et al. needed size-scaled
excitability for their rate model to oscillate. --adapt runs the adaptation variant.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

DEV = "cuda"  # --device cpu runs the executor on the CPU

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from flyweight import fly_layers, setup  # noqa: E402
from flyweight.lif import FlyLIF  # noqa: E402

PEAK = 5.0
CORE = ["IN17A001", "INXXX466", "IN16B036"]
PROMOTOR, REMOTOR = "Tergopleural/Pleural promotor MN", "Pleural remotor/abductor MN"
SEGS = [("fl", "L"), ("fl", "R"), ("ml", "L"), ("ml", "R"), ("hl", "L"), ("hl", "R")]


def groups_of(A, n):
    """Readout groups: the core, then per leg its promotors, then per leg its remotors."""
    pick = lambda f: [i for i in range(n) if f(i)]
    g = [("core", pick(lambda i: A.type[i] in CORE))]
    g += [(f"pro {s}{d}", pick(lambda i, s=s, d=d: A.type[i] == PROMOTOR and A.subclass[i] == s and A.side[i] == d)) for s, d in SEGS]
    g += [(f"rem {s}{d}", pick(lambda i, s=s, d=d: A.type[i] == REMOTOR and A.subclass[i] == s and A.side[i] == d)) for s, d in SEGS]
    return g


def run(con, conds, grp, ms, adapt):
    """-> (conditions, groups, ms bins) spike counts."""
    B, steps, n = len(conds), int(ms * 10), con.n
    ex = FlyLIF(con, batch=B, adapt=adapt, device=DEV)
    ton = torch.zeros(B, n, dtype=torch.float64, device=DEV)
    for b, (_, members, lv) in enumerate(conds):
        ton[b, torch.as_tensor(members, device=DEV)] = torch.as_tensor(lv * fly_layers.STEP * fly_layers.member_gain(members) * 1e-3, device=DEV)
    ex.tonic = ton
    gid = torch.full((n,), -1, dtype=torch.long, device=DEV)
    for k, (_, idx) in enumerate(grp):
        gid[torch.as_tensor(idx, dtype=torch.long, device=DEV)] = k
    G = len(grp)
    out = torch.zeros(B * G, steps // 10, dtype=torch.float64, device=DEV)
    for s in range(steps):
        flat = ex.step()
        if len(flat):
            g = gid[flat % n]
            keep = g >= 0
            if keep.any():
                out[:, s // 10].index_add_(0, (flat[keep] // n) * G + g[keep], torch.ones(int(keep.sum()), dtype=torch.float64, device=DEV))
    return out.view(B, G, -1).cpu().numpy()


def analyse(series, grp):
    """series: groups x ms. -> (peak Hz, peak / median, antiphase correlation, rates Hz per member)."""
    x = series[:, 100:]
    sizes = np.array([max(1, len(i)) for _, i in grp])[:, None]
    rates = x.sum(1) / sizes[:, 0] / (x.shape[1] * 1e-3)
    pop = x[0] + x[1:7].sum(0)  # the core and the promotors
    pop = pop - pop.mean()
    f = np.fft.rfftfreq(len(pop), 1e-3)
    pw = np.abs(np.fft.rfft(pop * np.hanning(len(pop)))) ** 2
    band = (f >= 2) & (f <= 30)
    k = np.argmax(pw[band])
    ratio = float(pw[band][k] / max(1e-12, np.median(pw[band])))
    sm = lambda v: np.convolve(v, np.ones(5) / 5, mode="same")
    cors = [np.corrcoef(sm(x[1 + l]), sm(x[7 + l]))[0, 1] for l in range(6) if x[1 + l].std() > 0 and x[7 + l].std() > 0]
    return float(f[band][k]), ratio, float(np.mean(cors)) if cors else float("nan"), rates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", type=float, default=1000)
    ap.add_argument("--adapt", action="store_true")
    ap.add_argument("--shuffles", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    global DEV
    DEV = a.device
    torch.set_num_threads(8)
    from check_legs import shuffled
    adapt = fly_layers.ADAPT if a.adapt else None
    con, A, world, groups, roles = setup.world_and_groups()
    grp = groups_of(A, con.n)
    dng = [i for i in range(con.n) if A.type[i] == "DNg100"]
    p9 = list(world.group("DNp09"))
    conds = [(f"DNg100 {lv}", dng, lv) for lv in (6, 9, 12, 15)] + [("P9 12", p9, 12)]
    print(f"{'adaptation variant; ' if adapt else ''}core {len(grp[0][1])} neurons; promotors {[len(i) for _, i in grp[1:7]]}; "
          f"remotors {[len(i) for _, i in grp[7:13]]}; DNg100 {len(dng)}", flush=True)
    found = []
    for tag, c in [("connectome", con)] + [(f"shuffle {s}", shuffled(con, s)) for s in range(a.shuffles)]:
        S = run(c, conds, grp, a.ms, adapt)
        for b, (nm, _, _) in enumerate(conds):
            fpk, ratio, cor, rates = analyse(S[b], grp)
            rhythm = ratio >= PEAK and cor < -0.2
            if tag == "connectome" and rhythm:
                found.append(nm)
            print(f"  {tag:10s} {nm:10s} peak {fpk:5.1f} Hz at {ratio:6.1f}x the median; promotor/remotor r {cor:+.2f}; "
                  f"core {rates[0]:5.1f} Hz, promotors {rates[1:7].mean():5.1f} Hz, remotors {rates[7:13].mean():5.1f} Hz"
                  f"{'  RHYTHM' if rhythm else ''}", flush=True)
        if tag == "connectome" and not found:
            print("  no rhythm on the connectome: the shuffles are not needed", flush=True)
            break
    print(f"rhythm on the connectome: {', '.join(found) if found else 'none'}")


if __name__ == "__main__":
    main()
