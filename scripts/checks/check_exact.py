"""Compare the GPU executor with a Brian2 reference run (scripts/checks/brian_ref.py), spike for spike.

    python scripts/checks/check_exact.py results/ref_sugar.npz [--fast] [--version 630]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from flyweight.connectome import load  # noqa: E402
from flyweight.lif import FlyLIF  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--version", default="630")
    a = ap.parse_args()
    r = np.load(a.ref)
    steps = int(round(float(r["ms"]) * 10))
    con = load(a.version)
    sim = FlyLIF(con, batch=1, exact=not a.fast)
    sim.set_poisson_targets(np.unique(r["ev_tgt"]))
    ev = {}
    for s, t in zip(r["ev_step"], r["ev_tgt"]):
        ev.setdefault(int(s), []).append(int(t))
    ev = {k: torch.tensor(sorted(v), device="cuda") for k, v in ev.items()}
    torch.cuda.synchronize(); t0 = time.time()
    out = sim.run(steps, lambda k: ev.get(k))
    torch.cuda.synchronize(); dt = time.time() - t0
    mine = {}
    for k, f in enumerate(out):
        for i in f.tolist():
            mine.setdefault(k, set()).add(i)
    ref = {}
    for s, i in zip(r["step"], r["idx"]):
        ref.setdefault(int(s), set()).add(int(i))
    keys = sorted(set(mine) | set(ref))
    bad = [k for k in keys if mine.get(k, set()) != ref.get(k, set())]
    n_m, n_r = sum(map(len, mine.values())), sum(map(len, ref.values()))
    print(f"{'fast' if a.fast else 'exact'}: {n_m} spikes vs Brian2 {n_r}; "
          f"steps that differ: {len(bad)} of {steps}" + (f" (first at step {bad[0]})" if bad else ""))
    print(f"executor: {dt:.2f} s for {steps} steps ({dt / steps * 1e6:.0f} us/step)")


if __name__ == "__main__":
    main()
