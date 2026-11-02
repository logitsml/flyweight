"""The fly's steps replayed as CUDA graphs (fly_layers.BlockGraphs) equal eager stepping, spike for spike.

    python scripts/checks/check_graphs.py [--steps 3000]

The whole CNS, one fly and a batch of four, registers switched every 500 steps with strong drives (so
many steps feed several blocks of one length: the case where block inputs must not be overwritten
before they are copied); the same schedule eagerly and with graphs. Passes when every step's spikes
and the final motor readout are identical, both runs have the same block check count, and steps with
several blocks occurred. Also reports the time per step of each.
"""

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from flyweight import fly_layers, setup  # noqa: E402
from flyweight.sets import REG  # noqa: E402

SCHEDULES = [[{"forward": 15, "JO-CE": 3}, {"turn_left": 12, "sugar": 13}, {"fly": 15, "groom": 12, "JO-CE": 3}],
             [{"forward": 9, "turn_right": 12}, {"back": 12}, {"takeoff": 15}],
             [{"back": 12}, {"groom": 12, "bitter": 13}, {"forward": 13, "turn_left": 8}],
             [{"groom": 12}, {"forward": 9}, {"turn_right": 12, "JO-CE": 3}]]


def run(con, groups, roles, batch, steps, graphs):
    fly_layers.FlyModel.graphs = graphs
    fm = fly_layers.FlyModel(con, groups, roles, batch=batch)
    spikes = []
    t0 = None
    for s in range(steps):
        if s % 500 == 0:
            for b in range(batch):
                fm.set_levels({REG[k]: v for k, v in SCHEDULES[b][(s // 500) % 3].items()}, b)
        if s == 100:
            torch.cuda.synchronize()
            t0 = time.time()
        spikes.append(fm.step())
    torch.cuda.synchronize()
    ms = (time.time() - t0) / (steps - 100) * 1e3
    multi = sum(max(len(p) for p in st) + 1 > fm.CHUNK for st in spikes)  # steps that fed more than one block
    out = (spikes, fm.move(), fm.mismatch, ms, sorted(fm.blocks.graphs) if fm.blocks else [], multi)
    del fm
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    a = ap.parse_args()
    con, A, world, groups, roles = setup.world_and_groups()
    ok = True
    for batch in (1, 4):
        e = run(con, groups, roles, batch, a.steps, False)
        g = run(con, groups, roles, batch, a.steps, True)
        first = next((s for s in range(a.steps) if e[0][s] != g[0][s]), None)
        same = first is None and (e[1] == g[1]).all() and e[2] == g[2] and g[5] > 0
        ok &= bool(same)
        n = sum(len(p) for st in e[0] for p in st)
        print(f"  {'ok  ' if same else 'FAIL'}  {batch} fly{'s' if batch > 1 else ''}, {a.steps} steps, {n:,} spikes, {g[5]} steps of several blocks: "
              f"{'identical' if first is None else f'first difference at step {first}'}; block check {e[2]} / {g[2]}; motor readout {'equal' if (e[1] == g[1]).all() else 'differs'}; "
              f"eager {e[3]:.2f} ms/step, graphs {g[3]:.2f} ms/step (block lengths {g[4]})", flush=True)
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
