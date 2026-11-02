"""Can the connectome steer a flying fly? Flies turn in flight by a left-right difference in wing stroke,
which the DNg02 population sets (Namiki et al. 2022). This drives DNg02 on both sides, on one side only,
and with a walking turn command, and reports the wing motor neurons of each side: the indirect power
motor neurons (DLMn, DVMn) and the steering motor neurons b1 and b2.

    python scripts/experiments/probe_flight_steer.py [--ms 300] [--bias]

--bias gives the wing steering motor neurons (b1, b2) a steady drive, as flight gives b1 in a real fly (it
fires once per wingbeat), and asks whether a turn command then moves them apart through its synapses.
"""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from flyweight import fly_layers, setup  # noqa: E402
from flyweight.sets import REG  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", type=float, default=300)
    ap.add_argument("--bias", action="store_true")
    a = ap.parse_args()
    con, A, world, groups, roles = setup.world_and_groups()
    side = lambda ids, s: [i for i in ids if A.side[i] == s]
    dng = world.group("DNg02")
    print(f"DNg02: {len(side(dng, 'L'))} left, {len(side(dng, 'R'))} right, {len(dng) - len(side(dng, 'L')) - len(side(dng, 'R'))} other")
    power = world.group("power")
    read = {"power L": side(power, "L"), "power R": side(power, "R"), "b1/b2 L": world.group("b_L"), "b1/b2 R": world.group("b_R"),
            "turn pool L": world.group("turn_L"), "turn pool R": world.group("turn_R")}
    print("read: " + ", ".join(f"{k} {len(v)}" for k, v in read.items()))
    steps = int(a.ms * 10)
    conds = [("DNg02 both 15", "both", {"fly": 15}), ("DNg02 left only 15", "L", {"fly": 15}), ("DNg02 right only 15", "R", {"fly": 15}),
             ("DNg02 both 15 + DNa02 left 12", "both", {"fly": 15, "turn_left": 12}),
             ("DNg02 both 15 + DNa02 right 12", "both", {"fly": 15, "turn_right": 12}),
             ("DNg02 left 15 + right 8", "LR", {"fly": 15})]
    if a.bias:  # the bitter register's slot drives b1 and b2 on both sides
        conds = [(f"b1/b2 bias {b}" + extra, which, {"fly": 15, "bitter": b, **lv}) for b in (6, 9, 12)
                 for extra, which, lv in (("", "both", {}), (" + DNa02 left 12", "both", {"turn_left": 12}), (" + DNa02 right 12", "both", {"turn_right": 12}),
                                          (" + DNg02 left only", "L", {}))]
    for nm, which, lv in conds:
        g = list(groups)
        if a.bias:
            g[REG["bitter"]] = np.asarray(world.group("b_L") + world.group("b_R"))
        if which in ("L", "R"):
            g[REG["fly"]] = np.asarray(side(dng, which))
        if which == "LR":  # both sides, the right one weaker: the fly register on the left, the right ones driven by the takeoff register's slot at a lower level
            g[REG["fly"]] = np.asarray(side(dng, "L")); g[REG["takeoff"]] = np.asarray(side(dng, "R")); lv = {"fly": 15, "takeoff": 8}
        fm = fly_layers.FlyModel(con, g, roles, batch=1)
        fm.set_levels({REG[k]: v for k, v in lv.items()}, 0)
        cnt = np.zeros(con.n)
        for s in range(steps):
            sp = fm.step()[0]
            if s >= steps // 3:
                np.add.at(cnt, np.asarray(sp, dtype=np.int64), 1)
        hz = cnt / ((steps - steps // 3) * 1e-4)
        print(f"  {nm:32s} " + "  ".join(f"{k} {hz[v].mean():6.1f} Hz" for k, v in read.items()), flush=True)
        del fm; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
