"""Which nerve-cord neurons each command recruits, on the fly's own layers (flyweight/fly_layers.py): one
batch row per command, its registers at full level, rates over the last two thirds of the run. For
choosing a body readout that never reads a neuron a register drives (scripts/calibrate/pick_readout.py).

    python scripts/calibrate/calib_motor.py [--ms 300]   # -> data/calib_motor.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from flyweight import fly_layers, malecns  # noqa: E402
from flyweight.flyworld import World  # noqa: E402
from flyweight.sets import DRIVEN, SCALE, SENSES, cns_sets  # noqa: E402

COMMANDS = ["rest", "forward", "turn_left", "turn_right", "takeoff", "fly", "back", "groom", "forward+turn_left", "forward+turn_right",
            "takeoff+fly"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", type=float, default=300)
    a = ap.parse_args()
    con = malecns.load(kc_kc=False); con.scale = SCALE
    A = malecns.annotations(con)
    sets = cns_sets(con, A)
    w = World(A, sets)
    groups = [np.asarray(w.group(nm)) for _, nm in DRIVEN] + [np.asarray(w.group(nm)) for nm in SENSES]
    driven = sorted({int(i) for g in groups for i in g})
    steps = int(a.ms * 10)
    cnt = np.zeros((len(COMMANDS), con.n))
    for g0 in range(0, len(COMMANDS), 5):  # five fly copies per pass
        grp = COMMANDS[g0:g0 + 5]
        fm = fly_layers.FlyModel(con, groups, [np.asarray(w.group("MN9"))], batch=len(grp))
        for b, c in enumerate(grp):
            fm.set_levels({r: 15 for r, (cmd, _) in enumerate(DRIVEN) if cmd in c.split("+")}, b)
        for s in range(steps):
            out = fm.step()
            if s >= steps // 3:
                for b, sp in enumerate(out):
                    np.add.at(cnt[g0 + b], sp, 1)
        print(f"{grp}: block check {fm.mismatch} set / {fm.swaps} order", flush=True)
        del fm; torch.cuda.empty_cache()
    hz = cnt / (a.ms * (2 / 3) / 1000)
    np.savez(ROOT / "data" / "calib_motor.npz", inputs=np.array(driven),
             **{c.replace("+", "__"): hz[k].astype(np.float32) for k, c in enumerate(COMMANDS)})
    print("saved data/calib_motor.npz")


if __name__ == "__main__":
    main()
