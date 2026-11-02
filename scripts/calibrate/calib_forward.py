"""What forward drive alone recruits in the body's turn pools, at every forward level, on the fly's own
layers: the curve the body subtracts from each turn pool at the forward pool's present rate
(flyweight/flyworld.py). The connectome's left and right turn circuits are not mirror images: forward
drive also recruits the left-turn pool, by a share that changes with the level.

    python scripts/calibrate/calib_forward.py [--ms 800]   # adds fwd_curve, turn_L_curve, turn_R_curve to data/readout.npz

Rows: P9 (DNp09) at levels 0..15, four flies per batch; rates of the three pools over the last three
quarters of each run, read from the MOVE rows as the body reads them. Rates in a chaotic network
vary from run to run, so each turn pool's rate is fitted as a non-decreasing function of the forward
pool's rate (pool-adjacent-violators), which the body interpolates.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from flyweight import fly_layers, setup  # noqa: E402
from flyweight.sets import ROLES  # noqa: E402

POOLS = ["fwd", "turn_L", "turn_R"]


def monotone(y):
    """The non-decreasing sequence closest to y in least squares (pool-adjacent-violators)."""
    blocks = []  # [mean, weight]
    for v in y:
        blocks.append([float(v), 1.0])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            (m2, w2), (m1, w1) = blocks.pop(), blocks.pop()
            blocks.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2])
    return np.concatenate([[m] * int(w) for m, w in blocks])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", type=float, default=800)
    a = ap.parse_args()
    con, A, world, groups, roles = setup.world_and_groups()
    steps = int(a.ms * 10)
    levels = list(range(16))
    hz = np.zeros((len(levels), len(POOLS)))
    for b0 in range(0, len(levels), 4):
        rows = levels[b0:b0 + 4]
        fm = fly_layers.FlyModel(con, groups, roles, batch=len(rows))
        for b, lv in enumerate(rows):
            fm.set_levels({0: lv}, b)
        acc, k = np.zeros((len(rows), len(ROLES))), 0
        for s in range(steps):
            fm.step()
            if s >= steps // 4 and s % 10 == 0:
                acc += fm.move(); k += 1
        for b, lv in enumerate(rows):
            hz[lv] = [acc[b, ROLES.index(p)] / k for p in POOLS]
            print(f"P9 level {lv:2d}: " + ", ".join(f"{p} {v:5.1f} Hz" for p, v in zip(POOLS, hz[lv])), flush=True)
        del fm; torch.cuda.empty_cache()
    order = np.argsort(hz[:, 0], kind="stable")  # the body interpolates on the forward pool's rate
    x = hz[order, 0]
    keep = np.concatenate([[True], np.diff(x) > 0])
    ro = dict(np.load(ROOT / "data" / "readout.npz"))
    ro.update(fwd_curve=x[keep], turn_L_curve=monotone(hz[order, 1])[keep], turn_R_curve=monotone(hz[order, 2])[keep])
    print("curve (forward Hz -> left, right turn pool Hz): "
          + "; ".join(f"{a:.1f} -> {b:.1f}, {c:.1f}" for a, b, c in zip(ro["fwd_curve"], ro["turn_L_curve"], ro["turn_R_curve"])))
    np.savez(ROOT / "data" / "readout.npz", **ro)
    print("saved data/readout.npz: fwd_curve, turn_L_curve, turn_R_curve")


if __name__ == "__main__":
    main()
