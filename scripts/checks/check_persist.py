"""How long each command's readout outlasts its drive, and the grooming latch.

    python scripts/checks/check_persist.py [--adapt]

The fly's layers alone, from rest: each command at level 12 for 150 ms, then released; reported is
when its readout pool falls below 20% of its peak. Then the antennal grooming command (aDN) at level
12 for 100 to 200 ms: in Shiu's model (no adaptation, no neuromodulation) the grooming network can
latch, its pool firing on with no drive. Passes when every walking and turning command releases within
150 ms; the latch is reported, not required either way. --adapt runs the adaptation variant.
"""

import argparse
import gc
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from flyweight import fly_layers, setup  # noqa: E402
from flyweight.sets import DRIVEN, REG, ROLES  # noqa: E402

ZERO = {REG[c]: 0 for c, _ in DRIVEN}


def run(con, groups, roles, cmd, role, on_ms, off_ms, adapt=None):
    fm = fly_layers.FlyModel(con, groups, roles, adapt=adapt)
    trace, n_on = [], on_ms * 10
    for s in range(n_on + off_ms * 10):
        if s % 200 == 0:
            fm.set_levels({**ZERO, REG[cmd]: 12} if s < n_on else ZERO)
        fm.step()
        if s % 100 == 99:
            trace.append(float(fm.move()[0][ROLES.index(role)]))
    del fm  # its CUDA graphs and cache: a model per run
    gc.collect(); torch.cuda.empty_cache()
    return trace, n_on // 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapt", action="store_true")
    adapt = fly_layers.ADAPT if ap.parse_args().adapt else None
    con, A, world, groups, roles = setup.world_and_groups()
    ok = True
    for cmd, role in (("forward", "fwd"), ("turn_left", "turn_L"), ("turn_right", "turn_R"), ("back", "bwd"), ("groom", "groom")):
        tr, k = run(con, groups, roles, cmd, role, 150, 300, adapt)
        peak = max(tr[:k])
        off = next((i + 1 for i, v in enumerate(tr[k:]) if v < 0.2 * peak), None)
        good = off is not None and off * 10 <= 150
        if cmd != "groom":
            ok &= good
        print(f"  {'ok  ' if good else ('FAIL' if cmd != 'groom' else '    ')}  {cmd:10s} peak {peak:6.1f} Hz; below 20% of peak "
              f"{f'{off * 10} ms after release' if off else 'not within 300 ms'}", flush=True)
    for on in (100, 120, 140, 160, 200):
        tr, k = run(con, groups, roles, "groom", "groom", on, 300, adapt)
        late = sum(tr[-10:]) / 10
        print(f"        grooming {on} ms: pool over the last 100 ms of 300 after release {late:.0f} Hz "
              f"({'latched' if late > 0.5 * max(tr[:k]) else 'released'})", flush=True)
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
