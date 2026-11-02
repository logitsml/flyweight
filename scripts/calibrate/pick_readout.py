"""Choose the body's walking readout from calibration runs of the fly's own layers (scripts/calibrate/calib_motor.py): for each of
forward, turn left, turn right, back and groom, the neurons that command recruits through the
connectome and the others do not, among nerve-cord and descending neurons. Stimulated neurons (command and sensory inputs) and neurons
with another role in the world are excluded, so the body never reads what Qwen writes.

    python scripts/calibrate/pick_readout.py [--min-hz 10] [--max-off 2] [--out data/readout.npz]   # from data/calib_motor.npz

Then scripts/calibrate/calib_forward.py adds what forward drive alone recruits in each turn pool, at every level.
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from flyweight import malecns  # noqa: E402
from flyweight.sets import SCALE, cns_sets  # noqa: E402
from flyweight.flyworld import World  # noqa: E402

TOP, MIN_HZ, MAX_OFF = 40, 10.0, 2.0


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-hz", type=float, default=MIN_HZ)
    ap.add_argument("--max-off", type=float, default=MAX_OFF)
    ap.add_argument("--out", default=str(ROOT / "data" / "readout.npz"))
    a = ap.parse_args()
    c = np.load(ROOT / "data" / "calib_motor.npz")
    r = {k: c[k] for k in c.files if k != "inputs"}
    con = malecns.load(kc_kc=False); con.scale = SCALE
    A = malecns.annotations(con)
    world = World.__new__(World); world.A, world.sets, world.readout, world._groups = A, cns_sets(con, A), {}, {}
    taken = set(c["inputs"].tolist())
    for nm in ("DNg100", "TTMn", "power", "b_L", "b_R", "MN9"):
        taken |= set(world.group(nm))
    motor_side = np.array([str(x).startswith(("vnc_", "ascending", "descending")) for x in A.superclass])
    ok = motor_side.copy(); ok[list(taken)] = False  # nerve cord and descending neurons only
    base = ["rest", "forward", "turn_left", "turn_right", "takeoff", "fly", "back", "groom"]
    want = {"fwd": "forward", "turn_L": "turn_left", "turn_R": "turn_right", "bwd": "back", "groom": "groom"}
    out, used = {}, np.zeros(con.n, bool)
    for nm, cmd in want.items():
        # turn pools are chosen against the other turn, takeoff, flight and rest, not against forward: the
        # same rule on both sides; what forward alone drives in them is subtracted when the body reads
        # them (scripts/calibrate/calib_forward.py), since the connectome's left and right circuits are not mirror
        # images (the left-turn neurons also fire under forward drive)
        others = [k for k in base if k != cmd and not (nm != "fwd" and k == "forward")]
        m = ok & ~used & (r[cmd] >= a.min_hz)
        for k in others:
            m &= r[k] <= (1.0 if k == "rest" else a.max_off)
        idx = np.flatnonzero(m)
        score = r[cmd][idx] - (r["forward"][idx] if nm != "fwd" else 0)
        idx = idx[np.argsort(-score)][:TOP]
        used[idx] = True
        out[nm] = idx
        out[nm + "_hz"] = float(r[cmd][idx].mean()) if len(idx) else 1.0
        kinds = Counter(f"{A.superclass[i]}:{A.type[i] or '?'}" for i in idx).most_common(8)
        print(f"{nm:7s} {len(idx):3d} neurons, {out[nm + '_hz']:.1f} Hz under {cmd}; {kinds}")
    kf = {nm: float(r["forward"][out[nm]].mean() / max(1e-9, r["forward"][out["fwd"]].mean())) for nm in ("turn_L", "turn_R")}
    print(f"forward's share in each turn pool at full forward drive, per unit of the forward pool's rate: L {kf['turn_L']:.3f}, "
          f"R {kf['turn_R']:.3f} (the body subtracts the share measured at every level: scripts/calibrate/calib_forward.py)")
    print("\ncross-talk: mean rate of each readout (rows) under each command (columns), as a fraction of full drive")
    cols = base + ["forward__turn_left", "forward__turn_right", "takeoff__fly"]
    print(f"{'':8s}" + "".join(f"{k.replace('__', '+'):>20s}" for k in cols))
    for nm in want:
        idx = out[nm]
        print(f"{nm:8s}" + "".join(f"{(r[k][idx].mean() / out[nm + '_hz'] if len(idx) else 0):20.2f}" for k in cols))
    np.savez(a.out, **out)


if __name__ == "__main__":
    main()
