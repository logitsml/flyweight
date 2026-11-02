"""The body's remaining readout calibrations, on the fly's own layers, read from the MOVE rows as the body reads
them (flyweight/flyworld.py). One fly per run, 250 ms each, rates over 50-250 ms:

  walk  P9 (DNp09) at levels 6, 9, 12 and 15: every ms's forward and turn pool rates past the onset (83-250 ms),
        and per 1 Hz of forward rate the mean of each turn pool: fwd_curve, turn_L_curve and turn_R_curve (what
        forward drive alone recruits in each turn pool at the forward pool's present rate, as the body reads it
        each ms), joined below the lowest bin by calib_forward.py's points; MDN at levels 9, 12 and 15: the backward and forward pools, giving bwd_curve and fwd_bwd_curve
        (what backward drive alone recruits in the forward pool, which the body subtracts)
  air   in flight as the body flies it (the giant fiber for 40 ms with DNg02 at 15, held): with no turn command for
        900 ms, the turn pools over 300-900 ms (air_base_L_hz, air_base_R_hz: what steady flight leaves in them; the
        jump recruits both pools for its first ~250 ms); with DNa02 left or right at 12 from 200 ms, the turn pools
        over 300-500 ms (air_turn_L_hz, air_turn_R_hz: what a full turn command gives); --side runs one
        of the three, --part airmerge adds them to the readout. The giant fiber for 40 ms with DNg02 at 0, 6, 9
        and 12: the jump motor neurons' peak rates (for the takeoff threshold)

    python scripts/calibrate/calib_body.py --part walk|air    # adds its keys to data/readout.npz
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
from flyweight.sets import REG, ROLES  # noqa: E402


def run(con, groups, roles, sched, ms=250):
    fm = fly_layers.FlyModel(con, groups, roles, batch=1)
    rows = []
    for s in range(ms * 10):
        if s in sched:
            fm.set_levels({REG[g]: sched[s].get(g, 0) for g in ("forward", "turn_left", "turn_right", "takeoff", "fly", "back", "groom")})
        fm.step()
        if s % 10 == 9:
            rows.append(fm.move()[0].tolist())
    del fm; gc.collect(); torch.cuda.empty_cache()
    return np.asarray(rows)  # per ms, the roles' rates


def monotone(y):  # pool-adjacent-violators: the closest non-decreasing sequence
    y = list(map(float, y)); out = []
    for v in y:
        out.append([v, 1])
        while len(out) > 1 and out[-2][0] > out[-1][0]:
            a, b = out.pop(), out.pop(); out.append([(a[0] * a[1] + b[0] * b[1]) / (a[1] + b[1]), a[1] + b[1]])
    return np.concatenate([[v] * n for v, n in out])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["walk", "fwd", "air", "airturn", "airmerge"], required=True)
    ap.add_argument("--side", choices=["base", "left", "right"], default=None)
    a = ap.parse_args()
    con, A, world, groups, roles = setup.world_and_groups()
    ix = {r: ROLES.index(r) for r in ROLES}
    path = ROOT / "data" / "readout.npz"
    R = dict(np.load(path))
    new = {}
    if a.part in ("walk", "fwd"):
        smp = []
        for lv in (6, 9, 12, 15):
            rows = run(con, groups, roles, {0: {"forward": lv}})[83:]
            smp.append(rows[:, [ix["fwd"], ix["turn_L"], ix["turn_R"]]])
            print(f"  forward {lv:2d}: fwd {rows[:, ix['fwd']].mean():.2f} Hz  turn_L {rows[:, ix['turn_L']].mean():.2f}  turn_R {rows[:, ix['turn_R']].mean():.2f}", flush=True)
        smp = np.concatenate(smp)
        bins = np.floor(smp[:, 0]).astype(int)
        keep = [b for b in np.unique(bins) if (bins == b).sum() >= 10]  # bins with enough ms to average
        bx = np.array([smp[bins == b, 0].mean() for b in keep])
        old = R["fwd_curve"] < bx.min()
        x = np.concatenate([R["fwd_curve"][old], bx])
        new["fwd_curve"] = x
        new["turn_L_curve"] = monotone(np.concatenate([R["turn_L_curve"][old], [smp[bins == b, 1].mean() for b in keep]]))
        new["turn_R_curve"] = monotone(np.concatenate([R["turn_R_curve"][old], [smp[bins == b, 2].mean() for b in keep]]))
        b, fb = [0.0], [0.0]
        for lv in ((9, 12, 15) if a.part == "walk" else ()):
            m = run(con, groups, roles, {0: {"back": lv}})[50:].mean(0)
            b.append(m[ix["bwd"]]); fb.append(m[ix["fwd"]])
            print(f"  back {lv:2d}: bwd {m[ix['bwd']]:.2f} Hz  fwd {m[ix['fwd']]:.2f}", flush=True)
        if len(b) > 1:
            o = np.argsort(b, kind="stable")
            new["bwd_curve"] = np.asarray(b)[o]; new["fwd_bwd_curve"] = monotone(np.asarray(fb)[o])
    elif a.part == "airmerge":
        for side in ("base", "left", "right"):
            z = np.load(ROOT / "data" / f"calib_air_{side}.npz")
            if side == "base":
                new["air_base_L_hz"], new["air_base_R_hz"] = z["L"], z["R"]
            else:
                new[f"air_turn_{side[0].upper()}_hz"] = z[side[0].upper()]
    elif a.part == "airturn":
        turn = {} if a.side == "base" else {f"turn_{a.side}": 12}
        m = run(con, groups, roles, {0: {"takeoff": 15, "fly": 15}, 400: {"fly": 15}, 2000: {"fly": 15, **turn}}, ms=900 if a.side == "base" else 500)[300:].mean(0)
        np.savez(ROOT / "data" / f"calib_air_{a.side}.npz", L=m[ix["turn_L"]], R=m[ix["turn_R"]])
        print(f"  flight, {a.side}: turn_L {m[ix['turn_L']]:.2f} Hz  turn_R {m[ix['turn_R']]:.2f}  power {m[ix['power']]:.1f}", flush=True)
        return
    else:
        for lv in (0, 6, 9, 12):
            m = run(con, groups, roles, {0: {"takeoff": 15, "fly": lv}, 400: {"fly": lv}}, ms=120)
            print(f"  takeoff + DNg02 {lv:2d}: TTMn peak L {m[:, ix['TTMn_L']].max():.1f} Hz  R {m[:, ix['TTMn_R']].max():.1f} Hz", flush=True)
    R = dict(np.load(path)); R.update(new)  # reread: the other part may have saved meanwhile
    np.savez(path, **R)
    print("added to data/readout.npz: " + ", ".join(new))


if __name__ == "__main__":
    main()
