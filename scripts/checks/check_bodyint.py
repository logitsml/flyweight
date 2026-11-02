"""The integer body (flyweight/bodyint.py) against the float body (flyweight/flyworld.py), on the same motor streams.

    python scripts/checks/check_bodyint.py [--episodes 20] [--ms 3000]

Each episode is a random motor stream: every 50 ms each readout role takes a new rate (walking pools,
grooming, flight power, wing steering, jump motor neurons in short bursts, MN9), so the bodies walk,
turn, back up, groom, jump, fly, steer, land and feed. Both start at the same random place. Passes
when, over every ms of every episode, positions agree within 0.05 mm, headings within 0.01 rad and
height within 0.05 mm; the walking/flying mode and the sense levels differ on under 0.5% of ms (a
threshold crossed a ms apart); and dust agrees within 0.01. An episode whose bodies part at a
threshold (one lands and the other flies on) stops being compared there and is reported.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from flyweight import malecns  # noqa: E402
from flyweight.bodyint import IntBody  # noqa: E402
from flyweight.flyworld import ARENA, Body, World  # noqa: E402
from flyweight.sets import ROLES, cns_sets  # noqa: E402


def stream(rng, ms):
    """Per ms, role -> Hz per member: piecewise constant over 50 ms blocks."""
    out, cur = [], {}
    for t in range(ms):
        if t % 50 == 0:
            mode = rng.choice(["walk", "turn", "back", "groom", "jump", "fly", "rest", "feed"], p=[.22, .18, .1, .1, .08, .2, .07, .05])
            cur = {r: 0.0 for r in ROLES}
            cur["fwd"] = rng.uniform(0, 25) if mode in ("walk", "turn") else rng.uniform(0, 3)
            cur["turn_L"], cur["turn_R"] = (rng.uniform(0, 40), rng.uniform(0, 10)) if rng.random() < 0.5 else (rng.uniform(0, 10), rng.uniform(0, 40))
            cur["bwd"] = rng.uniform(10, 50) if mode == "back" else rng.uniform(0, 4)
            cur["groom"] = rng.uniform(40, 110) if mode == "groom" else rng.uniform(0, 5)
            cur["power"] = rng.uniform(60, 160) if mode == "fly" else rng.uniform(0, 40)
            cur["b_L"], cur["b_R"] = rng.uniform(0, 60), rng.uniform(0, 60)
            cur["MN9"] = rng.uniform(10, 60) if mode == "feed" else 0.0
            cur["jump"] = mode == "jump"
        hz = dict(cur)
        burst = hz.pop("jump")
        hz["TTMn_L"] = hz["TTMn_R"] = 150.0 if burst and t % 50 < 15 else 0.0
        out.append(hz)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--ms", type=int, default=3000)
    a = ap.parse_args()
    con = malecns.load(kc_kc=False)
    A = malecns.annotations(con)
    ro = np.load(ROOT / "data" / "readout.npz")
    world = World(A, cns_sets(con, A), readout={k: ro[k] for k in ro.files})
    n_b = (len(world.group("b_L")), len(world.group("b_R")))
    worst = dict(pos=0.0, yaw=0.0, z=0.0)
    mode_bad, sense_bad, first, total, dust_err = 0, 0, [], 0, 0.0
    for e in range(a.episodes):
        rng = np.random.default_rng(e)
        x, y, yaw = rng.uniform(-ARENA + 5, ARENA - 5), rng.uniform(-ARENA + 5, ARENA - 5), rng.uniform(-math.pi, math.pi)
        dust = 1.0 if e % 3 == 0 else 0.0
        world.bodies = [Body("f", x, y, yaw=yaw, dust=dust)]
        world.t = 0
        ib = IntBody(x, y, yaw, world.readout, dust=dust)
        for t, hz in enumerate(stream(rng, a.ms)):
            world.t += 10
            world.update_body(0, "", hz)
            ib.update(hz, n_b)
            B = world.bodies[0]
            ix, iy, iz, iyaw, imode, idust, ifed = ib.state()
            dp, dz = math.hypot(B.x - ix, B.y - iy), abs(B.z - iz)
            dyaw = abs((B.yaw - iyaw + math.pi) % (2 * math.pi) - math.pi)
            total += 1
            if (B.mode == "air") != (imode == "air"):
                mode_bad += 1
                if dp > 0.05 or dz > 0.05:  # they have parted: one flies on, the other has landed
                    first.append((e, t, B.mode, imode))
                    break
                continue
            worst["pos"], worst["yaw"], worst["z"] = max(worst["pos"], dp), max(worst["yaw"], dyaw), max(worst["z"], dz)
            dust_err = max(dust_err, abs(B.dust - idust))
            if world.sense_levels(0) != ib.senses():
                sense_bad += 1
        fed_err = abs(world.bodies[0].fed_ms - ifed) / max(1.0, world.bodies[0].fed_ms)
        print(f"  episode {e:2d}: compared {t + 1} ms; end ({B.x:+.2f}, {B.y:+.2f}, {B.z:.2f}) vs ({ix:+.2f}, {iy:+.2f}, {iz:.2f}), mode {B.mode}/{imode}, "
              f"dust {B.dust:.3f}/{idust:.3f}, fed {B.fed_ms:.0f}/{ifed} ms", flush=True)
    ok = worst["pos"] < 0.05 and worst["yaw"] < 0.01 and worst["z"] < 0.05 and mode_bad < 0.005 * total and sense_bad < 0.005 * total and dust_err < 0.01
    print(f"over {total} ms: max differences position {worst['pos']:.4f} mm, heading {worst['yaw']:.5f} rad, height {worst['z']:.4f} mm, dust {dust_err:.4f}; "
          f"ms in different modes {mode_bad}, with different senses {sense_bad}; episodes parted at a threshold: {len(first)} {first[:4]}")
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
