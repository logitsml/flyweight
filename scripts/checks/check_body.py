"""Every behaviour is reachable through the connectome: each command (and the senses' reflexes) on the
fly's own layers, read by the body's own code (flyweight/flyworld.py).

    python scripts/checks/check_body.py [--ms 300] [--adapt]

One fly per condition, four per batch, registers held for --ms; the body reads the MOVE rows every
ms as in a session. Reported over the last two thirds: walking, backward walking, turning, grooming
and flight power (fractions of full drive), and whether the fly took off. Passes when forward walks
straight, each turn turns its own way (alone and while walking), back walks backward (at least 1 mm/s
net of what it recruits in the forward pool), groom grooms, no walking command takes off (a stray
giant fiber spike at a command's onset must not), the giant fiber takes off, DNg02 powers the wings,
and rest does nothing. The reflexes of Johnston's
organ (grooming) and of sugar (MN9) are reported, not required. --adapt runs the adaptation variant
(the body's readout is calibrated on Shiu's model, so this shows what adaptation costs it).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from flyweight import fly_layers, setup  # noqa: E402
from flyweight.flyworld import BACK_V, WALK_V, Body  # noqa: E402
from flyweight.sets import REG, ROLES  # noqa: E402

CONDS = [("rest", {}), ("forward", {"forward": 9}), ("forward + left", {"forward": 9, "turn_left": 12}),
         ("forward + right", {"forward": 9, "turn_right": 12}), ("left", {"turn_left": 12}), ("right", {"turn_right": 12}),
         ("back", {"back": 12}), ("groom", {"groom": 12}), ("takeoff", {"takeoff": 15}), ("fly", {"fly": 15}),
         ("forward 13 + left 8", {"forward": 13, "turn_left": 8}),
         ("Johnston's organ (reflex)", {"JO-CE": 3}), ("sugar (reflex)", {"sugar": 13})]
PASS = {"rest": lambda b: b["walk"] < 0.05 and b["back"] < 0.05 and b["groom"] < 0.05 and abs(b["turn"]) < 0.05,
        "forward": lambda b: b["walk"] >= 0.3 and abs(b["turn"]) <= 0.15 and not b["air"],
        "forward + left": lambda b: b["turn"] >= 0.15 and not b["air"], "forward + right": lambda b: b["turn"] <= -0.15 and not b["air"],
        "left": lambda b: b["turn"] >= 0.3 and not b["air"], "right": lambda b: b["turn"] <= -0.3 and not b["air"],
        "back": lambda b: BACK_V * b["back"] - WALK_V * b["walk"] >= 1.0 and not b["air"], "groom": lambda b: b["groom"] >= 0.3 and not b["air"],
        "takeoff": lambda b: b["air"], "fly": lambda b: b["power"] >= 0.3, "forward 13 + left 8": lambda b: not b["air"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", type=float, default=300)
    ap.add_argument("--adapt", action="store_true")
    a = ap.parse_args()
    adapt = fly_layers.ADAPT if a.adapt else None
    con, A, world, groups, roles = setup.world_and_groups()
    steps = int(a.ms * 10)
    ok = True
    for c0 in range(0, len(CONDS), 4):
        batch = CONDS[c0:c0 + 4]
        fm = fly_layers.FlyModel(con, groups, roles, batch=len(batch), adapt=adapt)
        for b, (_, lv) in enumerate(batch):
            fm.set_levels({REG[k]: v for k, v in lv.items()}, b)
        world.bodies = [Body(nm, 0.0, 0.0) for nm, _ in batch]
        world.t = 0
        mn9 = np.zeros(len(batch))
        for s in range(steps):
            fm.step()
            world.t += 1
            if world.t % 10 == 0:
                mv = fm.move()
                for b in range(len(batch)):
                    world.update_body(b, batch[b][0], dict(zip(ROLES, mv[b].tolist())))
                if s >= steps // 3:
                    mn9 += mv[:, ROLES.index("MN9")]
        for b, (nm, _) in enumerate(batch):
            log = world.bodies[b].log[len(world.bodies[b].log) // 3:]
            col = lambda k: float(np.mean([row[k] for row in log]))
            body = dict(walk=col(6), back=col(7), turn=col(8), power=col(9), groom=col(11), air=any(row[5] == "air" for row in world.bodies[b].log))
            good = PASS[nm](body) if nm in PASS else None
            ok &= good is not False
            print(f"  {'ok  ' if good else ('FAIL' if good is False else '    ')}  {nm:26s} walk {body['walk']:.2f}  back {body['back']:.2f}  "
                  f"turn {body['turn']:+.2f}  groom {body['groom']:.2f}  power {body['power']:.2f}  airborne {body['air']}  "
                  f"MN9 {mn9[b] / max(1, (steps - steps // 3) // 10):.0f} Hz", flush=True)
        del fm; torch.cuda.empty_cache()
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
