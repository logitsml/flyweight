"""How strong should the adaptation variant be? A sweep of kappa (fly_layers.ADAPT) on the float64 executor.

    python scripts/experiments/sweep_adapt.py [--kappas 0,2,4,8,16] [--tau 200] [--device cpu]

Per kappa:
- the grooming latch: aDN (DNg12_e) at 12 for 100, 110, ..., 200 ms from rest, then released; how many
  of those leave the grooming pool firing over 50 Hz 200-300 ms after release (the latch is a narrow
  state: in Shiu's model some durations latch and their neighbours do not);
- each command held at its level from rest for 300 ms: the motor pool the body reads, as a fraction of
  its rate at kappa 0 over the last 200 ms.
A useful strength releases the latch and keeps every command's pool well above the body's thresholds.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from flyweight import fly_layers, setup  # noqa: E402
from flyweight.lif import FlyLIF  # noqa: E402
from flyweight.sets import REG  # noqa: E402

CMDS = [("forward", 12, "fwd"), ("turn_left", 12, "turn_L"), ("back", 12, "bwd"), ("groom", 12, "groom"), ("fly", 15, "power"),
        ("takeoff", 15, "TTMn_L")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappas", default="0,2,4,8,16")
    ap.add_argument("--tau", type=float, default=fly_layers.ADAPT["tau_ms"])
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    torch.set_num_threads(8)
    dev = a.device
    con, A, world, groups, roles = setup.world_and_groups()
    pools = [world.group(p) for _, _, p in CMDS]
    durs = list(range(100, 201, 10))
    B = len(durs) + len(CMDS)
    base = None
    print("kappa  latched (of 11)   " + "  ".join(f"{c:>9s}" for c, _, _ in CMDS))
    for kappa in [float(k) for k in a.kappas.split(",")]:
        adapt = dict(tau_ms=a.tau, kappa_mv=kappa) if kappa > 0 else None
        ex = FlyLIF(con, batch=B, adapt=adapt, device=dev)

        def drive(b, cmd, lv):
            g = groups[REG[cmd]]
            t = ex.tonic if ex.tonic is not None else torch.zeros(B, con.n, dtype=torch.float64, device=dev)
            t[b] = 0.0
            if lv:
                t[b, torch.as_tensor(g, device=dev)] = torch.as_tensor(lv * fly_layers.STEP * fly_layers.member_gain(g) * 1e-3, device=dev)
            ex.tonic = t
        for k, d in enumerate(durs):
            drive(k, "groom", 12)
        for b, (c, lv, _) in enumerate(CMDS):
            drive(len(durs) + b, c, lv)
        groom = torch.as_tensor(world.group("groom"), device=dev)
        late, acc = np.zeros(len(durs)), np.zeros(len(CMDS))
        for s in range(int(durs[-1] * 10) + 3000):
            for k, d in enumerate(durs):
                if s == d * 10:
                    drive(k, "groom", 0)
            flat = ex.step()
            b, i = flat // con.n, flat % con.n
            for k, d in enumerate(durs):
                if d * 10 + 2000 <= s < d * 10 + 3000:
                    late[k] += float(torch.isin(i[b == k], groom).sum())
            if 1000 <= s < 3000:
                for k, p in enumerate(pools):
                    acc[k] += float(torch.isin(i[b == len(durs) + k], torch.as_tensor(p, device=dev)).sum())
        latched = int((late / len(groom) / 0.1 > 50).sum())
        hz = acc / np.array([len(p) for p in pools]) / 0.2
        if base is None:
            base = hz.copy()
        print(f"{kappa:5.1f}  {latched:4d}              " + "  ".join(f"{h / max(1e-9, b0):9.2f}" for h, b0 in zip(hz, base))
              + f"   (Hz: {', '.join(f'{h:.0f}' for h in hz)})", flush=True)


if __name__ == "__main__":
    main()
