"""The live server's model loop (flyweight/live.py) through every control a visitor has, without a browser.

    python scripts/checks/check_live.py [--fly-only] [--memory 4] [--adapt]

Qwen drives for a few windows; a message goes into its reports and is cleared; the visitor takes the
command registers (forward, left, right, back, groom, each at level 12) and the body must do that
move; reset puts the fly back at the start with clean antennae; pause stops the windows; the
registers the fly holds must equal the levels the visitor set. Also reports seconds per window by
driver and the process's memory growth over the run (the server runs for hours).
"""

import argparse
import base64
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from flyweight.flyworld import START  # noqa: E402
from flyweight.viewdata import BODY_COLS  # noqa: E402
from flyweight.live import LiveFly  # noqa: E402
from flyweight.sets import DRIVEN  # noqa: E402

CMDS = [c for c, _ in DRIVEN]
MOVES = [("forward", "walk", lambda b: b["walk"] >= 0.15), ("turn_left", "turn", lambda b: b["turn"] >= 0.1),
         ("turn_right", "turn", lambda b: b["turn"] <= -0.1), ("back", "back", lambda b: b["back"] >= 0.3),
         ("groom", "groom", lambda b: b["groom"] >= 0.5)]


def body(msgs):
    rows = np.concatenate([np.frombuffer(base64.b64decode(m["body"]), np.float32).reshape(-1, BODY_COLS) for m in msgs])
    return dict(walk=float(rows[:, 6].mean()), back=float(rows[:, 7].mean()), turn=float(rows[:, 8].mean()), groom=float(rows[:, 11].mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fly-only", action="store_true")
    ap.add_argument("--memory", type=int, default=0)
    ap.add_argument("--adapt", action="store_true")
    a = ap.parse_args()
    from flyweight import fly_layers
    live = LiveFly(fly_only=a.fly_only, memory=a.memory, adapt=fly_layers.ADAPT if a.adapt else None)
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    ok, secs = True, {"qwen": [], "you": []}

    def windows(n):
        out = []
        for _ in range(n):
            t = time.time()
            m = live.window()
            secs[m["state"]["driver"]].append(time.time() - t)
            out.append(m)
        return out

    def check(name, good, detail):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'ok  ' if good else 'FAIL'}  {name}: {detail}", flush=True)

    if not a.fly_only:
        w = windows(4)
        check("Qwen drives", all(any(s["kind"] == "qwen" for s in m["segments"]) for m in w), f"answers {[m['tokens'][0].strip() for m in w]}")
        live.apply({"type": "message", "text": "Hi, nice weather?"})
        w = windows(3)
        reads = [next(s for s in m["segments"] if s["kind"] == "qwen")["read"] for m in w]
        check("a message stays in every report", all('The user says: "Hi, nice weather?"' in r for r in reads) and w[-1]["state"]["message"] == "Hi, nice weather?", "3 of 3 reports")
        live.apply({"type": "message", "text": ""})
        w = windows(2)
        check("clearing the message", all("The user says" not in next(s for s in m["segments"] if s["kind"] == "qwen")["read"] for m in w) and w[-1]["state"]["message"] is None, "gone from the next report")
    live.apply({"type": "driver", "value": "you"})
    for c, what, good in MOVES:
        live.apply({"type": "world", "action": "reset"})
        live.apply({"type": "levels", "levels": {k: (12 if k == c else 0) for k in CMDS}})
        w = windows(8)
        b = body(w[3:])
        held = w[-1]["state"].get("held")
        check(f"you drive {c} at 12", good(b) and (held is None or held == [12.0 if k == c else 0.0 for k in CMDS]),
              f"{what} {b[what]:+.2f} (walk {b['walk']:.2f}, back {b['back']:.2f}, turn {b['turn']:+.2f}, groom {b['groom']:.2f}); registers held {held}")
    live.apply({"type": "levels", "levels": {k: 0 for k in CMDS}})
    live.apply({"type": "world", "action": "dust"})
    w = windows(1)
    check("dust", w[-1]["state"]["dust"] > 0, f"dust {w[-1]['state']['dust']:.2f}")
    live.apply({"type": "world", "action": "reset"})
    w = windows(1)
    st = w[-1]["state"]
    check("reset", abs(st["x"] - START[0]) < 0.5 and abs(st["y"] - START[1]) < 0.5 and st["dust"] == 0, f"at ({st['x']:.2f}, {st['y']:.2f}), dust {st['dust']:.2f}")
    live.apply({"type": "pause", "value": True})
    paused = live.window()
    live.apply({"type": "pause", "value": False})
    check("pause", paused is None and live.window() is not None, "no window while paused, windows again after")
    rss = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - rss0) / 1024
    for d, s in secs.items():
        if s:
            print(f"  {d} drives: {np.median(s):.2f} s per {live.report_ms} ms window (median of {len(s)})")
    print(f"  peak memory grew {rss:.0f} MB over the run")
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
