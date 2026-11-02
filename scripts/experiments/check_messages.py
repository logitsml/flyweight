"""What a visitor's message does to the fly (the live server, scripts/run/serve.py, puts it in every report).

    python scripts/experiments/check_messages.py [--states 30]

Qwen's instructions are the ones every run uses; nothing in them mentions a visitor. For random world
states (on the floor, with dusty antennae, fed, in flight) the one-stack model reads each report with
and without ' The user says: "..."' after the world's part, as the live server writes it, the fly's
printed words given as text. Reported per message: how often Qwen's move changes, and how far the fly's
seven register levels move (read from the fly's first layer, where Qwen's answer writes them).
"""

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from flyweight import report, setup  # noqa: E402
from flyweight.flyworld import ARENA, BITTER, PERCH, SUGAR  # noqa: E402
from flyweight.merged import QwenFlyRunner  # noqa: E402
from flyweight.sets import DRIVEN  # noqa: E402

MESSAGES = ["Hi, nice weather?", "asdf qwerty zxcv", "I love you, little fly.", "Go to the sugar.", "Go to the perch.",
            "Stop!", "Turn left.", "Fly away!", "Groom your antennae."]
KINDS = ["floor", "floor", "floor", "dusty", "fed", "air"]


def rel(x, y, yaw, p):
    d = np.asarray(p) - [x, y]
    return float(np.linalg.norm(d)), float((math.degrees(math.atan2(d[1], d[0]) - yaw) + 180) % 360 - 180)


def wall_ahead(x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    ts = [((ARENA if c > 0 else -ARENA) - x) / c if abs(c) > 1e-9 else math.inf,
          ((ARENA if s > 0 else -ARENA) - y) / s if abs(s) > 1e-9 else math.inf]
    return min(t for t in ts if t > 0)


def states(n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        x, y, yaw = rng.uniform(-25, 25), rng.uniform(-25, 25), rng.uniform(-math.pi, math.pi)
        kind = KINDS[i % len(KINDS)]
        air = kind == "air"
        head = report.head("in flight" if air else "on the floor", x, y, 6.0 if air else 0.0, rel(x, y, yaw, SUGAR),
                           rel(x, y, yaw, BITTER), rel(x, y, yaw, PERCH), 600.0 if kind in ("fed", "air") else 0.0, wall_ahead(x, y, yaw))
        words = [" nothing", " folded", " touching" if kind == "dusty" else " free", " beating" if air else " still"]
        out.append((kind, head, words))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=int, default=30)
    a = ap.parse_args()
    tok, qwen = setup.load_qwen()
    r = QwenFlyRunner(setup.standard(tok, qwen)["m"])
    r.text(tok(report.PROMPT).input_ids)
    r.snapshot_qwen()
    r.start_fly()
    wid = [tok(" " + w).input_ids[0] for w in setup.ANSWER_WORDS]
    cmds = [c for c, _ in DRIVEN]

    def read(head, words):
        lg, _ = setup.read_report(r, tok, 1000.0, head, words=words)
        return setup.ANSWER_WORDS[int(lg[wid].argmax())], np.asarray(r.levels(r.registers())[:len(cmds)])

    S = states(a.states)
    ref = [read(h, w) for _, h, w in S]
    print(f"{len(S)} states ({', '.join(f'{k} {v}' for k, v in Counter(k for k, _, _ in S).items())}); "
          f"Qwen's moves with no message: {dict(Counter(m for m, _ in ref))}")
    print(f"{'message':26s} {'move changed':>13s} {'mean |level change|':>20s} {'max':>5s}  moves")
    for msg in MESSAGES:
        got = [read(h + f' The user says: "{msg}"', w) for _, h, w in S]
        changed = sum(g[0] != f[0] for g, f in zip(got, ref))
        d = np.abs(np.array([g[1] - f[1] for g, f in zip(got, ref)]))
        print(f"{msg:26s} {changed:>6d} / {len(S):<5d} {d.mean():>20.2f} {d.max():>5.1f}  {dict(Counter(m for m, _ in got))}", flush=True)


if __name__ == "__main__":
    main()
