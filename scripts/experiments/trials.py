"""Controlled trials (docs/design.md): does the fly reach the sugar because Qwen read
its reports, and how well, against controls?

    python scripts/experiments/trials.py --cond answer|groups|constant|oracle|none --task orient|walk|groom|back|perch|request
        [--n 10] [--say TEXT] [--memory 4] [--adapt] [--tag NAME]

Conditions (one Qwen, one fly; episodes run one after another on the same model, the fly reset
between them; each report is read from Qwen's post-instructions state). The oracle and none
conditions read no text, so they run the fly's layers alone: the same computation as the fly inside
Qwen, step for step (scripts/checks/check_merged.py), in a quarter of the memory, beside a Qwen condition:
  answer    Qwen's own answer drives the commands (the default design)
  groups    the 48 situation neurons and the hand-written map drive them (rediscovered on this report)
  constant  as answer, but the residual the fly reads at the report's end is replaced by its mean over
            40 synthetic walking reports: the same drive whatever the report says
  oracle    no Qwen: the host pilot writes the commands with DRIVE tokens (15 x the pilot's targets); in the air it
            stops once the perch is within report.STOP_MM, as Qwen's instructions say
  none      no drive at all
Tasks (starts seeded per episode, identical across conditions):
  orient    500 ms on the floor, the sugar drop 10 mm away at bearings +-30, 60, 90, 120, 150 degrees:
            did the fly turn toward it? Scored on the signed heading change (more than 10 degrees
            toward the drop's side, or away from it), not on |bearing|, which wraps at 180 degrees
            and grows as a straight walker passes the drop; and whether |bearing| ever fell below 15.
  walk      up to 3 s, the drop 7 mm away at a random bearing: did the labellum reach it? Chance for a
            fly that only walks straight is the share of starts within ~30 degrees of the drop.
  groom     up to 600 ms on the floor with dust on the antennae (Johnston's organ fires): were they
            cleaned? The grooming pool cleans them; the connectome's own Johnston's organ -> aDN path can
            groom without any command, so the none condition measures that reflex.
  back      up to 1 s facing an arena wall, the antennae against it: did the fly get its head
            report.WALL_BACK (3 mm) clear of the wall, and did it walk backward to do so?
  perch     up to 2 s, fed, on the floor beside the sugar drop: did it take off, fly to the perch (a 12 mm
            post ~18 mm away) and land on its top? The whole flight: the giant fiber's jump, wing power
            from DNg02, steering in the air, and stopping in time for the fly to glide down onto the top.
  request   up to 300 ms on the floor with clean antennae, so Johnston's organ is silent and the grooming reflex
            off: did the fly groom (its grooming pool at half strength or more)? Run with --say "Groom your
            antennae." the visitor's request reaches the fly only through Qwen's answer.
--say puts a visitor's message in every report, as the live loop does (The user says: "...").
--memory and --adapt run the variants (Qwen's log; the fly's adaptation); --tag names the results file.
Rates with Wilson 95% intervals; results to results/trials_<cond><tag>_<task>.json.

    python scripts/experiments/trials.py --rescore   # every saved run, scored the same way -> results/trials_summary.json
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from flyweight import fly_layers, report, setup  # noqa: E402
from flyweight.flyworld import ANTENNA, ARENA, DROP_R, PERCH, PERCH_R, PERCH_TOP, SUGAR, Body  # noqa: E402
from flyweight.merged import ANSWER_WORDS, QH, QwenFlyRunner  # noqa: E402
from flyweight.sets import REG, ROLES, SENSES, WORDS  # noqa: E402

MAX_MS = {"orient": 500, "walk": 3000, "groom": 600, "back": 1000, "perch": 2000, "request": 300}

torch.backends.cuda.matmul.allow_tf32 = False


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 1.0)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (p, max(0.0, c - h), min(1.0, c + h))


def starts(task, n):
    """(x, y, yaw) per episode: the fly placed around the fixed sugar drop at the bearing wanted."""
    out = []
    for e in range(n):
        rng = np.random.default_rng(1000 + e)
        if task == "back":  # facing one of the four walls, the head 0.2-1.2 mm from it
            side, gap, edge = e % 4, rng.uniform(0.2, ANTENNA - 0.3), rng.uniform(-ARENA + 6, ARENA - 6)
            c = ARENA - 1.2 - gap
            out.append([(c, edge, 0.0), (edge, c, math.pi / 2), (-c, edge, math.pi), (edge, -c, -math.pi / 2)][side])
            continue
        if task == "orient":
            b = [30, -30, 60, -60, 90, -90, 120, -120, 150, -150][e % 10]
            d = 10.0
        elif task in ("groom", "request"):
            b, d = rng.uniform(-180, 180), 12.0
        elif task == "perch":  # fed, beside the drop, facing anywhere
            b, d = rng.uniform(-180, 180), 2.0
        else:
            b, d = rng.uniform(-180, 180), 7.0
        phi = rng.uniform(0, 2 * math.pi)  # where around the drop the fly stands
        x, y = SUGAR[0] + d * math.cos(phi), SUGAR[1] + d * math.sin(phi)
        to_drop = math.atan2(SUGAR[1] - y, SUGAR[0] - x)
        out.append((x, y, to_drop - math.radians(b)))  # bearing (+ = left) = direction to the drop - yaw
    return out


def bearing(B):
    return (math.degrees(math.atan2(SUGAR[1] - B.y, SUGAR[0] - B.x) - B.yaw) + 180) % 360 - 180


def bearing_xy(x, y, yaw):
    return (math.degrees(math.atan2(SUGAR[1] - y, SUGAR[0] - x) - yaw) + 180) % 360 - 180


def quantized(b):
    """The bearing as a report states it: whole degrees, "ahead" within 15 (what the oracle may use)."""
    return 0.0 if abs(b) < 15 else float(round(b))


def summarize(task, eps):
    n = len(eps)
    if task == "orient":
        turn = [math.degrees(e["end"][2] - e["start"][2]) * (1 if e["bearing0"] > 0 else -1) for e in eps]  # + = toward the drop's side
        aligned = [min(abs(bearing_xy(x, y, yw)) for _, x, y, yw in e["path"]) < 15 if e["path"] else False for e in eps]
        return dict(turned_toward=wilson(sum(t > 10 for t in turn), n), turned_away=wilson(sum(t < -10 for t in turn), n),
                    aligned_15=wilson(sum(aligned), n), mean_turn_toward_deg=float(np.mean(turn)),
                    sem_turn_deg=float(np.std(turn, ddof=1) / math.sqrt(n)) if n > 1 else 0.0)
    if task == "perch":
        ok = [e["done_ms"] is not None for e in eps]
        return dict(landed=wilson(sum(ok), n), took_off=wilson(sum(e.get("took_off", False) for e in eps), n),
                    median_ms=float(np.median([e["done_ms"] for e in eps if e["done_ms"]])) if any(ok) else None,
                    max_z_mm=float(np.mean([e.get("max_z", 0.0) for e in eps])))
    if task in ("groom", "back", "request"):
        ok = [e["done_ms"] is not None for e in eps]
        out = dict(done=wilson(sum(ok), n), median_ms=float(np.median([e["done_ms"] for e in eps if e["done_ms"]])) if any(ok) else None,
                   moved_mm=float(np.mean([math.hypot(e["end"][0] - e["start"][0], e["end"][1] - e["start"][1]) for e in eps])))
        if task == "back":  # walked backward: net displacement against the starting heading of at least 1 mm
            backed = [-(math.cos(e["start"][2]) * (e["end"][0] - e["start"][0]) + math.sin(e["start"][2]) * (e["end"][1] - e["start"][1])) >= 1.0 for e in eps]
            out["walked_back"] = wilson(sum(backed), n)
        return out
    ok = [e["reached_ms"] is not None for e in eps]
    return dict(reached=wilson(sum(ok), n), median_ms=float(np.median([e["reached_ms"] for e in eps if e["reached_ms"]])) if any(ok) else None,
                mean_dist_end=float(np.mean([e["dist_end"] for e in eps])),
                straight_chance=float(np.mean([abs(e["bearing0"]) < 30 for e in eps])))


def rescore():
    rows = {}
    for f in sorted((ROOT / "results").glob("trials_*_*.json")):
        d = json.loads(f.read_text())
        if d.get("n", 0) < 2:
            continue
        rows[f"{d['cond']}{d.get('tag', '')}/{d['task']}"] = dict(n=d["n"], **summarize(d["task"], d["episodes"]))
    (ROOT / "results" / "trials_summary.json").write_text(json.dumps(rows, indent=1))
    fmt = lambda w: f"{w[0]:.0%} [{w[1]:.0%}, {w[2]:.0%}]"
    for k, v in rows.items():
        if k.endswith("orient"):
            print(f"{k:18s} n={v['n']:2d}  toward {fmt(v['turned_toward'])}  away {fmt(v['turned_away'])}  aligned {fmt(v['aligned_15'])}  "
                  f"turn toward {v['mean_turn_toward_deg']:+.0f} +- {v['sem_turn_deg']:.0f} deg")
        elif k.endswith("perch"):
            print(f"{k:18s} n={v['n']:2d}  landed {fmt(v['landed'])}  took off {fmt(v['took_off'])}  median {v['median_ms']} ms  max height {v['max_z_mm']:.1f} mm")
        elif k.endswith(("groom", "back", "request")):
            print(f"{k:18s} n={v['n']:2d}  done {fmt(v['done'])}  median {v['median_ms']} ms  moved {v['moved_mm']:.1f} mm"
                  + (f"  walked back {fmt(v['walked_back'])}" if "walked_back" in v else ""))
        else:
            print(f"{k:18s} n={v['n']:2d}  reached {fmt(v['reached'])}  median {v['median_ms']} ms  end distance {v['mean_dist_end']:.1f} mm  "
                  f"(straight-walk chance {v['straight_chance']:.0%})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", choices=["answer", "groups", "constant", "oracle", "none"])
    ap.add_argument("--task", choices=list(MAX_MS))
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--report-ms", type=int, default=report.STEP_MS)
    ap.add_argument("--say", default="", help="a visitor's message, in every report")
    ap.add_argument("--memory", type=int, default=0, help="Qwen reads each report after its last N reports and answers")
    ap.add_argument("--adapt", action="store_true", help="the adaptation variant (fly_layers.ADAPT)")
    ap.add_argument("--tag", default="", help="suffix for the results file (variants)")
    a = ap.parse_args()
    adapt = fly_layers.ADAPT if a.adapt else None
    if a.rescore:
        return rescore()
    max_ms = MAX_MS[a.task]
    if a.cond in ("oracle", "none"):  # no text: the fly's layers alone
        con, A, world, groups, roles = setup.world_and_groups()
        fly = fly_layers.FlyModel(con, groups, roles, adapt=adapt)
        r = type("FlyOnly", (), dict(flybrain=fly, say_bad=0, start_fly=lambda self: fly._start()))()
    else:
        tok, qwen = setup.load_qwen()
        S = setup.standard(tok, qwen, bridge="groups" if a.cond == "groups" else "answer", adapt=adapt)
        m, world = S["m"], S["world"]
        r = QwenFlyRunner(m, memory=a.memory)
        pre = tok(report.PROMPT).input_ids
        r.text(pre); r.snapshot_qwen()
    if a.cond == "constant":  # the mean residual at the report's end over 40 synthetic walking reports
        acc, cnt = {}, [0]

        def grab(mod, args):
            x = args[0]
            if torch.cuda.is_current_stream_capturing():  # a fly step's block graph: fly tokens, never the report's end
                return
            if float(x[0, -1, QH]) != 0.0:
                acc["s"] = acc.get("s", 0) + x[0, -1].double()
                cnt[0] += 1
        hk = m.flayers[0].register_forward_pre_hook(grab)
        r.start_fly()
        rng = np.random.default_rng(5)
        for _ in range(40):
            h, words, st, tgt = report.situation(rng, "seek")
            setup.read_report(r, tok, 100, h, words=words)
        hk.remove()
        mean = (acc["s"] / cnt[0]).float()

        def const(mod, args):
            x = args[0]
            if torch.cuda.is_current_stream_capturing():
                return
            if float(x[0, -1, QH]) != 0.0:
                x = x.clone(); x[0, -1, :QH] = mean[:QH]
            return (x,)
        m.flayers[0].register_forward_pre_hook(const)
    steps = a.report_ms * 10
    wid = [tok(" " + w).input_ids[0] for w in ANSWER_WORDS] if a.cond not in ("oracle", "none") else None
    t0 = time.time()
    eps = []
    for e, (x0, y0, yaw0) in enumerate(starts(a.task, a.n)):
        r.start_fly()
        if hasattr(r, "reset_memory"):
            r.reset_memory()
        world.bodies = [Body("fly", x0, y0, yaw=yaw0, dust=1.0 if a.task == "groom" else 0.0, fed_ms=600.0 if a.task == "perch" else 0.0)]
        world.t = 0
        Bd = world.bodies[0]
        sense_now = {g: 0 for g in SENSES}
        ep = dict(start=[round(x0, 3), round(y0, 3), round(yaw0, 4)], bearing0=round(bearing(Bd), 1), reports=[], reached_ms=None, done_ms=None)
        while world.t < max_ms * 10 and not (a.task != "orient" and (ep["reached_ms"] or ep["done_ms"]) is not None):
            if a.cond in ("answer", "groups", "constant"):
                h = report.head(*world.observe(0)) + (f' The user says: "{a.say}"' if a.say else "")
                lg, words = setup.read_report(r, tok, world.t / 10, h)
                lv = r.levels(r.registers())[:len(WORDS)]
                p = torch.softmax(lg[wid].float(), 0).cpu().numpy()
                r.end_report(setup.answer_ids(tok, ANSWER_WORDS[int(p.argmax())]))
                ep["reports"].append(dict(t=world.t / 10, words="".join(words), answer=ANSWER_WORDS[int(p.argmax())],
                                          p=np.round(p, 3).tolist(), levels=np.round(lv, 2).tolist()))
            elif a.cond == "oracle":  # the pilot's move from what the report states (bearing rounded, antennae as printed)
                where, x, y, z, sug, bit, per, fed, wall = world.observe(0)
                touching = world.sense_levels(0)["JO-CE"] > 0
                if where == "in flight":
                    state, tgt = ("over_perch" if per[0] < report.STOP_MM else "flight"), per[1]
                else:
                    state, tgt = ("wall" if wall < report.WALL_BACK else ("dusty" if touching else ("seek" if fed < report.FEED_MS else "fed_floor"))), sug[1]
                t = report.pilot_targets(state, quantized(tgt))
                lv = [int(round(15 * t[w])) for w in WORDS]
                r.flybrain.set_levels(dict(enumerate(lv)))
                ep["reports"].append(dict(t=world.t / 10, levels=lv, state=state))
            for s_ in range(steps):
                sl = world.sense_levels(0)
                ch = {REG[g]: sl[g] for g in SENSES if sl[g] != sense_now[g]}
                if ch:
                    r.flybrain.set_levels(ch)
                    sense_now.update(sl)
                r.flybrain.step()
                world.t += 1
                if world.t % 10 == 0:
                    world.update_body(0, a.cond, dict(zip(ROLES, r.flybrain.move()[0].tolist())))
                    head = np.array([Bd.x + 1.2 * math.cos(Bd.yaw), Bd.y + 1.2 * math.sin(Bd.yaw)])
                    if a.task == "walk" and ep["reached_ms"] is None and np.linalg.norm(head - SUGAR) < DROP_R:
                        ep["reached_ms"] = world.t / 10
                    if Bd.mode == "air":
                        ep["took_off"] = True
                    ep["max_z"] = max(ep.get("max_z", 0.0), Bd.z)
                    landed = Bd.mode == "walk" and Bd.z >= PERCH_TOP - 0.1 and np.linalg.norm([Bd.x - PERCH[0], Bd.y - PERCH[1]]) < PERCH_R
                    if ep["done_ms"] is None and ((a.task == "groom" and Bd.dust == 0.0) or (a.task == "back" and world.wall_ahead(0) > report.WALL_BACK)
                                                  or (a.task == "perch" and landed) or (a.task == "request" and Bd.log[-1][11] >= 0.5)):
                        ep["done_ms"] = world.t / 10
        ep.update(bearing_end=round(bearing(Bd), 1), end=[round(Bd.x, 3), round(Bd.y, 3), round(Bd.yaw, 4)],
                  dist_end=round(float(np.linalg.norm([Bd.x - SUGAR[0], Bd.y - SUGAR[1]])), 2),
                  path=[(row[0], row[1], row[2], row[4]) for row in Bd.log[::10]])
        eps.append(ep)
        print(f"episode {e + 1}/{a.n} ({time.time() - t0:.0f} s): bearing {ep['bearing0']:+.0f} -> {ep['bearing_end']:+.0f} deg, "
              f"distance {ep['dist_end']:.1f} mm, reached {ep['reached_ms']}, done {ep['done_ms']}, dust {Bd.dust:.2f}", flush=True)
    B = a.n
    summ = summarize(a.task, eps)
    out = dict(cond=a.cond, task=a.task, n=B, report_ms=a.report_ms, summary=summ, episodes=eps, wall_s=time.time() - t0,
               tag=a.tag, memory=a.memory, adapt=adapt,
               block_check=[r.flybrain.mismatch, r.flybrain.swaps], say_bad=r.say_bad)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"trials_{a.cond}{a.tag}_{a.task}.json").write_text(json.dumps(out))
    print(f"{a.cond} / {a.task}: {json.dumps(summ)}; block check {r.flybrain.mismatch}/{r.flybrain.swaps}; "
          f"SAY off-slot {r.say_bad}; {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
