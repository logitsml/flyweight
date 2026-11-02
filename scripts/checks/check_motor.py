"""How well the fly moves: each command through the connectome, in the body's units (flyweight/flyworld.py).

    python scripts/checks/check_motor.py [--adapt] [--inside]

check_body asks whether each behaviour is reachable; this measures it. One fly at a time on the fly's own
layers (equal to the fly inside Qwen, check_merged), registers on a schedule, the body updated every ms:
  walking   forward at levels 6, 9, 12 and 15: speed (mm/s) and heading drift (deg/s); left and right turns
            alone and while walking (deg/s); backward walking (mm/s); grooming (fraction of full)
            and, reported with no criterion, forward at 12 and 15 for 2 s (drift over 250-2000 ms)
  takeoff   the giant fiber for 40 ms: does the fly leave the floor, how high and for how long
  flight    the giant fiber for 40 ms with DNg02 held at 15, 12 or 9: the share of the time aloft the fly
            stays up, its height, ground speed and distance; its yaw while taking off (to 300 ms) and its
            heading drift flying on (from 300 ms); a turn command in flight from 300 ms (deg/s, from 400 ms,
            past the body's 0.1 s rotational lag); DNg02 released mid-flight (how long the fly takes to come down).
            Yaw rates in flight count the fly's own turning and leave out its saccades away from the walls.
PASS: forward walks at least 5 mm/s at level 12, faster with level and straight (heading drift within 12
deg/s at every level); each turn turns its own way at least 45 deg/s; back walks backward at least 4 mm/s;
groom grooms; the giant fiber alone hops and lands, and with DNg02 at 9 too it leaves the floor; with DNg02
at 12 or 15 the fly stays aloft and covers ground; flying on, it holds its heading within 30 deg/s; a turn
command in flight turns it its own way at least 90 deg/s; released, it lands; over a long flight it spends
under a fifth of its time at a wall.
--set walk|flight|steer runs a third (three processes then cover the whole in a third of the time).
--inside runs the same schedules on the fly's layers inside the merged model (Qwen loaded, the registers
written by DRIVE tokens), where the answer bridge's inhibitions act: a turn command lowers forward drive.
Each run saves its per-condition measures and body paths to results/motor[_inside].json.
"""

import argparse
import gc
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from flyweight import fly_layers, setup  # noqa: E402
from flyweight.flyworld import Body, World  # noqa: E402
from flyweight.sets import REG, ROLES  # noqa: E402

WALKS = [(f"forward {lv}", {0: {"forward": lv}}) for lv in (6, 9, 12, 15)] + [
    ("left 12", {0: {"turn_left": 12}}), ("right 12", {0: {"turn_right": 12}}),
    ("forward 9 + left 12", {0: {"forward": 9, "turn_left": 12}}), ("forward 9 + right 12", {0: {"forward": 9, "turn_right": 12}}),
    ("back 12", {0: {"back": 12}}), ("groom 12", {0: {"groom": 12}})]
JUMP = 400  # steps (40 ms) of giant fiber drive
TURN = 3000  # steps: a turn command in flight starts at 300 ms, once the jump's recruitment of the turn pools has passed
FLIGHTS = [("takeoff alone", {0: {"takeoff": 15}, JUMP: {}}),
           ("takeoff + fly 15", {0: {"takeoff": 15, "fly": 15}, JUMP: {"fly": 15}}),
           ("takeoff + fly 12", {0: {"takeoff": 15, "fly": 12}, JUMP: {"fly": 12}}),
           ("takeoff + fly 9", {0: {"takeoff": 15, "fly": 9}, JUMP: {"fly": 9}}),
           ("fly 15, then left 12", {0: {"takeoff": 15, "fly": 15}, JUMP: {"fly": 15}, TURN: {"fly": 15, "turn_left": 12}}),
           ("fly 15, then right 12", {0: {"takeoff": 15, "fly": 15}, JUMP: {"fly": 15}, TURN: {"fly": 15, "turn_right": 12}}),
           ("fly 15, released at 250 ms", {0: {"takeoff": 15, "fly": 15}, JUMP: {"fly": 15}, 2500: {}}),
           ("long flight, fly 15", {0: {"takeoff": 15, "fly": 15}, JUMP: {"fly": 15}})]


def run(con, groups, roles, world, adapt, sched, ms, start, m=None):
    if m is not None:  # inside the merged model
        from flyweight.merged import QwenFlyRunner
        r = QwenFlyRunner(m); r.start_fly(); fm = r.flybrain
    else:
        fm = fly_layers.FlyModel(con, groups, roles, batch=1, adapt=adapt)
    world.bodies = [Body("fly", *start[:2], yaw=start[2])]
    world.t = 0
    for s in range(int(ms * 10)):
        if s in sched:
            fm.set_levels({REG[g]: sched[s].get(g, 0) for g in ("forward", "turn_left", "turn_right", "takeoff", "fly", "back", "groom")})
        fm.step()
        world.t += 1
        if world.t % 10 == 0:
            world.update_body(0, "fly", dict(zip(ROLES, fm.move()[0].tolist())))
    del fm
    if m is not None:
        del r
    gc.collect(); torch.cuda.empty_cache()  # each run's CUDA graphs go with it
    return world.bodies[0].log  # per ms: t, x, y, z, yaw, mode, fwd, back, turn, power, MN9, groom, ...


def walk_stats(log):
    L = log[len(log) // 3:]  # past the onset
    dt = (L[-1][0] - L[0][0]) / 1000
    dx, dy = L[-1][1] - L[0][1], L[-1][2] - L[0][2]
    yaw0, yaw1 = L[0][4], L[-1][4]
    ahead = dx * math.cos(yaw0) + dy * math.sin(yaw0)  # + forward, - backward
    return dict(speed=ahead / dt, turn=math.degrees(yaw1 - yaw0) / dt, groom=float(np.mean([r[11] for r in L])),
                air=any(r[5] == "air" for r in log))


def yaw_rate(rows):
    """deg/s of the fly's own turning over flight rows (per ms): each ms's heading change minus the saccade the body
    added in it, which World.saccade gives from the position and heading the ms started with."""
    own = sum(b[4] - a[4] - math.radians(World.saccade(SimpleNamespace(x=a[1], y=a[2], yaw=a[4]))) * 1e-3
              for a, b in zip(rows, rows[1:]) if b[0] - a[0] == 1)
    span = (rows[-1][0] - rows[0][0]) / 1000 if len(rows) > 1 else 0.0
    return math.degrees(own) / span if span else 0.0


def flight_stats(log, t_release=None, t_turn=0.0, t_end=1e9):
    air = [r for r in log if r[5] == "air"]
    if not air:
        return dict(took_off=False)
    t0 = air[0][0]
    after = [r for r in log if r[0] >= t0 and (t_release is None or r[0] < t_release)]
    aloft = float(np.mean([r[5] == "air" for r in after])) if after else 0.0
    fl = [r for r in after if r[5] == "air"]
    path = sum(math.hypot(b[1] - a[1], b[2] - a[2]) for a, b in zip(fl, fl[1:]))
    span = (fl[-1][0] - fl[0][0]) / 1000 if len(fl) > 1 else 0.0
    out = dict(took_off=True, t_off=t0, aloft=aloft, apex=max(r[3] for r in air), height=float(np.mean([r[3] for r in fl])),
               power=float(np.mean([r[9] for r in fl])), speed=path / span if span else 0.0, dist=path,
               turn=0.0, landed_at=None)
    out["turn"] = yaw_rate([r for r in fl if t_turn <= r[0] <= t_end])
    out["takeoff_yaw"] = yaw_rate([r for r in fl if r[0] <= 300]) * (min(300, fl[-1][0]) - t0) / 1000  # deg, to 300 ms
    out["drift"] = yaw_rate([r for r in fl if r[0] >= 300])  # flying on, from 300 ms
    for a, b in zip(log, log[1:]):
        if a[5] == "air" and b[5] == "walk" and (t_release is None or b[0] >= t_release):
            out["landed_at"] = b[0]; break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapt", action="store_true")
    ap.add_argument("--inside", action="store_true")
    ap.add_argument("--set", choices=["walk", "flight", "steer", "all"], default="all")
    ap.add_argument("--only", default="", help="flight conditions to run, by name, separated by ;")
    a = ap.parse_args()
    adapt = fly_layers.ADAPT if a.adapt else None
    m = None
    if a.inside:
        tok, qwen = setup.load_qwen()
        S = setup.standard(tok, qwen, adapt=adapt)
        m, con, world, groups, roles = S["m"], S["con"], S["world"], S["groups"], S["roles"]
    else:
        con, A, world, groups, roles = setup.world_and_groups()
    res, paths, ok = {}, {}, True
    for nm, sched in (WALKS if a.set in ("walk", "all") else []):
        log = run(con, groups, roles, world, adapt, sched, 250, (-20.0, 18.0, -0.3), m)
        res[nm] = s = walk_stats(log); paths[nm] = [r[:6] for r in log[::5]]
        print(f"  {nm:24s} speed {s['speed']:+6.2f} mm/s  turn {s['turn']:+7.1f} deg/s  groom {s['groom']:.2f}  airborne {s['air']}", flush=True)
    for lv in ((12, 15) if a.set in ("walk", "all") else ()):  # sustained walking, reported (no criterion): drift over 250-2000 ms
        nm = f"long walk, forward {lv}"
        log = run(con, groups, roles, world, adapt, {0: {"forward": lv}}, 2000, (-20.0, 18.0, -0.3), m)
        L = [r for r in log if r[0] >= 250]
        res[nm] = s = dict(speed=sum(math.hypot(b[1] - a[1], b[2] - a[2]) for a, b in zip(L, L[1:])) / ((L[-1][0] - L[0][0]) / 1000),
                           turn=math.degrees(L[-1][4] - L[0][4]) / ((L[-1][0] - L[0][0]) / 1000)); paths[nm] = [r[:6] for r in log[::5]]
        print(f"  {nm:24s} speed {s['speed']:+6.2f} mm/s  turn {s['turn']:+7.1f} deg/s over 250-2000 ms", flush=True)
    STEER = ("takeoff + fly 15", "fly 15, then left 12", "fly 15, then right 12", "long flight, fly 15")
    if a.only:
        FL = [(nm, sc) for nm, sc in FLIGHTS if nm in a.only.split(";")]
    for nm, sched in (FL if a.only else [(nm, sc) for nm, sc in FLIGHTS if a.set == "all" or (a.set == "steer") == (nm in STEER) and a.set != "walk"]):
        rel = 250.0 if "released" in nm else None
        log = run(con, groups, roles, world, adapt, sched, 900 if "long" in nm else 750 if "released" in nm else 600, (-26.0, 22.0, -0.45), m); paths[nm] = [r[:6] for r in log[::5]]
        res[nm] = s = flight_stats(log, rel, (TURN / 10 + 100) if "left" in nm or "right" in nm else 300.0, 600.0)
        if "long" in nm and s["took_off"]:  # the share of flight spent within 2.5 mm of a wall
            fl = [r for r in log if r[5] == "air"]; s["at_wall"] = float(np.mean([max(abs(r[1]), abs(r[2])) > 27.5 for r in fl]))
        if not s["took_off"]:
            print(f"  {nm:24s} never left the floor", flush=True); continue
        print(f"  {nm:24s} off at {s['t_off']:.0f} ms; aloft {100 * s['aloft']:.0f}% of the time after; apex {s['apex']:.1f} mm, "
              f"mean height {s['height']:.1f} mm; power {s['power']:.2f}; ground speed {s['speed']:.0f} mm/s over {s['dist']:.1f} mm; "
              f"yaw taking off {s['takeoff_yaw']:+.0f} deg; turn {s['turn']:+.0f} deg/s; drift flying on {s['drift']:+.0f} deg/s"
              + (f"; landed at {s['landed_at']:.0f} ms" if s["landed_at"] else "")
              + (f"; at a wall {100 * s['at_wall']:.0f}% of the flight" if "at_wall" in s else ""), flush=True)
    checks = {}
    if a.set in ("walk", "all"):
        f = [res[f"forward {lv}"]["speed"] for lv in (6, 9, 12, 15)]
        checks.update({
            "forward walks, faster with level": f[2] >= 5 and f[3] >= f[1] and not any(res[f"forward {lv}"]["air"] for lv in (6, 9, 12, 15)),
            "forward walks straight at every level": all(abs(res[f"forward {lv}"]["turn"]) <= 12 for lv in (6, 9, 12, 15)),
            "turns turn their own way": res["left 12"]["turn"] >= 45 and res["right 12"]["turn"] <= -45
                                        and res["forward 9 + left 12"]["turn"] > 0 > res["forward 9 + right 12"]["turn"],
            "back walks backward at 4 mm/s or more": res["back 12"]["speed"] <= -4.0,
            "groom grooms": res["groom 12"]["groom"] >= 0.3,
            "no walking command takes off": not any(res[nm]["air"] for nm, _ in WALKS)})
    if a.set != "walk":
        fl = lambda nm: res[nm]
        cand = {
            "the giant fiber alone hops and lands": lambda: fl("takeoff alone")["took_off"] and fl("takeoff alone").get("landed_at") is not None,
            "with DNg02 at 9 the fly leaves the floor": lambda: fl("takeoff + fly 9")["took_off"],
            "DNg02 at 12 or 15 keeps the fly aloft and moving": lambda: [fl(n).get("aloft", 0) >= 0.95 and fl(n).get("dist", 0) >= 10
                                                                         for n in ("takeoff + fly 12", "takeoff + fly 15") if n in res] and all(
                fl(n).get("aloft", 0) >= 0.95 and fl(n).get("dist", 0) >= 10 for n in ("takeoff + fly 12", "takeoff + fly 15") if n in res),
            "flying on, the fly holds its heading within 30 deg/s": lambda: abs(fl("long flight, fly 15")["drift"]) <= 30,
            "in flight, each turn turns its own way at 90 deg/s or more": lambda: fl("fly 15, then left 12").get("turn", 0) >= 90 and fl("fly 15, then right 12").get("turn", 0) <= -90,
            "released, the fly lands": lambda: fl("fly 15, released at 250 ms")["took_off"] and fl("fly 15, released at 250 ms").get("landed_at") is not None,
            "a long flight keeps off the walls (under a fifth of the time)": lambda: fl("long flight, fly 15").get("at_wall", 1.0) < 0.2}
        for k, v in cand.items():  # the checks whose conditions this set ran
            try:
                checks[k] = v() if callable(v) else v
            except KeyError:
                pass
    for k, v in checks.items():
        print(f"  {'ok  ' if v else 'FAIL'}  {k}")
        ok &= v
    out = ROOT / "results" / ("motor" + ("_inside" if a.inside else "") + ("_adapt" if a.adapt else "") + ("" if a.set == "all" else "_" + a.set) + ("_only" if a.only else "") + ".json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(dict(results=res, checks=checks, paths=paths), default=float))
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
