"""What Qwen reads: its instructions with worked examples, then each report of its fly.

A report is the world's description of the fly (position, the sugar drop, the bitter drop, the
perch, time fed, and on the floor the arena wall ahead), then the fly's own neurons print its senses (docs/design.md):
the host writes the scaffold (" Its neurons: taste", ", proboscis", ...) and at each slot a SAY
token makes the fly's layers choose the word, a real text token. The report ends with the question
whose last token ("):") is where Qwen's answer writes the fly's command registers.

`situation` makes synthetic reports in exactly this format (for the answer check, the neuron
discovery, the bridge checks and the trials' oracle), with the move a sensible pilot would make.
"""

import math

from .flyworld import ANTENNA, ARENA, PERCH_TOP

STEP_MS = 20  # one report per this much fly time
FEED_MS = 500
ASK = "\nBest move (forward, left, right, back, jump, fly, groom or stop):"
WALL_BACK = 3.0  # mm: with the arena wall this close ahead the pilot backs away
STOP_MM = 10.0  # mm: the trials' oracle stops the wings with the perch this close: a fly whose wings stop at full power
# glides about this far while it sinks to the perch's top (check_motor's released flight)
GOAL2 = " The first goal is reached; now jump, fly to the perch and land on it."
# the fly's printed senses: slot -> (scaffold before it, [(word, role, threshold Hz per member)]);
# the first word whose role fires above its threshold by the most wins, else the last (default) word
SAY = [(" Its neurons: taste", [(" sweet", "sugar", 20.0), (" bitter", "bitter", 20.0), (" nothing", None, 0.0)]),
       (", proboscis", [(" extended", "MN9", 15.0), (" folded", None, 0.0)]),
       (", antennae", [(" touching", "JO-CE", 10.0), (" free", None, 0.0)]),
       (", wings", [(" beating", "power", 80.0), (" still", None, 0.0)])]  # DNg02 from level 7 drives the power motor neurons past 80 Hz; walking, 35-41
SAY_PER_HZ = 1.0 / 8.0  # logit per Hz above threshold (the default word sits at 0)
TAIL = "." + ASK


def side(ang):
    return "ahead" if abs(ang) < 15 else (f"{abs(ang):.0f} degrees to its left" if ang > 0 else f"{abs(ang):.0f} degrees to its right")


def head(where, x, y, z, sugar, bitter, perch, fed, wall=None):
    """The world's part of a report (sugar, bitter, perch: (distance mm, bearing deg, + = left); wall: the
    distance to the arena wall ahead, stated on the floor)."""
    rel = lambda name, d_a: f"{name} {d_a[0]:.0f} mm, {side(d_a[1])}"
    s = (f"The fly: {where} at ({x:.0f}, {y:.0f}, {z:.0f}) mm; {rel('sugar drop', sugar)}; {rel('bitter drop', bitter)}; "
         f"{rel(f'perch ({PERCH_TOP:.0f} mm high)', perch)}; fed {fed:.0f} ms.")
    if wall is not None and where == "on the floor":
        s += f" The arena wall is {wall:.0f} mm ahead."
    return s + (GOAL2 if fed >= FEED_MS else "")


def text(t_ms, head_text, words):
    """The full report as Qwen reads it (words: the four printed words, with their spaces)."""
    s = f"{t_ms:.0f} ms. {head_text}"
    for (scaffold, _), w in zip(SAY, words):
        s += scaffold + w
    return s + TAIL


def ids(tok, t_ms, head_text, words=None):
    """Token ids, piece by piece as the stream holds them: [head + scaffold 0], word 0, [scaffold 1],
    ... ; with words=None the pieces between the slots (for a live report: the fly fills the slots)."""
    pieces = [tok(f"{t_ms:.0f} ms. {head_text}{SAY[0][0]}").input_ids]
    for k in range(1, len(SAY)):
        pieces.append(tok(SAY[k][0]).input_ids)
    pieces.append(tok(TAIL).input_ids)
    if words is None:
        return pieces
    out = list(pieces[0])
    for k, w in enumerate(words):
        out += tok(w).input_ids + pieces[k + 1]
    return out


def pilot_word(state, tgt_ang):
    """The move a sensible pilot makes."""
    if state == "wall":
        return "back"
    if state == "dusty":
        return "groom"
    if state == "seek":
        return "forward" if abs(tgt_ang) < 20 else ("left" if tgt_ang > 0 else "right")
    if state in ("on_sugar", "over_perch"):
        return "stop"
    if state == "fed_floor":
        return "jump"
    return "fly" if abs(tgt_ang) < 25 else ("left" if tgt_ang > 0 else "right")  # flight


def pilot_targets(state, tgt_ang):
    """Graded targets per command, for the neuron discovery and the oracle (0..1)."""
    t = dict.fromkeys(["left", "right", "forward", "stop", "jump", "fly", "back", "groom"], 0.0)
    if state == "wall":
        t["back"] = 1.0
    elif state == "dusty":
        t["groom"] = 1.0
    elif state in ("seek", "flight"):
        t["left"] = min(1.0, max(0.0, tgt_ang / 60))
        t["right"] = min(1.0, max(0.0, -tgt_ang / 60))
        t["forward"] = max(0.0, math.cos(math.radians(tgt_ang))) if state == "seek" else 0.0
        t["fly"] = 1.0 if state == "flight" else 0.0
    elif state in ("on_sugar", "over_perch"):
        t["stop"] = 1.0
    elif state == "fed_floor":
        t["jump"] = 1.0
    return t


def situation(rng, state=None):
    """A synthetic report: (head text, the words the fly's neurons would print, state, target bearing)."""
    state = state or rng.choice(["seek", "seek", "seek", "on_sugar", "fed_floor", "flight", "flight", "over_perch", "wall", "dusty"])
    sug = [rng.uniform(3, 30), rng.uniform(-180, 180)]
    per = [rng.uniform(3, 30), rng.uniform(-180, 180)]
    bit = [rng.uniform(5, 40), rng.uniform(-180, 180)]
    x, y = rng.uniform(-ARENA + 6, ARENA - 6), rng.uniform(-ARENA + 6, ARENA - 6)
    wall = rng.uniform(WALL_BACK + 1, 45)  # the wall ahead, as far as a heading allows
    if state == "wall":  # facing a wall within WALL_BACK; the antennae touch it within ANTENNA
        wall = rng.uniform(0, WALL_BACK)
        edge = rng.uniform(-ARENA + 3, ARENA - 3)
        x, y = [(ARENA - 1.2 - wall, edge), (edge, ARENA - 1.2 - wall), (-ARENA + 1.2 + wall, edge), (edge, -ARENA + 1.2 + wall)][rng.integers(4)]
    fed = 0 if state in ("seek", "wall", "dusty") else (int(rng.uniform(50, 450)) if state == "on_sugar" else int(rng.uniform(500, 900)))
    if state == "on_sugar" or (state == "fed_floor" and rng.random() < 0.5):
        sug = [rng.uniform(0, 2), rng.uniform(-10, 10)]
    if state == "over_perch":
        per[0] = rng.uniform(0, 4)
    z = 0 if state in ("seek", "on_sugar", "fed_floor", "wall", "dusty") else rng.uniform(13, 25)
    h = head("in flight" if z > 0 else "on the floor", x, y, z, sug, bit, per, fed, wall)
    words = [" sweet" if (z == 0 and sug[0] < 2) else (" bitter" if (z == 0 and bit[0] < 2) else " nothing"),
             " extended" if state == "on_sugar" else " folded",
             " touching" if state == "dusty" or (state == "wall" and wall < ANTENNA) else " free",
             " beating" if z > 0 else " still"]
    tgt = per[1] if state in ("flight", "over_perch") else sug[1]
    return h, words, state, tgt


def examples():
    """Worked reports for the instructions, one per kind of moment, with the pilot's move."""
    E = [  # state, where, x, y, z, sugar, bitter, perch, fed, words[, wall ahead]
        ("seek", "on the floor", -8, 4, 0, (14, 70), (22, -110), (19, 150), 0, [" nothing", " folded", " free", " still"]),
        ("dusty", "on the floor", -6, -1, 0, (12, 40), (17, 100), (15, -120), 0, [" nothing", " folded", " touching", " still"], 24),
        ("seek", "on the floor", 3, -6, 0, (9, -45), (17, 120), (11, -160), 0, [" nothing", " folded", " free", " still"]),
        ("seek", "on the floor", -2, -3, 0, (8, 6), (25, -140), (13, 95), 0, [" nothing", " folded", " free", " still"]),
        ("seek", "on the floor", 12, 9, 0, (16, -150), (9, 30), (24, 60), 0, [" nothing", " folded", " free", " still"]),
        ("wall", "on the floor", 27, 4, 0, (24, 150), (40, 120), (32, -130), 0, [" nothing", " folded", " touching", " still"], 1),
        ("seek", "on the floor", 20, -24, 0, (21, 120), (30, 80), (21, -100), 0, [" nothing", " folded", " free", " still"]),
        ("seek", "on the floor", 1, 2, 0, (5, -12), (16, 170), (17, -60), 0, [" nothing", " folded", " free", " still"]),
        ("seek", "on the floor", -15, 6, 0, (21, 35), (8, -20), (25, 110), 0, [" nothing", " folded", " free", " still"]),
        ("seek", "on the floor", 9, -14, 0, (17, -100), (27, 60), (10, 15), 0, [" nothing", " folded", " free", " still"]),
        ("seek", "on the floor", -20, -18, 0, (30, 160), (12, 90), (20, -30), 0, [" nothing", " folded", " free", " still"]),
        ("dusty", "on the floor", 4, 7, 0, (5, -70), (20, 140), (22, -100), 0, [" nothing", " folded", " touching", " still"], 11),
        ("wall", "on the floor", -10, -28, 0, (31, 80), (37, 100), (11, 30), 0, [" nothing", " folded", " touching", " still"], 0),
        ("wall", "on the floor", 11, 26, 0, (24, -60), (30, 150), (41, 170), 0, [" nothing", " folded", " free", " still"], 2),
        ("seek", "on the floor", 6, 12, 0, (10, -28), (24, -170), (27, 40), 0, [" nothing", " folded", " free", " still"]),
        ("seek", "on the floor", -4, 8, 0, (11, 18), (20, 45), (23, 175), 0, [" nothing", " folded", " free", " still"]),
        ("seek", "on the floor", 14, 2, 0, (10, 95), (30, -15), (20, -135), 0, [" nothing", " folded", " free", " still"]),
        ("on_sugar", "on the floor", 5, 3, 0, (1, 4), (20, -95), (18, 130), 180, [" sweet", " extended", " free", " still"], 26),
        ("dusty", "on the floor", 15, -9, 0, (14, 160), (31, 30), (15, -90), 0, [" nothing", " folded", " touching", " still"], 7),
        ("fed_floor", "on the floor", 5, 2, 0, (1, -3), (21, 100), (17, -120), 640, [" sweet", " folded", " free", " still"], 25),
        ("over_perch", "in flight", -2, -15, 16, (19, 60), (26, 110), (2, -90), 820, [" nothing", " folded", " free", " beating"]),
        ("flight", "in flight", 4, -2, 16, (5, 160), (19, 40), (13, 75), 640, [" nothing", " folded", " free", " beating"]),
        ("flight", "in flight", 2, -9, 18, (12, -170), (24, 10), (6, -40), 700, [" nothing", " folded", " free", " beating"]),
        ("flight", "in flight", 1, -12, 17, (15, 120), (26, -60), (7, 8), 720, [" nothing", " folded", " free", " beating"]),
        ("over_perch", "in flight", 0, -14, 14, (17, -30), (25, 70), (1, 2), 760, [" nothing", " folded", " free", " beating"]),
        ("flight", "in flight", -6, -4, 19, (13, 20), (14, -150), (14, -70), 680, [" nothing", " folded", " free", " beating"]),
        ("over_perch", "in flight", 2, -16, 15, (19, 110), (27, -20), (2, 140), 790, [" nothing", " folded", " free", " beating"]),
        ("over_perch", "in flight", 1, -13, 18, (16, -140), (24, 50), (3, 35), 750, [" nothing", " folded", " free", " beating"]),
    ]
    out, t = [], 40
    for k, (st, where, x, y, z, sg, bt, pr, fed, words, *wall) in enumerate(E):
        tgt = pr[1] if st in ("flight", "over_perch") else sg[1]
        ahead = wall[0] if wall else 9 + (k * 7) % 30  # on the floor every report states the wall ahead
        out.append(text(t, head(where, x, y, z, sg, bt, pr, fed, ahead), words) + " " + pilot_word(st, tgt) + "\n")
        t += 20
    return "".join(out)


PROMPT = (f"A fruit fly's entire central nervous system (Janelia's male CNS, 166,700 spiking neurons) runs inside you, "
          f"as layers of your own network: its spikes are tokens in your own token stream. Every {STEP_MS} ms of fly time you "
          f"read a report of the fly and answer with one move. Your answer drives the fly's descending neurons: how much "
          f"more you favour forward, left, right, back, jump, fly or groom than stop sets how hard that command is driven "
          f"until the next report (favouring stop drives nothing; left and right hold each other back, so only the one you "
          f"favour more turns). At the end of each report the fly's own neurons print what it senses. Moves: forward "
          f"(walk), left and right (turn on the spot), back (walk backward), jump (take off), fly (wing power, which keeps "
          f"it in the air; fly while turning left or right steers), groom (the front legs clean the antennae while the fly "
          f"stands), stop. Walking reaches about 14 mm/s, backward about 8 mm/s and turning about 160 degrees per "
          f"second; flight up to 120 mm/s. The arena is "
          f"{2 * ARENA:.0f} x {2 * ARENA:.0f} mm. The perch is a {PERCH_TOP:.0f} mm high post, reached only from the air by "
          f"sinking onto its top. On the sugar drop the fly feeds by itself if it stops there. Goals, in order: walk to the "
          f"sugar drop and feed for {FEED_MS} ms; then jump, fly to the perch and land on it. How to choose: on the floor, "
          f"if the arena wall is within {WALL_BACK:.0f} mm ahead, answer back; if the antennae are touching with the wall "
          f"farther away, they carry dust: answer groom until they are free. Otherwise, on the floor, if the sugar drop is "
          f"more than about 20 degrees to its left, answer "
          f"left; more than about 20 degrees to its "
          f"right, answer right; within about 20 degrees (or ahead), answer forward; on the drop, "
          f"stop until fed for {FEED_MS} ms; once fed, jump; in the air, fly, or turn left or right toward the perch while "
          f"it is more than about 25 degrees off; over the perch (within about 3 mm of it), stop, so the fly sinks onto "
          f"it.\n\nExamples:\n{examples()}\nLog:\n")
