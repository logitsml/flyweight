"""The report and its pilot: each situation's printed words, the wall in the world's text, and the
move the worked examples teach."""

import numpy as np
import pytest

from flyweight import report


@pytest.mark.parametrize("state", ["seek", "on_sugar", "fed_floor", "flight", "over_perch", "wall", "dusty"])
def test_situations_are_consistent(state):
    rng = np.random.default_rng(3)
    for _ in range(20):
        h, words, st, tgt = report.situation(rng, state)
        assert st == state
        if state != "wall":  # facing a wall the antennae touch only within reach
            assert (words[2] == " touching") == (state == "dusty")
        assert ("arena wall" in h) == ("on the floor" in h)  # the wall ahead is stated on the floor
        if state == "wall":
            assert any(f"arena wall is {d} mm ahead" in h for d in (0, 1, 2, 3))


def test_pilot_words():
    assert report.pilot_word("wall", 50) == "back" and report.pilot_word("dusty", -50) == "groom"
    assert report.pilot_word("seek", 40) == "left" and report.pilot_word("seek", -40) == "right" and report.pilot_word("seek", 5) == "forward"
    t = report.pilot_targets("dusty", 0.0)
    assert t["groom"] == 1.0 and sum(t.values()) == 1.0


def test_examples_teach_every_move():
    ex = report.examples()
    for w in ("forward", "left", "right", "back", "jump", "fly", "groom", "stop"):
        assert f" {w}\n" in ex, w
