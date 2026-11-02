"""The body and the world: forward drive alone never turns the body, back walks backward, grooming
stands still and cleans the antennae, and the antennae sense dust or a wall ahead."""

import math

import numpy as np

from flyweight.flyworld import ANTENNA, ARENA, BACK_V, GROOM_S, SENSE_LEVELS, TAKEOFF_HZ, WALK_V, Body, World
from flyweight.sets import ROLES


def world(x=0.0, y=0.0, yaw=0.0, dust=0.0):
    w = World.__new__(World)
    w.A, w.sets, w.t = None, {}, 0
    w.readout = {"fwd_hz": 20.0, "turn_L_hz": 30.0, "turn_R_hz": 30.0, "bwd_hz": 20.0, "groom_hz": 20.0,
                 "fwd_curve": np.array([0.0, 10.0, 20.0]), "turn_L_curve": np.array([0.0, 12.0, 18.0]), "turn_R_curve": np.array([0.0, 1.0, 2.0])}
    w._groups = {nm: np.arange(4) for nm in ROLES}
    w.bodies = [Body("fly", x, y, yaw=yaw, dust=dust)]
    return w


def rates(**hz):
    return {nm: hz.get(nm, 0.0) for nm in ROLES}


def test_forward_drive_alone_walks_straight():
    w = world()
    for _ in range(100):  # 100 ms with the turn pools at what forward alone recruits
        w.update_body(0, "", rates(fwd=10.0, turn_L=12.0, turn_R=1.0))
    B = w.bodies[0]
    assert B.yaw == 0.0 and abs(B.x - 0.1 * WALK_V * 0.5) < 1e-9


def test_back_walks_backward():
    w = world()
    for _ in range(100):
        w.update_body(0, "", rates(bwd=20.0))
    assert abs(w.bodies[0].x + 0.1 * BACK_V) < 1e-9


def test_grooming_stands_and_cleans_the_antennae():
    w = world(dust=1.0)
    for _ in range(int(GROOM_S * 1000) + 1):
        w.update_body(0, "", rates(groom=20.0, fwd=20.0))
    B = w.bodies[0]
    assert B.dust == 0.0 and B.x == 0.0


def test_a_burst_takes_off_and_a_stray_spike_does_not():
    w = world()
    w.update_body(0, "", rates(TTMn_L=35.0, TTMn_R=35.0))  # one jump motor spike each, from a stray giant fiber spike
    assert w.bodies[0].mode == "walk"
    w.update_body(0, "", rates(TTMn_L=TAKEOFF_HZ + 20, TTMn_R=TAKEOFF_HZ + 20))
    assert w.bodies[0].mode == "air"


def test_antennae_sense_dust_and_a_wall_ahead():
    assert world(dust=0.5).sense_levels()["JO-CE"] == SENSE_LEVELS["JO-CE"]
    w = world(x=ARENA - 1.2 - 0.5 * ANTENNA)
    assert abs(w.wall_ahead() - 0.5 * ANTENNA) < 1e-9 and w.sense_levels()["JO-CE"] == SENSE_LEVELS["JO-CE"]
    w = world(x=ARENA - 1.2 - 0.5 * ANTENNA, yaw=math.pi)  # the wall behind: nothing touches
    assert w.sense_levels()["JO-CE"] == 0
