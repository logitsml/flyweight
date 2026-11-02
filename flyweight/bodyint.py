"""The body and the senses in integer fixed point: flyweight/flyworld.py's update_body and sense_levels as
integer arithmetic (adds, multiplies, shifts, compares and one table), the form a circuit compiled into
stock layers computes exactly (the first stage of moving the body inside the model; docs/design.md).

Units: positions in 2^-20 mm, heading in 2^-24 turns, fractions (walk, back, groom, power, turn) in
Q16 (65,536 = full), rates in 1/1024 Hz, vertical speed in 2^-20 mm/s, dust in Q32. Sine and cosine come
from a 65,536-entry table (Q14). Divisions round to nearest, so no bias builds up over a flight. Every constant is the float body's, rounded once. scripts/checks/check_bodyint.py
runs both on the same motor streams.
"""

import math

import numpy as np

from .flyworld import (AIR_TAU, ANTENNA, ARENA, BACK_V, CRUISE_H, DROP_R, GROOM_S, LIFT, PERCH, PERCH_R, PERCH_TOP, POWER_HZ,
                       SACCADE, SENSE_LEVELS, SUGAR, BITTER, TAKEOFF_HZ, WALK_V, WALL_H, WALL_NEAR, WALL_SEE)

P20 = 1 << 20  # position units per mm
TURN24 = 1 << 24  # heading units per turn
Q12, Q14, Q16 = 1 << 12, 1 << 14, 1 << 16
QF = Q16  # fractions (walk, back, groom, power, turn, steer)
TAB = 1 << 16
COS = [int(round(math.cos(2 * math.pi * i / TAB) * Q14)) for i in range(TAB)]
SIN = [int(round(math.sin(2 * math.pi * i / TAB) * Q14)) for i in range(TAB)]
WV = int(round(WALK_V * 1e-3 * P20 * QF))  # mm per ms at full walk, in 2^-36 mm
BV = int(round(BACK_V * 1e-3 * P20 * QF))
FV = int(round(120.0 * 1e-3 * P20 * QF))  # flight speed at full power
YT_WALK = int(round(160.0 / 360.0 * 1e-3 * TURN24))  # heading per ms at full turn
YT_AIR = int(round(220.0 / 360.0 * 1e-3 * TURN24))
YT_SAC = int(round(SACCADE / 360.0 * 1e-3 * TURN24))  # heading per ms of a full saccade
DUST_K = int(round(2 ** 32 / (GROOM_S * 1e3 * QF)))  # dust (Q32) cleaned per ms per unit of groom (Q16)
LIFT0 = int(round(LIFT * (1 << 24)))  # the power that holds the fly up, Q24
LIFT16 = int(round(LIFT * QF))
TAU_MS = int(round(AIR_TAU * 1000))
rdiv = lambda a, b: (a + b // 2) // b  # integer division, rounded to nearest (b > 0)
HEAD = int(round(1.2 * P20))
ARENA_IN = int(round((ARENA - 1) * P20))
ARENA_P = int(round(ARENA * P20))
DROP2 = int(round(DROP_R * P20)) ** 2
PERCH2 = int(round(PERCH_R * P20)) ** 2
TOP = int(round(PERCH_TOP * P20))
ZFLOOR_WALK = int(round((PERCH_TOP - 0.1) * P20))
ZMAX = int(round((WALL_H - 2) * P20))
CRUISE = int(round(CRUISE_H * P20))
SEE, NEAR = int(round(WALL_SEE * P20)), int(round(WALL_NEAR * P20))
VZMAX = 150 * P20
HZQ = 1024  # rate units per Hz
HZ16 = lambda hz: int(round(hz * HZQ))


def frac(hz16, full16):
    """A rate as a fraction of full drive, clipped to 0..1, in Q16 (rounded)."""
    return max(0, min(QF, rdiv(hz16 * QF, full16)))


class IntBody:
    def __init__(self, x, y, yaw, readout, dust=0.0, z=0.0):
        self.x, self.y, self.z = int(round(x * P20)), int(round(y * P20)), int(round(z * P20))
        self.yaw = int(round(yaw / (2 * math.pi) * TURN24)) % TURN24
        self.air, self.vz, self.wz, self.fed_ms = False, 0, 0, 0  # wz: yaw rate in flight, heading units per ms x QF
        self.v = 0  # ground speed in flight, as FV
        self.dust = int(round(dust * 2 ** 32))
        R = readout
        self.full = {k: HZ16(R[f"{k}_hz"]) for k in ("fwd", "bwd", "groom", "turn_L", "turn_R")}
        self.fc = [HZ16(v) for v in R["fwd_curve"]]
        self.tlc = [HZ16(v) for v in R["turn_L_curve"]]
        self.trc = [HZ16(v) for v in R["turn_R_curve"]]
        self.bc = [HZ16(v) for v in R["bwd_curve"]] if "bwd_curve" in R else None
        self.fbc = [HZ16(v) for v in R["fwd_bwd_curve"]] if "bwd_curve" in R else None
        self.air_full = (HZ16(R["air_turn_L_hz"]), HZ16(R["air_turn_R_hz"])) if "air_turn_L_hz" in R else None
        self.air_base = (HZ16(R["air_base_L_hz"]), HZ16(R["air_base_R_hz"])) if "air_base_L_hz" in R else (0, 0)
        self.pw = HZ16(POWER_HZ)
        self.tk = HZ16(TAKEOFF_HZ)

    def _interp(self, x, xs, ys):
        """np.interp on integers: clamped at both ends, linear between, rounded to nearest."""
        if x <= xs[0]:
            return ys[0]
        for k in range(1, len(xs)):
            if x <= xs[k]:
                dx = xs[k] - xs[k - 1]
                return ys[k - 1] + rdiv((ys[k] - ys[k - 1]) * (x - xs[k - 1]), dx) if dx else ys[k]
        return ys[-1]

    def cs(self):
        i = (self.yaw >> 8) % TAB
        return COS[i], SIN[i]

    def head(self):
        c, s = self.cs()
        return self.x + rdiv(HEAD * c, Q14), self.y + rdiv(HEAD * s, Q14)

    def saccade(self):
        """Heading units this ms of turn away from the walls (flyworld.World.saccade, in integers)."""
        c, s = self.cs()
        nx = ny = 0
        for ux, uy, d in ((1, 0, ARENA_P - self.x), (-1, 0, ARENA_P + self.x), (0, 1, ARENA_P - self.y), (0, -1, ARENA_P + self.y)):
            near = max(0, min(QF, rdiv((SEE - d) * QF, SEE - NEAR)))
            nx += near * ux; ny += near * uy
        n = math.isqrt(nx * nx + ny * ny)
        if n == 0:
            return 0
        into = max(0, min(QF, rdiv((rdiv(c * nx + s * ny, n) + Q14 // 2) * QF, Q14 // 2)))
        w = rdiv(YT_SAC * min(QF, n) * into, QF * QF)
        return w if nx * s - ny * c >= 0 else -w

    def update(self, hz, n_b):
        """One ms. hz: role -> Hz per member (the MOVE rows); n_b: (members of b_L, of b_R)."""
        h = {k: HZ16(v) for k, v in hz.items()}
        fwd_own = h["fwd"] - (self._interp(h["bwd"], self.bc, self.fbc) if self.bc else 0)  # minus backward drive's share
        fwd, back, groom = frac(fwd_own, self.full["fwd"]), frac(h["bwd"], self.full["bwd"]), frac(h["groom"], self.full["groom"])
        tl = h["turn_L"] - self._interp(h["fwd"], self.fc, self.tlc)  # signed excesses, as the float body
        tr = h["turn_R"] - self._interp(h["fwd"], self.fc, self.trc)
        turn = max(-QF, min(QF, rdiv(tl * QF, self.full["turn_L"]) - rdiv(tr * QF, self.full["turn_R"])))
        power = frac(h["power"], self.pw)
        if not self.air:
            stand = QF - groom
            v = rdiv(rdiv(WV * fwd - BV * back, QF) * stand, QF)  # 2^-36 mm per ms
            self.yaw = (self.yaw + rdiv(YT_WALK * turn * stand, QF * QF)) % TURN24
            self.dust = max(0, self.dust - groom * DUST_K)
            c, s = self.cs()
            nx, ny = self.x + rdiv(v * c, QF * Q14), self.y + rdiv(v * s, QF * Q14)
            dx, dy = nx - int(round(PERCH[0] * P20)), ny - int(round(PERCH[1] * P20))
            if not (self.z < ZFLOOR_WALK and dx * dx + dy * dy < PERCH2):  # the perch is a wall from the floor
                self.x, self.y = nx, ny
            if h["TTMn_L"] > self.tk and h["TTMn_R"] > self.tk:
                self.air, self.vz = True, 100 * P20
            px, py = self.x - int(round(PERCH[0] * P20)), self.y - int(round(PERCH[1] * P20))
            if self.z > 0 and px * px + py * py > PERCH2:
                self.z = 0
        else:
            if self.air_full:  # excess over flight's own share, against a full turn command's
                (bl, br), (fl, fr) = self.air_base, self.air_full
                air_turn = rdiv((h["turn_L"] - bl) * QF, fl - bl) - rdiv((h["turn_R"] - br) * QF, fr - br)
            else:
                air_turn = turn
            steer = max(-QF, min(QF, rdiv((n_b[1] * h["b_R"] - n_b[0] * h["b_L"]) * QF, 120 * HZQ) + air_turn))
            self.wz += rdiv(YT_AIR * steer - self.wz, TAU_MS)
            self.yaw = (self.yaw + rdiv(self.wz, QF) + self.saccade()) % TURN24
            lift = power if power >= LIFT16 else 0  # the wings beat or they stop
            self.v += rdiv(rdiv(FV * lift, QF) - self.v, TAU_MS)
            c, s = self.cs()
            ox, oy = self.x, self.y
            self.x += rdiv(self.v * c, QF * Q14)
            self.y += rdiv(self.v * s, QF * Q14)
            if lift > 0:  # the height the wings' power holds, at up to 150 mm/s (2^-20 mm/s)
                vt = max(-VZMAX, min(VZMAX, 10 * (rdiv(CRUISE * (lift - LIFT16), QF - LIFT16) - self.z)))
                self.vz += rdiv(vt - self.vz, 50)
            else:  # falling: gravity against drag
                self.vz += rdiv(rdiv(-2400 * LIFT0 * P20, 1 << 24) - 3 * self.vz, 1000)
            z0 = self.z
            self.z += rdiv(self.vz, 1000)
            px, py = self.x - int(round(PERCH[0] * P20)), self.y - int(round(PERCH[1] * P20))
            over = px * px + py * py < PERCH2
            if over and z0 < TOP and self.z < TOP:  # into the perch's side: bounce off
                self.x, self.y = ox, oy
                over = False
            floor = TOP if over and z0 >= TOP - 1 else 0
            if self.z <= floor and self.vz <= 0:
                self.z, self.vz, self.wz, self.v, self.air = floor, 0, 0, 0, False
            if self.z >= ZMAX:  # the lid
                self.z, self.vz = ZMAX, min(self.vz, 0)
        self.x = max(-ARENA_IN, min(ARENA_IN, self.x))
        self.y = max(-ARENA_IN, min(ARENA_IN, self.y))
        hx, hy = self.head()
        sx, sy = hx - int(round(SUGAR[0] * P20)), hy - int(round(SUGAR[1] * P20))
        if self.z < P20 // 2 and sx * sx + sy * sy < DROP2 and h["MN9"] > 15 * HZQ:
            self.fed_ms += 1
        return dict(fwd=fwd / QF, back=back / QF, turn=turn / QF, power=power / QF, groom=groom / QF)

    def senses(self):
        """The sense registers' levels, as flyworld.World.sense_levels gives them."""
        hx, hy = self.head()
        grounded = self.z < P20 // 2
        near = lambda p: (hx - int(round(p[0] * P20))) ** 2 + (hy - int(round(p[1] * P20))) ** 2 < DROP2
        c, s = self.cs()
        wall = False  # a wall within ANTENNA ahead, without dividing: along each axis, distance x |direction| < ANTENNA x |direction|
        for u, p in ((c, hx), (s, hy)):
            if u != 0:
                gap = ARENA_P - p if u > 0 else ARENA_P + p
                wall |= gap * Q14 < int(round(ANTENNA * P20)) * abs(u)
        on = {"sugar": grounded and near(SUGAR), "bitter": grounded and near(BITTER), "JO-CE": self.dust > 0 or (grounded and wall)}
        return {nm: (lv if on[nm] else 0) for nm, lv in SENSE_LEVELS.items()}

    def state(self):
        return (self.x / P20, self.y / P20, self.z / P20, self.yaw / TURN24 * 2 * math.pi, "air" if self.air else "walk", self.dust / 2 ** 32, self.fed_ms)
