"""The body and the world around the fly inside Qwen, as NeuroMechFly or flybody would be.

Every neuron is in the model (flyweight/merged.py). This module turns the model's motor readout into
movement and the fly's surroundings into its sense registers.

Motor side: the MOVE rows of the model's LM head (each role's rate, Hz per member, integrated in the
fly's layers over 30 ms). No readout neuron is one that a register drives directly: every movement
comes through the connectome.
  walking        forward, left-turn, right-turn and backward populations in the nerve cord, the
                 neurons each command recruits selectively (scripts/calibrate/calib_motor.py, pick_readout.py);
                 each turn pool minus what forward drive alone recruits in it at the forward pool's
                 present rate (scripts/calibrate/calib_forward.py, extended to the top levels by calib_body.py),
                 the turn the difference of the two signed excesses,
                 and the forward pool minus what backward drive alone recruits in it (calib_body.py)
  grooming       the grooming pool (what the antennal grooming descending neuron recruits): the fly
                 stands and its front legs clean the antennae
  takeoff        both jump motor neurons (TTMn left and right) above TAKEOFF_HZ: under the giant fiber
                 (DNp01) the right one reaches ~120 Hz and the left one 70-83 Hz (calib_body.py), with or
                 without DNg02; a stray giant fiber spike at a walking command's onset fires them once
                 (~35 Hz)
  flight power   the indirect flight muscle motor neurons DLMn and DVMn, full at POWER_HZ (what
                 DNg02 at level 12 gives): the ground speed and the height held (CRUISE_H at full power);
                 below LIFT the wings stop and the fly falls (walking commands, and the flight circuit
                 after DNg02 lets go, leave them at 35-55 Hz, below lift). Near a wall a flying fly turns
                 away in a saccade (SACCADE, from WALL_SEE): the model has no vision to steer by
  flight turns   wing steering motor neurons b1/b2 (more on the right yaws left), plus the turn
                 populations read against what a full turn command recruits in them with the wings
                 powered, each minus what steady flight leaves in it (calib_body.py): under DNg02 the circuit silences b1/b2 and recruits the turn
                 pools about a tenth as strongly as on the floor. Yaw follows with the body's rotational
                 inertia (AIR_TAU), which smooths the pools' sparse spikes
  feeding        MN9 (proboscis), while the labellum is on the sugar drop
The stepping and wingbeat rhythms are drawn by the viewer: the LIF connectome has no rhythm
generator of its own.

Sensory side: drive levels for the sense registers (labellum on the sugar drop -> sugar GRNs, on the
bitter drop -> bitter GRNs, antennae against a wall ahead or carrying dust -> Johnston's organ C/E
neurons).
"""

import math
from dataclasses import dataclass, field

import numpy as np

from .sets import LEGS

ARENA = 30.0  # half-width, mm
WALL_H = 30.0
SUGAR = np.array([5.0, 3.0]); BITTER = np.array([-14.0, 10.0]); PERCH = np.array([0.0, -15.0])
DROP_R, PERCH_R, PERCH_TOP = 3.5, 7.0, 12.0
START = (-2.0, -3.0, -0.2)  # x, y, yaw
SENSE_LEVELS = {"sugar": 13, "bitter": 4, "JO-CE": 3}  # drive levels: ~167, ~54 and ~34 Hz on an unconnected neuron
ANTENNA = 1.5  # mm ahead of the head that the antennae reach
WALK_V, BACK_V = 14.0, 8.0  # mm/s at full drive of the forward and backward pools
TAKEOFF_HZ = 55.0  # Hz per jump motor neuron, both sides: between a stray spike (~35) and the giant fiber's weakest burst (70)
POWER_HZ = 140.0  # Hz per flight power motor neuron for full power
LIFT = 0.42  # the power that holds the fly up; below it the wings stop
AIR_TAU = 0.1  # s: the flying body's yaw rate and speed follow the steering command and the wing power with this time constant
CRUISE_H = 20.0  # mm: the height a flying fly holds at full wing power (lower power holds lower; at LIFT it sinks to the floor)
SACCADE = 1500.0  # deg/s: the fastest turn away from a wall, a flying fly's collision-avoidance saccade
WALL_SEE, WALL_NEAR = 12.0, 7.0  # mm from a wall at which a flying fly starts to turn away from it, and turns at the full rate
GROOM_S = 0.3  # s of full grooming that cleans dust off the antennae


@dataclass
class Body:
    name: str
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0  # radians, 0 = +x
    mode: str = "walk"  # walk | air
    vz: float = 0.0
    wz: float = 0.0  # yaw rate in flight, rad/s
    v: float = 0.0  # ground speed in flight, mm/s
    fed_ms: float = 0.0
    dust: float = 0.0  # dust on the antennae, 0..1
    log: list = field(default_factory=list)  # per ms: (t, x, y, z, yaw, mode, fwd, back, turn, power, MN9 Hz, groom, label, leg Hz x 6)


class World:
    def __init__(self, A, sets, readout=None):
        """readout: {"fwd", "turn_L", "turn_R": neuron indices, "<name>_hz": the rate that
        counts as full drive} from scripts/calibrate/pick_readout.py, and the turn pools' rates under forward
        drive alone against the forward pool's rate ("fwd_curve", "turn_L_curve", "turn_R_curve")
        from scripts/calibrate/calib_forward.py."""
        self.A, self.sets = A, sets
        self.readout = readout or {}
        self._groups = {}
        self.bodies = [Body("fly", START[0], START[1], yaw=START[2])]
        self.t = 0  # 0.1 ms steps

    def group(self, nm):
        if nm not in self._groups:
            self._groups[nm] = np.asarray(self._group(nm), np.int64)
        return self._groups[nm]

    def _group(self, nm):
        A = self.A
        if nm in self.sets:
            return self.sets[nm]
        if nm in ("fwd", "turn_L", "turn_R", "bwd", "groom"):
            return list(self.readout[nm])
        if nm in ("TTMn_L", "TTMn_R"):
            return [i for i in range(len(A.type)) if A.type[i] == "TTMn" and A.side[i] == nm[-1]]
        if nm in ("DNa02_L", "DNa02_R"):
            return [i for i in range(len(A.type)) if A.type[i] == "DNa02" and A.side[i] == nm[-1]]
        if nm == "power":
            return [i for i in range(len(A.type)) if A.type[i].startswith(("DLMn", "DVMn"))]
        if nm in ("b_L", "b_R"):
            return [i for i in range(len(A.type)) if A.type[i] in ("b1 MN", "b2 MN") and A.side[i] == nm[-1]]
        if nm == "DNg02":
            return [i for i in range(len(A.type)) if A.type[i].startswith("DNg02")]
        if nm.startswith("leg_"):  # every leg motor neuron of one leg: leg neuromere (front, middle, hind) and side
            seg = {"f": "fl", "m": "ml", "h": "hl"}[nm[4]]
            return [i for i in range(len(A.type)) if A.superclass[i] == "vnc_motor" and A.subclass[i] == seg and A.side[i] == nm[5]]
        return [i for i in range(len(A.type)) if A.type[i] == nm]

    def wall_ahead(self, b=0):
        """Distance (mm) from the head to the arena wall along the heading."""
        B = self.bodies[b]
        c, s = math.cos(B.yaw), math.sin(B.yaw)
        hx, hy = B.x + 1.2 * c, B.y + 1.2 * s
        d = [(ARENA * math.copysign(1.0, u) - p) / u for u, p in ((c, hx), (s, hy)) if abs(u) > 1e-9]
        return max(0.0, min(d))

    def sense_levels(self, b=0):
        """The senses as drive levels for the fly's sense registers."""
        B = self.bodies[b]
        head = np.array([B.x + 1.2 * math.cos(B.yaw), B.y + 1.2 * math.sin(B.yaw)])
        grounded = B.z < 0.5
        on = {"sugar": grounded and np.linalg.norm(head - SUGAR) < DROP_R,
              "bitter": grounded and np.linalg.norm(head - BITTER) < DROP_R,
              "JO-CE": B.dust > 0 or (grounded and self.wall_ahead(b) < ANTENNA)}
        return {nm: (lv if on[nm] else 0) for nm, lv in SENSE_LEVELS.items()}

    @staticmethod
    def saccade(B):
        """deg/s of turn away from the arena walls for a flying fly: from WALL_SEE mm of a wall, harder the nearer it is (the
        full rate from WALL_NEAR) and the more the fly heads into it, until it heads away at 30 degrees or more; to the side
        its heading already leans. Near a corner the two walls count as one, along their summed normals."""
        c, s, nx, ny = math.cos(B.yaw), math.sin(B.yaw), 0.0, 0.0
        for ux, uy, d in ((1, 0, ARENA - B.x), (-1, 0, ARENA + B.x), (0, 1, ARENA - B.y), (0, -1, ARENA + B.y)):
            near = max(0.0, min(1.0, (WALL_SEE - d) / (WALL_SEE - WALL_NEAR)))
            nx += near * ux; ny += near * uy
        n = math.hypot(nx, ny)
        if n == 0:
            return 0.0
        into = max(0.0, min(1.0, ((c * nx + s * ny) / n + 0.5) / 0.5))
        return SACCADE * min(1.0, n) * into * (1.0 if nx * s - ny * c >= 0 else -1.0)

    def update_body(self, b, label, move):
        """One ms of body movement from the model's motor readout (move: Hz per member, by role)."""
        B = self.bodies[b]
        hz = dict(move)
        tot = {nm: hz[nm] * len(self.group(nm)) for nm in hz}
        dt = 1e-3
        clip = lambda v: max(0.0, min(1.0, v))
        R = self.readout
        fwd_own = hz["fwd"] - (float(np.interp(hz["bwd"], R["bwd_curve"], R["fwd_bwd_curve"])) if "bwd_curve" in R else 0.0)  # minus backward drive's share
        fwd, back, groom = clip(fwd_own / R["fwd_hz"]), clip(hz["bwd"] / R["bwd_hz"]), clip(hz["groom"] / R["groom_hz"])
        fc = R["fwd_curve"]  # minus what forward drive alone recruits in each turn pool at this forward rate
        tl = hz["turn_L"] - float(np.interp(hz["fwd"], fc, R["turn_L_curve"]))  # each pool's excess over forward's share, signed:
        tr = hz["turn_R"] - float(np.interp(hz["fwd"], fc, R["turn_R_curve"]))  # clipping it would turn the fluctuations into a veer
        turn = max(-1.0, min(1.0, tl / R["turn_L_hz"] - tr / R["turn_R_hz"]))
        power = clip(hz["power"] / POWER_HZ)
        if B.mode == "walk":
            stand = 1.0 - groom  # a grooming fly stands on its middle and hind legs
            v = (WALK_V * fwd - BACK_V * back) * stand
            B.yaw += math.radians(160) * turn * stand * dt
            B.dust = max(0.0, B.dust - groom * dt / GROOM_S)
            nx, ny = B.x + v * math.cos(B.yaw) * dt, B.y + v * math.sin(B.yaw) * dt
            if not (B.z < PERCH_TOP - 0.1 and np.linalg.norm([nx, ny] - PERCH) < PERCH_R):  # the perch is a wall from the floor
                B.x, B.y = nx, ny
            if hz["TTMn_L"] > TAKEOFF_HZ and hz["TTMn_R"] > TAKEOFF_HZ:  # the jump: both jump motor neurons in a burst
                B.mode, B.vz = "air", 100.0  # a few mm of jump; the wings must hold it up
            if B.z > 0 and np.linalg.norm([B.x, B.y] - PERCH) > PERCH_R:  # walked off the perch
                B.z = 0.0
        else:
            if "air_turn_L_hz" in R:  # each pool's excess over what steady flight leaves in it, against a full turn command's
                bl, br = (float(R["air_base_L_hz"]), float(R["air_base_R_hz"])) if "air_base_L_hz" in R else (0.0, 0.0)
                air_turn = (hz["turn_L"] - bl) / (float(R["air_turn_L_hz"]) - bl) - (hz["turn_R"] - br) / (float(R["air_turn_R_hz"]) - br)
            else:
                air_turn = turn
            steer = max(-1.0, min(1.0, (tot["b_R"] - tot["b_L"]) / 120 + air_turn))
            B.wz += (math.radians(220) * steer - B.wz) * dt / AIR_TAU
            B.yaw += (B.wz + math.radians(self.saccade(B))) * dt
            lift = power if power >= LIFT else 0.0  # the wings beat or they stop
            B.v += (120.0 * lift - B.v) * dt / AIR_TAU
            v = B.v
            B.x += v * math.cos(B.yaw) * dt; B.y += v * math.sin(B.yaw) * dt
            if lift > 0:  # the wings hold a height set by their power, climbing or sinking at up to 150 mm/s
                vt = max(-150.0, min(150.0, 10.0 * (CRUISE_H * (lift - LIFT) / (1 - LIFT) - B.z)))
                B.vz += (vt - B.vz) * dt / 0.05
            else:  # the wings stopped: it falls
                B.vz += (-2400.0 * LIFT - 3.0 * B.vz) * dt
            z0 = B.z
            B.z += B.vz * dt
            over = np.linalg.norm([B.x, B.y] - PERCH) < PERCH_R
            if over and z0 < PERCH_TOP and B.z < PERCH_TOP:  # flew into the perch's side: bounce off
                B.x -= v * math.cos(B.yaw) * dt; B.y -= v * math.sin(B.yaw) * dt
                over = False
            floor = PERCH_TOP if over and z0 >= PERCH_TOP - 1e-6 else 0.0
            if B.z <= floor and B.vz <= 0:
                B.z, B.vz, B.wz, B.v, B.mode = floor, 0.0, 0.0, 0.0, "walk"
            if B.z >= WALL_H - 2:  # the lid: no climbing through it, no speed stored against it
                B.z, B.vz = WALL_H - 2, min(B.vz, 0.0)
        B.x = max(-ARENA + 1, min(ARENA - 1, B.x)); B.y = max(-ARENA + 1, min(ARENA - 1, B.y))
        head = np.array([B.x + 1.2 * math.cos(B.yaw), B.y + 1.2 * math.sin(B.yaw)])
        if B.z < 0.5 and np.linalg.norm(head - SUGAR) < DROP_R and hz["MN9"] > 15:
            B.fed_ms += 1
        B.log.append((self.t / 10, round(B.x, 3), round(B.y, 3), round(B.z, 3), round(B.yaw, 4), B.mode,
                      round(fwd, 3), round(back, 3), round(turn, 3), round(power, 3), round(hz["MN9"], 1),
                      round(groom, 3), label, *(round(hz.get(nm, 0.0), 2) for nm in LEGS)))

    def observe(self, b=0):
        """The world's part of the report Qwen reads, as report.head's arguments: where, x, y, z, and
        (distance mm, bearing deg, + = left) of the sugar drop, the bitter drop and the perch, fed ms, and
        the distance to the arena wall ahead."""
        B = self.bodies[b]

        def rel(p):
            d = np.array(p) - [B.x, B.y]
            ang = (math.degrees(math.atan2(d[1], d[0]) - B.yaw) + 180) % 360 - 180
            return float(np.linalg.norm(d)), float(ang)
        where = "in flight" if B.mode == "air" else ("on the perch" if B.z >= PERCH_TOP - 0.1 else "on the floor")
        return where, B.x, B.y, B.z, rel(SUGAR), rel(BITTER), rel(PERCH), B.fed_ms, self.wall_ahead(b)
