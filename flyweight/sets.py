"""The operating point and the named neuron sets of the male CNS.

Shiu et al.'s LIF model on Janelia's male CNS v1.0: neurons are bodies with a superclass, Kenyon cell
-> Kenyon cell synapses removed, synapse counts scaled by 0.73 (without both, the network ignites).
"""

SCALE = 0.73


def cns_sets(con, A):
    """Taste neurons (types matched through FlyWire's annotations), antennal JO-C/E neurons, and the
    descending and motor neurons the fly's registers and readouts use. The senses are bilateral: the
    labellum meets a drop with both halves, and a wall meets both antennae."""
    n = con.n
    pick = lambda t: [i for i in range(n) if A.type[i] == t]
    from . import annot as fw
    f = fw.load("783")
    tastes = {s: {t for t, c in zip(f["name"], f["sub_class"]) if c == s and t} for s in ("sugar", "bitter", "water")}
    sets = {s: [i for i in range(n) if A.cls[i] == "gustatory" and (A.type[i] in ts or A.flywire[i] in ts)] for s, ts in tastes.items()}
    sets["JO-CE"] = [i for i in range(n) if A.type[i].startswith(("JO-C", "JO-E"))]
    for t in ("MDN", "DNp09", "DNg100", "DNge031", "DNg12_e", "MN9", "DNp01"):
        sets[t] = pick(t)
    sets["DNa02_L"] = [i for i in pick("DNa02") if A.side[i] == "L"]
    sets["DNa02_R"] = [i for i in pick("DNa02") if A.side[i] == "R"]
    sets["DNg02"] = [i for i in range(n) if A.type[i].startswith("DNg02")]
    return sets


# the fly's registers: commands (written by Qwen's answer), then senses (written by the world). Each command is
# the descending neuron type that drives one behaviour: P9 forward walking (Bidaye et al. 2020), DNa02 steering, the
# giant fiber takeoff, DNg02 wing power (Namiki et al. 2022), MDN backward walking (Bidaye et al. 2014), the antennal
# grooming descending neuron (aDN; Hampel et al. 2015). WORDS: the move word of Qwen's answer that drives each.
DRIVEN = [("forward", "DNp09"), ("turn_left", "DNa02_L"), ("turn_right", "DNa02_R"), ("takeoff", "DNp01"), ("fly", "DNg02"),
          ("back", "MDN"), ("groom", "DNg12_e")]
WORDS = ["forward", "left", "right", "jump", "fly", "back", "groom"]
SENSES = ["sugar", "bitter", "JO-CE"]
REG = {nm: i for i, nm in enumerate([c for c, _ in DRIVEN] + SENSES)}  # register index by command or sense
# the motor readout (MOVE rows): walking pools (forward, turns, backward), the grooming pool, jump and flight motor
# neurons, proboscis; the sense neurons whose rates the fly prints (flyweight/report.py SAY); and every leg motor neuron,
# by leg (front, middle, hind; left, right: the annotations' leg neuromere and side)
LEGS = ["leg_fL", "leg_fR", "leg_mL", "leg_mR", "leg_hL", "leg_hR"]
ROLES = ["fwd", "turn_L", "turn_R", "bwd", "groom", "TTMn_L", "TTMn_R", "power", "b_L", "b_R", "MN9", "sugar", "bitter", "JO-CE"] + LEGS
assert len(WORDS) == len(DRIVEN)
