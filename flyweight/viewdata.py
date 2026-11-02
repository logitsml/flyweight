"""The viewer's data (viz/index.html), encoded the same way for a recorded session (scripts/run/export_viz.py)
and a live one (scripts/run/serve.py).

window: per fly, spikes (step, neuron), input tokens (step, neuron), membrane frames (step, neuron,
level). bodies: per fly, per ms: t, x, y, z, yaw, air, fwd, back, turn, power, MN9 Hz, groom, then each leg's motor neurons in Hz
(front, middle, hind; left, right). qwen:
per token, per layer, the top-k MLP activations (index uint16, value float16). neurons: every
neuron's position, class, leg and role. Groups: the neuron sets the bridge panel and the CNS halos
follow.
"""

import struct
from pathlib import Path

import numpy as np

from .sets import cns_sets

ROOT = Path(__file__).resolve().parents[1]
KEEP = 48  # Qwen MLP neurons kept per layer per token
BODY_COLS = 18  # floats per body row
LEG_HZ = 12.0  # a leg bar in the viewer is full at this rate (Hz per member): the front legs' motor neurons while grooming (scripts/experiments/check_legs.py)
CLASSES = ["optic", "central", "sensory", "visual_projection", "ascending", "descending",
           "sensory_ascending", "visual_centrifugal", "motor", "vnc", "unknown"]
LEGS = ["", "flL", "flR", "mlL", "mlR", "hlL", "hlR"]
NEURON_ROLES = ["", "MN9", "DNge031", "DNg12_e", "MDN", "DNa02_L", "DNa02_R", "DNp09", "DNg100"]
VOTE_WORDS = ["forward", "left", "right", "back", "jump", "fly", "groom", "stop"]


def u32(a):
    return np.asarray(a, np.uint32).tobytes()


def klass(sc):
    if sc.startswith("ol_"): return 0
    if sc == "cb_intrinsic": return 1
    if sc.endswith("_sensory"): return 2
    if sc == "vnc_intrinsic" or sc == "vnc_efferent": return 9
    if sc.endswith("_motor"): return 8
    return CLASSES.index(sc.replace("_neuron", "")) if sc.replace("_neuron", "") in CLASSES else 10


def neurons_payload(con, A):
    """Every neuron's position (int16 xyz in a normalised CNS frame), class, leg (motor neurons) and role."""
    n = con.n
    xyz = A.xyz.astype(np.float64) / 1000.0  # um
    c = np.median(xyz, 0)
    s = np.abs(xyz - c).max()
    q = np.clip(np.round((xyz - c) / s * 32767), -32767, 32767).astype(np.int16)
    cls = np.array([klass(str(x)) for x in A.superclass], np.uint8)
    leg = np.zeros(n, np.uint8)
    for i in range(n):
        if A.superclass[i] == "vnc_motor" and A.subclass[i] in ("fl", "ml", "hl") and A.side[i] in ("L", "R"):
            leg[i] = LEGS.index(A.subclass[i] + A.side[i])
    role = np.zeros(n, np.uint8)
    for r, t in enumerate(NEURON_ROLES):
        if not t: continue
        tt, side = (t[:-2], t[-1]) if t.endswith(("_L", "_R")) else (t, None)
        for i in np.flatnonzero(A.type == tt):
            if side is None or A.side[i] == side:
                role[i] = r
    return struct.pack("<I", n) + q.tobytes() + cls.tobytes() + leg.tobytes() + role.tobytes()


def window_payload(w, A=None, labels=None):
    """One window's spikes, input tokens and membrane frames, per fly; labels (optional) gains every
    spiking neuron's type and side."""
    parts = [struct.pack("<I", len(w["spikes"]))]
    for b in range(len(w["spikes"])):
        sp = np.asarray(w["spikes"][b], np.int64).reshape(-1, 2)
        po = np.asarray(w["pois"][b], np.int64).reshape(-1, 2)
        mem = w["membrane"][b]
        starts = np.cumsum([0] + [len(m[1]) for m in mem]).astype(np.uint32)
        midx = np.concatenate([m[1] for m in mem]).astype(np.uint32) if mem else np.zeros(0, np.uint32)
        mlev = np.concatenate([np.clip(m[2] / 7.0 * 255, 0, 255) for m in mem]).astype(np.uint8) if mem else np.zeros(0, np.uint8)
        mstep = np.asarray([m[0] for m in mem], np.uint32)
        parts.append(struct.pack("<IIII", len(sp), len(po), len(mem), len(midx)))
        parts += [u32(sp[:, 0]), u32(sp[:, 1]), u32(po[:, 0]), u32(po[:, 1]), mstep.tobytes(), starts.tobytes(),
                  midx.tobytes(), mlev.tobytes(), b"\0" * ((-len(mlev)) % 4)]
        if labels is not None and A is not None:
            for i in np.unique(sp[:, 1]) if len(sp) else []:
                labels[int(i)] = f"{A.type[i] or A.superclass[i] or 'neuron'}{' (' + A.side[i] + ')' if A.side[i] else ''}"
    return b"".join(parts)


def window_meta(w, size):
    return dict(cmds=w["cmds"], ms=w["ms"], t0=int(w["t0"]), obs=w["obs"], forwards=int(w["forwards"]),
                wall=float(w["wall"]), n_spikes=[len(x) for x in w["spikes"]], n_poi=[len(x) for x in w["pois"]],
                raw_mb=round(size / 1e6, 2), probs=w.get("probs"), tokens=w.get("tokens"),
                levels=w.get("levels"), words=w.get("words"), nerve=w.get("nerve"))


def body_rows(log):
    """A body log (flyweight/flyworld.py Body.log) as float32 rows of BODY_COLS."""
    return np.array([[r[0], r[1], r[2], r[3], r[4], 1.0 if r[5] == "air" else 0.0, r[6], r[7], r[8], r[9], r[10], r[11],
                      *(list(r[13:19]) + [0.0] * (19 - max(13, len(r))))] for r in log], np.float32).reshape(-1, BODY_COLS)


def bodies_payload(logs):
    parts = [struct.pack("<I", len(logs))]
    for log in logs:
        rows = body_rows(log)
        parts += [struct.pack("<I", len(rows)), rows.tobytes()]
    return b"".join(parts)


def qwen_payload(acts_per_token, L, K=KEEP):
    """Per token: per layer (top indices, values) as recorded at the report's last token."""
    blobs = []
    for a in acts_per_token:
        for idx, val, _ in a:
            blobs.append(np.asarray(idx[:K], np.uint16).tobytes()); blobs.append(np.asarray(val[:K], np.float16).tobytes())
    return struct.pack("<III", len(acts_per_token), L, K) + b"".join(blobs)


def groups_and_roles(con, A, readout):
    """The groups the viewer's bridge panel and the CNS shader follow (what Qwen's answer drives, what the
    fly prints, what the body moves from), and labels for the named roles."""
    from .flyworld import World
    sets = cns_sets(con, A)
    wv = World.__new__(World)
    wv.A, wv.sets, wv.readout, wv._groups = A, sets, readout or {}, {}
    roles = {nm: [int(i) for i in np.flatnonzero(A.type == t)] for nm, t in
             (("MN9", "MN9"), ("DNp09", "DNp09"), ("MDN", "MDN"), ("DNp01", "DNp01"), ("DNg12", "DNg12_e"), ("TTMn", "TTMn"))}
    if readout:
        for nm, lab in (("fwd", "forward pool"), ("turn_L", "left-turn pool"), ("turn_R", "right-turn pool"),
                        ("bwd", "backward pool"), ("groom", "grooming pool")):
            roles[lab] = [int(i) for i in readout[nm]]
    labels = {}
    for nm, idx in roles.items():
        for i in idx:
            side = f" ({A.side[i]})" if A.side[i] else ""
            labels[i] = f"{A.type[i] or 'neuron'}{side} · {nm}" if nm.endswith("pool") else f"{nm}{side}"
    gdef = [("forward", "command", "P9 (DNp09)", sets["DNp09"]), ("left", "command", "DNa02 left", sets["DNa02_L"]),
            ("right", "command", "DNa02 right", sets["DNa02_R"]), ("jump", "command", "giant fiber DNp01", sets["DNp01"]),
            ("fly", "command", "DNg02 (wing power)", sets["DNg02"]), ("back", "command", "MDN (moonwalker)", sets["MDN"]),
            ("groom", "command", "aDN (antennal grooming)", sets["DNg12_e"]),
            ("taste", "sense", "sugar taste neurons", sets["sugar"]), ("taste", "sense", "bitter taste neurons", sets["bitter"]),
            ("antennae", "sense", "Johnston's organ C/E", sets["JO-CE"]), ("proboscis", "motor", "MN9 (proboscis)", list(wv.group("MN9"))),
            ("wings", "motor", "flight motor neurons", list(wv.group("power"))), ("jump", "motor", "jump motor neurons TTMn", list(wv.group("TTMn_L")) + list(wv.group("TTMn_R")))]
    if readout:
        gdef += [("forward", "pool", "forward pool", list(wv.group("fwd"))), ("left", "pool", "left-turn pool", list(wv.group("turn_L"))),
                 ("right", "pool", "right-turn pool", list(wv.group("turn_R"))), ("back", "pool", "backward pool", list(wv.group("bwd"))),
                 ("groom", "pool", "grooming pool", list(wv.group("groom")))]
    groups = [dict(key=k, kind=kind, name=nm, idx=[int(i) for i in idx]) for k, kind, nm, idx in gdef]
    return groups, roles, labels


def qwen_votes():
    """Every Qwen MLP neuron's direct vote on the move words: its down-projection column read through
    the final norm's gains and the unembedding, centred over the words (logit per unit of activation,
    ignoring later layers and the norm's scale). int8 per neuron per word, one global scale. Returns
    (payload, meta)."""
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoTokenizer
    snap = Path(snapshot_download("Qwen/Qwen3.5-4B", local_files_only=True))
    tok = AutoTokenizer.from_pretrained(snap)
    ids = [tok(" " + w).input_ids[0] for w in VOTE_WORDS]
    T = {}
    for f in sorted(snap.glob("*.safetensors")):
        with safe_open(f, "pt") as fh:
            for k in fh.keys():  # noqa: SIM118
                if k.endswith(("embed_tokens.weight", "language_model.norm.weight")) or ".mlp.down_proj.weight" in k and "mtp" not in k:
                    T[k] = fh.get_tensor(k)
    emb = next(v for k, v in T.items() if k.endswith("embed_tokens.weight"))
    g = 1.0 + next(v for k, v in T.items() if k.endswith("language_model.norm.weight")).float()
    U = emb[ids].float() * g
    U = U - U.mean(0)
    layers = sorted({int(k.split(".layers.")[1].split(".")[0]) for k in T if ".mlp.down_proj.weight" in k})
    V = torch.stack([U @ next(v for k, v in T.items() if f".layers.{l}.mlp.down_proj.weight" in k).float() for l in layers])
    scale = float(V.abs().max()) / 127.0
    q = torch.clamp(torch.round(V / scale), -127, 127).to(torch.int8).permute(0, 2, 1).contiguous()
    return struct.pack("<IIIf", q.shape[0], q.shape[1], q.shape[2], scale) + q.numpy().tobytes(), dict(words=VOTE_WORDS, scale=scale)


ENGINE = ("one stack: Qwen3.5-4B's 32 layers, then the fly's whole CNS as 3 stock Qwen3.5 Gated DeltaNet layers; one "
          "residual, vocabulary and token stream; every spike and every printed sense word is the model's own greedy token")
