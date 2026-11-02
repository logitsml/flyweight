"""The standard one-stack model, built the same way by every script: Qwen3.5-4B, the male CNS at the
operating point of flyweight/sets.py, the body's readout, and the bridge (Qwen's answer by default,
or the 48 situation neurons), with the fly's printed senses."""

import json
from pathlib import Path

import numpy as np
import torch

from . import malecns, report
from .flyworld import World
from .merged import ANSWER_WORDS, QwenFlyModel, load_drive
from .sets import DRIVEN, ROLES, SCALE, SENSES, cns_sets

ROOT = Path(__file__).resolve().parents[1]


def load_qwen(device="cuda"):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    path = snapshot_download("Qwen/Qwen3.5-4B", local_files_only=True)
    tok = AutoTokenizer.from_pretrained(path)
    qwen = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16, device_map=device).eval()
    return tok, qwen


def say_spec(tok):
    """report.SAY as the merged model's printed-word rows: (slot, token id, [(role, per Hz)], bias)."""
    out = []
    for k, (_, words) in enumerate(report.SAY):
        for w, role, th in words:
            t = tok(w).input_ids
            assert len(t) == 1, w
            out.append((k, t[0], [] if role is None else [(ROLES.index(role), report.SAY_PER_HZ)],
                        -th * report.SAY_PER_HZ if role else 0.0))
    return out


def world_and_groups():
    con = malecns.load(kc_kc=False); con.scale = SCALE
    A = malecns.annotations(con); sets = cns_sets(con, A)
    ro = np.load(ROOT / "data" / "readout.npz")
    world = World(A, sets, readout={k: ro[k] for k in ro.files})
    groups = [np.asarray(world.group(nm)) for _, nm in DRIVEN] + [np.asarray(world.group(nm)) for nm in SENSES]
    roles = [np.asarray(world.group(nm)) for nm in ROLES]
    return con, A, world, groups, roles


def standard(tok, qwen, bridge="answer", gain=2.5, drive_path=None, answer_path=None, say=True, device="cuda", adapt=None):
    """-> dict(m, con, A, world, groups, roles, cls_id). bridge: "answer" (Qwen's own answer drives the
    commands), "groups" (the 48 situation neurons and the hand-written map) or None (no bridge).
    adapt: the adaptation variant (fly_layers.ADAPT) or None."""
    con, A, world, groups, roles = world_and_groups()
    cls_id = tok(report.PROMPT + "0 ms. The fly." + report.ASK).input_ids[-1]
    kw = {}
    if bridge == "answer":
        am = json.loads(Path(answer_path or ROOT / "data" / "answer_map.json").read_text())
        kw["answer"] = dict(ids=[tok(" " + w).input_ids[0] for w in ANSWER_WORDS], **{k: am[k] for k in ("ref", "theta", "delta", "forward_c", "groom_c")})
    elif bridge == "groups":
        kw["drive"] = load_drive(drive_path or ROOT / "data" / "drive_neurons.npz")
    m = QwenFlyModel(qwen, con, groups, roles, device=device, cls_ids={cls_id}, gain=gain, say=say_spec(tok) if say else None, adapt=adapt, **kw)
    return dict(m=m, con=con, A=A, world=world, groups=groups, roles=roles, cls_id=cls_id)


def answer_ids(tok, word):
    """A move word as Qwen's answer joins its log: the word, then a newline (the worked examples' form)."""
    return tok(" " + word).input_ids + tok("\n").input_ids


def read_report(r, tok, t_ms, head_text, row=0, words=None):
    """One report through the one stack, from Qwen's post-instructions state: the world's text, the
    fly's printed senses (SAY tokens; or `words` given as text), then the question. Returns (Qwen's
    logits at the report's last token, the printed words)."""
    pieces = report.ids(tok, t_ms, head_text)
    lead = r.begin_report()  # from the instructions' state, or after Qwen's log (QwenFlyRunner memory)
    r.text(lead + pieces[0], row=row)
    printed = []
    for k in range(len(report.SAY)):
        w = r.say(k, row=row) if words is None else tok(words[k]).input_ids[0]
        printed.append(w)
        lg = r.text([w] + pieces[k + 1], row=row)
    r.report_ids = list(pieces[0]) + [t for k, w in enumerate(printed) for t in [w] + list(pieces[k + 1])]
    return lg, [tok.decode([w]) for w in printed]
