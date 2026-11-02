"""Pack a session (scripts/run/qwenfly_session.py) for the viewer (viz/index.html).

    python scripts/run/export_viz.py results/qwenfly.pt viz

Formats in flyweight/viewdata.py (the live server, scripts/run/serve.py, streams the same ones). Writes
viz/cns/neurons.gz.b64.txt and viz/agent/: windows<pack> (PACK windows per file), bodies, qwen,
votes and session.json (everything else).
"""

import base64
import gzip
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from flyweight import malecns, viewdata  # noqa: E402

PACK = 50  # windows per file (a 6 s session has 300 of them)


def write(path, payload):
    path.write_text(base64.b64encode(gzip.compress(payload, 9)).decode())
    return len(payload)


def main():
    src, root = Path(sys.argv[1]), Path(sys.argv[2])
    out = root / "agent"
    out.mkdir(parents=True, exist_ok=True)
    (root / "cns").mkdir(parents=True, exist_ok=True)
    d = torch.load(src, weights_only=False)
    con = malecns.load(kc_kc=False)
    A = malecns.annotations(con)
    write(root / "cns" / "neurons.gz.b64.txt", viewdata.neurons_payload(con, A))
    labels, wins, payloads = {}, [], []
    for w in d["windows"]:
        payload = viewdata.window_payload(w, A, labels)
        payloads.append(payload)
        wins.append(viewdata.window_meta(w, len(payload)))
    for f in out.glob("window*.gz.b64.txt"):
        f.unlink()
    n_packs = 0
    for c0 in range(0, len(payloads), PACK):
        chunk = payloads[c0:c0 + PACK]
        write(out / f"windows{n_packs}.gz.b64.txt", struct.pack("<I", len(chunk)) + b"".join(struct.pack("<I", len(x)) + x for x in chunk))
        n_packs += 1
    write(out / "bodies.gz.b64.txt", viewdata.bodies_payload([B["log"] for B in d["bodies"]]))
    toks, acts, segs = [], [], []
    K = min(viewdata.KEEP, len(d["segments"][0]["acts"][0][0][0]))  # the strongest K per layer (topk is sorted)
    for sgm in d["segments"]:
        if sgm["kind"] == "fly":
            segs.append({"kind": "fly", "run": sgm["run"], "summary": sgm["summary"], "args": sgm["args"]}); continue
        if sgm["kind"] == "user":
            segs.append({"kind": "user", "text": sgm["text"]}); continue
        start = len(toks)
        toks += sgm["tokens"]; acts += sgm["acts"]
        segs.append({"kind": "qwen", "start": start, "count": len(toks) - start, "read": sgm.get("read"), "probs": sgm.get("probs"),
                     "words": sgm.get("words"), "levels": sgm.get("levels")})
    L = len(d["segments"][0]["acts"][0])
    write(out / "qwen.gz.b64.txt", viewdata.qwen_payload(acts, L, K))
    ro_path = ROOT / "data" / "readout.npz"
    readout = {k: np.load(ro_path)[k] for k in np.load(ro_path).files} if ro_path.exists() else {}
    groups, roles, role_labels = viewdata.groups_and_roles(con, A, readout)
    labels.update(role_labels)
    vpay, votes = viewdata.qwen_votes()
    write(out / "votes.gz.b64.txt", vpay)
    session = dict(prompt=d["prompt"], segments=segs, windows=wins, tokens=toks, labels={str(k): v for k, v in labels.items()}, groups=groups, votes=votes,
                   world=d["world"], flies=[b["name"] for b in d["bodies"]], fed_ms=[b["fed_ms"] for b in d["bodies"]],
                   moves=d.get("moves"), done=d.get("done"), packs=n_packs, pack=PACK, design=d.get("design"), answer=d.get("answer"),
                   n_neurons=int(con.n), mismatch=d.get("mismatch"), block_mismatch=d.get("block_mismatch"), say_bad=d.get("say_bad"), roles=roles,
                   qwen=dict(layers=L, neurons=9216, topk=K), block_swaps=d.get("block_swaps"), stream_len=d.get("stream_len"),
                   vocab=d.get("vocab"), engine=viewdata.ENGINE, body_cols=viewdata.BODY_COLS, leg_hz=viewdata.LEG_HZ)
    (out / "session.json").write_text(json.dumps(session))
    for p in sorted(out.glob("*")):
        print(f"{p.name:24s} {p.stat().st_size / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
