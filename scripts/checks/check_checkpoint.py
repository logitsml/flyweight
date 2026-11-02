"""The checkpoint holds the model.

    python scripts/checks/check_checkpoint.py [--out checkpoint]

Builds the standard model (Qwen's answer bridge, the printed senses), records its outputs on a test
stream (the instructions, three reports read with the fly printing its senses, 300 fly steps with
sense registers changing), writes the checkpoint, frees everything, loads the checkpoint alone and
replays the stream: every logit must match bit for bit (the built model is replayed twice first, so a
replay that is not repeatable shows apart from a checkpoint that lost something). Both runs multiply the connectome by columns
(SparseLinear.deterministic): cuSPARSE's matrix-vector kernel sums its long rows in an order that
varies from run to run. Then checks the manifest's hashes.
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from flyweight import checkpoint, report, setup  # noqa: E402
from flyweight.merged import QwenFlyRunner  # noqa: E402
from flyweight.sets import REG  # noqa: E402
from flyweight.sparse import SparseLinear  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False


def stream_outputs(m, tok):
    r = QwenFlyRunner(m)
    out = [r.text(tok(report.PROMPT).input_ids)]
    r.snapshot_qwen(); r.start_fly()
    rng = np.random.default_rng(3)
    for k in range(3):
        h, words, st, tgt = report.situation(rng, "seek")
        lg, said = setup.read_report(r, tok, 20 * k, h)
        out += [lg, torch.as_tensor(r.registers())]
        if k == 1:
            r.flybrain.set_levels({REG["sugar"]: 13, REG["JO-CE"]: 3})
        for _ in range(100):
            r.flybrain.step()
            out.append(r.flybrain.logits[0])
    return [o.detach().float().cpu() for o in out], r.stream


def compare(a, b):
    """(max |diff| over every output, index of the first output that differs or None)."""
    d = [float((x - y).abs().max()) for x, y in zip(a, b)]
    return max(d), next((k for k, v in enumerate(d) if v != 0.0), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "checkpoint"))
    a = ap.parse_args()
    SparseLinear.deterministic = True
    tok, qwen = setup.load_qwen()
    m = setup.standard(tok, qwen)["m"]
    ref, ref_stream = stream_outputs(m, tok)
    again, again_stream = stream_outputs(m, tok)  # the built model twice: is the replay itself repeatable?
    rep_worst, rep_first = compare(ref, again)
    if rep_first is not None:
        d = (ref[rep_first] - again[rep_first]).abs().reshape(-1)
        top = torch.topk(d, min(5, len(d)))
        print(f"  output {rep_first} ({ref[rep_first].numel()} values) differs at {top.indices.tolist()}: "
              f"{ref[rep_first].reshape(-1)[top.indices].tolist()} vs {again[rep_first].reshape(-1)[top.indices].tolist()}", flush=True)
    print(f"built vs built: max |diff| {rep_worst:.3g}{'' if rep_first is None else f' (first at output {rep_first})'}; "
          f"token streams identical: {ref_stream == again_stream}", flush=True)
    t0 = time.time()
    man = checkpoint.save(m, a.out, tok)
    gb = sum(e["bytes"] for e in man.values()) / 1e9
    print(f"saved {a.out}: {len(man)} files, {gb:.1f} GB, in {time.time() - t0:.0f} s", flush=True)
    del m, qwen; gc.collect(); torch.cuda.empty_cache()
    t0 = time.time()
    m2, tok2 = checkpoint.load(a.out)
    print(f"loaded from the checkpoint alone in {time.time() - t0:.0f} s; GPU {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)
    got, got_stream = stream_outputs(m2, tok2)
    worst, first = compare(ref, got)
    same = ref_stream == got_stream
    ok_hash = checkpoint.verify(a.out)
    print(f"loaded vs built: {len(ref)} outputs (text logits, registers, fly logits), max |diff| {worst:.3g}"
          f"{'' if first is None else f' (first at output {first})'}; "
          f"token streams identical: {same} ({len(ref_stream)} tokens); manifest hashes match: {ok_hash}")
    print("PASS" if worst == 0.0 and same and ok_hash and rep_worst == 0.0 else "FAIL")


if __name__ == "__main__":
    main()
