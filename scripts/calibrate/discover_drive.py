"""Find the Qwen3.5-4B MLP neurons that represent where the fly's target is and what state it is
in, as Qwen reads a report of its fly: the neurons that will drive the fly's descending neurons.

Synthetic reports in the live format (flyweight/report.py: the world's text and the fly's printed
senses) vary the target's bearing and distance and the fly's state. For each drive (left, right, forward, stop, jump, fly) the neurons whose activation at the
report's last token correlates best with that drive's target are picked (no weights are trained;
neurons are only selected), in the style of qwen-steer's contrastive discovery. Each picked
neuron is scaled by its 5th-95th percentile over the training reports. Checked on held-out reports.

    python scripts/calibrate/discover_drive.py [--n 600] [--k 8]

These groups drive the fly only in the trials' "groups" condition; by default Qwen's answer does.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

DRIVES = ["left", "right", "forward", "stop", "jump", "fly"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "data" / "drive_neurons.npz"))
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from flyweight import report
    path = snapshot_download("Qwen/Qwen3.5-4B", local_files_only=True)
    tok = AutoTokenizer.from_pretrained(path)
    qwen = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
    lm = qwen.model.language_model
    nL, nN = len(lm.layers), lm.layers[0].mlp.down_proj.in_features
    acts = torch.zeros(nL, nN, device="cuda")

    def hook(li):
        def f(mod, inp):
            acts[li].copy_(inp[0][0, -1].float())
        return f
    for li in range(nL):
        lm.layers[li].mlp.down_proj.register_forward_pre_hook(hook(li))
    rng = np.random.default_rng(0)
    pre = tok(report.PROMPT).input_ids
    with torch.no_grad():  # the long instructions once, then each report from that cache
        out = qwen(input_ids=torch.tensor([pre], device="cuda"), use_cache=True)
        past0 = out.past_key_values
    import copy
    X = np.zeros((a.n, nL * nN), np.float32)
    Y = np.zeros((a.n, len(DRIVES)), np.float32)
    for i in range(a.n):
        h, words, st, tgt = report.situation(rng)
        t = report.pilot_targets(st, tgt)
        ids = report.ids(tok, int(rng.uniform(0, 4000)), h, words)
        with torch.no_grad():
            qwen(input_ids=torch.tensor([ids], device="cuda"), past_key_values=copy.deepcopy(past0), use_cache=True, logits_to_keep=1)
        X[i] = acts.flatten().cpu().numpy()
        Y[i] = [t[d] for d in DRIVES]
        if i % 80 == 0:
            print(f"report {i}/{a.n}", flush=True)
    tr, te = slice(0, int(a.n * 0.75)), slice(int(a.n * 0.75), a.n)
    Xc = X[tr] - X[tr].mean(0)
    sx = Xc.std(0) + 1e-6
    lo_l = 8 * nN  # only layers 8 and up: past the token-identity layers
    out = {}
    print("\ndrive     picked neurons (r on training reports)                      held-out r")
    for d_i, d in enumerate(DRIVES):
        y = Y[tr, d_i]
        yc = (y - y.mean()) / (y.std() + 1e-9)
        r = (Xc * yc[:, None]).mean(0) / sx
        r[:lo_l] = 0
        pick = np.argsort(-r)[:a.k]
        p5, p95 = np.percentile(X[tr][:, pick], 5, axis=0), np.percentile(X[tr][:, pick], 95, axis=0)
        drive = np.clip(((X[:, pick] - p5) / np.maximum(1e-6, p95 - p5)).mean(1), 0, 1)  # linear mean, then clip: what the weights compute
        r_te = np.corrcoef(drive[te], Y[te, d_i])[0, 1]
        out[d] = np.stack([pick // nN, pick % nN], 1)
        out[d + "_p5"], out[d + "_p95"] = p5, p95
        print(f"{d:8s}  " + ", ".join(f"L{p // nN} N{p % nN} ({r[p]:.2f})" for p in pick[:4]) + f" ...   {r_te:.2f}")
    np.savez(a.out, **out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
