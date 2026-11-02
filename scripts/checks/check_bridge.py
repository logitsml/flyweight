"""Qwen's answer drives the fly's registers, and the fly prints its senses, in the weights
(docs/design.md).

    python scripts/checks/check_bridge.py [--n 120]

1. Text fidelity of the merged model, every bridge and print row in place, vs base Qwen3.5-4B.
2. On synthetic reports: the registers the model writes at the report's last token vs float64 on
   the host from the captured residual (layer 1's input norm, each move word's row minus its reference's,
   the convolution's SiLU, the per-head norm), the levels they make (each register's inhibitors,
   offset and slope); and the host's s_w vs Qwen's own logits there.
3. The fly's layers only on report-end tokens == on every text token (registers and logits; both
   read the text 16 tokens per forward, so Qwen's own rounding is the same).
4. The printed words: after 60 ms with each sense (none, sugar, bitter, Johnston's organ, wing power),
   the word at every SAY token vs the host's rule on the MOVE rates at that moment.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from flyweight import report, setup  # noqa: E402
from flyweight.merged import ANSWER_REGS, ANSWER_WORDS, LAM, QH, QwenFlyRunner, answer_levels, answer_refs  # noqa: E402
from flyweight.sets import REG, ROLES  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    a = ap.parse_args()
    tok, qwen = setup.load_qwen()
    lm, tc = qwen.model.language_model, qwen.config.text_config
    probe = "Fruit flies taste sugar with neurons on their legs and proboscis; when"
    pids = tok(probe).input_ids
    with torch.no_grad():
        lp_base = torch.log_softmax(qwen(input_ids=torch.tensor([pids], device="cuda")).logits[0].float(), -1)
    wid = [tok(" " + w).input_ids[0] for w in ANSWER_WORDS]
    U = lm.embed_tokens.weight[wid].detach().double().cpu().numpy()
    iw = ANSWER_WORDS.index
    am = json.loads((ROOT / "data" / "answer_map.json").read_text())
    refs = answer_refs(am)
    C = U[[iw(w) for w in ANSWER_REGS]] - U[[iw(r) for r in refs]]
    S = setup.standard(tok, qwen)
    m = S["m"]
    nw1 = m.flayers[0].input_layernorm.weight.detach().double().cpu().numpy()
    eps1 = m.flayers[0].input_layernorm.eps
    # 1. text fidelity
    r = QwenFlyRunner(m)
    lg = torch.cat([m.text(pids[t:t + 16], list(range(t, min(len(pids), t + 16))), r.cache, keep=min(16, len(pids) - t))[0, :, :m.n_text]
                    for t in range(0, len(pids), 16)])
    lq = torch.log_softmax(lg.float(), -1)
    kl = (lp_base.exp() * (lp_base - lq)).sum(-1)
    print(f"1. text fidelity, all bridge and print rows in place: KL mean {float(kl.mean()):.2e}, max {float(kl.max()):.2e}; "
          f"top-1 {int((lp_base.argmax(-1) == lq.argmax(-1)).sum())}/{len(pids)}", flush=True)
    # 2. registers vs float64 from the residual layer 1 reads
    r = QwenFlyRunner(m)
    pre = tok(report.PROMPT).input_ids
    r.text(pre); r.snapshot_qwen(); r.start_fly()
    cap = {}
    def grab(mod, args):  # the report-end text token only (TXT marker set), not the PADs or SAYs that follow
        if float(args[0][0, -1, QH]) != 0.0:
            cap["h"] = args[0][0, -1].detach().double().cpu().numpy()
    hk = m.flayers[0].register_forward_pre_hook(grab)
    rng = np.random.default_rng(7)
    err_reg, err_lv, err_s = [], [], []
    for i in range(a.n):
        h, words, st, tgt = report.situation(rng)
        lg, _ = setup.read_report(r, tok, int(rng.uniform(0, 4000)), h, words=words)
        regs = r.registers()[:len(ANSWER_REGS)]
        x = cap["h"] / np.sqrt(np.mean(cap["h"] ** 2) + eps1) * (1.0 + nw1)
        s = C @ x[:QH]
        v = (LAM * s).astype(np.float32).astype(np.float64)
        reg_ref = v / (1.0 + np.exp(-v)) / LAM  # SiLU, then the exact per-head norm (|v| < 1024)
        lv_ref = answer_levels(reg_ref, am)
        err_reg.append(np.abs(regs - reg_ref).max()); err_lv.append(np.abs(r.levels(regs) - lv_ref).max())
        lgq = lg.double().cpu().numpy()
        err_s.append(np.abs((lgq[[wid[iw(w)] for w in ANSWER_REGS]] - lgq[[wid[iw(r)] for r in refs]]) - s).max())
    hk.remove()
    print(f"2. registers written by Qwen's answer vs float64 from the residual, {a.n} reports: max |error| {max(err_reg):.2e} "
          f"(levels {max(err_lv):.2e}); the host's s_w vs Qwen's own logits (bf16 head): max |diff| {max(err_s):.3f}", flush=True)
    # 3. fly layers on report-end tokens only vs on every text token
    rs = [QwenFlyRunner(m, fly_all=False), QwenFlyRunner(m, fly_all=True)]
    for rr in rs:
        rr.chunk = 16
        rr.text(pre); rr.snapshot_qwen(); rr.start_fly()
    rng2 = np.random.default_rng(11)
    worst_r = worst_l = 0.0
    for i in range(4):
        h, words, st, tgt = report.situation(rng2)
        out = []
        for rr in rs:
            lg, _ = setup.read_report(rr, tok, 5, h, words=words)
            out.append((rr.registers(), lg))
        worst_r = max(worst_r, float(np.abs(out[0][0] - out[1][0]).max()))
        worst_l = max(worst_l, float((out[0][1] - out[1][1]).abs().max()))
    print(f"3. fly layers on report-end tokens only vs on every text token: max register diff {worst_r:.2e}, max logit diff {worst_l:.2e}", flush=True)
    # 4. printed words vs the host's rule on the MOVE rates
    conds = {"none": {}, "sugar": {REG["sugar"]: 13}, "bitter": {REG["bitter"]: 13}, "JO-CE": {REG["JO-CE"]: 3}, "wing power": {REG["fly"]: 15},
             "sugar + JO-CE": {REG["sugar"]: 13, REG["JO-CE"]: 3}}
    bad, said = 0, {}
    for nm, lv in conds.items():
        r = QwenFlyRunner(m); r.start_fly()
        r.flybrain.set_levels(lv)
        for _ in range(600):
            r.flybrain.step()
        hz = dict(zip(ROLES, r.flybrain.move()[0].tolist()))
        words = []
        for k, (_, ws) in enumerate(report.SAY):
            sc = [((hz[role] - th) * report.SAY_PER_HZ if role else 0.0) for _, role, th in ws]
            want = ws[int(np.argmax(sc))][0]
            got = tok.decode([r.say(k)])
            bad += got != want
            words.append(got.strip())
        said[nm] = " ".join(words)
    print(f"4. printed words vs the host rule: {bad} mismatches; " + "; ".join(f"{k}: {v}" for k, v in said.items())
          + f"; block check {r.flybrain.mismatch}/{r.flybrain.swaps}, SAY argmax off the slot's words {r.say_bad}")
    ok = float(kl.mean()) < 1e-3 and max(err_lv) < 1e-3 and worst_l == 0.0 and worst_r < 1e-4 and bad == 0
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
