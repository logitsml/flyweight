"""Checks for the one-stack model (flyweight/merged.py).

    python scripts/checks/check_merged.py

(i) text fidelity: next-token distributions of the merged model vs base Qwen3.5-4B on prompts, with
    Qwen's layers on the dims a text token holds (the fast path) and on the whole width;
(ii) fast path == full forward (all 35 layers on every token, the whole width) on a mixed text/fly
     sequence, with a report whose senses the fly prints (SAY tokens) and whose last token writes
     the registers;
(iii) the fly inside the merged model == the fly alone (FlyModel), spike for spike.

PASS: text KL mean below 1e-3 and top-1 agreement at least 95% at both widths; the fly's logits exactly equal,
each SAY's word the same, text logits within 0.25 between the paths, no SAY mismatch; the fly inside and alone
identical at every step, with a clean block check. The command levels after the report are reported, not
required: each path's registers follow its own logits exactly (check_bridge), and the bridge's slope (16-64
levels per nat) turns the paths' bf16 rounding of Qwen's text logits into level differences.
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from flyweight import fly_layers, report, setup  # noqa: E402
from flyweight.merged import QwenFlyRunner  # noqa: E402
from flyweight.sets import DRIVEN, REG  # noqa: E402

levels = lambda **lv: [lv.get(nm, 0) for nm in REG]  # every register's level, by command or sense name

torch.backends.cuda.matmul.allow_tf32 = False
PROMPTS = ["The fruit fly Drosophila melanogaster has a brain of about 140,000 neurons, and",
           "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
           "Write a haiku about a fly landing on a sugar cube:\n",
           "In 1905, Albert Einstein published four papers that"]


def main():
    tok, qwen = setup.load_qwen()
    base = []
    with torch.no_grad():
        for pr in PROMPTS:
            ids = tok(pr).input_ids
            lp = torch.log_softmax(qwen(input_ids=torch.tensor([ids], device="cuda")).logits[0].float(), -1)
            base.append((ids, lp))
    S = setup.standard(tok, qwen)  # the answer bridge and the printed senses, as in a session
    m, con, groups, roles = S["m"], S["con"], S["groups"], S["roles"]
    print(f"merged: hidden {fly_layers.H_FLY:,}; vocabulary {m.n_text:,} + {m.L.vocab:,}; GPU {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    # (i) text fidelity, the fast path's dims and the whole width
    lqs, fails = {}, []
    for narrow in (True, False):
        kls, agree, tot, lqs[narrow] = [], 0, 0, []
        for ids, lp in base:
            cache = QwenFlyRunner(m).cache
            lg = []
            for t0 in range(0, len(ids), 16):
                lg.append(m.text(ids[t0:t0 + 16], list(range(t0, min(len(ids), t0 + 16))), cache, keep=min(16, len(ids) - t0),
                                 narrow=narrow)[0, :, :m.n_text])
            lq = torch.log_softmax(torch.cat(lg).float(), -1)
            lqs[narrow].append(lq)
            kl = (lp.exp() * (lp - lq)).sum(-1)
            kls.append(kl); agree += int((lp.argmax(-1) == lq.argmax(-1)).sum()); tot += len(ids)
        kl = torch.cat(kls)
        print(f"(i) text fidelity vs base Qwen3.5-4B, Qwen's layers on {'the dims a text token holds' if narrow else 'the whole width'}: "
              f"KL mean {float(kl.mean()):.2e}, max {float(kl.max()):.2e}; top-1 agreement {agree}/{tot}")
        if float(kl.mean()) >= 1e-3 or agree < 0.95 * tot:
            fails.append("text fidelity" + (" (narrow)" if narrow else " (whole width)"))
    d = max(float((a - b).abs().max()) for a, b in zip(lqs[True], lqs[False]))
    print(f"    narrow vs whole width: max |log-prob diff| {d:.2e}")
    # (ii) fast path vs full forward on a mixed sequence: text, a fly window, text, a fly window, a report
    # (the fly prints its senses; the last token writes the registers). The fly's logits must match
    # exactly; Qwen's text logits differ by the rounding of a chunked prefill against token-by-token
    # decoding (both paths run Qwen's own layers; HF's chunked and recurrent delta rules round apart)
    seq_text1 = tok("The fly is on the floor, the sugar drop is 12 mm ahead.").input_ids
    seq_text2 = tok(" Best move:").input_ids
    r = QwenFlyRunner(m)
    marks = {}
    lg = r.text(seq_text1); marks["text 1 end"] = (len(r.stream) - 1, lg.clone(), "text")
    r.start_fly()
    r.flybrain.set_levels(levels(forward=12, sugar=9))
    for s_ in range(60):
        r.flybrain.step()
    marks["fly TICK before text 2"] = (len(r.stream) - 1, r.flybrain.logits[0].clone(), "fly")
    lg = r.text(seq_text2); marks["text 2 end"] = (len(r.stream) - 1, lg.clone(), "text")
    for s_ in range(30):
        r.flybrain.step()
    marks["fly TICK before the report"] = (len(r.stream) - 1, r.flybrain.logits[0].clone(), "fly")
    pieces = report.ids(tok, 9, report.head("on the floor", 1, 2, 0, (5, 30), (20, -60), (14, 120), 0))
    r.text(pieces[0])
    for k in range(len(report.SAY)):
        wd = r.say(k)
        marks[f"SAY {k}"] = (len(r.stream) - 1, r.say_logits.clone(), "all")
        lg = r.text([wd] + pieces[k + 1])
    marks["report end"] = (len(r.stream) - 1, lg.clone(), "text")
    regs_fast = r.registers()
    stream = list(r.stream)
    cache = m.new_cache()
    fly_cols, tpos, full_logits = [], 0, []
    for i, t in enumerate(stream):
        is_fly = t >= m.n_text
        fly_cols.append(is_fly)
        lgf = m.full([t], is_fly, [tpos], cache, fly_cols)
        if not is_fly:
            tpos += 1
        full_logits.append(lgf[0, -1])
    out = []
    for nm, (q, fast, part) in marks.items():
        full = full_logits[q]
        full = full[:m.n_text] if part == "text" else (full[m.n_text:] if part == "fly" else full)
        dmax = float((full - fast).abs().max())
        if (part == "fly" and dmax != 0) or (part == "text" and dmax > 0.25) or (part == "all" and int(full.argmax()) != int(fast.argmax())):
            fails.append(nm)
        out.append(f"{nm} {float((full - fast).abs().max()):.3g}" + (f" (word {tok.decode([int(full.argmax())])!r} both)"
                                                                        if part == "all" and int(full.argmax()) == int(fast.argmax()) else ""))
    # the registers the report wrote, read by one PAD through all 35 layers
    reg = {}
    hk = m.flayers[0].register_forward_hook(lambda mod, i_, o: reg.__setitem__("x", o))
    fly_cols.append(True); m.full([m.n_text + m.L.tok_pad], True, [tpos], cache, fly_cols); hk.remove()
    regs_full = reg["x"][0, -1, m.L.reg].double().cpu().numpy()
    lv_f, lv_s = r.levels(regs_full)[:len(DRIVEN)], r.levels(regs_fast)[:len(DRIVEN)]
    print(f"(ii) fast path vs full forward ({len(stream)} tokens), max |logit diff|: " + "; ".join(out)
          + f"; command levels after the report: fast {np.round(lv_s, 3).tolist()}, full {np.round(lv_f, 3).tolist()}"
          f" (max diff {np.abs(lv_s - lv_f).max():.2e}); SAY mismatches {r.say_bad}")
    if r.say_bad:
        fails.append("SAY")
    # (iii) the fly inside vs the fly alone
    r = QwenFlyRunner(m); r.text(seq_text1); r.start_fly()
    alone = fly_layers.FlyModel(con, groups, roles)
    # (the answer bridge's inhibitions are inside Qwen only, so no two registers that inhibit each other are driven together)
    sched = {0: levels(forward=12, sugar=9), 300: levels(turn_left=13), 450: levels(back=12), 525: levels(groom=10), 600: levels(forward=8, fly=14, **{"JO-CE": 4})}
    diff = 0
    for s in range(900):
        if s in sched:
            r.flybrain.set_levels(sched[s]); alone.set_levels(sched[s])
        a, b = sorted(r.flybrain.step()[0]), sorted(alone.step()[0])
        diff += a != b
    print(f"(iii) the fly inside Qwen vs the fly alone, 90 ms: steps that differ {diff}; block check inside "
          f"{r.flybrain.mismatch} set / {r.flybrain.swaps} order")
    if diff or r.flybrain.mismatch or r.flybrain.swaps:
        fails.append("the fly inside vs alone")
    print("PASS" if not fails else "FAIL: " + ", ".join(fails))


if __name__ == "__main__":
    main()
