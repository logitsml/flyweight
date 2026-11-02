"""Is Qwen's own answer a sensible move? Base Qwen3.5-4B with the instructions and worked examples of
flyweight/report.py, on synthetic reports in the live format (the fly's printed senses included).

    python scripts/calibrate/check_answers.py [--n 400] [--history K]   # -> data/answer_map.json
    python scripts/calibrate/check_answers.py --refit                   # the map again from the saved logits, no GPU

--history K: before each report, K earlier synthetic reports, each followed by Qwen's own greedy
answer, as one continuous stream would hold them (no reset between reports); writes nothing.

Per report: the 8 words' logits at the report's last token; Qwen's argmax word against the pilot's
move, per kind of moment. Then the command map (flyweight/merged.py): register w holds max(s_w, 0), s_w =
logit(w) - logit(w's reference word), and drives at clip(15 (m_w - theta_w) / delta_w, 0, 15) on its
margin m_w: the steering pair inhibit each other (m = s_w - relu(s_other)), forward is inhibited by both
(- relu(s_left - c) - relu(s_right - c)), groom inhibits forward, left, right and back (- relu(s_groom -
c_g)), and jump, fly and groom read s_w. Every parameter is fitted to the pilot's levels (15 x
report.pilot_targets) by least squares over the reports where Qwen's own move is the pilot's (the map
scales Qwen's decisions; it is not fitted to undo the ones the pilot disagrees with): jump, fly and
groom each a reference word (stop, doing nothing, or forward, walking on), theta and delta; then c_g
together with the walking registers' fits: the steering pair one delta, forward c, theta and delta,
back a reference, theta and delta. Each delta is a value whose slope 15 / delta is a power of two,
each theta a multiple of 1/4 (both exact in the weights). Reported: each register's error against the
pilot over all reports, in sample and held out (two halves, each scored under the map fitted on the
other's agreeing reports).
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from flyweight import report  # noqa: E402
from flyweight.merged import ANSWER_REGS, ANSWER_WORDS, TURNS, answer_levels, answer_refs, inhibitors  # noqa: E402

STATES = ["seek", "on_sugar", "fed_floor", "flight", "over_perch", "wall", "dusty"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", default=str(ROOT / "data" / "answer_map.json"))
    ap.add_argument("--history", type=int, default=0)
    ap.add_argument("--refit", action="store_true")
    a = ap.parse_args()
    if a.refit:
        z = np.load(Path(a.out).with_suffix(".npz"), allow_pickle=True)
        fit(z["S"], z["pilot"], z["state"], z["bearing"], a, agreement(z["S"], z["pilot"], z["state"]))
        return
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    path = snapshot_download("Qwen/Qwen3.5-4B", local_files_only=True)
    tok = AutoTokenizer.from_pretrained(path)
    qwen = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
    wid = []
    for w in ANSWER_WORDS:
        t = tok(" " + w).input_ids
        assert len(t) == 1, (w, t)
        wid.append(t[0])
    for _, words in report.SAY:
        for w, _, _ in words:
            assert len(tok(w).input_ids) == 1, w
    pre = tok(report.PROMPT).input_ids
    with torch.no_grad():
        past0 = qwen(input_ids=torch.tensor([pre], device="cuda"), use_cache=True).past_key_values
    rng = np.random.default_rng(101)
    S, P, ST, TG = [], [], [], []
    for i in range(a.n):
        past, t_ms = copy.deepcopy(past0), 0
        for _ in range(a.history):  # earlier reports and Qwen's own answers, in one stream
            h0, w0, _, _ = report.situation(rng)
            with torch.no_grad():
                lg0 = qwen(input_ids=torch.tensor([report.ids(tok, t_ms, h0, w0)], device="cuda"), past_key_values=past, use_cache=True,
                           logits_to_keep=1).logits[0, -1].float()
                ans = " " + ANSWER_WORDS[int(lg0[wid].argmax())] + "\n"
                qwen(input_ids=torch.tensor([tok(ans).input_ids], device="cuda"), past_key_values=past, use_cache=True, logits_to_keep=1)
            t_ms += report.STEP_MS
        h, words, st, tgt = report.situation(rng)
        ids = report.ids(tok, t_ms if a.history else int(rng.uniform(0, 4000)), h, words)
        with torch.no_grad():
            lg = qwen(input_ids=torch.tensor([ids], device="cuda"), past_key_values=past, use_cache=True,
                      logits_to_keep=1).logits[0, -1].float()
        w = lg[wid].cpu().numpy().astype(np.float64)
        S.append(w - w.mean()); P.append(report.pilot_word(st, tgt)); ST.append(st); TG.append(tgt)
    S, P, ST, TG = np.array(S), np.array(P), np.array(ST), np.array(TG)
    print(f"prompt: {len(pre)} tokens; {a.n} synthetic reports")
    res = agreement(S, P, ST)
    if a.history:
        fit(S, P, ST, TG, a, res, save=False)
        print(f"with {a.history} earlier reports and answers in context (nothing written)")
        return
    np.savez(Path(a.out).with_suffix(".npz"), S=S, pilot=P, state=ST, bearing=TG)
    fit(S, P, ST, TG, a, res)


def agreement(S, P, ST):
    """Qwen's argmax word against the pilot's move, overall and per kind of moment."""
    am = np.array([ANSWER_WORDS[k] for k in S.argmax(1)])
    res = {"agreement": float((am == P).mean()), "by_state": {}}
    for st in STATES:
        sel = ST == st
        if not sel.any():
            continue
        words = {w: int((am[sel] == w).sum()) for w in ANSWER_WORDS if (am[sel] == w).any()}
        res["by_state"][st] = dict(n=int(sel.sum()), agree=float((am[sel] == P[sel]).mean()), words=words)
        print(f"  {st:10s} n={sel.sum():3d}  Qwen's move = the pilot's in {(am[sel] == P[sel]).mean():5.0%}   Qwen's words {words}")
    return res


DY = [15.0 / 2.0 ** k for k in range(-2, 7)]  # deltas whose slope 15 / delta is a power of two (1/4 .. 64): DRIVE writes stay exact
TH = np.arange(0.0, 4.01, 0.25)  # offsets: a register holds max(s, 0), so only theta >= 0 is in the weights
CS = np.arange(0.0, 6.01, 0.25)  # the inhibition offsets: forward's by the steering pair (c), the walking registers' by groom (c_g)
REFS = ("stop", "forward")  # the reference words a single command may be read against
# Wing power is not left to least squares, which weighs every report alike: the moments just after a jump, when
# Qwen puts fly about 1 nat over forward (under the 1.5-2.1 of the synthetic in-flight reports), came out below
# the power that holds the fly up, so it hopped. Fly is read from no offset at the slope that takes a 0.6 nat
# margin to level 9.6 (DNg02 at 9 already holds the fly up, check_motor.py); every state on the floor stays
# below -2 nats, at level 0.
FLY_MAP = (0.0, 0.9375)


def registers(S, am):
    """What each register holds per report: max(s_w, 0), s_w = logit(w) - logit(w's reference word)."""
    iw = ANSWER_WORDS.index
    return np.stack([np.maximum(S[:, iw(w)] - S[:, iw(ref)], 0.0) for w, ref in zip(ANSWER_REGS, answer_refs(am))], 1)


def calibrate(S, PL):
    """The answer map that best reproduces the pilot's levels PL (reports x 7) by least squares."""
    iw = ANSWER_WORDS.index
    g = ANSWER_REGS.index
    reg = lambda w, ref: np.maximum(S[:, iw(w)] - S[:, iw(ref)], 0.0)
    lv = lambda x, th, d: np.clip(15.0 * (x - th) / d, 0, 15)
    sse = lambda L, w: float(((L - PL[:, g(w)]) ** 2).sum())
    am = dict(ref={}, theta={}, delta={})
    for w in ("jump", "fly", "groom"):  # each weighed against doing nothing or against walking on
        _, ref, th, d = min((sse(lv(reg(w, ref), th, d), w), ref, th, d) for ref in REFS for th in TH for d in DY)
        am["ref"][w], am["theta"][w], am["delta"][w] = ref, float(th), d
    am["ref"]["fly"], (am["theta"]["fly"], am["delta"]["fly"]) = "forward", FLY_MAP
    Dl, Dr, Df, Dg = reg("left", "stop"), reg("right", "stop"), reg("forward", "stop"), reg("groom", am["ref"]["groom"])
    best = None
    for cg in CS:  # groom's offset: it inhibits every walking register
        G = np.maximum(Dg - cg, 0.0)
        e_t, d_t = min((sse(lv(Dl - Dr - G, 0, d), "left") + sse(lv(Dr - Dl - G, 0, d), "right"), d) for d in DY)  # one slope, no offset
        inh = lambda c: Df - np.maximum(Dl - c, 0.0) - np.maximum(Dr - c, 0.0) - G
        e_f, c, th_f, d_f = min((sse(lv(inh(c), th, d), "forward"), c, th, d) for c in CS for th in TH for d in DY)
        e_b, ref_b, th_b, d_b = min((sse(lv(reg("back", ref) - G, th, d), "back"), ref, th, d) for ref in REFS for th in TH for d in DY)
        if best is None or e_t + e_f + e_b < best[0]:
            best = (e_t + e_f + e_b, cg, d_t, c, th_f, d_f, ref_b, th_b, d_b)
    _, cg, d_t, c, th_f, d_f, ref_b, th_b, d_b = best
    am["groom_c"], am["forward_c"] = float(cg), float(c)
    for w in TURNS:
        am["ref"][w], am["theta"][w], am["delta"][w] = "stop", 0.0, d_t
    am["ref"]["forward"], am["theta"]["forward"], am["delta"]["forward"] = "stop", float(th_f), d_f
    am["ref"]["back"], am["theta"]["back"], am["delta"]["back"] = ref_b, float(th_b), d_b
    return am


def fit(S, P, ST, TG, a, res, save=True):
    """The command map from the 8 words' logits (S, centred per report), fitted to the pilot's levels."""
    PL = np.array([[15.0 * report.pilot_targets(st, b)[w] for w in ANSWER_REGS] for st, b in zip(ST, TG)])
    agree = np.array([ANSWER_WORDS[k] for k in S.argmax(1)]) == P  # fitted where Qwen's move is the pilot's
    am = calibrate(S[agree], PL[agree])
    lv = answer_levels(registers(S, am), am)
    rmse = np.sqrt(((lv - PL) ** 2).mean(0))
    perm = np.random.default_rng(0).permutation(len(S))
    held = np.zeros_like(lv)
    for tr, te in ((perm[::2], perm[1::2]), (perm[1::2], perm[::2])):  # each half's levels under the map fitted on the other
        tr = tr[agree[tr]]
        am_h = calibrate(S[tr], PL[tr])
        held[te] = answer_levels(registers(S[te], am_h), am_h)
    rmse_held = np.sqrt(((held - PL) ** 2).mean(0))
    am_word = np.array([ANSWER_WORDS[k] for k in S.argmax(1)])
    own = np.array([w if w in ANSWER_REGS else "stop" for w in am_word])
    driven = np.array([lv[i, ANSWER_REGS.index(w)] >= 0.5 for i, w in enumerate(own) if w != "stop"])
    lr = np.isin(own, TURNS)
    il, ir = ANSWER_REGS.index("left"), ANSWER_REGS.index("right")
    sign_ok = np.sign(lv[lr, il] - lv[lr, ir]) == np.where(own[lr] == "left", 1, -1)
    quiet = (lv[own == "stop"] < 0.5).all(1)
    both = float(((lv[:, il] > 0) & (lv[:, ir] > 0)).mean())
    seek = ST == "seek"
    iw = ANSWER_WORDS.index
    res.update(am, rmse=dict(zip(ANSWER_REGS, np.round(rmse, 2).tolist())), rmse_held_out=dict(zip(ANSWER_REGS, np.round(rmse_held, 2).tolist())),
               answer_driven=float(driven.mean()), turn_sign=float(sign_ok.mean()), stop_quiet=float(quiet.mean()), both_sides=both,
               r_turn=float(np.corrcoef(TG[seek], S[seek, iw("left")] - S[seek, iw("right")])[0, 1]),
               r_turn_levels=float(np.corrcoef(TG[seek], lv[seek, il] - lv[seek, ir])[0, 1]),
               mean_levels={st: np.round(lv[ST == st].mean(0), 2).tolist() for st in STATES if (ST == st).any()},
               pilot_levels={st: np.round(PL[ST == st].mean(0), 2).tolist() for st in STATES if (ST == st).any()})
    print(f"overall agreement {res['agreement']:.0%}; walking: r(sugar bearing, s_left - s_right) = {res['r_turn']:.2f}")
    for w in ANSWER_REGS:
        inh = "".join(f" - relu(s_{ANSWER_REGS[p_]} - {c_:g})" for p_, c_ in inhibitors(am).get(ANSWER_REGS.index(w), ()))
        print(f"  {w:7s} s = logit({w}) - logit({am['ref'][w]}); level = 15 (s{inh} - {am['theta'][w]:g}) / {am['delta'][w]:g}; "
              f"rmse against the pilot {rmse[ANSWER_REGS.index(w)]:.2f} (held out {rmse_held[ANSWER_REGS.index(w)]:.2f})")
    print(f"Qwen's answer drives its own register in {res['answer_driven']:.0%} of answers with one; turn side = Qwen's in "
          f"{res['turn_sign']:.0%} of left/right answers; stop answers drive nothing in {res['stop_quiet']:.0%}; both sides driven in "
          f"{both:.0%} of reports; walking: r(sugar bearing, left - right levels) = {res['r_turn_levels']:.2f}")
    for st in res["mean_levels"]:
        print(f"  {st:10s} mean levels {res['mean_levels'][st]}  pilot {res['pilot_levels'][st]}")
    ifl = ANSWER_REGS.index("fly")
    floor = ~np.isin(ST, ["flight", "over_perch"])
    res["fly_on_floor_max"] = float(lv[floor, ifl].max())
    print(f"wing power on the floor: at most level {res['fly_on_floor_max']:.1f} over {floor.sum()} reports; in flight: "
          f"{np.mean(lv[ST == 'flight', ifl] >= 9):.0%} of reports at level 9 or more")
    if save:
        Path(a.out).write_text(json.dumps(res, indent=1))
        print("saved", a.out)


if __name__ == "__main__":
    main()
