"""One model, one fly: Qwen3.5-4B with the fly's whole CNS inside it, one stack (flyweight/merged.py).

Every --report-ms of fly time Qwen reads a report of its fly (from its post-instructions state): the
world's description, then the fly's own neurons print its senses (SAY tokens: taste, proboscis,
antennae, wings), then the question. At the question's last token Qwen's own answer, its logits of
the move words over their reference words, writes the fly's command registers inside the same forward pass
(docs/design.md). The fly's layers then run, spike token by spike token; its
descending neurons fire from those registers through their own dynamics. The world writes the fly's
sense registers and moves the body from the motor rates the model's LM head reads out (MOVE rows).
The fly starts with dust on its antennae (--dust 0 for a clean start): grooming comes first. --start
places it (mm, mm, radians) and --fed-ms starts it fed, as the perch trials do (beside the drop, fed 600 ms).

    python scripts/run/qwenfly_session.py --out results/qwenfly.pt [--report-ms 20] [--max-ms 6000] [--dust 1]
        [--start X,Y,YAW] [--fed-ms 0]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from flyweight import report, setup  # noqa: E402
from flyweight.flyworld import ARENA, BITTER, DROP_R, PERCH, PERCH_R, PERCH_TOP, SUGAR  # noqa: E402
from flyweight.merged import ANSWER_WORDS, QwenFlyRunner  # noqa: E402
from flyweight.sets import DRIVEN, REG, ROLES, SENSES  # noqa: E402

TOPK = 96


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "qwenfly.pt"))
    ap.add_argument("--report-ms", type=int, default=report.STEP_MS)
    ap.add_argument("--max-ms", type=int, default=6000)
    ap.add_argument("--dust", type=float, default=1.0)
    ap.add_argument("--start", default=None)
    ap.add_argument("--fed-ms", type=float, default=0.0)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    tok, qwen = setup.load_qwen()
    nL = len(qwen.model.language_model.layers)
    t0 = time.time()
    S = setup.standard(tok, qwen)
    m, world, con = S["m"], S["world"], S["con"]
    r = QwenFlyRunner(m)
    pre = tok(report.PROMPT).input_ids
    r.text(pre); r.snapshot_qwen(); r.start_fly()
    wid = [tok(" " + w).input_ids[0] for w in ANSWER_WORDS]
    print(f"one stack: Qwen3.5-4B's 32 layers, then the fly's 3 (whole CNS, {con.n:,} neurons); one residual of "
          f"{m.L.H:,} dims; vocabulary {m.n_text:,} + {m.L.vocab:,}; built in {time.time() - t0:.0f} s; "
          f"GPU {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)
    # Qwen's MLP neurons at the report's last token (the viewer's wall)
    buf = torch.zeros(nL, qwen.config.text_config.intermediate_size, device="cuda")
    def hook(li):
        def f(mod, inp):  # returns None: a forward pre-hook that returns a value replaces the module's input
            buf[li].copy_(inp[0][0, -1].float())
        return f
    for li, layer in enumerate(m.qlayers):
        layer.mlp.down_proj.register_forward_pre_hook(hook(li))

    def snapshot():
        top = buf.abs().topk(TOPK, dim=1)
        val = torch.gather(buf, 1, top.indices)
        return [(top.indices[l].to(torch.int16).cpu(), val[l].half().cpu(), float(buf[l].abs().mean())) for l in range(nL)]

    B = world.bodies[0]
    B.dust, B.fed_ms = a.dust, a.fed_ms
    if a.start:
        B.x, B.y, B.yaw = (float(v) for v in a.start.split(","))
    segments, windows, done = [], [], False
    sense_now = {g: 0 for g in SENSES}
    steps = a.report_ms * 10
    fed_note = False
    FEED_MS = report.FEED_MS
    while world.t < a.max_ms * 10 and not done:
        head = report.head(*world.observe(0))
        fed_note = fed_note or B.fed_ms >= FEED_MS
        tq = time.time()
        lg, words = setup.read_report(r, tok, world.t / 10, head)  # the fly prints its senses; Qwen's answer writes the registers
        line = report.text(world.t / 10, head, words)
        acts = snapshot()
        levels = r.levels(r.registers())[:len(DRIVEN)]
        p = torch.softmax(lg[wid].float(), 0).cpu().numpy()
        tq = time.time() - tq
        answer = ANSWER_WORDS[int(p.argmax())]
        segments.append({"kind": "qwen", "read": line, "words": [w.strip() for w in words], "tokens": [" " + answer], "acts": [acts],
                         "probs": {w: float(v) for w, v in zip(ANSWER_WORDS, p)}, "levels": levels.tolist()})
        cmd_now = [c for (c, _), lv in zip(DRIVEN, levels) if lv >= 3] or ["rest"]
        spikes, membr = [], []
        tw = time.time(); f0 = r.flybrain.n_forward; s0 = len(r.stream)
        for s in range(steps):
            sl = world.sense_levels(0)
            ch = {REG[g]: sl[g] for g in SENSES if sl[g] != sense_now[g]}
            if ch:  # the world writes the sense registers
                r.flybrain.set_levels(ch)
                sense_now.update(sl)
            out = r.flybrain.step()[0]
            spikes += [(world.t, i) for i in out]
            if s % 50 == 0:
                idx, mv = r.flybrain.membrane()[0]
                membr.append((world.t, idx, mv))
            world.t += 1
            if world.t % 10 == 0:
                move = dict(zip(ROLES, r.flybrain.move()[0].tolist()))
                world.update_body(0, "+".join(cmd_now), move)
        nf, wall = r.flybrain.n_forward - f0, time.time() - tw
        windows.append(dict(cmds={"fly": cmd_now}, ms=a.report_ms, spikes=[spikes], pois=[[]], membrane=[membr], wall=wall,
                            forwards=nf, obs=[line], t0=world.t - steps, tokens=len(r.stream) - s0, probs=segments[-1]["probs"],
                            levels=levels.tolist(), words=segments[-1]["words"]))
        segments.append({"kind": "fly", "run": len(windows) - 1, "summary": "", "args": {"fly": cmd_now}})
        if len(windows) % 10 == 1:
            print(f"--- {world.t / 10:.0f} ms: the fly prints {' '.join(segments[-2]['words'])}; Qwen answers {answer} "
                  f"({', '.join(f'{k} {v:.2f}' for k, v in segments[-2]['probs'].items())}); "
                  f"drive levels {dict(zip([c for c, _ in DRIVEN], np.round(levels, 1)))} (Qwen {tq:.2f} s); fly {len(spikes)} "
                  f"spikes in {wall:.1f} s; at ({B.x:.1f}, {B.y:.1f}, {B.z:.1f}) {B.mode}, fed {B.fed_ms:.0f} ms; "
                  f"block check {r.flybrain.mismatch}/{r.flybrain.swaps}; SAY off-slot {r.say_bad}", flush=True)
        if fed_note and B.z >= PERCH_TOP - 0.1 and B.mode == "walk":
            done = True
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(segments=segments, windows=windows, prompt=report.PROMPT, messages=[], done=done,
                    bodies=[dict(name=b.name, log=b.log, fed_ms=b.fed_ms) for b in world.bodies],
                    world=dict(sugar=SUGAR.tolist(), bitter=BITTER.tolist(), perch=PERCH.tolist(), perch_top=PERCH_TOP,
                               arena=ARENA, drop_r=DROP_R, perch_r=PERCH_R),
                    block_mismatch=r.flybrain.mismatch, block_swaps=r.flybrain.swaps, stream_len=len(r.stream),
                    vocab=dict(text=m.n_text, fly=m.L.vocab), moves=dict(DRIVEN), roles=ROLES,
                    mismatch=None, design="one stack: Qwen's answer drives, the fly prints its senses", answer=m.answer,
                    say_bad=r.say_bad, report_ms=a.report_ms), a.out)
    print("saved", a.out, "done" if done else "goals not reached")


if __name__ == "__main__":
    main()
