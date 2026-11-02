"""Text benchmarks: is Qwen with the fly inside it as capable as Qwen3.5-4B? (docs/design.md)

The tasks, prompts, task loaders and scoring are adapted from QwenPU's chipformer/q35/text_eval.py,
at smaller sizes (github.com/divagr18/QwenPU; Copyright (c) 2026 Divyansh Agrawal; MIT License).

    python scripts/experiments/text_eval.py run base|eager|merged [--tasks mmlu,arc,hellaswag,gsm8k,ifeval,ppl,niah]
    python scripts/experiments/text_eval.py table

Models: base (Qwen3.5-4B, sdpa), eager (the same, eager attention: the noise floor, which changes
only the rounding inside Qwen's full-attention layers), merged (the one stack with the fly, Qwen's
answer bridge and the printed-sense rows, read through its fast path, 512 tokens per forward).
Tasks (chat template, thinking off, except the ARC/HellaSwag contexts and the perplexity text):
  mmlu       500 seeded test questions, argmax over A/B/C/D at the first answer token
  arc        500 seeded ARC-Challenge test items, continuation log-likelihood (acc, acc_norm)
  hellaswag  500 seeded validation items, continuation log-likelihood (acc_norm)
  gsm8k      100 seeded test problems, greedy (512 new tokens), \\boxed{} answer
  ifeval     100 seeded prompts, greedy (640 new tokens), lm-eval's checker (prompt-level strict/loose)
  ppl        2 disjoint 4,096-token wikitext-103 windows: mean NLL and greedy next-token agreement
  niah       a 7-digit number in a 8,192-token wikitext-103 haystack at depths 0.1, 0.5, 0.9
Per-item records go to results/text_eval/<model>/<task>.json (a task whose file exists is skipped);
`table` compares merged and eager against base, paired (agreement of per-item outcomes, exact McNemar).
What the merged column measures: the one stack's arithmetic on text (a float32 residual, norms
averaging over its 1.57M dims, 512-token prefill), computed on the dims a text token holds
(flyweight/merged.py; scripts/checks/check_merged.py compares it with the whole width). The fly's layers run
only on a report-end token ("):"), which benchmark text rarely contains, and never write Qwen's dims.
"""

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "text_eval"
CITIES = ["Lisbon", "Nairobi", "Osaka", "Quito", "Tbilisi", "Winnipeg"]
CH = 512  # merged: tokens per forward


class HF:
    def __init__(self, attn):
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
        path = snapshot_download("Qwen/Qwen3.5-4B", local_files_only=True)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.m = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda",
                                                                  attn_implementation=attn).eval()

    @torch.no_grad()
    def logits(self, ids):
        """Next-token logits after every position (float32, on the GPU)."""
        return self.m(input_ids=torch.tensor([ids], device="cuda")).logits[0].float()

    @torch.no_grad()
    def prefix(self, ids):
        out = self.m(input_ids=torch.tensor([ids], device="cuda"), use_cache=True)
        return (out.past_key_values, len(ids)), out.logits[0, -1].float()

    @torch.no_grad()
    def extend(self, state, ids):
        import copy
        cache, n = state
        out = self.m(input_ids=torch.tensor([ids], device="cuda"), past_key_values=copy.deepcopy(cache), use_cache=True)
        return out.logits[0].float()

    @torch.no_grad()
    def nll(self, ids, chunk=512):
        """Per-token NLL of ids[1:] and whether the greedy prediction was right, prefilled in chunks."""
        from transformers import DynamicCache
        cache, nll, hit, prev = DynamicCache(config=self.m.config.text_config), [], [], None
        for c0 in range(0, len(ids), chunk):
            ch = ids[c0:c0 + chunk]
            lg = self.m(input_ids=torch.tensor([ch], device="cuda"), past_key_values=cache, use_cache=True).logits[0].float()
            lg = torch.cat([prev[None], lg]) if prev is not None else lg
            prev = lg[-1]
            n_t = len(ch) if c0 else len(ch) - 1
            tgt = torch.tensor(ids[c0 + (0 if c0 else 1):c0 + len(ch)], device="cuda")
            use = lg[:len(tgt)]
            nll.append(-torch.log_softmax(use, -1)[torch.arange(len(tgt)), tgt]); hit.append(use.argmax(-1) == tgt)
        return torch.cat(nll), torch.cat(hit)

    @torch.no_grad()
    def generate(self, ids, max_new, eos):
        out = self.m.generate(input_ids=torch.tensor([ids], device="cuda"), max_new_tokens=max_new, do_sample=False,
                              eos_token_id=list(eos), pad_token_id=self.tok.pad_token_id)
        return out[0, len(ids):].tolist()


class Merged:
    def __init__(self):
        from flyweight import setup
        from flyweight.merged import QwenFlyRunner
        self.tok, qwen = setup.load_qwen()
        self.m = setup.standard(self.tok, qwen)["m"]
        self.Runner = QwenFlyRunner

    def _feed(self, r, ids, keep_all):
        out = []
        for c0 in range(0, len(ids), CH):
            ch = ids[c0:c0 + CH]
            lg = self.m.text(ch, list(range(r.text_pos, r.text_pos + len(ch))), r.cache, keep=len(ch) if keep_all else 1)
            r.text_pos += len(ch)
            out.append(lg[0, :, :self.m.n_text].float())
        return torch.cat(out) if keep_all else out[-1][-1]

    @torch.no_grad()
    def logits(self, ids):
        return self._feed(self.Runner(self.m), ids, True)

    @torch.no_grad()
    def nll(self, ids):
        r, nll, hit, prev = self.Runner(self.m), [], [], None
        for c0 in range(0, len(ids), CH):
            lg = self._feed(r, ids[c0:c0 + CH], True)
            lg = torch.cat([prev[None], lg]) if prev is not None else lg
            prev = lg[-1]
            tgt = torch.tensor(ids[c0 + (0 if c0 else 1):c0 + CH], device=lg.device)
            use = lg[:len(tgt)]
            nll.append(-torch.log_softmax(use, -1)[torch.arange(len(tgt)), tgt]); hit.append(use.argmax(-1) == tgt)
        return torch.cat(nll), torch.cat(hit)

    @torch.no_grad()
    def prefix(self, ids):
        r = self.Runner(self.m)
        lg = self._feed(r, ids, False)
        r.snapshot_qwen()
        return r, lg

    @torch.no_grad()
    def extend(self, r, ids):
        r.restore_qwen()
        return self._feed(r, ids, True)

    @torch.no_grad()
    def generate(self, ids, max_new, eos):
        r = self.Runner(self.m)
        lg = self._feed(r, ids, False)
        out = []
        for _ in range(max_new):
            t = int(lg.argmax())
            out.append(t)
            if t in eos:
                break
            lg = self._feed(r, [t], False)
        return out


def chat(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                   return_tensors="pt", enable_thinking=False)["input_ids"][0].tolist()


def eos_ids(tok):
    return {t for t in (tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>"), tok.eos_token_id)
            if isinstance(t, int) and t >= 0}


def task_mmlu(M, n=500):
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    letters = [M.tok.encode(c, add_special_tokens=False)[0] for c in "ABCD"]
    items = []
    for i in sorted(random.Random(0).sample(range(len(ds)), n)):
        d = ds[i]
        q = (f"The following is a multiple choice question about {d['subject'].replace('_', ' ')}.\n\n{d['question'].strip()}\n"
             + "".join(f"{c}. {t}\n" for c, t in zip("ABCD", d["choices"])) + "\nAnswer with the letter of the correct option only.")
        _, lg = M.prefix(chat(M.tok, q))
        lp = torch.log_softmax(lg, -1)[letters].tolist()
        items.append(dict(idx=i, gold=d["answer"], lp=lp, pred=int(np.argmax(lp)), correct=int(np.argmax(lp)) == d["answer"]))
    return items


def choice(M, docs):
    items = []
    for c, conts, gold in docs:
        cids = M.tok.encode(c, add_special_tokens=False)
        state, last = M.prefix(cids)
        lps = []
        for x in conts:
            xids = M.tok.encode(x, add_special_tokens=False)
            lg = torch.cat([last[None], M.extend(state, xids)[:-1]]) if len(xids) > 1 else last[None]
            lp = torch.log_softmax(lg, -1)
            lps.append(float(lp[torch.arange(len(xids)), torch.tensor(xids, device=lp.device)].sum()))
        norm = [v / len(x) for v, x in zip(lps, conts)]
        items.append(dict(gold=gold, lp=lps, pred=int(np.argmax(lps)), pred_norm=int(np.argmax(norm)),
                          correct=int(np.argmax(lps)) == gold, correct_norm=int(np.argmax(norm)) == gold))
    return items


def task_arc(M, n=500):
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    idx = sorted(random.Random(0).sample(range(len(ds)), n))
    docs = [(f"Question: {ds[i]['question']}\nAnswer:", [" " + t for t in ds[i]["choices"]["text"]],
             ds[i]["choices"]["label"].index(ds[i]["answerKey"])) for i in idx]
    return choice(M, docs)


def _hs(text):
    text = text.strip().replace(" [title]", ". ")
    return re.sub(r"\[.*?\]", "", text).replace("  ", " ")


def task_hellaswag(M, n=500):
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation")
    idx = sorted(random.Random(0).sample(range(len(ds)), n))
    docs = [(_hs(ds[i]["activity_label"] + ": " + ds[i]["ctx_a"] + " " + ds[i]["ctx_b"].capitalize()),
             [" " + _hs(e) for e in ds[i]["endings"]], int(ds[i]["label"])) for i in idx]
    return choice(M, docs)


def gsm_answer(text):
    boxed = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", text)
    s = boxed[-1] if boxed else text
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", s.replace("$", ""))
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def task_gsm8k(M, n=100, max_new=512):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    eos = eos_ids(M.tok)
    items = []
    for i in sorted(random.Random(0).sample(range(len(ds)), n)):
        g = M.generate(chat(M.tok, ds[i]["question"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."), max_new, eos)
        text = M.tok.decode(g, skip_special_tokens=True)
        gold = ds[i]["answer"].split("####")[-1].strip().replace(",", "")
        pred = gsm_answer(text)
        try:
            ok = pred is not None and abs(float(pred) - float(gold)) < 1e-6
        except ValueError:
            ok = False
        items.append(dict(idx=i, gold=gold, pred=pred, correct=ok, n=len(g), tokens=g))
    return items


def ifeval_utils():
    import importlib.util
    import types
    root = Path(importlib.util.find_spec("lm_eval").submodule_search_locations[0])
    for name, p in (("lm_eval", root), ("lm_eval.tasks", root / "tasks")):
        if name not in sys.modules:
            mod = types.ModuleType(name); mod.__path__ = [str(p)]; sys.modules[name] = mod
    from lm_eval.tasks.ifeval import utils
    return utils


def task_ifeval(M, n=100, max_new=640):
    from datasets import load_dataset
    utils = ifeval_utils()
    ds = load_dataset("google/IFEval", split="train")
    eos = eos_ids(M.tok)
    items = []
    for i in sorted(random.Random(0).sample(range(len(ds)), n)):
        d = ds[i]
        g = M.generate(chat(M.tok, d["prompt"]), max_new, eos)
        text = M.tok.decode(g, skip_special_tokens=True)
        s = utils.process_results(d, [text])
        items.append(dict(key=d["key"], strict=s["prompt_level_strict_acc"], loose=s["prompt_level_loose_acc"],
                          correct=s["prompt_level_strict_acc"], n=len(g), tokens=g))
    return items


def wikitext_ids(tok):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    return tok.encode("".join(ds["text"]), add_special_tokens=False)


def task_ppl(M, length=4096, docs=2):
    ids = wikitext_ids(M.tok)
    items = []
    for k in range(docs):
        w = ids[k * length:(k + 1) * length]
        nll, hit = M.nll(w)
        items.append(dict(window=k, nll=float(nll.mean()), agree=float(hit.float().mean()), hit=hit.int().tolist()))
    return items


def task_niah(M, length=8192, depths=(0.1, 0.5, 0.9)):
    hay = wikitext_ids(M.tok)[50000:50000 + length]
    eos = eos_ids(M.tok)
    items = []
    for j, dpt in enumerate(depths):
        rng = random.Random(j)
        num = str(rng.randrange(10 ** 6, 10 ** 7))
        city = CITIES[j % len(CITIES)]
        needle = M.tok.encode(f" The magic number for {city} is {num}. ", add_special_tokens=False)
        cut = int(len(hay) * dpt)
        doc = M.tok.decode(hay[:cut] + needle + hay[cut:])
        q = chat(M.tok, doc + f"\n\nWhat is the magic number for {city}? Answer with the number only.")
        g = M.generate(q, 12, eos)
        text = M.tok.decode(g, skip_special_tokens=True)
        items.append(dict(depth=dpt, number=num, text=text, correct=num in text, prompt_tokens=len(q)))
    return items


TASKS = dict(mmlu=task_mmlu, arc=task_arc, hellaswag=task_hellaswag, gsm8k=task_gsm8k, ifeval=task_ifeval, ppl=task_ppl, niah=task_niah)


def run(model, tasks):
    torch.backends.cuda.matmul.allow_tf32 = False
    M = Merged() if model == "merged" else HF("eager" if model == "eager" else "sdpa")
    d = OUT / model; d.mkdir(parents=True, exist_ok=True)
    for t in tasks:
        f = d / f"{t}.json"
        if f.exists():
            continue
        t0 = time.time()
        items = TASKS[t](M)
        f.write_text(json.dumps(dict(task=t, model=model, wall_s=time.time() - t0, items=items)))
        acc = np.mean([it["correct"] for it in items]) if "correct" in items[0] else np.mean([it["nll"] for it in items])
        print(f"{model} {t}: {acc:.3f} ({len(items)} items, {time.time() - t0:.0f} s)", flush=True)


def mcnemar(b01, b10):
    """Exact two-sided McNemar p-value from the discordant pairs."""
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, c - h, c + h


def table():
    rows = []
    for t in TASKS:
        base = OUT / "base" / f"{t}.json"
        if not base.exists():
            continue
        B = json.loads(base.read_text())["items"]
        line = f"{t:10s}"
        for mdl in ("base", "eager", "merged"):
            f = OUT / mdl / f"{t}.json"
            if not f.exists():
                line += f"{'-':>34s}"; continue
            X = json.loads(f.read_text())["items"]
            if t == "ppl":
                line += f"{'NLL %.4f, agree %.3f' % (np.mean([x['nll'] for x in X]), np.mean([x['agree'] for x in X])):>34s}"
                if mdl != "base":
                    same = np.mean([np.mean(np.array(x["hit"]) == np.array(b["hit"])) for x, b in zip(X, B)])
                    line += f" (greedy hit = base's {same:.4f})"
                continue
            k = sum(x["correct"] for x in X); p, lo, hi = wilson(k, len(X))
            s = f"{p:.3f} [{lo:.3f}, {hi:.3f}]"
            if mdl != "base":
                same = np.mean([x["correct"] == b["correct"] for x, b in zip(X, B)])
                b01 = sum(1 for x, b in zip(X, B) if b["correct"] and not x["correct"])
                b10 = sum(1 for x, b in zip(X, B) if x["correct"] and not b["correct"])
                s += f" same {same:.3f}, +{b10}/-{b01}, p={mcnemar(b01, b10):.2f}"
            line += f"{s:>34s}"
        rows.append(line)
    print(f"{'task':10s}{'base':>34s}{'eager (noise floor)':>34s}{'merged (fly inside)':>34s}")
    print("\n".join(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "table"])
    ap.add_argument("model", nargs="?", default="base", choices=["base", "eager", "merged"])
    ap.add_argument("--tasks", default=",".join(TASKS))
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.model, a.tasks.split(","))
    else:
        table()


if __name__ == "__main__":
    main()
