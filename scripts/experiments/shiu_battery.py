"""Shiu et al.'s figures on the fly inside Qwen (docs/design.md).

    python scripts/experiments/shiu_battery.py [--part poisson|inside|all] [--device cuda|cpu] [--trials 10] [--ms 1000]
    # -> results/shiu_battery.json (parts merge into it)

Three columns, the same experiments:
  FlyWire     Shiu's model on FlyWire v783 (his public release): Poisson input to the stimulated neurons
              (rate r, each event 250 x w_syn, no refractory period), the exact executor, `trials` runs
  male CNS    the same Poisson input on the male CNS at our operating point (KC->KC removed, counts
              x 0.73): what connectome and retuning change
  in Qwen     the male CNS inside Qwen3.5-4B (the merged model's fly layers): the stimulated neurons
              driven by a register (tonic input, the level whose rate on an unconnected neuron is
              nearest r); deterministic, one run
Experiments (bilateral stimulation; readouts are means over both sides):
  Fig. 1  sugar GRNs at 50, 100, 150 Hz and the top level -> MN9
  Fig. 3  sugar GRNs at 100 Hz + bitter GRNs at 50, 100 Hz and the top level -> MN9 (suppression)
  Fig. 4  water GRNs at 100 Hz and the top level -> MN9
  Fig. 5  Johnston's organ C/E neurons at 100 Hz and the top level -> the DNg12 family (antennal
          descending neurons; FlyWire's annotations name no aDN1/aDN2, so the family is read)
  Fig. 1F silencing: sugar at 100 Hz with each of the five cell types giving MN9 the most excitatory
          synapses in the male CNS silenced (all their outgoing synapses zeroed), matched in FlyWire by
          type name (male CNS flywireType)
Neurons are found by annotation in each connectome (FlyWire's cell_sub_class and cell_type; the
male CNS's own types and flywireType), since FlyWire v630 root ids (Shiu's lists) do not carry over.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from flyweight import connectome, fly_layers, malecns  # noqa: E402
from flyweight.annot import load as fw_load  # noqa: E402
from flyweight.connectome import P, Connectome  # noqa: E402
from flyweight.lif import FlyLIF  # noqa: E402
from flyweight.sets import SCALE  # noqa: E402

RATES = [50, 100, 150]


def level_rates():
    """Rate of an unconnected neuron (gain 1) under each register level: Shiu's LIF with tonic input."""
    a_m = math.exp(-P["dt"] / P["t_mbr"])
    out = []
    for lv in range(fly_layers.LEVELS):
        v, last, n = 0.0, -10 ** 9, 0
        for k in range(20000):
            if k - last >= 22:
                v = v * a_m + lv * fly_layers.STEP
                if v > 7.0:
                    v, last, n = 0.0, k, n + 1
        out.append(n / 2.0)
    return np.array(out)


def flywire_sets(con):
    f = pd.read_csv(ROOT / "external" / "flywire_annotations_v783.tsv", sep="\t", low_memory=False)
    lut = {int(r): i for i, r in enumerate(con.flyids.tolist())}
    idx = lambda m: sorted(lut[int(r)] for r in f.root_id[m] if int(r) in lut)
    ct = f.cell_type.fillna("").astype(str)
    sc = f.cell_sub_class.fillna("").astype(str)
    s = {"sugar": idx(sc == "sugar"), "bitter": idx(sc == "bitter"), "water": idx(sc == "water"),
         "JO-CE": idx(ct.str.startswith(("JO-C", "JO-E"))), "MN9": idx(ct == "CB0701"), "DNg12": idx(ct.str.startswith("DNg12"))}
    return s, ct, lut, f


def malecns_sets(con, A):
    fw = fw_load("783")
    tastes = {k: {t for t, c in zip(fw["name"], fw["sub_class"]) if c == k and t} for k in ("sugar", "bitter", "water")}
    s = {k: [i for i in range(con.n) if A.cls[i] == "gustatory" and (A.type[i] in ts or A.flywire[i] in ts)] for k, ts in tastes.items()}
    s["JO-CE"] = [i for i in range(con.n) if A.type[i].startswith(("JO-C", "JO-E"))]
    s["MN9"] = [i for i in range(con.n) if A.type[i] == "MN9"]
    s["DNg12"] = [i for i in range(con.n) if A.type[i].startswith("DNg12")]
    return s


def experiments(top_rate):
    E = [(f"sugar {r}", {"sugar": r}, "MN9") for r in RATES + [top_rate]]
    E += [(f"sugar 100 + bitter {r}", {"sugar": 100, "bitter": r}, "MN9") for r in [50, 100, top_rate]]
    E += [(f"water {r}", {"water": r}, "MN9") for r in [100, top_rate]]
    E += [(f"JO-CE {r}", {"JO-CE": r}, "DNg12") for r in [100, top_rate]]
    return E


def run_poisson(con, sets, stim, read, trials, steps, silence=(), device="cuda"):
    """Shiu's paradigm on the executor: Poisson events (rfc 0 targets), `trials` batch rows."""
    sim = FlyLIF(con, batch=trials, silence=silence, device=device)
    targets = sorted({i for k in stim for i in sets[k]})
    sim.set_poisson_targets(targets)
    rate = torch.zeros(con.n, dtype=torch.float64, device=device)
    for k, r in stim.items():
        rate[torch.as_tensor(sets[k], device=device)] = r
    tg = torch.as_tensor(targets, device=device)
    pr = (rate[tg] * P["dt"]).expand(trials, -1)
    off = (torch.arange(trials, device=device) * con.n)[:, None]
    gen = torch.Generator(device=device).manual_seed(0)
    cnt = torch.zeros(trials * con.n, dtype=torch.float64, device=device)
    for _ in range(steps):
        hit = torch.rand(pr.shape, device=device, generator=gen, dtype=torch.float64) < pr
        ev = (off + tg[None])[hit]
        sp = sim.step(ev)
        if len(sp):
            cnt.index_add_(0, sp, torch.ones(len(sp), dtype=torch.float64, device=device))
    hz = cnt.view(trials, con.n) / (steps * P["dt"])
    res = {nm: (float(hz[:, sets[nm]].mean()), float(hz[:, sets[nm]].mean(1).std())) for nm in [read] + list(stim)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--ms", type=float, default=1000)
    ap.add_argument("--part", default="all", choices=["poisson", "inside", "all"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    path = ROOT / "results" / "shiu_battery.json"
    path.parent.mkdir(exist_ok=True)
    steps = int(a.ms * 10)
    lr = level_rates()
    top_lv = fly_layers.LEVELS_M1
    top_rate = int(round(lr[top_lv]))
    to_level = lambda r: int(np.argmin(np.abs(lr - r)))
    print(f"register levels -> rate of an unconnected neuron: " + ", ".join(f"{lv}: {lr[lv]:.0f} Hz" for lv in range(4, 16)), flush=True)
    E = experiments(top_rate)
    out = json.loads(path.read_text()) if path.exists() else {}
    out.update({"level_rates": lr.tolist(), "experiments": [e[0] for e in E]})
    out.setdefault("columns", {})
    # male CNS (operating point) and its sets
    mc = malecns.load(kc_kc=False); mc.scale = SCALE
    A = malecns.annotations(mc)
    ms = malecns_sets(mc, A)
    # the five types with the most excitatory synapses onto MN9
    mn9 = set(ms["MN9"])
    not_gust = np.asarray(A.cls) != "gustatory"
    sel = np.isin(mc.post, list(mn9)) & (mc.count > 0) & not_gust[mc.pre]
    tcount = pd.Series(mc.count[sel]).groupby(pd.Series([A.type[i] or f"#{i}" for i in mc.pre[sel]])).sum().sort_values(ascending=False)
    tcount = tcount[[not any(s in t for s in ("JO-", "MN9")) for t in tcount.index]]
    top_types = [t for t in tcount.index if not t.startswith("#")][:5]
    sil_mc = {t: [i for i in range(mc.n) if A.type[i] == t] for t in top_types}
    out["silenced_types"] = {t: dict(synapses_onto_MN9=int(tcount[t]), n=len(sil_mc[t]), flywire=sorted({A.flywire[i] for i in sil_mc[t]})) for t in top_types}
    print("male CNS: top excitatory input types to MN9: " + ", ".join(f"{t} ({int(tcount[t])} syn, {len(sil_mc[t])} cells)" for t in top_types), flush=True)
    print("male CNS sets: " + ", ".join(f"{k} {len(v)}" for k, v in ms.items()), flush=True)
    # FlyWire v783
    fw = connectome.load("783")
    fs, fct, flut, fdf = flywire_sets(fw)
    sil_fw = {}
    for t in top_types:
        names = {A.flywire[i] for i in sil_mc[t] if A.flywire[i]}
        sil_fw[t] = sorted(flut[int(r)] for r, c in zip(fdf.root_id, fct) if c in names and int(r) in flut)
    print("FlyWire sets: " + ", ".join(f"{k} {len(v)}" for k, v in fs.items()) + "; silenced types matched: "
          + ", ".join(f"{t} {len(v)}" for t, v in sil_fw.items()), flush=True)
    t0 = time.time()
    for col, con, sets, sil in ((("FlyWire", fw, fs, sil_fw), ("male CNS", mc, ms, sil_mc)) if a.part != "inside" else ()):
        res = {}
        for nm, stim, read in E:
            if any(len(sets[k]) == 0 for k in list(stim) + [read]):
                continue
            res[nm] = run_poisson(con, sets, stim, read, a.trials, steps, device=a.device)
        for t, ids in sil.items():
            if ids and sets["sugar"] and sets["MN9"]:
                res[f"sugar 100, {t} silenced"] = run_poisson(con, sets, {"sugar": 100}, "MN9", a.trials, steps, silence=ids, device=a.device)
        out["columns"][col] = res
        path.write_text(json.dumps(out, indent=1))
        print(f"{col} ({time.time() - t0:.0f} s): " + "; ".join(f"{k}: {v[list(v)[0]][0]:.1f} Hz" for k, v in res.items()), flush=True)
        torch.cuda.empty_cache()
    # inside Qwen: the merged model's fly layers, one batch row per experiment (registers, tonic input)
    from flyweight import setup
    from flyweight.merged import QwenFlyModel, QwenFlyRunner
    keys = ["sugar", "bitter", "water", "JO-CE"]
    res = {}

    def run_inside(con_v, rows, label):
        tok, qwen = setup.load_qwen()
        m = QwenFlyModel(qwen, con_v, [np.asarray(ms[k]) for k in keys], [np.asarray(ms["MN9"])])
        r = QwenFlyRunner(m, batch=len(rows)); r.start_fly()
        for b, (nm, stim, read) in enumerate(rows):
            r.flybrain.set_levels({keys.index(k): to_level(v) for k, v in stim.items()}, b)
        cnt = np.zeros((len(rows), con_v.n))
        for _ in range(steps):
            for b, sp in enumerate(r.flybrain.step()):
                np.add.at(cnt[b], sp, 1)
        hz = cnt / (steps * P["dt"])
        for b, (nm, stim, read) in enumerate(rows):
            res[nm] = {k: (float(hz[b, ms[k]].mean()), 0.0) for k in [read] + list(stim)}
            res[nm]["levels"] = {k: to_level(v) for k, v in stim.items()}
        print(f"in Qwen, {label} ({time.time() - t0:.0f} s): block check {r.flybrain.mismatch}/{r.flybrain.swaps}", flush=True)
        del m, r, qwen; torch.cuda.empty_cache()
    if a.part != "poisson":
        run_inside(mc, E, "experiments")
        for t, ids in sil_mc.items():
            keep = ~np.isin(mc.pre, ids)
            cv = Connectome(flyids=mc.flyids, pre=mc.pre[keep], post=mc.post[keep], count=mc.count[keep]); cv.scale = SCALE
            run_inside(cv, [(f"sugar 100, {t} silenced", {"sugar": 100}, "MN9")], f"{t} silenced")
        out["columns"]["in Qwen"] = res
    out = {**json.loads(path.read_text()), **out} if path.exists() else out
    path.write_text(json.dumps(out, indent=1))
    print("\nexperiment".ljust(36) + "".join(c.rjust(16) for c in out["columns"]))
    names = list(dict.fromkeys(k for c in out["columns"].values() for k in c))
    for nm in names:
        row = nm.ljust(36)
        for c in out["columns"].values():
            v = c.get(nm)
            row += (f"{v[list(v)[0]][0]:.1f} Hz".rjust(16) if v else "-".rjust(16))
        print(row)
    print("saved results/shiu_battery.json")


if __name__ == "__main__":
    main()
