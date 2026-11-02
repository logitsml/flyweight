"""Brian2 reference: Shiu et al.'s own create_model, with PoissonInput swapped for the same
events delivered deterministically, so another simulator can be compared spike for spike.

    python scripts/checks/brian_ref.py --ms 300 --rate 100 --seed 0 --out results/ref_sugar.npz [--target numpy]

The swap keeps Brian2's timing: a SpikeGeneratorGroup fires in the thresholds slot and a
zero-delay Synapses(on_pre='v += f_poi*w_syn') adds to v in the synapses slot, where
PoissonInput adds too; the targets keep rfc = 0 as in model.poi.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external" / "Drosophila_brain_model"))

from flyweight.connectome import SUGAR_R, VERSIONS, load  # noqa: E402


def poisson_events(n_targets, steps, rate_hz, seed, dt=0.0001):
    rng = np.random.default_rng(seed)
    hit = rng.random((steps, n_targets)) < rate_hz * dt
    st, tg = np.nonzero(hit)
    return st, tg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", type=float, default=300)
    ap.add_argument("--rate", type=float, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", default="numpy")
    ap.add_argument("--version", default="630")
    ap.add_argument("--out", default=str(ROOT / "results" / "ref_sugar.npz"))
    a = ap.parse_args()

    import brian2 as b2
    from model import create_model, default_params

    b2.prefs.codegen.target = a.target
    params = dict(default_params)
    con = load(a.version)
    COMP, CONN = VERSIONS[a.version]
    exc = con.index(SUGAR_R)
    steps = int(round(a.ms * 10))
    st, tg = poisson_events(len(exc), steps, a.rate, a.seed)

    t0 = time.time()
    neu, syn, spk = create_model(str(COMP), str(CONN), params)
    gen = b2.SpikeGeneratorGroup(len(exc), tg, st * b2.defaultclock.dt)
    inp = b2.Synapses(gen, neu, on_pre="v += f_poi * w_syn", namespace=params, name="poisson_det")
    inp.connect(i=np.arange(len(exc)), j=np.array(exc))
    for i in exc:
        neu[i].rfc = 0 * b2.ms
    net = b2.Network(neu, syn, spk, gen, inp)
    t1 = time.time()
    net.run(a.ms * b2.ms)
    t2 = time.time()
    steps_out = np.round(np.asarray(spk.t[:] / b2.defaultclock.dt)).astype(np.int64)
    idx = np.asarray(spk.i[:], dtype=np.int64)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, step=steps_out, idx=idx, ev_step=st, ev_tgt=np.array(exc)[tg],
                        ms=a.ms, rate=a.rate, seed=a.seed)
    print(f"build {t1 - t0:.0f} s, run {t2 - t1:.0f} s ({a.target}); {len(idx)} spikes from "
          f"{len(np.unique(idx))} neurons -> {a.out}")


if __name__ == "__main__":
    main()
