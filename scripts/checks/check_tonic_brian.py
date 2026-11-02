"""The executor's tonic drive against Brian2 , spike for spike.

    python scripts/checks/check_tonic_brian.py [--ms 300] [--adapt]

Brian2 runs Shiu et al.'s equations, threshold, reset and synapses on the connectome (built from the
same arrays the executor uses), plus a per-neuron constant `tonic` added in the synapses slot after
the synaptic pathway, unless refractory: v = v + tonic * int(not_refractory). The register schedule
is check_fly_layers.py's (random levels on the command and sense groups, each member scaled by its fixed
gain), applied between Brian2 runs and to the executor's `tonic` at the same steps. Both run in
float64 on the CPU (Brian2's numpy target; the executor in its exact summation order).

--adapt: both run the adaptation variant (fly_layers.ADAPT). Brian2 holds a threshold rise `theta` that,
just before the thresholds, becomes a_th theta + (1 - a_th) kappa r, r = 1 if the neuron spiked in
the last 22 steps; the threshold is v > v_th + theta.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "external" / "Drosophila_brain_model"))

from flyweight import fly_layers  # noqa: E402
from flyweight.connectome import P  # noqa: E402
from flyweight.lif import FlyLIF  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", type=float, default=300)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--adapt", action="store_true")
    a = ap.parse_args()
    adapt = fly_layers.ADAPT if a.adapt else None
    torch.set_num_threads(a.threads)
    from check_fly_layers import schedule_for
    from flyweight import setup
    con, A, world, groups, roles = setup.world_and_groups()
    steps = int(a.ms * 10)
    rng = np.random.default_rng(0)
    sched = schedule_for(len(groups), steps, rng, hi=12, p_on=0.35, every=(100, 400))
    tonic_at = {}
    for s, lv in sched.items():
        ton = np.zeros(con.n)
        for g, members in enumerate(groups):
            ton[members] = lv[g] * fly_layers.STEP * fly_layers.member_gain(members) * 1e-3  # volts, as check_fly_layers gives the executor
        tonic_at[s] = ton
    # executor (float64, Brian2's summation order)
    t0 = time.time()
    ex = FlyLIF(con, exact=True, device="cpu", adapt=adapt)
    ex_sp = []
    for s in range(steps):
        if s in tonic_at:
            ex.tonic = torch.as_tensor(tonic_at[s][None], dtype=torch.float64)
        ex_sp.append(sorted(ex.step().tolist()))
    t_ex = time.time() - t0
    # Brian2
    import brian2 as b2
    from model import default_params
    b2.prefs.codegen.target = "numpy"
    b2.defaultclock.dt = P["dt"] * b2.second
    params = dict(default_params)
    eqs = params["eqs"] + "tonic : volt (constant)\n" + ("theta : volt\n" if adapt else "")
    if adapt:
        params["a_th"] = float(np.exp(-P["dt"] / (adapt["tau_ms"] * 1e-3)))
        params["kappa_th"] = adapt["kappa_mv"] * 1e-3 * b2.volt
    neu = b2.NeuronGroup(con.n, model=eqs, method="linear", threshold="v > v_th + theta" if adapt else params["eq_th"],
                         reset=params["eq_rst"], refractory="rfc", name="default_neurons", namespace=params)
    neu.v = params["v_0"]; neu.g = 0; neu.rfc = params["t_rfc"]; neu.tonic = 0 * b2.volt
    syn = b2.Synapses(neu, neu, "w : volt", on_pre="g += w", delay=params["t_dly"], name="default_synapses")
    syn.connect(i=con.pre, j=con.post)
    syn.w = con.weights(P["w_syn"]) * b2.volt  # count x (w_syn x scale), the executor's own array
    drive = neu.run_regularly("v = v + tonic * int(not_refractory)", when="synapses", order=1, name="tonic_drive")
    mon = b2.SpikeMonitor(neu)
    objs = [neu, syn, drive, mon]
    if adapt:
        neu.theta = 0 * b2.volt
        objs.append(neu.run_regularly("theta = a_th * theta + (1.0 - a_th) * kappa_th * int(t - lastspike < 22.5 * dt)",
                                      when="thresholds", order=-1, name="adapt"))
    net = b2.Network(*objs)
    t1 = time.time()
    marks = sorted(tonic_at) + [steps]
    done = 0
    for s0, s1 in zip(marks[:-1], marks[1:]):
        if s0 > done:
            net.run((s0 - done) * b2.defaultclock.dt); done = s0
        neu.tonic = tonic_at[s0] * b2.volt
        net.run((s1 - s0) * b2.defaultclock.dt); done = s1
    t_b2 = time.time() - t1
    st = np.round(np.asarray(mon.t[:] / b2.defaultclock.dt)).astype(np.int64)
    br = [[] for _ in range(steps)]
    for k, i in zip(st, np.asarray(mon.i[:], np.int64)):
        br[k].append(int(i))
    br = [sorted(x) for x in br]
    bad = [k for k in range(steps) if br[k] != ex_sp[k]]
    print(f"tonic drive, {con.n:,} neurons, {a.ms:.0f} ms, {len(sched)} register changes: executor {sum(map(len, ex_sp))} spikes, "
          f"Brian2 {sum(map(len, br))}; steps that differ: {len(bad)}" + (f" (first {bad[0]})" if bad else "")
          + f"; executor {t_ex:.0f} s, Brian2 {t_b2:.0f} s")
    print("PASS" if not bad else "FAIL")


if __name__ == "__main__":
    main()
