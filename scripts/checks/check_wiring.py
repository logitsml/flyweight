"""Audit of which neurons the model uses and how its weights reach them, read back from the built
weights (CPU only; no forward pass).

    python scripts/checks/check_wiring.py

1. The neuron sets: every register's group and every readout role, with cell types and sides; no
   readout neuron is one a register drives; the senses are bilateral; the taste sets are exactly
   FlyWire's "sugar" (LB3c) and "bitter" (LB1a-d) labellar types; and the biology each set stands
   for, in the connectome's signed synapse counts: sugar GRNs excite MN9 and bitter GRNs inhibit it,
   the giant fiber synapses on the jump motor neurons, DNg02 drives the flight power motor neurons,
   each DNa02 projects to its own side, synapses on its side's turn pool and inhibits the other's,
   MDN's output goes mostly to the leg neuromeres and excites the backward pool, the antennal
   grooming descending neuron's goes mostly to the front-leg neuromere and excites the grooming pool.
2. The fly's layers, built as in a session (Qwen's answer bridge, the printed senses):
   - every register's DRIVE tokens write values the tonic rows read back as exactly their level;
   - each register's tonic rows reach exactly its group: gate = alpha K2 x register (minus the
     opposing register's relu for the steering pair, from layer 1's MLP), closed by its member's
     refractory flag; up = the member's fixed gain x STEP, read off the TICK flag;
   - each readout role's MOVE row reads exactly its members' refractory flags, in Hz per member;
   - the connectome product holds every synapse once, at count x 0.275 mV x 0.73, sign by channel;
   - each printed word reads exactly its sense role, at 1/8 per Hz above its threshold;
   - MOVE tokens are inert if fed.
3. The answer bridge: each command register's 2,560 coefficients are 64 x (Qwen's unembedding row of
   its word minus stop's), and layer 1's input norm is Qwen's final norm.
4. The control bridge's 48 Qwen neurons (L#N#), and where their summed direct vote lands.
"""

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from flyweight import fly_layers  # noqa: E402
from flyweight.connectome import P  # noqa: E402
from flyweight.merged import (ANSWER_REGS, ANSWER_WORDS, BALL1, CLS, EPS_HEAD, FLY_BASE, K_KEY, LAM, QH,  # noqa: E402
                            SITS, TXT, H, answer_refs, inhibitors)
from flyweight.sets import DRIVEN, ROLES, SENSES  # noqa: E402

FAIL = []


def check(ok, what):
    print(("  ok    " if ok else "  FAIL  ") + what, flush=True)
    if not ok:
        FAIL.append(what)


def csr(sl):
    """(rows, cols, values) of a SparseLinear's CSR weight."""
    w = sl.w
    crow, col, val = w.crow_indices().long().numpy(), w.col_indices().long().numpy(), w.values().double().numpy()
    return np.repeat(np.arange(len(crow) - 1), np.diff(crow)), col, val


def describe(A, idx):
    t = Counter(A.type[i] or "?" for i in idx).most_common(3)
    sd = Counter(A.side[i] or "?" for i in idx)
    more = len(set(A.type[i] for i in idx)) - len(t)
    return f"{len(idx):4d} neurons; sides {dict(sd)}; types {', '.join(f'{k} x{v}' for k, v in t)}{f' (+{more} more)' if more > 0 else ''}"


def main():
    from flyweight import setup
    con, A, world, groups, roles = setup.world_and_groups()
    names = [nm for _, nm in DRIVEN] + list(SENSES)
    print(f"1. Neuron sets (male CNS v1.0, {con.n:,} neurons, {len(con.post):,} connections)")
    for (cmd, nm), g in zip(DRIVEN + [(s, s) for s in SENSES], groups):
        print(f"  register {cmd:10s} {nm:8s} {describe(A, g)}")
    for nm, r in zip(ROLES, roles):
        print(f"  readout  {nm:19s} {describe(A, r)}")
    driven = set(np.concatenate(groups).tolist())
    body = set(np.concatenate([roles[ROLES.index(nm)] for nm in ROLES if nm not in SENSES]).tolist())
    check(not (body & driven), f"no motor readout neuron is driven by a register ({len(body & driven)} shared)")
    check(all(len(g) for g in groups) and all(len(r) for r in roles), "no empty group or role")
    for s_ in SENSES:
        g = groups[len(DRIVEN) + SENSES.index(s_)]
        sides = Counter(A.side[i] for i in g)
        check(sides.get("L", 0) > 0 and sides.get("R", 0) > 0, f"sense {s_} is bilateral: {dict(sides)}")
    for (cmd, nm), g in zip(DRIVEN, groups):
        want = {"DNa02_L": {"L"}, "DNa02_R": {"R"}}.get(nm)
        if want:
            check(set(A.side[i] for i in g) == want, f"{nm} holds only side {want}")
    from flyweight import annot
    fw = annot.load("783")
    label = {}
    for nm_, sc in zip(fw["name"], fw["sub_class"]):
        label.setdefault(nm_, set()).add(sc)
    for s_, want in (("sugar", "sugar"), ("bitter", "bitter")):
        g = groups[len(DRIVEN) + SENSES.index(s_)] if s_ in SENSES else []
        types = sorted({A.type[i] for i in g})
        check(all(label.get(x) == {want} for x in types), f"{s_} GRNs are FlyWire's {want!r} labellar types {types}")
    import scipy.sparse as sp
    n = con.n
    t = np.array(A.type, dtype=str); side = np.array(A.side)
    Wc = sp.csr_matrix((con.count.astype(np.float64), (con.post, con.pre)), shape=(n, n))  # signed synapse counts, post x pre
    def reach(src, k):
        v = np.zeros(n); v[src] = 1.0
        for _ in range(k):
            v = Wc @ v
        return v
    typ = lambda x: np.flatnonzero(t == x)
    grp = lambda nm_: groups[[nm for _, nm in DRIVEN].index(nm_)] if nm_ in [nm for _, nm in DRIVEN] else groups[len(DRIVEN) + SENSES.index(nm_)]
    mn9 = roles[ROLES.index("MN9")]
    s3, b3 = reach(grp("sugar"), 3)[mn9].sum() / len(grp("sugar")), reach(grp("bitter"), 3)[mn9].sum() / len(grp("bitter"))
    check(s3 > 0 > b3, f"sugar GRNs excite MN9 three synapses away ({s3:+.2g} per GRN), bitter GRNs inhibit it ({b3:+.2g})")
    from flyweight.malecns import GF_COUPLING
    tt = roles[ROLES.index("TTMn_L")].tolist() + roles[ROLES.index("TTMn_R")].tolist()
    gf_tt = float(Wc[tt][:, grp("DNp01")].sum())
    same = all(float(Wc[t_][:, [g_ for g_ in grp("DNp01") if A.side[g_] == A.side[t_]]].sum()) > GF_COUPLING for t_ in tt)
    check(same, f"each giant fiber (DNp01) drives its own side's jump motor neuron TTMn: {gf_tt - 2 * GF_COUPLING:.0f} chemical "
                f"synapses plus the electrical junctions ({GF_COUPLING} each)")
    pw = reach(grp("DNg02"), 2)[roles[ROLES.index("power")]].mean()
    check(pw > 0, f"DNg02 excites the flight power motor neurons DLMn/DVMn two synapses away ({pw:+.3g} per motor neuron)")
    for s_ in ("L", "R"):
        a = grp(f"DNa02_{s_}")
        d = np.asarray(abs(Wc[:, a]).sum(1)).ravel()
        own = d[side == s_].sum() / d.sum()
        mine, other = roles[ROLES.index(f"turn_{s_}")], roles[ROLES.index("turn_" + ("R" if s_ == "L" else "L"))]
        ipsi, cross = float(Wc[mine][:, a].sum()), float(Wc[other][:, a].sum())
        contra = reach(a, 2)[other].sum()
        check(own > 0.8 and ipsi > 10 * max(cross, 1.0) and contra < 0,
              f"DNa02_{s_}: {own:.0%} of its synapses on its own side; {ipsi:+.0f} synapses on turn_{s_} ({cross:+.0f} on the other pool), "
              f"which it inhibits two synapses away ({contra:+.3g})")
    nmr = np.array(A.neuromere)
    for name, legs, pool, what in (("MDN", ("T1", "T2", "T3"), "bwd", "the leg neuromeres"), ("DNg12_e", ("T1",), "groom", "the front-leg neuromere")):
        a = grp(name)
        d = np.asarray(abs(Wc[:, a]).sum(1)).ravel()  # each target's synapses from the whole group
        frac = d[np.isin(nmr, legs)].sum() / d.sum()
        exc = reach(a, 2)[roles[ROLES.index(pool)]].sum()
        check(frac > 0.5 and exc > 0, f"{name}: {frac:.0%} of its output synapses in {what}; excites the {pool} pool two synapses away ({exc:+.3g})")

    # the fly's layers as a session builds them (CPU), with the answer bridge's coefficients from Qwen's weights
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoTokenizer
    snap = Path(snapshot_download("Qwen/Qwen3.5-4B", local_files_only=True))
    tok = AutoTokenizer.from_pretrained(snap)
    Tq = {}
    for f in sorted(snap.glob("*.safetensors")):
        with safe_open(f, "pt") as fh:
            for k in fh.keys():  # noqa: SIM118
                if "mtp" in k:
                    continue
                if k.endswith(("embed_tokens.weight", "language_model.norm.weight")) or ".mlp.down_proj.weight" in k:
                    Tq[k] = fh.get_tensor(k)
    emb = next(v for k, v in Tq.items() if k.endswith("embed_tokens.weight"))
    gfin = next(v for k, v in Tq.items() if k.endswith("language_model.norm.weight")).float()
    am = json.loads((ROOT / "data" / "answer_map.json").read_text())
    iw = ANSWER_WORDS.index
    wid = [tok(" " + w).input_ids[0] for w in ANSWER_WORDS]
    U = emb[wid].double().numpy()
    refs = answer_refs(am)
    Cw = U[[iw(w) for w in ANSWER_REGS]] - U[[iw(r) for r in refs]]
    delta, theta = am["delta"], am["theta"]
    affine = {r: (fly_layers.LEVELS_M1 / delta[w], 1.0 - fly_layers.LEVELS_M1 * theta[w] / delta[w]) for r, w in enumerate(ANSWER_REGS)}
    inhibit = inhibitors(am)
    scale_q = math.sqrt(QH / H)
    eps_res = 1e-6 * QH / H
    qread = ((1.0 + gfin) * scale_q - 1.0).numpy()
    cls = dict(dim=CLS, k_key=K_KEY, lam=LAM, eps_head=EPS_HEAD, ball1=BALL1, qread=(qread, eps_res),
               answer={r: list(zip(range(QH), Cw[r])) for r in range(len(ANSWER_REGS))}, affine=affine, inhibit=inhibit)
    say = setup.say_spec(tok)
    W = con.count * P["w_syn"] * con.scale * 1000.0
    c = fly_layers.fly_components(con.n, con.post, con.pre, W, groups, roles, device="cpu", H=H, base=FLY_BASE, layer_offset=32, txt=TXT,
                             markers=[QH + 1 + g for g in range(len(SITS))] + [CLS], cls=cls, say=[(sl, terms, b) for sl, _, terms, b in say])
    L, (l1, l2, l3) = c["L"], c["layers"]
    print("2. The fly's layers")
    # DRIVE tokens -> register value -> the level the tonic rows read (float64 algebra of the stored constants)
    Er, Ec, Ev = c["E"].arrays()
    q1r, q1c, q1v = csr(l1.linear_attn.in_proj_qkv)
    o1r, o1c, o1v = csr(l1.linear_attn.out_proj)
    worst = 0.0  # the per-head norm divides by exactly ballast / 2 and its weight and gate multiply that back: reg = value x c_sig x c_out
    for g in range(L.G):
        h = fly_layers.NPH + g
        vrow = 2 + h * 4  # value channel 0 of head h (one key head of dim 1)
        c_sig = q1v[(q1r == vrow) & (q1c == L.drv_val.start + g)][0]
        c_out = o1v[(o1r == L.reg.start + g) & (o1c == h * 4)][0]
        al, be = affine.get(g, (1.0, 0.0))
        for lv in range(fly_layers.LEVELS):
            dv = Ev[(Er == L.tok_drv + g * fly_layers.LEVELS + lv) & (Ec == L.drv_val.start + g)]
            val = float(dv[0]) if len(dv) else 0.0
            reg = val * c_sig * c_out
            if g in affine and lv == 0:  # an answer register's level 0 writes 0, which reads as level 0
                worst = max(worst, abs(reg), max(0.0, be - 1.0))
            else:
                worst = max(worst, abs(al * reg + be - (1.0 + lv)))
    check(worst < 1e-9, f"DRIVE_g_l tokens: every register reads back 1 + l exactly through layer 1, an answer register's level 0 as 0 (max error {worst:.1e})")
    g1r, g1c, g1v = csr(l1.mlp.gate_proj)
    u1r, u1c, u1v = csr(l1.mlp.up_proj)
    d1r, d1c, d1v = csr(l1.mlp.down_proj)
    rect_reg = {}  # rect dim -> (register, offset): it holds relu(register - offset)
    ok = len(u1r) == len(d1r) == L.n_rect
    for k in range(L.n_rect):
        gk = dict(zip(g1c[g1r == k].tolist(), g1v[g1r == k].tolist()))
        p_ = next(c_ for c_ in gk if L.reg.start <= c_ < L.reg.start + L.G)
        ok &= gk.pop(p_) == fly_layers.KR and set(gk) <= {L.ONE} and u1c[u1r == k][0] == L.ONE and u1v[u1r == k][0] == 1.0 / fly_layers.KR
        ok &= d1r[d1c == k][0] == L.rect.start + k and d1v[d1c == k][0] == 1.0
        rect_reg[L.rect.start + k] = (p_ - L.reg.start, 0.0 - gk.get(L.ONE, 0.0) / fly_layers.KR)
    want = sorted({(p_, float(c_)) for t in inhibit.values() for p_, c_ in t})
    check(ok and sorted(rect_reg.values()) == want,
          f"layer 1's MLP: {', '.join(f'relu({names[p_]} - {c_:g})' for p_, c_ in sorted(rect_reg.values()))}, on fly tokens only")
    gr, gc, gv = csr(l2.mlp.gate_proj)
    ur, uc, uv = csr(l2.mlp.up_proj)
    dr, dc, dv_ = csr(l2.mlp.down_proj)
    down_of = np.full(int(dc.max()) + 1, -1)
    down_of[dc] = dr
    gate = {}
    for r_, c_, v_ in zip(gr, gc, gv):
        if r_ >= 4 * n:
            gate.setdefault(r_, {})[c_] = v_
    up = {}
    for r_, c_, v_ in zip(ur, uc, uv):
        if r_ >= 4 * n:
            up.setdefault(r_, {})[c_] = v_
    for g, members in enumerate(groups):
        al, be = affine.get(g, (1.0, 0.0))
        rows = sorted(r_ for r_, cols in gate.items() if L.reg.start + g in cols)
        tgt = sorted({int(down_of[r_]) - L.inp.start for r_ in rows})
        ok = tgt == sorted(members.tolist()) and len(rows) == 2 * len(members)
        for r_ in rows:
            m_ = int(down_of[r_]) - L.inp.start
            gc_ = dict(gate[r_])
            rds = [d for d in gc_ if d in rect_reg]  # each inhibitor's relu, at minus the same slope
            ok &= sorted(rect_reg[d] for d in rds) == sorted((p_, float(c_)) for p_, c_ in inhibit.get(g, ()))
            ok &= all(gc_.pop(d) == -al * fly_layers.K2 for d in rds)
            lo = be - gc_.get(L.ONE, 0.0) / fly_layers.K2
            ok &= set(gc_) == {L.reg.start + g, L.ONE, L.refr22.start + m_} or (set(gc_) == {L.reg.start + g, L.refr22.start + m_} and abs(lo - 1) < 1e-12)
            ok &= abs(gc_[L.reg.start + g] - al * fly_layers.K2) < 1e-9 and gc_[L.refr22.start + m_] == -fly_layers.BIGG and min(abs(lo - 1), abs(lo - 16)) < 1e-9
            ok &= set(up[r_]) == {L.is_tick} and abs(abs(up[r_][L.is_tick]) - fly_layers.STEP * fly_layers.member_gain(m_) / fly_layers.K2) < 1e-15
        minus = "".join(f" - relu({names[p_]} - {c_:g})" for p_, c_ in inhibit.get(g, ()))
        check(ok, f"register {names[g]:9s}: tonic reaches exactly its {len(members)} neurons{' (' + names[g] + minus + ')' if minus else ''}, "
                  f"closed by each one's refractory flag, read off the TICK flag")
    o2r, o2c, o2v = csr(l2.linear_attn.out_proj)
    Ua, Ub, Uw = c["U"].arrays()
    for r, members in enumerate(roles):
        heads = sorted((o2c[o2r == L.role.start + r] - n).tolist())
        rate = Uw[(Ua == L.move + r) & (Ub == L.rate.start + r)]
        check(heads == sorted(members.tolist()) and len(rate) == 1 and abs(rate[0] - 1.0 / (fly_layers.RFC * P["dt"] * len(members))) < 1e-9,
              f"readout {ROLES[r]:8s}: the MOVE row reads exactly its {len(members)} members' refractory flags, in Hz per member")
    Wf = np.asarray(W, np.float64).astype(np.float32).astype(np.float64)
    pos = (o2r >= L.dgp.start) & (o2r < L.dgp.start + n)
    neg = (o2r >= L.dgn.start) & (o2r < L.dgn.start + n)
    check(int(pos.sum()) == int((Wf > 0).sum()) and int(neg.sum()) == int((Wf < 0).sum()) and (o2c[pos | neg] < n).all(),
          f"connectome: {int(pos.sum()):,} excitatory and {int(neg.sum()):,} inhibitory connections, each once, read from the delayed spike heads")
    key_m = np.where(pos, o2r - L.dgp.start, o2r - L.dgn.start)[pos | neg] * n + o2c[pos | neg]
    val_m = np.where(pos, o2v, -o2v)[pos | neg]
    order = np.argsort(key_m)
    key_m, val_m = key_m[order], val_m[order]
    key_c = np.asarray(con.post, np.int64) * n + np.asarray(con.pre, np.int64)
    at = np.searchsorted(key_m, key_c)
    found = (at < len(key_m)) & (key_m[np.minimum(at, len(key_m) - 1)] == key_c)
    err = np.abs(val_m[np.minimum(at, len(key_m) - 1)] - Wf)
    check(found.all() and float(err.max()) == 0.0, f"connectome: every connection post <- pre at count x {P['w_syn'] * 1000:g} mV x {con.scale} (max error {float(err.max()):.1e} mV)")
    Sa, Sb, Sw = c["SAYW"].arrays()
    for j, (sl, tid, terms, bias) in enumerate(say):
        row = set(Sb[Sa == j].tolist())
        want = {L.is_say.start + sl} | {L.rate.start + r for r, _ in terms}
        okc = all(abs(Sw[(Sa == j) & (Sb == L.rate.start + r)][0] - ph / (fly_layers.RFC * P["dt"] * len(roles[r]))) < 1e-12 for r, ph in terms)
        okb = abs(Sw[(Sa == j) & (Sb == L.is_say.start + sl)][0] - (fly_layers.SAY_BIG + bias)) < 1e-9
        check(row == want and okc and okb, f"printed word {tok.decode([tid])!r:12s} (slot {sl}): reads {ROLES[terms[0][0]] + ' rate' if terms else 'its slot only'}")
    move_other = all(((Er == L.move + r) & (Ec == L.is_other)).any() for r in range(L.R))
    check(move_other, "MOVE tokens carry the class flag that makes every layer skip them")
    print("3. The answer bridge")
    for r, w in enumerate(ANSWER_REGS):
        vrow = 2 + (fly_layers.NPH + r) * 4
        sel = (q1r == vrow) & (q1c < QH)
        got = np.zeros(QH); got[q1c[sel]] = q1v[sel]
        e = np.abs(got - (LAM * Cw[r]).astype(np.float32)).max()
        minus = "".join(f" - relu(s_{ANSWER_REGS[p_]} - {c_:g})" for p_, c_ in inhibit.get(r, ()))
        check(e == 0.0, f"register {w:7s}: 2,560 coefficients = 64 x (U_{w} - U_{refs[r]}) (max error {e:.1e}); level = 15 x "
                        f"(s{minus} - {theta[w]:g}) / {delta[w]:g}")
    a1r, a1c, a1v = csr(l1.linear_attn.in_proj_a)
    want = {(fly_layers.NPH + r, CLS, 2e9) for r in range(len(ANSWER_REGS))} | {(fly_layers.NPH + r, TXT, -1e9) for r in range(len(ANSWER_REGS))}
    check(set(zip(a1r.tolist(), a1c.tolist(), a1v.tolist())) == want and float(l1.linear_attn.dt_bias.min()) == -200.0,
          "the answer registers forget at the report's end (decay exp(-softplus(2e9 CLS - 1e9 TXT - 200)): 0 there, 1 elsewhere) and only there")
    nw1 = l1.input_layernorm.weight.detach().double().numpy()[:QH]
    check(np.abs(nw1 - qread.astype(np.float32)).max() == 0.0 and l1.input_layernorm.eps == eps_res, "layer 1's input norm on Qwen's dims = Qwen's final norm (gains and eps)")
    check(l1.linear_attn.norm.variance_epsilon == EPS_HEAD, "layer 1's per-head norm eps = 2^42 (a ballast-free register is divided by exactly 2^21)")
    print("4. The control bridge's 48 Qwen neurons (data/drive_neurons.npz) and their summed direct vote on Qwen's move words")
    z = np.load(ROOT / "data" / "drive_neurons.npz")
    Ug = (emb[wid].float() * (1.0 + gfin)).double().numpy()
    Ug -= Ug.mean(0)
    for nm in SITS:
        pairs = z[nm][:, :2].astype(int)
        votes = np.zeros(len(ANSWER_WORDS))
        for l_, n_ in pairs:
            d = next(v for k, v in Tq.items() if f".layers.{l_}.mlp.down_proj.weight" in k)[:, n_].double().numpy()
            votes += Ug @ d
        print(f"  {nm:8s} {', '.join(f'L{l_}N{n_}' for l_, n_ in pairs)}; vote leans to {ANSWER_WORDS[int(np.argmax(votes))]!r}")
    print("PASS" if not FAIL else f"FAIL: {len(FAIL)}")


if __name__ == "__main__":
    main()
