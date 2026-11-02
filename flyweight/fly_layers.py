"""The fly's whole CNS as three stock Qwen3.5 decoder layers (docs/design.md).

Every neuron is a vocabulary token, a spike is a greedy token, a 0.1 ms step is a TICK token.
- Layer 1: the phase register (which TICK came last) and the drive registers. Every input, command
  or sense, is a register written by a DRIVE_g_l token (l = 0..15) or, inside Qwen, by Qwen's own
  neurons (flyweight/merged.py); there are no input events.
- Layer 2: a 23-slot spike ring keyed by phase. One read type looks 19 slots back (the 1.8 ms
  synaptic delay) and its output projection is the connectome; the other reads the last 22 slots
  (the 2.2 ms refractory period). Its MLP adds synaptic input and, at every TICK, each register's
  tonic input (level x STEP mV x the neuron's gain) to every member that is not refractory.
- Layer 3: each neuron's membrane (two decay modes, + and - parts as channels), decayed by the
  delta rule's beta and reset by the neuron's own spike token; and the motor readout, each role's
  refractory flags integrated over 30 ms. The LM head's spike logits are membrane potentials
  (greedy decoding is the threshold) and its MOVE rows are the roles' rates in Hz.
- Adaptation (a named variant, off by default; Shiu's model has none): each neuron's threshold rises
  by kappa x the fraction of recent time it spent refractory, filtered with time constant tau (an
  adaptive threshold). One more layer-3 head per neuron integrates the neuron's own refractory flag,
  as the motor readout integrates its roles, and writes minus it into the neuron's membrane dim, which
  only the spike logit reads (lif.FlyLIF: the same rule).
- Exact normalisation: fly tokens carry 2^33 in 6 ballast dims of a 6 x 4^9-wide residual, so every
  residual RMSNorm on a fly token divides by exactly 2^24, which the weights fold in. Layer 2's heads
  hold only 0 or 32 (spike indicators): one value channel, and the per-head RMSNorm returns 0 or 1
  (1 to within 6e-8: the delta rule scales queries by 1/sqrt(23)). Layers 1 and 3 hold continuous
  values: four channels with a power-of-two ballast, so the per-head RMSNorm divides by exactly half
  of it.

Constants are powers of two where they scale a signal (IND = KC = ZG = 32: silu(32) == 32 in fp32;
KAPPA = 1024; STEP = 2^-6 mV).
"""

import math

import numpy as np
import torch
from torch import nn

from .sparse import COO, SparseEmbedding, SparseLinear
from .connectome import P

NPH = 23  # ring slots
DELAY = 19  # TICK_t reads slot t-19: spikes that arrive at step t-1 (Brian2: 1.8 ms delay)
RFC = 22  # 2.2 ms refractory period
M_BALLAST = 6
H_FLY = M_BALLAST * 4 ** 9  # 1,572,864: M x 2^66 / H = 2^48, an exact power of four
B0 = 2.0 ** 33  # residual ballast, in each of the M ballast dims
B1 = 2.0 ** 30  # per-head ballast (layers 1 and 3)
IND = 32.0  # indicator value in a head
KC = 32.0  # one-hot keys and queries
ZG = 32.0  # output gate (silu(32) == 32)
BIG = 104.0  # sigmoid(+-104) == 1 / 0 in fp32
GATE = 256.0
MEM_S = 1024.0  # membrane value scale: the convolution's SiLU passes MEM_S x a/(1-a) x input unchanged above 17,
# which every synaptic input and any drive above 1/150 of a level reaches
B3 = 2.0 ** 40  # layer 3's ballast: the per-head norm stays exact while MEM_S x |v| < 2^28 (|v| < 2^18 mV)
KAPPA = 1024.0  # spike logits per mV
LEVELS = 16
LEVELS_M1 = LEVELS - 1
STEP = 2.0 ** -6  # mV of tonic input per drive level per 0.1 ms (level 13 ~ 165 Hz on an unconnected neuron)
TAU_RATE = 30.0  # ms, the motor readout's integration time
ADAPT = dict(tau_ms=200.0, kappa_mv=16.0)  # the adaptation variant: a neuron firing at 100 Hz (refractory 22% of the time) has its threshold ~3.5 mV up
MOVE_OFF = 2.0 ** 14  # MOVE logits sit this far below zero, never decoded; the host adds it back
SAY_BIG = 2.0 ** 16  # a printed word's logit at its SAY slot, above every fly row (text rows sit at -2^13 on fly tokens)


def _route_fly_layers_to_recurrent():
    """HF runs a block of several tokens through its chunked delta rule, whose rounding moves a
    membrane by ~1e-3 mV; its recurrent rule (the one HF uses for cached decoding: flash-linear-
    attention's fused recurrent kernel when installed, else torch_recurrent_gated_delta_rule) is the
    same computation done step by step, so a block and single tokens give the same numbers. The
    fly's layers are recognised by their value-head size (<= 4; Qwen3.5's own heads are 128 wide) and
    always take it; Qwen's layers keep the chunked path. Transformers looks the rule up in its module
    at every call, so replacing the module's function routes every layer."""
    import inspect

    import transformers.models.qwen3_5.modeling_qwen3_5 as M
    if getattr(M, "_fly2_routed", False):
        return
    chunk, recurrent = M.torch_chunk_gated_delta_rule, M.torch_recurrent_gated_delta_rule

    def call(rule, query, key, value, g, beta, kw):
        """HF's rule on the GPU; on the CPU the torch reference under HF's kernel wrapper (the fused
        kernels need CUDA), given only the arguments it takes."""
        if not query.is_cuda:
            rule = getattr(rule, "__wrapped__", rule)
            ok = inspect.signature(rule).parameters
            kw = {k: v for k, v in kw.items() if k in ok}
        return rule(query, key, value, g=g, beta=beta, **kw)

    def dispatch(query, key, value, g, beta, **kw):
        if value.shape[-1] <= 4 or query.shape[2] > 20000:
            kw.pop("chunk_size", None)
            return call(recurrent, query, key, value, g, beta, kw)
        return call(chunk, query, key, value, g, beta, kw)

    M.torch_chunk_gated_delta_rule = dispatch
    M.torch_recurrent_gated_delta_rule = lambda query, key, value, g, beta, **kw: call(recurrent, query, key, value, g, beta, kw)
    M._fly2_routed = M._fly_routed = True


_route_fly_layers_to_recurrent()


class FlyTokenNorm:
    """Residual RMSNorms on fly tokens. A fly token's ballast makes the norm divide by exactly 2^24
    (Exact normalisation, above), so its output is (x 2^-24)(1 + w), dim by dim, the reference's own two
    products without its squares, mean and rsqrt. Used only inside forwards whose every token is a fly
    token (FlyModel, QwenFlyModel.fly); the full forward keeps the reference, and check_merged compares
    them."""

    on = False

    class Mode:
        def __enter__(self):
            FlyTokenNorm.on = True

        def __exit__(self, *exc):
            FlyTokenNorm.on = False

    @staticmethod
    def install(norm):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

        class Norm(type(norm)):
            def forward(self, x):
                if not FlyTokenNorm.on:
                    return super().forward(x)
                if x.shape[-1] != self.weight.shape[0]:
                    return super().forward(x)
                return (x.float() * 2.0 ** -24) * (1.0 + self.weight.float())

        assert isinstance(norm, Qwen3_5RMSNorm)
        norm.__class__ = Norm


def mode_constants(p=P):
    dt, tm, ts = p["dt"], p["t_mbr"], p["tau"]
    a_m, a_s = math.exp(-dt / tm), math.exp(-dt / ts)
    return a_m, a_s, ts / (ts - tm)


class Layout:
    """Residual dims (relative to `base`) and vocabulary of the fly's layers."""

    def __init__(self, n, groups, roles, H=H_FLY, base=0, n_say=0, n_rect=0):
        self.n, self.G, self.R, self.H, self.base, self.S, self.n_rect = n, len(groups), len(roles), H, base, n_say, n_rect
        d = base

        def take(k):
            nonlocal d
            s = slice(d, d + k)
            d += k
            return s

        self.ball = take(M_BALLAST)
        self.ONE = self.ball.start
        self.f_tick = take(NPH)
        self.f_spk = take(n)
        self.is_tick, self.is_init, self.is_spk, self.is_other = (take(1).start for _ in range(4))
        self.is_say = take(n_say)  # SAY_k: the fly prints a word for slot k (merged model)
        self.f_drv = take(self.G)
        self.drv_val = take(self.G)
        self.phase = take(NPH)  # L1
        self.reg = take(self.G)  # L1: 1 + level
        self.rect = take(n_rect)  # L1 MLP: relu(register - offset) for each inhibition (cls["inhibit"])
        self.dgp = take(n)  # L2: synaptic input, mV
        self.dgn = take(n)
        self.refr22 = take(n)
        self.role = take(self.R)  # L2: role members that spiked in the last 22 steps
        self.inp = take(4 * n)  # L2 MLP: gated input per membrane channel (m+, m-, s+, s-), mV
        self.vdev = take(n)  # L3: mV above rest
        self.rate = take(self.R)  # L3: leaky role spike counts
        self.end = d
        assert d <= H, (d, H)
        self.tok_init = 0
        self.tok_tick = 1
        self.tok_spk = 1 + NPH
        self.tok_pad = 1 + NPH + n
        self.tok_drv = 2 + NPH + n
        self.move = 2 + NPH + n + self.G * LEVELS  # MOVE rows (read, never decoded)
        self.tok_say = self.move + self.R  # SAY_k tokens (fed by the host; write nothing)
        self.vocab = self.tok_say + n_say
        self.levels = LEVELS


def build_fly_layers(n, W_rows, W_cols, W_vals_mV, groups, roles, p=P, device="cuda", adapt=None):
    """The fly on its own: a stock Qwen3_5ForCausalLM with the three fly layers.
    groups: neuron-index arrays, one per register (commands, then senses). roles: neuron-index
    arrays for the motor readout. W as COO: W[post, pre] in mV."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextRotaryEmbedding
    c = fly_components(n, W_rows, W_cols, W_vals_mV, groups, roles, p=p, device=device, adapt=adapt)
    L, dev = c["L"], device
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(c["cfgs"][0])
    m = model.model
    m.layers = nn.ModuleList(c["layers"])
    for layer in c["layers"]:
        FlyTokenNorm.install(layer.input_layernorm); FlyTokenNorm.install(layer.post_attention_layernorm)
    m.embed_tokens = SparseEmbedding(*c["E"].arrays(), L.vocab, L.H, dev)
    with torch.device(dev):
        m.rotary_emb = Qwen3_5TextRotaryEmbedding(c["cfgs"][0])
    m.norm.weight = nn.Parameter(c["nw"].clone().to(dev), requires_grad=False)
    FlyTokenNorm.install(m.norm)
    model.lm_head = SparseLinear(*c["U"].arrays(), L.vocab, L.H, dev)
    left = [nm for nm, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    assert not left, left
    return model.eval(), L


K2 = 1024.0  # tonic rows: silu(K2 x) / K2 == relu(x) for every level step (exact on integer levels)
KR = 1024.0  # layer 1's relu rows: silu(KR x) / KR == relu(x) for |x| > 1/60


def member_gain(idx):
    """Each driven neuron's fixed share of its register's drive: (16 + k) / 16, k in -3..3, from a hash
    of its index (exact in fp32, so the model and the executor add the same current). A population
    driven by one register then fires at a spread of rates instead of in lockstep."""
    idx = np.asarray(idx, np.int64)
    return (16.0 + ((idx * 2654435761) >> 7) % 7 - 3) / 16.0
BIGG = 2.0 ** 60  # tonic rows: closes a refractory member's row for any register below 2^50 levels (x 0 exactly otherwise)


def fly_components(n, W_rows, W_cols, W_vals_mV, groups, roles, p=P, device="cuda", H=H_FLY, base=0,
                    layer_offset=0, txt=None, K_TXT=1e9, markers=(), cls=None, say=None, adapt=None):
    """The three fly decoder layers, plus the fly's embedding rows (E), LM head rows (U) and
    residual-norm weights (nw), over a residual of width H whose fly dims start at `base`.
    txt: index of the text-token marker dim when the fly sits inside Qwen: every fly head's
    write strength and every fly LM row get -K_TXT x TXT, so on a text token the fly writes
    nothing and is never decoded (docs/design.md, One stack).
    markers: further text-side dims the fly reads (normalised like TXT).
    cls: Qwen's neurons drive registers (the control bridge): dict(dim=the report-end marker dim,
    regs={register: (TXT coefficient, [(readout dim, coefficient), ...])}, c_b=reference scale,
    k_key=key scale). At the marker the register is written with channel 0 = the linear readout
    and channel 3 = c_b x TXT; the per-head RMSNorm later returns 2^30 x ch0 / ch3, free of the
    residual's scale at that token.
    cls["answer"]: Qwen's own answer drives registers instead: {register: [(dim, coef)]}
    reading Qwen's final hidden state through layer 1's input norm (cls["qread"]: its weights on
    Qwen's dims and its eps), times cls["lam"]. On a text token the ballast is 0, so layer 1's
    per-head norm divides by sqrt(cls["eps_head"]) exactly (2^42: 2^21) and the register holds the
    readout; layer 1's ballast is cls["ball1"] (2^40), so DRIVE tokens (ballast heads) still write
    exactly and eps 2^42 stays below half an ulp of the ballast heads' mean.
    cls["affine"]: {register: (alpha, beta)}: the tonic rows read alpha x reg + beta as 1 + level,
    and that register's DRIVE tokens write reg = (1 + level - beta) / alpha (exact for dyadic alpha).
    cls["inhibit"]: {register: [(other, offset c)]}: inhibition. Layer 1's MLP writes relu(other - c) and
    the register's tonic rows read alpha x (reg - sum relu(other - c)) + beta. Two registers inhibiting
    each other at c = 0 drive at most one, by its margin over the other.
    say: the fly's printed words, [(slot, [(role, per_hz)], bias)]: the returned SAYW
    rows (one per word) give 2^16 at the word's own SAY slot + per_hz x the role's rate + bias there.
    adapt: the adaptation variant, dict(tau_ms, kappa_mv) (ADAPT), or None (Shiu's model)."""
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

    groups = [np.asarray(g, np.int64) for g in groups]
    roles = [np.asarray(r, np.int64) for r in roles]
    n_say = 1 + max(sl for sl, _, _ in say) if say else 0
    inhibit = (cls or {}).get("inhibit", {})
    rect_of = {rc: k for k, rc in enumerate(sorted({(ri, float(c)) for t in inhibit.values() for ri, c in t}))}  # rect dim k holds relu(register ri - c)
    L = Layout(n, groups, roles, H=H, base=base, n_say=n_say, n_rect=len(rect_of))
    answer = (cls or {}).get("answer", {})
    affine = (cls or {}).get("affine", {})
    H, G, R = L.H, L.G, L.R
    n_drv = sum(len(g) for g in groups)
    IM = 4 * n + 2 * n_drv
    shapes = [(1, 1, NPH + G, 4), (2, NPH, 2 * n, 1), (1, 1, 2 * n + R + (n if adapt else 0), 4)]  # (key heads, key dim, value heads, value dim)
    base_cfg = dict(vocab_size=L.vocab, hidden_size=H, intermediate_size=IM, num_hidden_layers=layer_offset + 3,
                    layer_types=["linear_attention"] * (layer_offset + 3), linear_conv_kernel_dim=4, num_attention_heads=1,
                    num_key_value_heads=1, head_dim=8, rms_norm_eps=1e-6, tie_word_embeddings=False,
                    max_position_embeddings=1 << 22, hidden_act="silu", attn_output_gate=True, mtp_num_hidden_layers=0)
    cfgs = [Qwen3_5TextConfig(**base_cfg, linear_num_key_heads=nk, linear_key_head_dim=dk, linear_num_value_heads=nh,
                              linear_value_head_dim=dv) for nk, dk, nh, dv in shapes]
    with torch.device("meta"):
        layers = [Qwen3_5DecoderLayer(c, layer_offset + li) for li, c in enumerate(cfgs)]
    a_m, a_s, A = mode_constants(p)
    a_r = math.exp(-p["dt"] / (TAU_RATE * 1e-3))
    j, ph = np.arange(n), np.arange(NPH)
    logit = lambda x: math.log(x / (1.0 - x))

    # embeddings
    E = COO((L.vocab, H))
    for b in range(M_BALLAST):
        E.add(np.arange(L.vocab), L.ball.start + b, B0)
    E.add(L.tok_init, L.is_init, 1.0)
    E.add(L.tok_tick + ph, L.f_tick.start + ph, 1.0); E.add(L.tok_tick + ph, L.is_tick, 1.0)
    E.add(L.tok_spk + j, L.f_spk.start + j, 1.0); E.add(L.tok_spk + j, L.is_spk, 1.0)
    E.add(L.tok_pad, L.is_other, 1.0)
    lv = np.arange(LEVELS)
    for g in range(G):
        al, be = affine.get(g, (1.0, 0.0))
        unit = 1.0  # register value per unit of drv_val (an answer register's output coefficient is not 1 / IND)
        if g in answer:
            unit = IND * math.sqrt(cls["eps_head"]) / (cls["ball1"] / 2 * cls["lam"])
        val = (1.0 + lv - be) / al / unit
        if g in answer:  # level 0 writes 0, the bottom of the register's dead zone: it reads as level 0 and inhibits nothing
            val = np.where(lv == 0, 0.0, val)
        E.add(L.tok_drv + g * LEVELS + lv, L.f_drv.start + g, 1.0)
        E.add(L.tok_drv + g * LEVELS + lv, L.drv_val.start + g, val)
        E.add(L.tok_drv + g * LEVELS + lv, L.is_other, 1.0)
    for k in range(n_say):
        E.add(L.tok_say + k, L.is_other, 1.0); E.add(L.tok_say + k, L.is_say.start + k, 1.0)
    E.add(L.move + np.arange(R), L.is_other, 1.0)  # MOVE rows are read, never fed; if fed they write nothing

    class Gdn:
        def __init__(self, li, ball=B1):
            self.nk, self.dk, self.nh, self.dv, self.ball = *shapes[li], ball
            self.qkv = COO((2 * self.nk * self.dk + self.nh * self.dv, H))
            self.b = COO((self.nh, H))
            self.a = COO((self.nh, H))  # decay input: 0 everywhere but the answer registers at the report's end
            self.o = COO((H, self.nh * self.dv))

        def q(self, kh, d): return kh * self.dk + d
        def k(self, kh, d): return self.nk * self.dk + kh * self.dk + d
        def v(self, h, c): return 2 * self.nk * self.dk + h * self.dv + c

        def head(self, h, signal, beta, out, base=None, signal2=None, ref=None):
            """Value channel 0 = signal (channel 1 = signal2), channel dv-1 = ballast when dv > 1;
            out reads channel 0 (minus channel 1). beta logit = base (default -BIG) + beta."""
            h = np.asarray(h)
            for dim, c in signal:
                self.qkv.add(self.v(h, 0), dim, c)
            for dim, c in (signal2 or []):
                self.qkv.add(self.v(h, 1), dim, c)
            if self.dv > 1:
                self.qkv.add(self.v(h, self.dv - 1), L.ONE, self.ball)
                for dim, c in (ref or []):  # extra reference terms on the ballast channel (text-side writes)
                    self.qkv.add(self.v(h, self.dv - 1), dim, c)
            for dim, c in (base if base is not None else [(L.ONE, -BIG)]) + beta:
                self.b.add(h, dim, c)
            if txt is not None:  # inside Qwen: never write on a text token
                self.b.add(h, txt, -K_TXT)
            for dim, c in out:
                self.o.add(dim, h * self.dv, c)
                if signal2:
                    self.o.add(dim, h * self.dv + 1, -np.asarray(c))

        def const_key(self):
            self.qkv.add(self.q(0, 0), L.ONE, KC); self.qkv.add(self.k(0, 0), L.ONE, KC)

    # layer 1: phase register and the drive registers (4 channels: the per-head norm divides by exactly ball / 2)
    g1 = Gdn(0, cls["ball1"] if answer else B1)
    g1.const_key()
    if cls is not None:  # a valid key on text tokens too (dk = 1: any positive key normalises to 1)
        g1.qkv.add(g1.k(0, 0), txt, cls["k_key"])
    lam = (cls or {}).get("lam", 1.0)
    # an answer register is read back x (ball/2 / ZG) x ZG / sqrt(eps_head) per unit written at the report's end
    out_answer = math.sqrt(cls["eps_head"]) / (g1.ball / 2 * lam) if answer else None
    g1.head(ph, [(L.f_tick.start + ph, IND)], [(L.is_tick, 2 * BIG)], [(L.phase.start + ph, 1.0 / IND)])
    for g in range(G):
        sig, beta, ref = [(L.drv_val.start + g, IND)], [(L.f_drv.start + g, 2 * BIG)], None
        out_c = 1.0 / IND
        if cls is not None and g in cls.get("regs", {}):  # written by Qwen's neurons at the report's end
            tc, terms = cls["regs"][g]
            sig = sig + [(txt, tc)] + list(terms)
            beta = beta + [(cls["dim"], 2 * K_TXT)]
            ref = [(txt, cls["c_b"])]
        if g in answer:  # Qwen's answer: lam x readout; the register then holds the readout itself
            sig = sig + [(d, lam * c_) for d, c_ in answer[g]]
            beta = beta + [(cls["dim"], 2 * K_TXT)]
            out_c = out_answer
            # and it forgets first (decay exp(-softplus(a - 200)) = 0 there, 1 on every other token), so the write is
            # exact whatever the register held (a DRIVE token's value is ~2^18 x larger inside the head)
            g1.a.add(NPH + g, cls["dim"], 2 * K_TXT); g1.a.add(NPH + g, txt, -K_TXT)
        g1.head(NPH + g, sig, beta, [(L.reg.start + g, out_c)], ref=ref)
    # layer 1 MLP: relu of each inhibiting register above its offset, on fly tokens only (up reads the ballast, 0 on text)
    G1, U1, D1 = COO((IM, H)), COO((IM, H)), COO((H, IM))
    for (ri, c), k in rect_of.items():
        G1.add(k, L.reg.start + ri, KR)
        if c:
            G1.add(k, L.ONE, -c * KR)
        U1.add(k, L.ONE, 1.0 / KR); D1.add(L.rect.start + k, k, 1.0)
    # layer 2: the spike ring (one channel: outputs 0 or 1)
    g2 = Gdn(1)
    for kh in range(2):
        g2.qkv.add(g2.k(kh, ph), L.phase.start + ph, KC)
    f = L.f_tick.start + ph
    g2.qkv.add(g2.q(0, (ph - DELAY) % NPH), f, KC)
    for d in range(1, RFC + 1):
        g2.qkv.add(g2.q(1, (ph - d) % NPH), f, KC)
    spk = L.f_spk.start + j
    write = [(L.is_tick, 2 * BIG), (spk, 2 * BIG)]  # a TICK clears the current slot; a spike token sets it
    g2.head(j, [(spk, IND)], write, [])
    Wr, Wc, Wv = np.asarray(W_rows), np.asarray(W_cols), np.asarray(W_vals_mV, np.float64).astype(np.float32).astype(np.float64)
    pos, neg = Wv > 0, Wv < 0
    g2.o.add(L.dgp.start + Wr[pos], Wc[pos], Wv[pos])
    g2.o.add(L.dgn.start + Wr[neg], Wc[neg], -Wv[neg])
    g2.head(n + j, [(spk, IND)], write, [(L.refr22.start + j, 1.0)])
    for r, members in enumerate(roles):  # the refractory read doubles as the motor readout's 2.2 ms window
        g2.o.add(L.role.start + r, n + members, 1.0)
    # layer 2 MLP: synaptic and tonic input, dropped while refractory (Brian2: "unless refractory")
    half = GATE / 2
    Gp, Up, Dn = COO((IM, H)), COO((IM, H)), COO((H, IM))
    srcs = ([(L.dgp.start + j, -A)], [(L.dgn.start + j, -A)], [(L.dgp.start + j, 1.0)], [(L.dgn.start + j, 1.0)])
    for q, src in enumerate(srcs):
        k = 4 * j + q
        Gp.add(k, L.ONE, GATE - half); Gp.add(k, L.refr22.start + j, -GATE)
        for s_, c in src:
            Up.add(k, s_, c / half)
        Dn.add(L.inp.start + q * n + j, k, 1.0)
    row = 4 * n
    for g, members in enumerate(groups):  # tonic drive straight onto v, only at a TICK, clipped to levels 0..15
        al, be = affine.get(g, (1.0, 0.0))
        for lo, sign in ((1.0, 1.0), (16.0, -1.0)):  # relu(r - 1) - relu(r - 16), r = alpha reg + beta, in level steps
            rr = row + np.arange(len(members)); row += len(members)
            Gp.add(rr, L.reg.start + g, al * K2); Gp.add(rr, L.ONE, (be - lo) * K2); Gp.add(rr, L.refr22.start + members, -BIGG)
            for ri, c in inhibit.get(g, ()):  # minus each inhibiting register's part above its offset
                Gp.add(rr, L.rect.start + rect_of[(ri, float(c))], -al * K2)
            Up.add(rr, L.is_tick, sign * STEP * member_gain(members) / K2)  # exactly 0 off a TICK, whatever the register holds
            Dn.add(L.inp.start + members, rr, 1.0)
    # layer 3: membranes (two decay modes, + and - parts as channels) and the motor integrators
    g3 = Gdn(2, B3)
    g3.const_key()
    base3 = lambda a: [(L.is_tick, logit(1.0 - a)), (L.is_spk, -BIG), (L.is_other, -BIG), (L.is_init, BIG)]
    for q, (a, coef) in enumerate(((a_m, 1.0), (a_s, A))):
        sc = MEM_S * a / (1.0 - a)
        g3.head(q * n + j, [(L.inp.start + 2 * q * n + j, sc)], [(L.f_spk.start + j, 2 * BIG)],
                [(L.vdev.start + j, coef / MEM_S)], base=base3(a), signal2=[(L.inp.start + (2 * q + 1) * n + j, sc)])
    for r in range(R):
        g3.head(2 * n + r, [(L.role.start + r, IND)], [], [(L.rate.start + r, 1.0 / IND)], base=base3(a_r))
    if adapt:  # threshold rise: kappa x the neuron's refractory flag, leaky over tau; not reset by its spikes. It is
        # written as -theta into the membrane's own dims, the only ones the spike logit reads: they then hold v - theta
        a_t = math.exp(-p["dt"] / (adapt["tau_ms"] * 1e-3))
        g3.head(2 * n + R + j, [(L.refr22.start + j, IND)], [], [(L.vdev.start + j, -adapt["kappa_mv"] / IND)], base=base3(a_t))
    # LM head
    U = COO((L.vocab, H))
    vth = (p["v_th"] - p["v_0"]) * 1000.0
    U.add(L.tok_spk + j, L.vdev.start + j, KAPPA); U.add(L.tok_spk + j, L.ONE, -vth * KAPPA)
    for t in range(NPH):
        others = np.array([x for x in range(NPH) if x != (t - 1) % NPH])
        U.add(L.tok_tick + t, L.phase.start + others, -1e3)
    U.add(L.tok_init, L.ONE, -1e4); U.add(L.tok_pad, L.ONE, -1e4)
    U.add(np.arange(L.tok_drv, L.move), L.ONE, -1e4)
    U.add(L.tok_say + np.arange(n_say), L.ONE, -1e4)
    for r, members in enumerate(roles):  # Hz per member: (spikes in the 22-step window / 22) / dt / members
        U.add(L.move + r, L.rate.start + r, 1.0 / (RFC * p["dt"] * max(1, len(members))))
        U.add(L.move + r, L.ONE, -MOVE_OFF)
    if txt is not None:  # inside Qwen: never decoded at a text token
        U.add(np.arange(L.vocab), txt, -K_TXT)
    SAYW = COO((len(say or []), H))  # the printed words' logits (added to text rows by the merged head)
    for j, (sl, terms, bias) in enumerate(say or []):
        SAYW.add(j, L.is_say.start + sl, SAY_BIG + bias)
        for r, per_hz in terms:
            SAYW.add(j, L.rate.start + r, per_hz / (RFC * p["dt"] * max(1, len(roles[r]))))

    # assemble
    dev = device
    real = lambda shape, fill: nn.Parameter(torch.full(shape, float(fill), device=dev), requires_grad=False)
    nw = torch.full((H,), -1.0)  # dims that are not the fly's: weight 0
    nw[base:L.end] = 2.0 ** 24 - 1.0  # x / 2^24 x 2^24: exact (the norm divides by B0 / 2^9 = 2^24)
    nw[L.ball] = 2.0 ** -9 - 1.0  # ballast dims normalise to exactly 1
    if txt is not None:
        nw[txt] = 2.0 ** 20 - 1.0  # the text marker (2^-20 on text tokens) reads ~1 / rms there
    for d in markers:
        nw[d] = 2.0 ** 20 - 1.0
    norm_w = {1: 1.0 / ZG, 4: 2.0 ** 29 / ZG}  # per-head output: 1 channel -> x / |x| (0 or 1); 4 channels -> the value itself
    for layer, gd in zip(layers, (g1, g2, g3)):
        for norm in (layer.input_layernorm, layer.post_attention_layernorm):
            norm.weight = nn.Parameter(nw.clone().to(dev), requires_grad=False)
        if gd is g1 and answer:  # layer 1 sees Qwen's final hidden state as Qwen's LM head does
            qw, in_eps = cls["qread"]
            layer.input_layernorm.weight.data[:len(qw)] = torch.as_tensor(qw, dtype=torch.float32, device=dev)
            layer.input_layernorm.eps = in_eps
        at = layer.linear_attn
        at.in_proj_qkv = SparseLinear(*gd.qkv.arrays(), 2 * gd.nk * gd.dk + gd.nh * gd.dv, H, dev)
        zr = np.arange(gd.nh * gd.dv)
        at.in_proj_z = SparseLinear(zr, np.full_like(zr, L.ONE), np.full(len(zr), ZG), gd.nh * gd.dv, H, dev)
        at.in_proj_b = SparseLinear(*gd.b.arrays(), gd.nh, H, dev)
        at.in_proj_a = SparseLinear(*gd.a.arrays(), gd.nh, H, dev)
        at.out_proj = SparseLinear(*gd.o.arrays(), H, gd.nh * gd.dv, dev)
        at.out_proj.skip_zero = gd is g2  # the connectome: only a block's TICK reads the ring
        cw = torch.zeros(at.conv_dim, 1, 4, device=dev); cw[:, 0, -1] = 1.0
        at.conv1d.weight = nn.Parameter(cw, requires_grad=False)
        at.dt_bias = real((gd.nh,), -200.0)
        at.A_log = real((gd.nh,), 0.0)
        at.norm.weight = real((gd.dv,), norm_w[gd.dv] if gd.dv == 1 else gd.ball / 2 / ZG)
        at.norm.variance_epsilon = cls["eps_head"] if (gd is g1 and answer) else 1e-30
        mlp = layer.mlp
        if gd is g2 or (gd is g1 and rect_of):
            gp, up, dn = (Gp, Up, Dn) if gd is g2 else (G1, U1, D1)
            mlp.gate_proj = SparseLinear(*gp.arrays(), IM, H, dev)
            mlp.up_proj = SparseLinear(*up.arrays(), IM, H, dev)
            mlp.down_proj = SparseLinear(*dn.arrays(), H, IM, dev)
        else:
            empty = (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0))
            mlp.gate_proj = SparseLinear(*empty, IM, H, dev)
            mlp.up_proj = SparseLinear(*empty, IM, H, dev)
            mlp.down_proj = SparseLinear(*empty, H, IM, dev)
    return dict(layers=layers, E=E, U=U, nw=nw, L=L, cfgs=cfgs, SAYW=SAYW)


class BlockGraphs:
    """A step's forwards as CUDA graphs, one per block length: the same kernels in the same order,
    replayed without Python per layer (a step is otherwise bound by kernel launches). The DeltaNet
    cache is updated in place, so its addresses are fixed. A step's block is [spikes, DRIVE tokens,
    PAD, next TICK]; only the TICK reads the spike ring, so the connectome's product runs on the last
    row (SparseLinear.LastRow) and `bad` counts any block whose other rows reached it."""

    def __init__(self, forward, device):
        self.forward, self.dev = forward, device  # forward(ids on the device) -> logits
        self.graphs, self.pool = {}, None
        self.bad = torch.zeros((), dtype=torch.long, device=device)

    def __call__(self, rows):
        ids = torch.tensor(rows, dtype=torch.long)
        T = ids.shape[1]
        if T in self.graphs:
            g, static, out = self.graphs[T]
            # a fresh pinned block per call: the host does not wait for the GPU, and the caching host
            # allocator keeps each block until its copy has run (one step can feed several blocks of a length)
            static.copy_(ids.pin_memory(), non_blocking=True)
            g.replay()
            return out
        static = ids.to(self.dev)
        cur = torch.cuda.current_stream(self.dev)
        side = torch.cuda.Stream(self.dev)
        side.wait_stream(cur)
        with torch.cuda.stream(side), SparseLinear.LastRow(self.bad):
            out = self.forward(static)  # this block's forward, eagerly (it also warms the kernels up)
        cur.wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool), SparseLinear.LastRow(self.bad):
            gout = self.forward(static)  # recorded, not run: the cache is not touched
        self.pool = g.pool()
        self.graphs[T] = (g, static, gout)
        return out


class FlyModel:
    """Decodes the fly's layers greedily, one forward per 0.1 ms step (blocks of <= CHUNK tokens).
    Several flies can run as rows of one batch (PAD keeps them in lockstep). On the GPU a step's
    blocks replay as CUDA graphs (BlockGraphs; `graphs = False` runs them eagerly)."""

    graphs = True

    @property
    def CHUNK(self):
        """Tokens per forward. SparseLinear runs up to 16 token rows as one matrix-vector product each,
        so a block, a batch of two and a lone token give the same numbers. From three flies on (more
        than 16 rows) it runs one sparse-dense product, whose summation order differs: rows then agree
        with a lone fly to rounding, not bit for bit."""
        return 16 if self.B == 1 else 8

    def __init__(self, con, groups, roles, batch=1, device="cuda", adapt=None):
        import time
        from transformers import DynamicCache
        t0 = time.time()
        W = con.count * P["w_syn"] * getattr(con, "scale", 1.0) * 1000.0
        self.adapt = adapt
        self.model, self.L = build_fly_layers(con.n, con.post, con.pre, W, groups, roles, device=device, adapt=adapt)
        self.build_s = time.time() - t0
        self.n, self.B, self.dev = con.n, batch, device
        self.cache = DynamicCache(config=self.model.config)
        self.mask = {"full_attention": None, "linear_attention": None}
        self.pos = 0
        self.t = 0
        self.n_forward = 0
        self.mismatch = self.swaps = 0
        self.bad_log = []  # (step, fly, position, chosen token, expected token) for every set mismatch
        self.blocks = None
        self.plan = None
        self._start()

    def _blocks(self, rows):
        """The step's block graphs, or None where they don't apply: the CPU, the deterministic product, and
        blocks of more than 16 token rows, whose sparse products take cuSPARSE's matrix-matrix kernel
        (equal to eager stepping only to rounding; up to 16 rows they are bit for bit the same)."""
        if not self.graphs or str(self.dev) == "cpu" or SparseLinear.deterministic or len(rows) * len(rows[0]) > 16:
            return None
        if self.blocks is None:
            self.blocks = BlockGraphs(self._forward, self.dev)
        return self.blocks

    def _forward(self, ids):
        with FlyTokenNorm.Mode():
            return self.model(input_ids=ids, position_ids=torch.zeros_like(ids), past_key_values=self.cache, use_cache=True,
                              attention_mask=self.mask).logits

    def _start(self):
        B, L = self.B, self.L
        self.tokens = [[] for _ in range(B)]
        self.pending = [[] for _ in range(B)]
        prefix = [L.tok_drv + g * LEVELS for g in range(L.G)] + [L.tok_tick + p for p in range(NPH)] + [L.tok_init]
        for c0 in range(0, len(prefix), self.CHUNK):  # every register at level 0, one turn of the ring (clears its slots), then INIT (membranes and integrators to 0)
            self._feed([prefix[c0:c0 + self.CHUNK]] * B)
        self.t = 0
        self.logits = self._feed([[L.tok_tick]] * B)[:, -1]  # TICK_0: step 0's membrane potentials
        self.plan = None

    @torch.no_grad()
    def _feed(self, rows, block=False):
        """One forward over rows of token ids (one row per fly). block: a step's block (BlockGraphs)."""
        bg = self._blocks(rows) if block else None
        logits = bg(rows) if bg is not None else self._forward(torch.tensor(rows, device=self.dev))
        self.pos += len(rows[0])
        self.n_forward += 1
        for b in range(self.B):
            self.tokens[b] += rows[b]
        return logits

    def set_levels(self, levels, b=0):
        """DRIVE tokens (register g to level l) for fly b, written with the next step. levels: a list
        over all registers, or a dict {register: level} for some of them."""
        L = self.L
        items = levels.items() if isinstance(levels, dict) else enumerate(levels)
        self.pending[b] += [L.tok_drv + g * LEVELS + int(lv) for g, lv in items]

    def move(self):
        """Each role's rate (Hz per member) from the LM head's MOVE rows at this TICK."""
        L = self.L
        return (self.logits[:, L.move:L.move + L.R].double() + MOVE_OFF).cpu().numpy()

    def membrane(self, top=2500, min_mv=1.4):
        L = self.L
        v = self.logits[:, L.tok_spk:L.tok_spk + self.n] / KAPPA + (P["v_th"] - P["v_0"]) * 1000.0
        out = []
        for b in range(self.B):
            k = torch.topk(v[b], top)
            keep = k.values > min_mv
            out.append((k.indices[keep].cpu().numpy(), k.values[keep].cpu().numpy()))
        return out

    PLAN = 64  # spikes per fly a step's plan carries; a step with more fetches its whole order

    def _plan(self, lg, k=None):
        """On the GPU, from a TICK's logits: per fly, how many spikes they choose, the model's argmax (the
        check of the block's first token), and the first k chosen neurons in the order they are fed
        (spike logit descending, ties by neuron, as a stable sort of the chosen ones gives)."""
        L = self.L
        nxt = L.tok_tick + (self.t + 1) % NPH
        spk = lg[:, L.tok_spk:L.tok_spk + self.n]
        above = spk > lg[:, nxt:nxt + 1]
        k = min(k or self.PLAN, self.n)
        v, i = torch.topk(torch.where(above, spk, float("-inf")), k, dim=1)  # the chosen ones (fewer than k: then -inf after)
        i, a = torch.sort(i, dim=1)  # by neuron, then stably by logit: ties keep neuron order
        i = torch.gather(i, 1, torch.sort(torch.gather(v, 1, a), dim=1, descending=True, stable=True).indices)
        return torch.cat([above.sum(1, keepdim=True), lg.argmax(-1, keepdim=True), i], 1)

    def step(self, strict=False):
        """One 0.1 ms step for every fly in one forward: the spikes the model's logits at this TICK
        choose (every spike logit above the next TICK's, in descending order), then
        [spikes, DRIVE tokens, PAD.., next TICK], checking the model's argmax at every position. The
        next step's spikes are chosen on the GPU and come back with this step's checks: one transfer."""
        L, B = self.L, self.B
        nxt = L.tok_tick + (self.t + 1) % NPH
        plan = self.plan if self.plan is not None else self._plan(self.logits).tolist()
        if any(p[0] > len(p) - 2 for p in plan):  # more spikes than the plan carries
            plan = self._plan(self.logits, max(p[0] for p in plan)).tolist()
        props = [p[2:2 + p[0]] for p in plan]
        body = [[L.tok_spk + i for i in props[b]] + self.pending[b] for b in range(B)]
        self.pending = [[] for _ in range(B)]
        m = max(len(x) for x in body)
        rows = [body[b] + [L.tok_pad] * (m - len(body[b])) + [nxt] for b in range(B)]
        picks = []  # the model's argmax after every token it is fed, kept on the GPU
        for c0 in range(0, m + 1, self.CHUNK):
            logits = self._feed([r[c0:c0 + self.CHUNK] for r in rows], block=True)
            picks.append(logits.argmax(-1))
        self.logits = logits[:, -1].clone() if self.blocks is not None else logits[:, -1]  # a graph's output is rewritten by its next replay
        self.t += 1
        if self.blocks is not None:
            picks.append(self.blocks.bad.expand(B, 1))
        back = torch.cat(picks + [self._plan(self.logits)], 1).tolist()  # one transfer per step
        nb = m + 1 + (self.blocks is not None)
        if self.blocks is not None and back[0][m + 1]:
            raise RuntimeError("a step's block fed the connectome from a token other than its TICK; run with FlyModel.graphs = False")
        choice = [[plan[b][1]] + back[b][:m + 1] for b in range(B)]
        self.plan = [back[b][nb:] for b in range(B)]
        for b in range(B):
            n = len(props[b])
            for q in range(m + 1):
                got, want = choice[b][q], (rows[b][q] if q < n else nxt)
                if got == want:
                    continue
                if q < n and got in rows[b][q + 1:n]:
                    self.swaps += 1
                else:
                    assert not strict, (b, self.t - 1, q, got, want)
                    self.mismatch += 1
                    self.bad_log.append((self.t - 1, b, q, got, want, rows[b]))
        return props
