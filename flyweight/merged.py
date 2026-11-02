"""One model, one residual stream: Qwen3.5-4B's 32 decoder layers, then the fly's three
(docs/design.md, One stack).

- Residual: H = 6 x 4^9 dims. Qwen's dims are 0..2559; dim 2560 is the text marker TXT (2^-20 on
  text tokens); the fly's dims follow. One vocabulary: Qwen's 248,320 text tokens, then the fly's.
- Qwen's layers keep their own weights (bf16), read and write only dims 0..2559 (BlockLinear: a
  dense block inside a zero-padded matrix), and carry mute columns on the fly's ballast: on a fly
  token their DeltaNet decay is 1, write 0, output 0, and attention output gate 0, exactly. Qwen's
  residual norms average over H, so their weights carry sqrt(2560 / H) and eps 1e-6 x 2560 / H.
- The fly's layers get -K x TXT on every write strength and LM row: on a text token they write
  nothing and are never decoded; their own ballast is exactly 0 there, so they output exactly 0.
- Fast path (`text`, `fly`): Qwen's layers are skipped on fly tokens, and their convolution windows
  are filled with the zeros those tokens would have put there. On text tokens Qwen's layers run on
  the residual's first NARROW dims, the only ones a text token holds before the fly's layers: their
  norms still average over all H dims, so this is the same computation without the zeros. On fly
  tokens the LM head computes only the fly's rows and the printed words' rows (Qwen's dims are 0
  there, so every other text row is the same constant). `full` runs all 35 layers on every token
  over the whole residual, with Qwen's full attention masked from fly positions (the standard mask
  input); the checks compare the two.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

from . import fly_layers
from .sparse import SparseEmbedding, SparseLinear
from .connectome import P
from .sets import WORDS

QH = 2560
TXT = QH  # the text-token marker dim
TXT_VAL = 2.0 ** -20
SITS = ["left", "right", "forward", "stop", "jump", "fly"]  # Qwen's situation groups (scripts/calibrate/discover_drive.py)
SIT0 = QH + 1  # their readout dims
CLS = QH + 1 + len(SITS)  # the report-end marker dim (the last token of the question)
FLY_BASE = CLS + 1
NARROW = FLY_BASE + 1  # what a text token holds inside Qwen's layers: Qwen's dims, the markers, the fly's ballast dim (0)
KAPPA_SIT = 2.0 ** -16  # readout scale (keeps Qwen's residual unchanged to ~1e-10)
C_B = 2.0 ** 36  # register reference channel per unit of TXT
K_KEY = 1e4
# commands (the fly's registers 0..4: P9, DNa02 L, DNa02 R, DNp01, DNg02) as linear maps of the groups;
# "bias" is a dead zone (a constant in the register's TXT coefficient): the fed-on-the-floor and in-flight
# groups sit at 0.2-0.4 in ordinary walking reports and near 1 in their own situations. Stop does not
# inhibit takeoff: right after feeding the fly is still at the sugar, where the stop group stays up
COMMAND_MAP = lambda G: [{"forward": G, "stop": -G}, {"left": 2 * G, "right": -2 * G, "stop": -G},
                         {"right": 2 * G, "left": -2 * G, "stop": -G}, {"jump": G, "bias": -0.45 * G},
                         {"fly": G, "stop": -G, "bias": -0.3 * G}]
MUTE = -1024.0
H = fly_layers.H_FLY
# the answer bridge: each command register reads s_w, Qwen's logit of its move word over its logit of a reference
# word (stop; for fly, forward), and holds max(s_w, 0) (the readout passes the layer's SiLU). The level is
# clip(15 (m_w - theta_w) / delta_w, 0, 15) on the register's margin m_w (scripts/calibrate/check_answers.py fits the
# references, theta, delta and the offsets c and c_g to the pilot's levels):
# - left and right inhibit each other, m = s_w - relu(s_other): at most one side turns, by how much Qwen favours it
#   over both stop and the other side;
# - forward is inhibited by both, m = s_f - relu(s_left - c) - relu(s_right - c): a strong pull to one side stops
#   the walk, so the fly turns in place (in the connectome, P9 and DNa02 drive suppress each other's pools);
# - groom inhibits every walking register (forward, left, right, back) by relu(s_groom - c_g): a fly that grooms
#   stands, and Qwen's pull toward back or a turn for dusty antennae would walk it away from the grooming;
# - jump, fly and groom: m = s_w.
ANSWER_WORDS = ["forward", "left", "right", "back", "jump", "fly", "groom", "stop"]
ANSWER_REGS = WORDS  # register order: P9, DNa02 L, DNa02 R, DNp01, DNg02, MDN, aDN (flyweight/sets.py DRIVEN)
TURNS = ("left", "right")


def answer_refs(am):
    """Per register (ANSWER_REGS order): the reference word its move word is read against."""
    return [am.get("ref", {}).get(w, "stop") for w in ANSWER_REGS]


def inhibitors(am):
    """By register index: [(inhibiting register, offset)], as fly_layers's cls["inhibit"] takes them, for an answer map
    with forward_c and groom_c."""
    ix = ANSWER_REGS.index
    c, cg = float(am["forward_c"]), float(am["groom_c"])
    out = {ix("left"): [(ix("right"), 0.0)], ix("right"): [(ix("left"), 0.0)],
           ix("forward"): [(ix("left"), c), (ix("right"), c)], ix("back"): []}
    for w in ("forward", "left", "right", "back"):
        out[ix(w)].append((ix("groom"), cg))
    return out


def answer_levels(D, am):
    """Levels 0..15 per register from the register values D (..., 7), under an answer map (data/answer_map.json)."""
    D = np.asarray(D, np.float64)
    M = D.copy()
    for g, terms in inhibitors(am).items():
        for p, c in terms:
            M[..., g] -= np.maximum(D[..., p] - c, 0.0)
    th = np.array([am["theta"][w] for w in ANSWER_REGS])
    dl = np.array([am["delta"][w] for w in ANSWER_REGS])
    return np.clip(fly_layers.LEVELS_M1 * (M - th) / dl, 0, fly_layers.LEVELS_M1)
LAM = 64.0  # SiLU(64 s) == 64 s for s > 0.27; below, the drive is under 0.27 / delta of a level
EPS_HEAD = 2.0 ** 42  # layer 1's per-head norm eps: a register written on a text token (no ballast) is divided by exactly 2^21
BALL1 = 2.0 ** 40  # layer 1's ballast with the answer bridge: eps stays below half an ulp of its heads' mean (2^78)


class ResidualRMSNorm(Qwen3_5RMSNorm):
    """Qwen's residual norm over the stack's H dims. Given only the residual's first k dims (the rest
    exactly 0), it computes the same: the mean of squares still divides by H."""

    def forward(self, x):
        if x.shape[-1] == self.weight.shape[0]:
            return super().forward(x)
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).sum(-1, keepdim=True) / self.weight.shape[0] + self.eps)
        return (y * (1.0 + self.weight[:x.shape[-1]].float())).type_as(x)


def narrow_norms(qlayers):
    for layer in qlayers:
        for norm in (layer.input_layernorm, layer.post_attention_layernorm):
            norm.__class__ = ResidualRMSNorm


class BlockLinear(nn.Module):
    """y[..., out_offset:out_offset+o] = W x[..., :i] (+ mute columns and extra sparse rows)."""

    residual = H  # the residual width the stack runs at; QwenFlyModel.text narrows it to NARROW

    def __init__(self, W, out_features, mute_rows=None, mute_col=None, extra=None):
        super().__init__()
        self.W = W  # dense (o, i), Qwen's own weight tensor
        self.in_features, self.out_features = W.shape[1], out_features
        self.mute_rows, self.mute_col = mute_rows, mute_col
        self.extra = extra  # (sparse (g, i) on the input, output dims)

    @property
    def weight(self):
        return self.W

    def forward(self, x):
        xi = x[..., :self.W.shape[1]]
        y = F.linear(xi.to(self.W.dtype), self.W).float()
        if self.mute_rows is not None:
            y[..., self.mute_rows] += MUTE * x[..., self.mute_col:self.mute_col + 1].float()
        width = BlockLinear.residual if self.out_features == H else self.out_features
        if width == self.W.shape[0] and self.extra is None:
            return y
        out = torch.zeros(*x.shape[:-1], width, device=x.device)
        out[..., :self.W.shape[0]] = y
        if self.extra is not None:
            S, dims = self.extra
            out[..., dims] += torch.sparse.mm(S, xi.reshape(-1, xi.shape[-1]).float().t()).t().reshape(*x.shape[:-1], -1)
        return out


class MergedEmbedding(nn.Module):
    def __init__(self, qwen_embed, fly_embed, n_text, cls_ids=()):
        super().__init__()
        self.q, self.f, self.n_text = qwen_embed, fly_embed, n_text
        self.register_buffer("cls_ids", torch.as_tensor(sorted(cls_ids), dtype=torch.long, device=qwen_embed.weight.device),
                             persistent=False)

    def forward(self, ids, width=H):
        out = torch.zeros(*ids.shape, width, device=ids.device)
        text = ids < self.n_text
        if text.any():
            out[text, :QH] = self.q(ids[text]).float()
            out[text, TXT] = TXT_VAL
            if len(self.cls_ids):
                is_cls = torch.isin(ids, self.cls_ids)
                out[is_cls, CLS] = TXT_VAL
        if (~text).any():
            out[~text] = self.f(ids[~text] - self.n_text)
        return out


class MergedHead(nn.Module):
    def __init__(self, qwen_W, fly_head, one, say=None):
        super().__init__()
        self.W, self.f, self.one = qwen_W, fly_head, one
        self.say = say  # (SparseLinear: residual -> the fly's printed words, their text token ids)

    def forward(self, x):
        q = F.linear(x[..., :QH].to(self.W.dtype), self.W).float() - 2.0 ** 13 * x[..., self.one:self.one + 1]
        if self.say is not None:  # zero on text tokens (they read only fly dims); 2^16 up at a SAY token
            S, ids = self.say
            q.index_add_(q.dim() - 1, ids, S(x))
        return torch.cat([q, self.f(x)], dim=-1)  # Qwen's rows never win on a fly token (its ballast reads 1)

    def on_fly(self, x, n_text, full=False):
        """forward() on fly tokens without Qwen's 248,320-row product: Qwen's dims are exactly 0 there, so
        every text row is -2^13 x the ballast (plus the printed words' rows). Returns the fly's rows, or
        with full=True the whole vocabulary's logits, equal to forward()'s."""
        fly = self.f(x)
        if not full:
            return fly
        q = (-2.0 ** 13 * x[..., self.one:self.one + 1]).expand(*x.shape[:-1], n_text).contiguous()
        if self.say is not None:
            S, ids = self.say
            q.index_add_(q.dim() - 1, ids, S(x))
        return torch.cat([q, fly], dim=-1)


def _mute_gdn(attn, one):
    """Qwen DeltaNet on a fly token: a -> -1024 (decay exp(0) = 1), b -> -1024 (write 0), z -> -1024 (output 0)."""
    attn.in_proj_qkv = BlockLinear(attn.in_proj_qkv.weight, attn.in_proj_qkv.weight.shape[0])
    for nm in ("in_proj_a", "in_proj_b", "in_proj_z"):
        W = getattr(attn, nm).weight
        setattr(attn, nm, BlockLinear(W, W.shape[0], mute_rows=torch.arange(W.shape[0], device=W.device), mute_col=one))
    attn.out_proj = BlockLinear(attn.out_proj.weight, H)


def _mute_attention(attn, one):
    """Qwen full attention on a fly token: the output gate (second half of each head's q_proj rows) -> -1024."""
    W = attn.q_proj.weight
    hd = attn.head_dim
    rows = torch.cat([torch.arange(h * 2 * hd + hd, (h + 1) * 2 * hd) for h in range(W.shape[0] // (2 * hd))]).to(W.device)
    attn.q_proj = BlockLinear(W, W.shape[0], mute_rows=rows, mute_col=one)
    attn.k_proj = BlockLinear(attn.k_proj.weight, attn.k_proj.weight.shape[0])
    attn.v_proj = BlockLinear(attn.v_proj.weight, attn.v_proj.weight.shape[0])
    attn.o_proj = BlockLinear(attn.o_proj.weight, H)


class QwenFlyModel(nn.Module):
    """Qwen3.5-4B (its text layers) and the fly, one stack. `qwen`: a loaded
    Qwen3_5ForConditionalGeneration; its modules are rewired in place."""

    def __init__(self, qwen, con, groups, roles, device="cuda", drive=None, cls_ids=(), gain=1.5, answer=None, say=None, adapt=None):
        """drive: Qwen's situation groups, {name: (layers, neurons, p5, p95)} (scripts/calibrate/discover_drive.py);
        with it, registers 0..4 (the commands) are written by Qwen's neurons at every cls_ids token.
        answer: instead, Qwen's own answer drives them: dict(ids=the 8 ANSWER_WORDS' token ids, and
        the answer map's ref, theta, delta and forward_c): register w holds s_w, Qwen's logit of w minus
        its logit of w's reference word, read from its final hidden state at the token; its level is
        clip(15 (m_w - theta_w) / delta_w, 0, 15), m_w = s_w after its inhibitors (inhibitors()).
        say: the fly's printed words, [(slot, text token id, [(role, per_hz)], bias)]."""
        super().__init__()
        from transformers import Qwen3_5TextConfig
        lm = qwen.model.language_model
        tc = qwen.config.text_config
        self.n_text = tc.vocab_size
        W = con.count * P["w_syn"] * getattr(con, "scale", 1.0) * 1000.0
        cls, extras = None, {}
        scale_q = math.sqrt(QH / H)
        eps_res = tc.rms_norm_eps * QH / H
        if answer is not None:
            U = (lm.embed_tokens.weight if tc.tie_word_embeddings else qwen.lm_head.weight)[list(answer["ids"])].detach().double().cpu().numpy()
            iw = ANSWER_WORDS.index
            C = [U[iw(w)] - U[iw(ref)] for w, ref in zip(ANSWER_REGS, answer_refs(answer))]
            dims = np.arange(QH)
            delta, theta = answer["delta"], answer["theta"]
            cls = dict(dim=CLS, k_key=K_KEY, lam=LAM, eps_head=EPS_HEAD, ball1=BALL1,
                       qread=(((1.0 + lm.norm.weight.detach().float()) * scale_q - 1.0).cpu().numpy(), eps_res),
                       answer={r: list(zip(dims, C[r])) for r in range(len(ANSWER_REGS))},
                       affine={r: (fly_layers.LEVELS_M1 / delta[w], 1.0 - fly_layers.LEVELS_M1 * theta[w] / delta[w]) for r, w in enumerate(ANSWER_REGS)},
                       inhibit=inhibitors(answer))
        elif drive is not None:
            # D_g = mean_n (a_n - p5_n) / (p95_n - p5_n), read in the weights as alpha SIT_g / TXT - P_g
            alpha = TXT_VAL * 2.0 ** 20 / (2.0 ** 20 * KAPPA_SIT)
            Pg = {}
            for g, nm in enumerate(SITS):
                Ls, Ns, p5, p95 = drive[nm]
                dlt = np.maximum(np.asarray(p95) - np.asarray(p5), 1e-6)
                Pg[nm] = float(np.mean(np.asarray(p5) / dlt))
                for l_, n_, d_ in zip(Ls, Ns, dlt):
                    extras.setdefault(int(l_), []).append((g, int(n_), KAPPA_SIT / (len(Ns) * d_)))
            k = 32.0 * C_B / 2.0 ** 30
            regs = {}
            for r_i, mp in enumerate(COMMAND_MAP(gain)):
                lin = {nm: c_ for nm, c_ in mp.items() if nm != "bias"}
                tc_ = k * (1.0 + fly_layers.LEVELS_M1 * (mp.get("bias", 0.0) - sum(c_ * Pg[nm] for nm, c_ in lin.items())))
                regs[r_i] = (tc_, [(SIT0 + SITS.index(nm), k * fly_layers.LEVELS_M1 * alpha * c_) for nm, c_ in lin.items()])
            cls = dict(dim=CLS, regs=regs, c_b=C_B, k_key=K_KEY)
        self.drive, self.gain, self.answer = drive, gain, answer
        c = fly_layers.fly_components(con.n, con.post, con.pre, W, groups, roles, device=device, H=H, base=FLY_BASE,
                                 layer_offset=len(lm.layers), txt=TXT, markers=[SIT0 + g for g in range(len(SITS))] + [CLS],
                                 cls=cls, say=[(sl, terms, b) for sl, _, terms, b in (say or [])], adapt=adapt)
        self.affine = (cls or {}).get("affine", {})
        self.inhibit = (cls or {}).get("inhibit", {})
        self.say_words = [(sl, tid) for sl, tid, _, _ in (say or [])]
        self.L = L = c["L"]
        self.qcfg, self.fly_cfgs = tc, c["cfgs"]  # for flyweight/checkpoint.py
        one = L.ONE
        self.one = one

        def qnorm(norm):
            w = torch.full((H,), -1.0, device=device)
            w[:QH] = (1.0 + norm.weight.float()) * scale_q - 1.0
            w[one] = 2.0 ** -9 - 1.0  # the fly's ballast reads exactly 1 on a fly token (the mute columns use it)
            norm.weight = nn.Parameter(w, requires_grad=False)
            norm.eps = eps_res
            norm.__class__ = ResidualRMSNorm

        for layer in lm.layers:
            qnorm(layer.input_layernorm); qnorm(layer.post_attention_layernorm)
            if layer.block_type == "linear_attention":
                _mute_gdn(layer.linear_attn, one)
            else:
                _mute_attention(layer.self_attn, one)
            mlp = layer.mlp
            mlp.gate_proj = BlockLinear(mlp.gate_proj.weight, mlp.gate_proj.weight.shape[0])
            mlp.up_proj = BlockLinear(mlp.up_proj.weight, mlp.up_proj.weight.shape[0])
            ex = None
            li = list(lm.layers).index(layer)
            if li in extras:  # Qwen's own neurons, read into the situation readout dims (new rows only)
                rr = np.array([e[0] for e in extras[li]]); cc = np.array([e[1] for e in extras[li]]); vv = np.array([e[2] for e in extras[li]])
                S = SparseLinear(rr, cc, vv, len(SITS), mlp.down_proj.weight.shape[1], device).w
                ex = (S, torch.arange(SIT0, SIT0 + len(SITS), device=device))
            mlp.down_proj = BlockLinear(mlp.down_proj.weight, H, extra=ex)
        self.qlayers = lm.layers
        self.flayers = nn.ModuleList(c["layers"])
        for layer in self.flayers:
            fly_layers.FlyTokenNorm.install(layer.input_layernorm); fly_layers.FlyTokenNorm.install(layer.post_attention_layernorm)
        self.layer_types = list(tc.layer_types) + ["linear_attention"] * 3
        self.cfg = Qwen3_5TextConfig(**{**tc.to_dict(), "hidden_size": H, "num_hidden_layers": len(self.layer_types),
                                        "layer_types": self.layer_types, "vocab_size": self.n_text + L.vocab})
        self.rotary = lm.rotary_emb
        self.embed = MergedEmbedding(lm.embed_tokens, SparseEmbedding(*c["E"].arrays(), L.vocab, H, device), self.n_text, cls_ids)
        fn = torch.full((H,), -1.0, device=device)
        fn[:] = c["nw"].to(device)
        fn[:QH] = (1.0 + lm.norm.weight.float()) * scale_q - 1.0
        self.norm = lm.norm
        self.norm.weight = nn.Parameter(fn, requires_grad=False)
        self.norm.eps = eps_res
        fly_layers.FlyTokenNorm.install(self.norm)
        sayh = None
        if say:
            sayh = (SparseLinear(*c["SAYW"].arrays(), len(say), H, device), torch.tensor([t for _, t, _, _ in say], device=device))
        self.head = MergedHead(lm.embed_tokens.weight if tc.tie_word_embeddings else qwen.lm_head.weight,
                               SparseLinear(*c["U"].arrays(), L.vocab, H, device), one, sayh)
        self.dev = device

    def new_cache(self):
        from transformers import DynamicCache
        return DynamicCache(config=self.cfg)

    def _rot(self, h, pos):
        T = h.shape[1]
        p = torch.as_tensor(pos, device=self.dev).view(1, 1, T)
        return self.rotary(h, p.expand(3, 1, T)), p.view(1, T)

    @torch.no_grad()
    def text(self, ids, pos, cache, keep=1, fly_all=False, row=0, batch=1, narrow=True):
        """Text tokens through the stack. pos: text positions. The fly's layers write nothing and output
        exactly 0 on text tokens other than the report's end (their write strengths are -K x TXT there,
        their convolution kernels read only the current token), so unless fly_all they run only on
        report-end tokens; `check_bridge` compares the two. With a fly batch (trials: one Qwen, many
        episodes), the report-end token is written into fly row `row`; the other rows get PAD.
        narrow: Qwen's layers run on the residual's first NARROW dims (see the module's notes);
        `check_merged` compares it with the whole width."""
        width = NARROW if narrow else H
        ids_t = torch.as_tensor(ids, device=self.dev).view(1, -1)
        h = self.embed(ids_t, width=width)
        pe, tp = self._rot(h, pos)
        T, past = h.shape[1], int(pos[0])  # Qwen's attention cache holds exactly the text before this chunk
        mask = None
        if T > 1:  # explicit causal mask, aligned to the end of the cache (sdpa's is_causal aligns top-left)
            j = torch.arange(past + T, device=self.dev)[None]
            i = torch.arange(T, device=self.dev)[:, None]
            mask = torch.where(j <= past + i, 0.0, float("-inf"))[None, None]
        BlockLinear.residual = width
        try:
            for li, layer in enumerate(self.qlayers):
                am = mask if self.layer_types[li] == "full_attention" else None
                h = layer(h, position_embeddings=pe, attention_mask=am, position_ids=tp, past_key_values=cache, use_cache=True)
        finally:
            BlockLinear.residual = H
        if fly_all:
            sel = torch.arange(T, device=self.dev)
        else:
            sel = torch.nonzero(torch.isin(ids_t[0], self.embed.cls_ids)).squeeze(1) if len(self.embed.cls_ids) else ids_t.new_zeros(0)
        out = F.pad(h[:, -keep:], (0, H - width))  # the last `keep` tokens as the whole residual holds them
        if len(sel):
            hs = F.pad(h[:, sel], (0, H - width))
            if batch > 1:
                rows = self.embed(torch.full((batch, len(sel)), self.n_text + self.L.tok_pad, device=self.dev))
                rows[row] = hs[0]
                hs = rows
            for layer in self.flayers:
                hs = layer(hs, position_embeddings=None, attention_mask=None, position_ids=None, past_key_values=cache, use_cache=True)
            last = sel >= T - keep
            if last.any():
                out[:, sel[last] - (T - keep)] = hs[row:row + 1, last]
        return self.head(self.norm(out))

    @torch.no_grad()
    def fly(self, rows, cache, full=False):
        """Fly tokens (merged ids) for a batch of rows: the fly's layers only (Qwen's are muted on
        them: their states and output are exactly unchanged, which `full` checks). Returns the LM head's
        fly rows, or with full=True the whole vocabulary's logits (MergedHead.on_fly)."""
        h = self.embed.f(torch.as_tensor(rows, device=self.dev) - self.n_text)
        with fly_layers.FlyTokenNorm.Mode():
            for layer in self.flayers:
                h = layer(h, position_embeddings=None, attention_mask=None, position_ids=None, past_key_values=cache, use_cache=True)
            x = self.norm(h)
        return self.head.on_fly(x, self.n_text, full=full)

    def zero_qwen_conv(self, cache, k):
        """What k fly tokens do to Qwen's DeltaNet convolution windows in the full forward: shift in
        k zero columns (Qwen's dims are exactly 0 on fly tokens, so its conv inputs are 0)."""
        for i, t in enumerate(self.layer_types[:len(self.qlayers)]):
            if t != "linear_attention" or not cache.layers[i].conv_states:
                continue
            cs = cache.layers[i].conv_states[0]
            kk = min(k, cs.shape[-1])
            if kk:
                cs.copy_(torch.cat([cs[..., kk:], torch.zeros_like(cs[..., :kk])], dim=-1))

    @torch.no_grad()
    def full(self, ids, is_fly, pos, cache, fly_cols):
        """One token through all 35 layers. fly_cols: bool list over cached positions so far (incl.
        this one) marking fly positions, which Qwen's full attention must not see."""
        h = self.embed(torch.as_tensor(ids, device=self.dev).view(1, -1))
        pe, tp = self._rot(h, pos)
        kv = len(fly_cols)
        mask = torch.zeros(1, 1, 1, kv, device=self.dev)
        mask[..., torch.as_tensor(fly_cols, device=self.dev)] = float("-inf")
        for i, layer in enumerate(self.qlayers):
            am = mask if self.layer_types[i] == "full_attention" else None
            h = layer(h, position_embeddings=pe, attention_mask=am, position_ids=tp, past_key_values=cache, use_cache=True)
        for layer in self.flayers:
            h = layer(h, position_embeddings=pe, attention_mask=None, position_ids=tp, past_key_values=cache, use_cache=True)
        return self.head(self.norm(h))


class QwenFlyRunner:
    """Drives a QwenFlyModel: text through the whole stack, the fly's 0.1 ms steps through its
    layers (greedy, with FlyModel's checks), one cache, one token stream (merged ids)."""

    def __init__(self, model, fly_all=False, batch=1, narrow=True, memory=0):
        self.m = model
        # memory: Qwen reads each report after its last `memory` reports and answers (its log), instead of from
        # its post-instructions state; the log rolls on in the cache and restarts from the instructions, with
        # the last `memory` entries re-read, when it reaches twice that
        self.memory, self.log, self.carry, self.report_ids = memory, [], [], []
        self.fly_all = fly_all  # run the fly's layers on every text token (equivalent; slower)
        self.narrow = narrow  # Qwen's layers on the residual's first NARROW dims (equivalent; much faster)
        self.chunk = 512 if narrow and not fly_all else 16  # text tokens per forward (a whole-width residual is 1.57M dims)
        self.B = batch  # fly rows (trials); Qwen reads each row's reports in turn
        self.cache = model.new_cache()
        self.text_pos = 0
        self.fly_since_text = 0
        self.streams = [[] for _ in range(batch)]
        self.say_bad = 0
        runner = self

        class _Fly(fly_layers.FlyModel):
            def __init__(self):
                self.L, self.n, self.B, self.dev = model.L, model.L.n, batch, model.dev
                self.n_forward = self.mismatch = self.swaps = 0
                self.bad_log = []
                self.t = 0
                self.blocks = None

            def _forward(self, ids):
                return model.fly(ids, runner.cache)

            def _feed(self, rows, block=False):
                ids = [[model.n_text + t for t in r] for r in rows]
                for b in range(len(ids)):
                    runner.streams[b] += ids[b]
                runner.fly_since_text += len(ids[0])
                self.n_forward += 1
                for b in range(self.B):
                    self.tokens[b] += rows[b]
                bg = self._blocks(ids) if block else None
                return bg(ids) if bg is not None else model.fly(ids, runner.cache)

        self.flybrain = _Fly()
        self.started = False

    @property
    def stream(self):
        return self.streams[0]

    def start_fly(self):
        self.flybrain._start()
        self.started = True

    def snapshot_qwen(self):
        """Save Qwen's layers' cache (after the instructions): each report is read from here."""
        import copy
        self.snap = [copy.deepcopy(self.cache.layers[i]) for i in range(len(self.m.qlayers))]
        self.snap_pos = self.text_pos

    def restore_qwen(self):
        import copy
        for i, lay in enumerate(self.snap):
            self.cache.layers[i] = copy.deepcopy(lay)
        self.text_pos = self.snap_pos
        self.fly_since_text = 0  # the restored windows hold text from before any fly token

    def begin_report(self):
        """Before reading a report. Without memory: back to the instructions' state. With memory: on after
        the log, restarted from the instructions with its last `memory` entries when it is long. Returns
        the tokens to read just before the report (the last answer, when the log rolls on)."""
        if not self.memory:
            self.restore_qwen()
            return []
        if len(self.log) >= 2 * self.memory:
            self.restore_qwen()
            for e in self.log[-self.memory:]:
                self.text(e)
                self.fly_since_text = 4  # between reports the fly's tokens shift zeros through Qwen's convolutions
            self.log, self.carry = self.log[-self.memory:], []
        out, self.carry = self.carry, []
        return out

    def end_report(self, answer_ids):
        """After a report: its answer (the move word's tokens, then a newline) joins the log."""
        if self.memory:
            self.log.append(list(self.report_ids) + list(answer_ids))
            self.carry = list(answer_ids)

    def reset_memory(self):
        self.log, self.carry = [], []
        if self.memory:
            self.restore_qwen()

    def registers(self, row=0):
        """The fly's register values as the model holds them: one PAD token through the fly's layers
        (it writes nothing), reading layer 1's register outputs. Affine registers (the answer bridge)
        hold Qwen's centred logit; `levels` converts."""
        L, out = self.m.L, {}
        h = self.m.flayers[0].register_forward_hook(lambda mod, i, o: out.__setitem__("x", o))
        self.flybrain._feed([[L.tok_pad]] * self.B)
        h.remove()
        return out["x"][row, -1, L.reg].double().cpu().numpy()

    def levels(self, regs):
        """Drive levels 0..15 the tonic rows apply, from register values."""
        out = []
        for g, r in enumerate(regs):
            al, be = self.m.affine.get(g, (1.0, 0.0))
            for p, c in self.m.inhibit.get(g, ()):
                r = r - max(regs[p] - c, 0.0)
            out.append(float(np.clip(al * r + be - 1.0, 0, fly_layers.LEVELS_M1)))
        return np.array(out)

    def say(self, slot, row=0):
        """SAY_slot through the fly's layers (it writes nothing): the greedy next token is the word
        the fly's neurons print, a text token. Returns its id."""
        L = self.m.L
        rows = [[self.m.n_text + L.tok_pad]] * self.B
        rows = [r if b != row else [self.m.n_text + L.tok_say + slot] for b, r in enumerate(rows)]
        lg = self.m.fly(rows, self.cache, full=True)
        self.say_logits = lg[row, -1]
        self.streams[row] += rows[row]
        self.fly_since_text += 1
        word = int(lg[row, -1].argmax())
        if word not in [t for sl, t in self.m.say_words if sl == slot]:
            self.say_bad += 1
        return word

    def text(self, ids, row=0):
        if self.fly_since_text:
            self.m.zero_qwen_conv(self.cache, self.fly_since_text)
            self.fly_since_text = 0
        pos = list(range(self.text_pos, self.text_pos + len(ids)))
        lg = None
        k = self.chunk
        for c0 in range(0, len(ids), k):
            lg = self.m.text(ids[c0:c0 + k], pos[c0:c0 + k], self.cache, fly_all=self.fly_all, row=row, batch=self.B, narrow=self.narrow)
        self.text_pos += len(ids)
        self.streams[row] += list(ids)
        return lg[0, -1, :self.m.n_text]


def load_drive(path):
    z = np.load(path)
    return {nm: (z[nm][:, 0], z[nm][:, 1], z[nm + "_p5"], z[nm + "_p95"]) for nm in SITS}
