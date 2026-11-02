"""The one-stack model as a checkpoint : every tensor in safetensors,
a config, the tokenizer and a manifest of sha256 hashes; `load` rebuilds the model from these alone
(no connectome, no base download).

Layout of the directory:
  config.json        Qwen's text config, the fly layers' configs, the layout and the bridge constants
  modules.json       per module path: its class and constructor arguments (BlockLinear, SparseLinear,
                     SparseEmbedding) and every norm's eps
  model.safetensors  all tensors: Qwen's dense blocks (bf16), sparse matrices as CSR parts
                     (crow, col, val), norms, embeddings, the fly's embedding and LM-head rows
  tokenizer files    Qwen3.5-4B's tokenizer
  manifest.json      sha256 of every file
"""

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from . import fly_layers
from .merged import BlockLinear, MergedEmbedding, MergedHead, QwenFlyModel, narrow_norms
from .sparse import SparseEmbedding, SparseLinear


def _csr_parts(prefix, w, T):
    T[prefix + ".crow"] = w.crow_indices().contiguous().cpu()
    T[prefix + ".col"] = w.col_indices().contiguous().cpu()
    T[prefix + ".val"] = w.values().contiguous().cpu()


def _csr(prefix, T, shape, device):
    return torch.sparse_csr_tensor(T[prefix + ".crow"], T[prefix + ".col"], T[prefix + ".val"], shape).to(device)


def save(m: QwenFlyModel, out, tok=None):
    from safetensors.torch import save_file
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    T, mods = {}, {}
    custom = set()
    for name, mod in m.named_modules():
        if isinstance(mod, BlockLinear):
            mods[name] = dict(cls="BlockLinear", out=mod.out_features, mute_col=mod.mute_col, extra=mod.extra is not None)
            T[name + ".W"] = mod.W.detach().contiguous().cpu()
            if mod.mute_rows is not None:
                T[name + ".mute_rows"] = mod.mute_rows.cpu()
            if mod.extra is not None:
                S, dims = mod.extra
                S = S.to_sparse_csr() if S.layout != torch.sparse_csr else S
                _csr_parts(name + ".extra", S, T); T[name + ".extra.dims"] = dims.cpu()
                mods[name]["extra_shape"] = list(S.shape)
            custom.add(name)
        elif isinstance(mod, SparseLinear):
            mods[name] = dict(cls="SparseLinear", out=mod.out_features, inp=mod.in_features, skip_zero=mod.skip_zero)
            _csr_parts(name, mod.w, T); custom.add(name)
        elif isinstance(mod, SparseEmbedding):
            mods[name] = dict(cls="SparseEmbedding", hidden=mod.hidden)
            T[name + ".C"] = mod.C.cpu(); T[name + ".V"] = mod.V.cpu(); custom.add(name)
        for attr in ("eps", "variance_epsilon"):
            if hasattr(mod, attr) and isinstance(getattr(mod, attr), float):
                mods.setdefault(name, {})[attr] = getattr(mod, attr)
    for name, p in m.named_parameters():
        if not any(name == c or name.startswith(c + ".") for c in custom):
            T[name] = p.detach().contiguous().cpu()
    if m.head.W.data_ptr() != m.embed.q.weight.data_ptr():  # an untied LM head
        T["head.W"] = m.head.W.detach().contiguous().cpu()
    if m.head.say is not None:
        S, ids = m.head.say
        _csr_parts("head.say", S.w, T); T["head.say.ids"] = ids.cpu()
    T["embed.cls_ids"] = m.embed.cls_ids.cpu()
    for name, b in m.rotary.named_buffers():  # recomputed on another device, the frequencies differ in their last bits
        T["rotary." + name] = b.detach().contiguous().cpu()
    L = m.L
    cfg = dict(text_config=m.qcfg.to_dict(), attn_implementation=m.qcfg._attn_implementation,
               fly_configs=[c.to_dict() for c in m.fly_cfgs], n_text=m.n_text,
               layer_types=m.layer_types, layout=dict(n=L.n, G=L.G, R=L.R, H=L.H, base=L.base, S=L.S, rect=L.n_rect), one=m.one,
               affine={str(k): list(v) for k, v in m.affine.items()}, inhibit={str(k): [list(t) for t in v] for k, v in m.inhibit.items()},
               say_words=m.say_words,
               answer=None if m.answer is None else {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in m.answer.items()},
               gain=m.gain, bridge="answer" if m.answer is not None else ("groups" if m.drive is not None else None),
               say_shape=list(m.head.say[0].w.shape) if m.head.say is not None else None)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    (out / "modules.json").write_text(json.dumps(mods, indent=1))
    save_file(T, str(out / "model.safetensors"))
    if tok is not None:
        tok.save_pretrained(out)
    man = {}
    for f in sorted(out.iterdir()):
        if f.is_file() and f.name != "manifest.json":
            h = hashlib.sha256()
            with open(f, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 24), b""):
                    h.update(chunk)
            man[f.name] = dict(sha256=h.hexdigest(), bytes=f.stat().st_size)
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    return man


def verify(path):
    """True if every file matches the manifest's sha256."""
    path = Path(path)
    man = json.loads((path / "manifest.json").read_text())
    for nm, e in man.items():
        h = hashlib.sha256()
        with open(path / nm, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        if h.hexdigest() != e["sha256"]:
            return False
    return True


def _set(root, path, mod):
    parent, _, leaf = path.rpartition(".")
    obj = root.get_submodule(parent) if parent else root
    setattr(obj, leaf, mod)


def load(path, device="cuda"):
    """Rebuild the one-stack model from a checkpoint directory. Returns (model, tokenizer)."""
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5RMSNorm, Qwen3_5TextRotaryEmbedding
    path = Path(path)
    cfg = json.loads((path / "config.json").read_text())
    mods = json.loads((path / "modules.json").read_text())
    T = load_file(str(path / "model.safetensors"))
    tc = Qwen3_5TextConfig(**cfg["text_config"])
    tc._attn_implementation = cfg["attn_implementation"]  # to_dict drops it; without it the rebuilt attention runs eager
    fcs = [Qwen3_5TextConfig(**c) for c in cfg["fly_configs"]]
    m = QwenFlyModel.__new__(QwenFlyModel)
    nn.Module.__init__(m)
    nq = tc.num_hidden_layers
    with torch.device("meta"):
        m.qlayers = nn.ModuleList([Qwen3_5DecoderLayer(tc, i) for i in range(nq)])
        m.flayers = nn.ModuleList([Qwen3_5DecoderLayer(c, nq + i) for i, c in enumerate(fcs)])
        m.norm = Qwen3_5RMSNorm(cfg["layout"]["H"])
    lay = cfg["layout"]
    H = lay["H"]
    for name, e in mods.items():
        if name.startswith(("head.", "embed.")):
            continue  # built below, once the embedding exists
        if e.get("cls") == "BlockLinear":
            W = T[name + ".W"].to(device)
            mr = T[name + ".mute_rows"].to(device) if name + ".mute_rows" in T else None
            ex = None
            if e["extra"]:
                ex = (_csr(name + ".extra", T, e["extra_shape"], device), T[name + ".extra.dims"].to(device))
            _set(m, name, BlockLinear(W, e["out"], mute_rows=mr, mute_col=e["mute_col"], extra=ex))
        elif e.get("cls") == "SparseLinear":
            sl = SparseLinear.__new__(SparseLinear); nn.Module.__init__(sl)
            sl.in_features, sl.out_features, sl.skip_zero = e["inp"], e["out"], e["skip_zero"]
            sl.register_buffer("w", _csr(name, T, (e["out"], e["inp"]), device), persistent=False)
            sl.empty, sl._csc = sl.w._nnz() == 0, None
            _set(m, name, sl)
        elif e.get("cls") == "SparseEmbedding":
            se = SparseEmbedding.__new__(SparseEmbedding); nn.Module.__init__(se)
            se.hidden = e["hidden"]
            se.register_buffer("C", T[name + ".C"].to(device), persistent=False)
            se.register_buffer("V", T[name + ".V"].to(device), persistent=False)
    # the Qwen embedding (tied with the LM head) and every remaining parameter
    q_embed = nn.Embedding(cfg["n_text"], tc.hidden_size, device="meta")
    m.embed = MergedEmbedding.__new__(MergedEmbedding); nn.Module.__init__(m.embed)
    m.embed.q = q_embed
    se = SparseEmbedding.__new__(SparseEmbedding); nn.Module.__init__(se)
    se.hidden = mods["embed.f"]["hidden"]
    se.register_buffer("C", T["embed.f.C"].to(device), persistent=False)
    se.register_buffer("V", T["embed.f.V"].to(device), persistent=False)
    m.embed.f, m.embed.n_text = se, cfg["n_text"]
    m.embed.register_buffer("cls_ids", T["embed.cls_ids"].to(device), persistent=False)
    for name, t in T.items():
        if any(name.startswith(p) for p in ("head.", "embed.cls_ids", "embed.f.", "rotary.")) or name.endswith((".crow", ".col", ".val", ".mute_rows", ".dims", ".C", ".V")):
            continue
        mod_path, _, leaf = name.rpartition(".")
        if mod_path in mods and mods[mod_path].get("cls") == "BlockLinear":
            continue
        obj = m.get_submodule(mod_path)
        setattr(obj, leaf, nn.Parameter(t.to(device), requires_grad=False))
    for name, e in mods.items():
        for attr in ("eps", "variance_epsilon"):
            if attr in e:
                setattr(m.get_submodule(name), attr, e[attr])
    fly_head = SparseLinear.__new__(SparseLinear); nn.Module.__init__(fly_head)
    e = mods["head.f"]
    fly_head.in_features, fly_head.out_features, fly_head.skip_zero = e["inp"], e["out"], e["skip_zero"]
    fly_head.register_buffer("w", _csr("head.f", T, (e["out"], e["inp"]), device), persistent=False)
    fly_head.empty, fly_head._csc = False, None
    say = None
    if cfg["say_shape"] is not None:
        sl = SparseLinear.__new__(SparseLinear); nn.Module.__init__(sl)
        sl.out_features, sl.in_features, sl.skip_zero = cfg["say_shape"][0], cfg["say_shape"][1], False
        sl.register_buffer("w", _csr("head.say", T, cfg["say_shape"], device), persistent=False)
        sl.empty, sl._csc = False, None
        say = (sl, T["head.say.ids"].to(device))
    m.head = MergedHead(T["head.W"].to(device) if "head.W" in T else m.embed.q.weight, fly_head, cfg["one"], say)
    with torch.device(device):
        m.rotary = Qwen3_5TextRotaryEmbedding(tc)
    for name, b in list(m.rotary.named_buffers()):
        if "rotary." + name in T:
            b.copy_(T["rotary." + name].to(device))
    left = [nm for nm, t in list(m.named_parameters()) + list(m.named_buffers()) if t.is_meta]
    assert not left, left[:5]
    narrow_norms(m.qlayers)
    for layer in m.flayers:
        fly_layers.FlyTokenNorm.install(layer.input_layernorm); fly_layers.FlyTokenNorm.install(layer.post_attention_layernorm)
    fly_layers.FlyTokenNorm.install(m.norm)
    # plain attributes
    m.n_text, m.one, m.dev, m.layer_types = cfg["n_text"], cfg["one"], device, cfg["layer_types"]
    m.L = fly_layers.Layout(lay["n"], [0] * lay["G"], [0] * lay["R"], H=H, base=lay["base"], n_say=lay["S"], n_rect=lay["rect"])
    m.affine = {int(k): tuple(v) for k, v in cfg["affine"].items()}
    m.inhibit = {int(k): [(int(p), float(c)) for p, c in v] for k, v in cfg["inhibit"].items()}
    m.say_words = [tuple(x) for x in cfg["say_words"]]
    m.answer, m.gain, m.drive = cfg["answer"], cfg["gain"], None
    m.qcfg, m.fly_cfgs = tc, fcs
    m.cfg = Qwen3_5TextConfig(**{**tc.to_dict(), "hidden_size": H, "num_hidden_layers": len(m.layer_types),
                                  "layer_types": m.layer_types, "vocab_size": m.n_text + m.L.vocab})
    tok = AutoTokenizer.from_pretrained(path)
    return m.eval(), tok
