"""The decision model: one causal pass over input + options (options in sorted key order, " <option> ... </option>"
delimiters) through a backbone (frozen or LoRA-tuned, optionally truncated), the joint reader over the option tokens,
a per-option scorer, and per-(question type, option-count bucket) temperatures.

Training: label-smoothed soft cross-entropy (+ ranked probability score on score questions), checkpoint chosen by mean
dev accuracy with patience, temperatures fitted on the calibration split. Defaults = the reported model (README).
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from decision_model import data as D
from decision_model.losses import fixed_sample, rps_sum, smooth_one_hot, soft_ce_sum
from decision_model.reader import BUCKETS, HEAD_D, N_TYPES, NEG, JointReader, bucket


@dataclass
class ModelConfig:
    model: str = "Qwen/Qwen3.5-9B"
    revision: str = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    readout_layer: int = 32                 # backbone truncated after this layer (32 = full depth)
    pooling: str = "marker"                 # "marker" = option's last token; "mean" = mean over its tokens
    lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lr: float = 2e-4                        # decision head
    lr_lora: float = 1e-4                   # LoRA adapters
    weight_decay: float = 0.01
    warmup_frac: float = 0.05
    n_train: int = 30_000
    batch: int = 32
    micro_batch: int = 8
    eval_every: int = 250
    patience: int = 4
    dev_per_source: int = 400
    calib_per_source: int = 300
    mix: str = "default"
    augment: tuple = ("bare_yes_no",)
    label_smoothing: float = 0.1
    w_rps: float = 1.0
    seed: int = 0
    max_steps: int = 0
    dtype: str = "bfloat16"
    out_dir: str = "results/run"


def option_order(it):
    keys = D.option_keys(it)
    return keys if it.qtype == "score" else sorted(keys)


class Backbone:
    """Text decoder truncated after `readout_layer` (layers above are never computed); optional LoRA on every linear
    projection that remains."""

    def __init__(self, cfg: ModelConfig, device="cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dtype = getattr(torch, cfg.dtype)
        self.tok = AutoTokenizer.from_pretrained(cfg.model, revision=cfg.revision)
        if "Qwen3.5" in cfg.model:
            from transformers import Qwen3_5ForConditionalGeneration
            full = Qwen3_5ForConditionalGeneration.from_pretrained(cfg.model, revision=cfg.revision, dtype=dtype,
                                                                   device_map={"": device})
            text = full.model.language_model
        else:
            full = AutoModelForCausalLM.from_pretrained(cfg.model, revision=cfg.revision, dtype=dtype,
                                                        device_map={"": device})
            text = full.model
        L = cfg.readout_layer
        text.layers = text.layers[:L]
        text.config.num_hidden_layers = L
        if getattr(text.config, "layer_types", None):
            text.config.layer_types = text.config.layer_types[:L]
        del full
        torch.cuda.empty_cache()
        text.requires_grad_(False)
        if cfg.lora:
            from peft import LoraConfig, get_peft_model
            text = get_peft_model(text, LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                                                   lora_dropout=cfg.lora_dropout, target_modules="all-linear"))
            text.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            text.config.use_cache = False
        self.text, self.lora, self.device = text, cfg.lora, device
        self.hidden = text.config.hidden_size
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0

    def lora_parameters(self):
        return [p for p in self.text.parameters() if p.requires_grad]

    def states(self, id_lists, train: bool):
        T = max(len(x) for x in id_lists)
        ids = torch.full((len(id_lists), T), self.pad_id, dtype=torch.long, device=self.device)
        att = torch.zeros((len(id_lists), T), dtype=torch.long, device=self.device)
        for i, x in enumerate(id_lists):
            ids[i, : len(x)] = torch.tensor(x, device=self.device)
            att[i, : len(x)] = 1
        self.text.train(train and self.lora)
        with torch.set_grad_enabled(train and self.lora):
            return self.text(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state


class DecisionHead(nn.Module):
    def __init__(self, hidden: int, d: int = HEAD_D, pooling: str = "marker"):
        super().__init__()
        assert pooling in ("marker", "mean"), pooling
        self.pooling = pooling
        self.reader = JointReader(hidden, d)
        self.score = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.register_buffer("temperature", torch.ones(N_TYPES, len(BUCKETS)))

    def forward(self, opt_h, opt_pad, marker_idx, opt_mask, qtype, span_start=None):
        W = self.reader(opt_h, opt_pad, qtype)
        if self.pooling == "mean":
            pos = torch.arange(W.shape[1], device=W.device)
            sel = ((pos[None, None, :] >= span_start[..., None]) & (pos[None, None, :] <= marker_idx[..., None])).float()
            m = (sel @ W) / sel.sum(-1, keepdim=True).clamp(min=1)
        else:
            m = torch.gather(W, 1, marker_idx.clamp(min=0)[..., None].expand(-1, -1, W.shape[-1]))
        return self.score(m).squeeze(-1).masked_fill(~opt_mask, NEG)

    def calibrated(self, logits, qtype, k):
        T = self.temperature[qtype, torch.tensor([bucket(int(x)) for x in k], device=logits.device)]
        return logits / T[:, None]


def build(bb: Backbone, items, train: bool):
    """-> (keys per item, inputs for DecisionHead, soft targets [B, k])."""
    seqs, plan = [], []
    for it in items:
        pre = D.prefix_ids(bb.tok, it)[0]
        keys = option_order(it)
        sfx, spans, _ = D.option_suffix_ids(bb.tok, [D.option_text(it, k) for k in keys])
        seqs.append(pre + sfx)
        plan.append((len(pre), len(sfx), spans, keys))
    H = bb.states(seqs, train).float()
    To, k = max(p[1] for p in plan), max(len(p[3]) for p in plan)
    B = len(items)
    opt_h = torch.zeros(B, To, H.shape[-1], device=H.device)
    opt_pad = torch.ones(B, To, dtype=torch.bool, device=H.device)
    marker = torch.full((B, k), -1, dtype=torch.long, device=H.device)
    start = torch.zeros((B, k), dtype=torch.long, device=H.device)
    mask = torch.zeros(B, k, dtype=torch.bool, device=H.device)
    tgt = torch.zeros(B, k, device=H.device)
    for i, (it, (np_, ns, spans, keys)) in enumerate(zip(items, plan)):
        opt_h[i, :ns] = H[i, np_: np_ + ns]
        opt_pad[i, :ns] = False
        for j, (a, e) in enumerate(spans):
            marker[i, j], start[i, j] = e - 1, a
        mask[i, : len(keys)] = True
        tgt[i, : len(keys)] = torch.tensor(D.restricted_target(it, keys), device=H.device)
    qtype = torch.tensor([D.QTYPE_ID[it.qtype] for it in items], device=H.device)
    return [p[3] for p in plan], dict(opt_h=opt_h, opt_pad=opt_pad, marker_idx=marker, opt_mask=mask, qtype=qtype,
                                      span_start=start), tgt


@torch.no_grad()
def dev_accuracy(bb, head, sample, mb):
    head.eval()
    per = {}
    for i in range(0, len(sample), mb):
        items = sample[i:i + mb]
        keys, inp, _ = build(bb, items, train=False)
        lg = head(**inp)
        for it, kk, l in zip(items, keys, lg):
            per.setdefault(it.source, []).append(kk[int(l[: len(kk)].argmax())] == it.gold)
    head.train()
    out = {s: {"n": len(v), "acc": float(np.mean(v))} for s, v in per.items()}
    out["_macro_acc"] = float(np.mean([v["acc"] for v in out.values()]))
    return out


@torch.no_grad()
def fit_temperatures(bb, head, sample, mb):
    rows = {}
    head.eval()
    for i in range(0, len(sample), mb):
        items = sample[i:i + mb]
        keys, inp, _ = build(bb, items, train=False)
        lg = head(**inp)
        for it, kk, l in zip(items, keys, lg):
            key = (D.QTYPE_ID[it.qtype], bucket(len(kk)))
            rows.setdefault(key, []).append((l[: len(kk)].float().cpu(), kk.index(it.gold)))
    report = {}
    for (q, b), rs in rows.items():
        logT = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=200)

        def closure():
            opt.zero_grad()
            loss = sum(F.cross_entropy((l / logT.exp())[None], torch.tensor([g])) for l, g in rs) / len(rs)
            loss.backward()
            return loss
        with torch.enable_grad():
            opt.step(closure)
        T = float(logT.exp().clamp(0.05, 20))
        head.temperature[q, b] = T
        report[f"{q}|{b}"] = {"n": len(rs), "T": round(T, 3)}
    head.train()
    return report


def train(cfg: ModelConfig, on_eval=None):
    t0 = time.time()
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    bb = Backbone(cfg)
    head = DecisionHead(bb.hidden, pooling=cfg.pooling).cuda()
    pools = D.load_all_splits(sources=D.MIXES[cfg.mix], augment=tuple(cfg.augment))
    D.assert_disjoint(*[[it for v in pools[s].values() for it in v] for s in ("train", "dev", "calib")])
    plan = D.sample_plan(pools["train"], cfg.n_train, cfg.seed, mix=D.MIXES[cfg.mix])
    D.plan_manifest(plan, out / "plan.json")
    dev_sample = fixed_sample(pools["dev"], cfg.dev_per_source, cfg.seed)
    calib_sample = fixed_sample(pools["calib"], cfg.calib_per_source, cfg.seed + 1)
    groups = [{"params": list(head.parameters()), "lr": cfg.lr}]
    if cfg.lora:
        groups.append({"params": bb.lora_parameters(), "lr": cfg.lr_lora})
    opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    steps = math.ceil(len(plan.items) / cfg.batch)
    if cfg.max_steps:
        steps = min(steps, cfg.max_steps)
    warm = max(1, int(cfg.warmup_frac * steps))

    def lr_factor(s):                        # linear warm-up, then cosine decay
        return min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))

    def snap():
        lora = {k: v.cpu() for k, v in bb.text.state_dict().items() if "lora_" in k} if cfg.lora else {}
        return {"head": {k: v.cpu() for k, v in head.state_dict().items()}, "lora": lora}

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    resume, start = out / "resume.pt", 0
    if resume.exists():
        rs = torch.load(resume, map_location="cpu", weights_only=False)
        assert rs["plan_digest"] == plan.digest()
        head.load_state_dict(rs["state"]["head"])
        if cfg.lora:
            bb.text.load_state_dict(rs["state"]["lora"], strict=False)
        opt.load_state_dict(rs["opt"]); sched.load_state_dict(rs["sched"])
        random.setstate(rs["py_rng"]); np.random.set_state(rs["np_rng"]); torch.set_rng_state(rs["torch_rng"])
        start, best, best_step, bad, run_loss = rs["step"], rs["best"], rs["best_step"], rs["bad"], rs["run_loss"]
        lines = [l for l in open(out / "log.jsonl") if l.strip() and json.loads(l)["step"] <= start]
        (out / "log.jsonl").write_text("".join(lines))
        log = open(out / "log.jsonl", "a")
    else:
        log = open(out / "log.jsonl", "w")
        base = dev_accuracy(bb, head, dev_sample, cfg.micro_batch)
        log.write(json.dumps({"step": 0, "dev": base}) + "\n"); log.flush()
        best, best_step, bad, run_loss = base["_macro_acc"], 0, 0, []
        torch.save(snap(), out / "best.pt")
    params = [p for g in groups for p in g["params"]]
    head.train()
    for step in range(start, steps):
        batch_items = plan.items[step * cfg.batch:(step + 1) * cfg.batch]
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for i in range(0, len(batch_items), cfg.micro_batch):
            items = batch_items[i:i + cfg.micro_batch]
            _, inp, tgt = build(bb, items, train=True)
            lg = head(**inp)
            tgt = smooth_one_hot(tgt, inp["opt_mask"], cfg.label_smoothing)
            s_loss, _ = soft_ce_sum(lg, tgt, inp["opt_mask"])
            if cfg.w_rps:
                s_loss = s_loss + cfg.w_rps * rps_sum(lg, tgt, inp["opt_mask"], inp["qtype"] == D.QTYPE_ID["score"])
            (s_loss / len(batch_items)).backward()
            tot += s_loss.item() / len(batch_items)
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        assert math.isfinite(gn), f"non-finite gradient at step {step}"
        opt.step(); sched.step()
        run_loss.append(tot)
        if (step + 1) % cfg.eval_every == 0 or step == steps - 1:
            dev = dev_accuracy(bb, head, dev_sample, cfg.micro_batch)
            rec = {"step": step + 1, "train_loss": float(np.mean(run_loss[-cfg.eval_every:])), "grad_norm": gn,
                   "dev": dev, "minutes": (time.time() - t0) / 60}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(json.dumps({k: rec[k] for k in ("step", "train_loss", "grad_norm", "minutes")}),
                  "dev macro", round(dev["_macro_acc"], 4), flush=True)
            if dev["_macro_acc"] > best + 1e-6:
                best, best_step, bad = dev["_macro_acc"], step + 1, 0
                torch.save(snap(), out / "best.pt")
            else:
                bad += 1
            torch.save({"step": step + 1, "plan_digest": plan.digest(), "state": snap(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "best": best, "best_step": best_step, "bad": bad,
                        "run_loss": run_loss, "py_rng": random.getstate(), "np_rng": np.random.get_state(),
                        "torch_rng": torch.get_rng_state()}, resume)
            if on_eval:
                on_eval()
            if bad >= cfg.patience:
                break
    st = torch.load(out / "best.pt", map_location="cpu")
    head.load_state_dict(st["head"])
    if cfg.lora:
        bb.text.load_state_dict(st["lora"], strict=False)
    temps = fit_temperatures(bb, head, calib_sample, cfg.micro_batch)
    final = dev_accuracy(bb, head, dev_sample, cfg.micro_batch)
    torch.save({"head": {k: v.cpu() for k, v in head.state_dict().items()}, "lora": st["lora"], "config": asdict(cfg)},
               out / "final.pt")
    summary = {"config": asdict(cfg), "plan_digest": plan.digest(), "best_step": best_step, "dev_best": final,
               "temperatures": temps, "minutes": (time.time() - t0) / 60,
               "trainable_head": sum(p.numel() for p in head.parameters()),
               "trainable_lora": sum(p.numel() for p in bb.lora_parameters()) if cfg.lora else 0}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    if on_eval:
        on_eval()
    return summary


def config_from_dict(d: dict) -> ModelConfig:
    """ModelConfig from a saved or user-supplied dict; keys this version does not know are ignored."""
    known = ModelConfig.__dataclass_fields__
    return ModelConfig(**{**ModelConfig().__dict__,
                          **{k: (tuple(v) if k == "augment" else v) for k, v in d.items() if k in known}})


def load_final(path, device="cuda"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = config_from_dict(ck["config"])
    bb = Backbone(cfg, device)
    if cfg.lora:
        bb.text.load_state_dict(ck["lora"], strict=False)
    head = DecisionHead(bb.hidden, pooling=cfg.pooling).to(device)
    head.load_state_dict(ck["head"])
    head.eval()
    return cfg, bb, head


@torch.no_grad()
def predict(bb, head, items, mb=8):
    """-> {item id: {option key: calibrated probability}} (every option is read; no shortlist)."""
    out = {}
    for i in range(0, len(items), mb):
        chunk = items[i:i + mb]
        keys, inp, _ = build(bb, chunk, train=False)
        k = inp["opt_mask"].sum(1)
        p = torch.softmax(head.calibrated(head(**inp), inp["qtype"], k.tolist()), -1)
        for it, kk, pp in zip(chunk, keys, p):
            out[it.id] = dict(zip(kk, pp[: len(kk)].tolist()))
    return out
