"""GPU jobs on Modal: train, predict, answer a query file.

    modal deploy modal_app/jobs.py
    python scripts/launch.py train --name lora                   # the default configuration
    python scripts/launch.py train --name frozen_l20_mean --set lora=false readout_layer=20 pooling=mean

Volume "decision-model": data/train/*.jsonl, data/eval/ and data/heldout.jsonl (built by scripts/build_*.py and
scripts/gate_sources.py, then uploaded), runs/<name>/ (checkpoints, logs), predictions/<name>/, hf/ (model cache).
"""
import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
ENV = {"HF_HOME": "/vol/hf", "PYTHONPATH": "/root/src", "TOKENIZERS_PARALLELISM": "false"}
IGN = ["**/__pycache__/**"]
if modal.is_local():
    image = (modal.Image.debian_slim(python_version="3.11").pip_install(
        "torch==2.11.0", "transformers==5.10.2", "peft", "flash-linear-attention", "tilelang", "accelerate", "safetensors",
        "huggingface_hub", "numpy", "scikit-learn", "scipy", "pillow", "torchvision")
        .env(ENV).add_local_dir(ROOT / "src", "/root/src", ignore=IGN)
        .add_local_dir(ROOT / "scripts", "/root/scripts", ignore=IGN))
else:
    image = modal.Image.debian_slim()
vol = modal.Volume.from_name("decision-model", create_if_missing=True)
app = modal.App("decision-model", image=image)


def _data():
    import os
    os.makedirs("/root/data", exist_ok=True)
    for name in ("train", "eval"):
        if not os.path.exists(f"/root/data/{name}"):
            os.symlink(f"/vol/data/{name}", f"/root/data/{name}")


@app.function(gpu="H100!", volumes={"/vol": vol}, timeout=20 * 3600, memory=131072, retries=2)
def train_h100(overrides: dict):
    """Frozen-backbone runs (forward only through the backbone). Resumes from resume.pt after a pre-emption."""
    _data()
    from decision_model.model import config_from_dict, train
    s = train(config_from_dict(overrides), on_eval=vol.commit)
    vol.commit()
    return json.dumps({k: s[k] for k in ("best_step", "minutes", "trainable_head", "trainable_lora")})


@app.function(gpu="A100-80GB", volumes={"/vol": vol}, timeout=20 * 3600, memory=131072, retries=2)
def train_a100(overrides: dict):
    """LoRA runs. The linear-attention kernels do not run their backward pass on H100 with the pinned Triton version,
    so training through the backbone uses A100s."""
    return train_h100.local(overrides)


@app.function(gpu="H100!", volumes={"/vol": vol}, timeout=6 * 3600, memory=131072, retries=1)
def predict(final: str, name: str, extra: str = ""):
    """Frozen test sets (+ optional extra jsonl items) -> predictions/<name>/ on the volume."""
    import os
    import subprocess
    _data()
    cmd = ["python", "scripts/predict.py", "--final", final, "--out", f"/vol/predictions/{name}"]
    if extra:
        cmd += ["--extra", extra]
    r = subprocess.run(cmd, cwd="/root", env={**os.environ})
    vol.commit()
    if r.returncode:
        raise RuntimeError(f"predict exited {r.returncode}")
    return "ok"


@app.function(gpu="H100!", volumes={"/vol": vol}, timeout=3600, memory=131072)
def answer_queries(final: str, queries_json: str) -> str:
    """A trained model's answers to a query file. -> JSON {query name: {option: probability}}."""
    _data()
    from decision_model import model as M
    from decision_model.schema import Item
    qs = json.loads(queries_json)
    cfg, bb, head = M.load_final(final)
    items = [Item(id=q["name"], source="query", split="dev", label_source="human", license="-", group=q["name"],
                  state=q["state"] or " ", qtype=q["type"], instructions=q["question"], criteria=q["options"],
                  target={q["expect"]: 1.0}, gold=q["expect"]) for q in qs]
    return json.dumps(M.predict(bb, head, items))


@app.function(volumes={"/vol": vol}, timeout=24 * 3600, memory=4096)
def train_then_predict(overrides: dict, name: str):
    """Server-side chain (survives the local machine): train, then predict on the frozen sets.
    Log: runs/<name>/chain.json on the volume."""
    out = Path(f"/vol/runs/{name}")
    out.mkdir(parents=True, exist_ok=True)
    overrides = {**overrides, "out_dir": str(out)}
    log = {}

    def save():
        (out / "chain.json").write_text(json.dumps(log, indent=1, default=str))
        vol.commit()
    c = (train_a100 if overrides.get("lora", True) else train_h100).spawn(overrides)
    log["train"] = c.object_id; save()
    try:
        log["train_result"] = c.get()
        p = predict.spawn(f"{out}/final.pt", name, extra="/vol/data/heldout.jsonl")
        log["predict"] = p.object_id; save()
        log["predict_result"] = p.get()
    except Exception as e:
        log["error"] = repr(e)[:3000]
    save()
    return json.dumps(log)
