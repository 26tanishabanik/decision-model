# decision-model

A decision model reads an input (a support ticket, an email, an agent's log, a document) and a typed question (pick
one option, answer yes or no, or give a score on a scale) and returns a probability for every allowed answer, in one
forward pass and without generating text.

This began as a learning project and grew into research on how such models decide. It is a work in progress.

## Where it started
Some decision models score the input and each answer separately, one vector each, compared by similarity. They often
cannot tell an answer that is merely *related* to the input from one that is actually *correct*. The project began by
finding out where that distinction gets lost.

## What was explored
- **Representations:** single vectors, token-level matching and joint reading, on claim verification and adversarial
  NLI. Joint reading with relational training is what works.
- **Training choices:** objective, backbone adaptation, hard negatives and calibration fixes for yes/no answers.
- **Interpretability:** where in the network the decision forms, and how it changes with model size.
- **Option order:** making answers independent of the order of the options.
- **Backbone, depth and fine-tuning:** a newer base model, reading a middle layer, and LoRA.

## The model
```
[ input + question ] [ <option> A </option> ] [ <option> B </option> ] ...    (options sorted by name)
          │  one pass through Qwen3.5-9B with LoRA (rank 16)
          ▼
  option-token states ──► joint reader (2 Transformer layers; every option sees every other option)
          ▼
  one score per option ──► temperature per question type ──► probabilities
```
- **Training:** 30,000 examples; cross-entropy with label smoothing, plus a ranked loss for score questions.
- **Training data:** VitaminC, SNLI, WANLI, CLINC150, MASSIVE, Banking77, ARC, BoolQ, Civil Comments, HelpSteer2,
  typed-decisions, two prompt-injection sets and ASSEBench. Each dataset's license is listed in `data/licenses.json`.
  Every yes/no item is also used as a bare question with no answer descriptions.

## Results
| Evaluation | Earlier version | Current |
|---|---|---|
| 12 frozen test sets (mean accuracy) | 0.771 | **0.866** |
| 267 scored queries written by four separate AI agents | 0.708 | **0.910** |
| 61 scored queries written by another AI agent | 0.754 | **0.918** |

The earlier version is a frozen Qwen3-8B with a separately trained option encoder, on the same training data. Details:
`reports/RESULTS.md`.

## What is left
- **Long inputs:** only the last 2,048 tokens are read, so evidence early in a long document is lost.
- **Counting and multi-step arithmetic:** still fail.
- **Intent sets with 77–151 labels:** 0.84–0.93 accuracy.
- **SciFact claim verification:** ranking is as good as the frozen models', but the decision threshold leans towards
  "contradict" (−10 points).
- **Next runs:** a second seed, LoRA with mid-depth reading, and more training data.

## Quick start
```bash
pip install -e ".[dev]"
pytest -q tests

python scripts/build_eval_sets.py               # frozen test sets (downloads SciFact on first run)
python scripts/build_train_data.py              # training sources
python scripts/gate_sources.py                  # quality gates, incl. removing near-duplicates of test items
python scripts/build_nli_yes_no.py
python scripts/build_injection_agent_data.py

# GPU jobs on Modal
modal volume put decision-model data/train /data/train
modal volume put decision-model data/eval /data/eval
modal volume put decision-model data/heldout.jsonl /data/heldout.jsonl
modal deploy modal_app/jobs.py
python scripts/launch.py train --name lora      # train, then predict on the frozen test sets
modal volume get decision-model /predictions/lora cache/predictions/lora
python scripts/report.py --runs lora=cache/predictions/lora
```

## Layout
| Path | Contents |
|---|---|
| `src/decision_model/` | model, joint reader, data pipeline, losses, metrics, test-set builders, one builder per source |
| `modal_app/jobs.py` | GPU jobs: train, predict, answer a query file |
| `scripts/` | data builders and quality gates, `launch.py`, `predict.py`, `report.py`, `queries_report.py`, `plot_progress.py` |
| `data/` | dataset licenses, test-set manifest, AI-written query sets (data files are rebuilt by the scripts) |
| `reports/RESULTS.md`, `results/` | results, and each model's answers to the query sets |
| `tests/` | unit tests |

## License
Apache License 2.0.
