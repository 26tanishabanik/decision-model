# Results

One training seed per model. Each comparison had its pass/fail rule fixed before the run; the frozen test sets were
used only for final scoring.

## Models
| Name | Backbone | Trained | Option summary | ms per decision (H100) |
|---|---|---|---|---|
| `lora` (current) | Qwen3.5-9B, all 32 layers | LoRA rank 16 + reader | last token | 29.4 |
| `frozen_l20_mean` | Qwen3.5-9B, first 20 layers | reader only | mean of option tokens | 12.8 |
| `frozen` | Qwen3.5-9B, all 32 layers | reader only | last token | 19.6 |
| `earlier` | Qwen3-8B with a separately trained option encoder | reader only | last token | – |

All four share the training data (30,000 examples). The first three also share the losses, checkpoint selection and
calibration.

## 12 frozen test sets (`results/frozen_eval_report.json`)
| Set | lora | frozen_l20_mean | frozen | earlier |
|---|---|---|---|---|
| ANLI | 0.633 | 0.623 | 0.598 | 0.495 |
| Banking77 (77 intents) | 0.843 | 0.752 | 0.706 | 0.665 |
| BFCL (tool selection) | 0.966 | 0.957 | 0.869 | 0.935 |
| CLINC150 (151 intents) | 0.928 | 0.809 | 0.692 | 0.794 |
| HellaSwag | 0.855 | 0.812 | 0.724 | 0.662 |
| OpenBookQA | 0.876 | 0.836 | 0.764 | 0.696 |
| Prompt injection | 0.793 | 0.638 | 0.672 | 0.647 |
| SciFact claim selection | 0.967 | 0.975 | 0.962 | 0.967 |
| SciFact claim verification | 0.827 | 0.930 | 0.928 | 0.906 |
| SciQ | 0.959 | 0.950 | 0.897 | 0.935 |
| Toxic-chat | 0.970 | 0.930 | 0.924 | 0.938 |
| Typed decisions | 0.780 | 0.663 | 0.569 | 0.615 |
| **Mean** | **0.866** | 0.823 | 0.775 | 0.771 |

- **LoRA vs frozen:** +9.1 points; better on 10 of 12 sets (paired bootstrap, 95% interval excludes 0), equal on
  SciFact claim selection, worse on SciFact claim verification.
  - On claim verification the ranking is unchanged (AUROC 0.971 vs 0.971); LoRA's threshold leans towards
    "contradict".
- **First 20 layers + mean of option tokens vs all 32 layers + last token (both frozen):** +4.7 points at 0.65× the
  time per decision.

## AI-written query sets (`data/queries/`, answers in `results/queries/`)
| Set | Scored | lora | frozen_l20_mean | frozen | earlier |
|---|---|---|---|---|---|
| Four AI agents (75 each) | 267 of 300 | 0.910 | 0.854 | 0.760 | 0.708 |
| One AI agent | 61 of 71 | 0.918 | 0.836 | 0.738 | 0.754 |

- Each set was written by separate AI agents that were not told which models would be tested and did not see the
  training data. The agents also chose the expected answers. The sets were fixed before any run; items whose answer
  is open to dispute are kept but not scored.
- The query sets are not used for training or model selection, and `lora` scores about the same on them as on the
  frozen test sets.

## Findings on frozen backbones
- **Base model:** Qwen3.5-9B's features beat Qwen3-8B's on our dev splits, so it replaced it.
- **Option summary:** averaging an option's tokens beats its last token at mid depth.
- **Differing tokens:** using only the words where options differ (as a summary, or to look up evidence in the input)
  is worse than using the whole option.
- **Common errors:** counting and multi-step arithmetic.
