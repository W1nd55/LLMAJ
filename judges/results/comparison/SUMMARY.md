# Judge Leaderboard vs `gemini`

Side-by-side comparison of 2 candidate judges against the baseline `gemini` across 8 TTS models on EmergentTTS-Eval.

**Candidates evaluated**: `qwen2-audio`, `unisrm`

---

## 1. Headline scoreboard

| Metric | `qwen2-audio` | `unisrm` | Better when |
|---|---:|---:|---|
| Failure rate | 25.1% | **0.1%** | lower |
| Aligned samples (out of 1600) | 1199 | **1598** | higher |
| Sample accuracy (3-class) | 0.4776 | **0.5192** | higher |
| Sample Cohen's κ | **0.0762** | 0.0510 | higher |
| Sample Pearson r (other_score) | -0.0223 | **0.2487** | higher |
| Model Spearman ρ | 0.5774 | **0.7619** | higher |
| Sample Spearman p-value | 0.1340 | **0.0280** | lower |
| Model Kendall W | 0.5000 | **0.8810** | higher |
| Winrate spread | 0.0034 | **0.4950** | higher |

- **Failure rate** = fraction of samples where the candidate produced no parseable verdict.
- **Aligned samples** = both judges produced valid winners; smaller means more data lost.
- **Sample κ < 0.20 = slight, 0.21–0.40 = fair, 0.41–0.60 = moderate** (Landis & Koch 1977).
- **Spearman p-value < 0.05** means the model-level rank agreement is statistically significant.
- **Winrate spread close to 0** indicates position bias (same winrate for every TTS).

---

## 2. Per-TTS winrate (baseline vs candidates)

| TTS model | `gemini` | `qwen2-audio` | `unisrm` |
|---|---:|---:|---:|
| `HumeAI` | 0.4400 | 0.5133 | 0.3869 |
| `Qwen2.5-Omni-7B` | 0.2900 | 0.5133 | 0.2650 |
| `Sesame1B` | 0.1167 | 0.5133 | 0.0200 |
| `deepgram` | 0.2767 | 0.5133 | 0.2300 |
| `eleven_multilingual_v2` | 0.3767 | 0.5133 | 0.1800 |
| `gpt-4o-audio-preview-2024-12-17` | 0.6846 | 0.5168 | 0.4724 |
| `gpt-4o-mini-tts` | 0.5967 | 0.5133 | 0.5150 |
| `orpheus-tts-0.1-finetune-prod` | 0.3300 | 0.5133 | 0.0750 |

### 2.1 Per-TTS ranks (best=1)

| TTS model | rank_gemini | rank_qwen2-audio | rank_unisrm |
|---|---:|---:|---:|
| `HumeAI` | 3.0 | 5.0 | 3.0 |
| `Qwen2.5-Omni-7B` | 6.0 | 5.0 | 4.0 |
| `Sesame1B` | 8.0 | 5.0 | 8.0 |
| `deepgram` | 7.0 | 5.0 | 5.0 |
| `eleven_multilingual_v2` | 4.0 | 5.0 | 6.0 |
| `gpt-4o-audio-preview-2024-12-17` | 1.0 | 1.0 | 2.0 |
| `gpt-4o-mini-tts` | 2.0 | 5.0 | 1.0 |
| `orpheus-tts-0.1-finetune-prod` | 5.0 | 5.0 | 7.0 |

---

## 3. Per-category breakdown

### 3.1 Failure rate per category

| Category | `qwen2-audio` | `unisrm` |
|---|---:|---:|
| Emotions | 6.1% | **0.0%** |
| Foreign Words | 10.3% | **0.4%** |
| Paralinguistics | 4.9% | **0.0%** |
| Pronunciation | 53.8% | **0.0%** |
| Questions | 84.8% | **0.0%** |
| Syntactic Complexity | 2.6% | **0.0%** |

### 3.2 Sample accuracy per category (vs baseline)

| Category | `qwen2-audio` | `unisrm` |
|---|---:|---:|
| Emotions | **0.6397** | 0.5703 |
| Foreign Words | 0.5048 | **0.6537** |
| Paralinguistics | **0.6122** | 0.6098 |
| Pronunciation | 0.4375 | **0.7308** |
| Questions | **0.2000** | 0.1932 |
| Syntactic Complexity | 0.2601 | **0.4638** |

### 3.3 Cohen's κ per category (vs baseline)

| Category | `qwen2-audio` | `unisrm` |
|---|---:|---:|
| Emotions | **0.2858** | 0.1106 |
| Foreign Words | 0.1460 | **0.3004** |
| Paralinguistics | **0.2673** | 0.1524 |
| Pronunciation | -0.0269 | **0.3441** |
| Questions | **0.0588** | 0.0077 |
| Syntactic Complexity | -0.2300 | **0.0663** |

---

## 4. Verdict

Weighted-best aggregation across 7 metrics:

- `unisrm`: **12** wins
- `qwen2-audio`: **1** wins

**Best candidate replacement for `gemini`**: `unisrm`.

`unisrm` shows statistically significant model-level ranking agreement with `gemini` (Spearman ρ = `0.762`, p = `0.028`) and a low failure rate of `0.1%`. It is a viable replacement when the goal is to rank TTS systems on the EmergentTTS-Eval benchmark; sample-level disagreement with the baseline remains substantial, so for fine-grained per-sample analyses the closed baseline is still preferable.

---

## Source files

- `gemini` vs `qwen2-audio`:
  - CSV: `judges/results/comparison/summary_gemini_vs_qwen2-audio.csv`
  - JSON: `judges/results/comparison/per_model_gemini_vs_qwen2-audio.json`
  - Full report: `judges/results/comparison/REPORT_gemini_vs_qwen2-audio.md`
- `gemini` vs `unisrm`:
  - CSV: `judges/results/comparison/summary_gemini_vs_unisrm.csv`
  - JSON: `judges/results/comparison/per_model_gemini_vs_unisrm.json`
  - Full report: `judges/results/comparison/REPORT_gemini_vs_unisrm.md`