# Judge Comparison Report: `gemini` vs `unisrm`

Sample-level and model-level agreement between two judges on EmergentTTS-Eval. Aligned by `unique_id_eval` across 8 TTS models.

---

## How to read this report

This report answers two questions about candidate judge `unisrm`: (a) **does it produce usable output** at all, and (b) **how well do its judgements agree with the baseline `gemini`?** Agreement is measured at two granularities — per individual sample (does each judgement match?) and per TTS model (do the two judges rank the 8 TTS systems the same way?). Below, every metric used in the tables is defined alongside concrete thresholds for deciding whether the value is **good enough** to call the candidate a viable replacement.

### Definitions

- **A sample** = one (text, baseline_audio, comparison_TTS_audio) triple. The judge outputs `{score_1, score_2, winner}`. EmergentTTS randomizes which audio goes to position 1 vs 2; `predicted_speech_index` records this so we can canonicalize.
- **Canonical winner**: 1 = baseline wins, 2 = comparison TTS wins, 0 = tie. After canonicalization the judge-side position randomization is removed, so the only thing being compared is the judgement.
- **A judge produces a parseable verdict** when the model output yields a JSON dict matching EmergentTTS's schema with `winner ∈ {0,1,2}`. Otherwise the framework records `winner = -1` and the sample is unusable.
- **Aligned samples** = both judges produced parseable verdicts on the same `unique_id_eval` AND chose the same audio-to-position assignment. Only aligned samples can be compared.

### Sample-level metrics

| Metric | What it measures | Good enough when |
|---|---|---|
| **Failure rate** | Fraction of samples for which the candidate produced no parseable verdict. | < 5% acceptable; 5-15% marginal; > 15% disqualifying for benchmark replacement. |
| **winrate_gemini** / **winrate_unisrm** | Among aligned samples for one TTS model, the fraction where the judge picked the comparison TTS over the baseline (ties = 0.5). Range 0-1. | Useful as a quality estimate per TTS — not a judge-quality metric on its own. Compare to baseline winrate to see if the judges agree on the TTS's quality. |
| **winrate_diff** | `winrate_gemini − winrate_unisrm` per TTS. | Sanity check on how generous one judge is vs the other for a given TTS. Large absolute values are not necessarily bad as long as the *ranking* is preserved. |
| **accuracy** | Per-TTS fraction of aligned samples where both judges output the same canonical winner ∈ {0,1,2}. | > 0.50 is non-trivial (tie class is rare so chance ≈ 0.5); > 0.65 is good; > 0.75 is strong. |
| **Cohen's κ** | Inter-rater agreement above chance on canonical winners. Corrects for both judges happening to pick the same class by accident. | Landis & Koch (1977): < 0 worse than chance; **0.0-0.20 slight; 0.21-0.40 fair; 0.41-0.60 moderate**; 0.61-0.80 substantial; > 0.80 almost perfect. For benchmark replacement aim for ≥ 0.40 (moderate). |
| **weighted κ (quadratic)** | Same as κ but penalises larger disagreements (e.g. winner=1 vs winner=2 is harsher than winner=1 vs winner=0). Treats the 3 classes as ordinal. | Same Landis & Koch thresholds. Negative values indicate systematic disagreement on the win/lose axis (worse than κ would suggest). |
| **Pearson r (other_score)** | Per-TTS correlation of the comparison-TTS's score (0-3) between the two judges. | 0 = no linear relationship; 0.3 weak; 0.5 moderate; 0.7+ strong. > 0.30 is the minimum to claim the candidate's *score signal* tracks the baseline. |
| **Pearson r (baseline_score)** | Same but for the baseline audio's score. | Often noisier (less variance), so 0.10-0.20 can already be informative. |

### Model-level metrics (across all 8 TTS systems)

| Metric | What it measures | Good enough when |
|---|---|---|
| **Spearman ρ** | Rank correlation of the 8-TTS winrate vectors of the two judges. Tells you whether they would draw the same leaderboard. Range -1 to 1. | ρ > 0.7 is strong; ρ > 0.9 near-perfect. Always check the **p-value**: with only n=8 TTS systems, ρ ≥ 0.74 is needed for p < 0.05 (two-sided). |
| **Spearman p-value** | Probability that the observed ρ arose by chance under no agreement. | **p < 0.05** = the rank agreement is statistically significant. |
| **Kendall's W** | Concordance among judges (here, m=2 raters); equivalent to Spearman ρ for n=2. Range 0-1. | W > 0.7 = good agreement; W ≈ 0.5 = no signal beyond chance. |
| **Winrate spread** (max − min across TTS) | How widely the candidate's winrates vary across the 8 TTS systems. Large = the judge distinguishes good vs bad TTS. Small = position bias / inability to discriminate. | Spread should be **comparable to the baseline's spread**. A spread < 0.05 while the baseline is > 0.20 is strong evidence of position bias. |

### How to decide *enough*

Pass / fail is task-dependent. Two common bars:

- **Leaderboard-replacement** (you only need `unisrm` to *rank* TTS systems the same way `gemini` does, e.g. for ML benchmark reporting): require **failure rate < 5%**, **winrate spread comparable to baseline**, and **Spearman ρ > 0.7 with p < 0.05**. Sample-level κ may stay low.
- **Per-sample replacement** (you need `unisrm` to render the same verdict as `gemini` on individual examples, e.g. for fine-grained TTS error analysis or reward-model training): additionally require **Cohen's κ ≥ 0.40** (moderate) and **Pearson r ≥ 0.30** on at least the categories of interest.

---

## 1. Headline numbers

- **Overall sample-level accuracy** (3-class winner): `0.5192`
- **Overall Cohen's kappa**: `0.0510`  (≥0.20 = slight, ≥0.40 = moderate)
- **Overall weighted kappa (quadratic)**: `-0.0069`
- **Overall winrate gap** (`gemini` − `unisrm`): `+0.1167`
- **Model-level Spearman ρ** (8-TTS ranking agreement): `0.7619` (p = `0.0280`)
- **Model-level Kendall's W**: `0.8810`
- **`unisrm` parsing failure rate**: `1/1600` = `0.1%`

---

## 2. Data validity (unisrm parsing failure)

Sample is a *failure* when the judge produces no valid `{score_1, score_2, winner}` JSON.

| TTS model | failure rate (unisrm) |
|---|---:|
| `HumeAI` | 1/200 = **0.5%** |
| `Qwen2.5-Omni-7B` | 0/200 = **0.0%** |
| `Sesame1B` | 0/200 = **0.0%** |
| `deepgram` | 0/200 = **0.0%** |
| `eleven_multilingual_v2` | 0/200 = **0.0%** |
| `gpt-4o-audio-preview-2024-12-17` | 0/200 = **0.0%** |
| `gpt-4o-mini-tts` | 0/200 = **0.0%** |
| `orpheus-tts-0.1-finetune-prod` | 0/200 = **0.0%** |
| **Total** | **1/1600 = 0.1%** |

### 2.1 Failure rate by category (across all TTS models)

| Category | failure rate (unisrm) | aligned & valid | accuracy vs `gemini` | Cohen's κ |
|---|---:|---:|---:|---:|
| Emotions | 0/264 = **0.0%** | 263 | 0.5703 | 0.1106 |
| Foreign Words | 1/232 = **0.4%** | 231 | 0.6537 | 0.3004 |
| Paralinguistics | 0/328 = **0.0%** | 328 | 0.6098 | 0.1524 |
| Pronunciation | 0/208 = **0.0%** | 208 | 0.7308 | 0.3441 |
| Questions | 0/264 = **0.0%** | 264 | 0.1932 | 0.0077 |
| Syntactic Complexity | 0/304 = **0.0%** | 304 | 0.4638 | 0.0663 |

### 2.2 Failure mode classification

Heuristic categorization of 1 failed samples:

| Mode | count | share |
|---|---:|---:|
| `other` | 1 | 100.0% |
| `preamble_no_json` | 0 | 0.0% |
| `refusal` | 0 | 0.0% |
| `list_returned_by_parser` | 0 | 0.0% |

- `refusal`: model says "I'm sorry", "please provide", or "unable to analyze" → no judgement attempted.
- `preamble_no_json`: model writes free-form analysis but never produces `{...}`; runs out of `max_new_tokens`.
- `list_returned_by_parser`: model wrote valid JSON, but `json_repair.loads` returned a list — schema check rejects it. (Fixable parser bug.)
- `other`: rare malformations (truncated mid-token, score out of range, etc.).

---

## 3. Sample-level agreement (per TTS model)

| TTS model | n_aligned | winrate_gemini | winrate_unisrm | winrate_diff | accuracy | Cohen's κ | weighted κ | Pearson r (other_score) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `HumeAI` | 199 | 0.4296 | 0.3869 | +0.0427 | 0.4523 | 0.0671 | -0.0278 | 0.2188 |
| `Qwen2.5-Omni-7B` | 200 | 0.2975 | 0.2650 | +0.0325 | 0.5900 | 0.1672 | -0.0769 | 0.5267 |
| `Sesame1B` | 200 | 0.1500 | 0.0200 | +0.1300 | 0.7900 | 0.0411 | 0.0027 | 0.5332 |
| `deepgram` | 200 | 0.3125 | 0.2300 | +0.0825 | 0.5300 | 0.0433 | -0.0260 | 0.2726 |
| `eleven_multilingual_v2` | 200 | 0.3475 | 0.1800 | +0.1675 | 0.4950 | 0.0096 | 0.0495 | 0.0828 |
| `gpt-4o-audio-preview-2024-12-17` | 199 | 0.6407 | 0.4724 | +0.1683 | 0.4020 | 0.0128 | 0.0160 | 0.0396 |
| `gpt-4o-mini-tts` | 200 | 0.5725 | 0.5150 | +0.0575 | 0.3750 | 0.0163 | 0.0143 | 0.0674 |
| `orpheus-tts-0.1-finetune-prod` | 200 | 0.3275 | 0.0750 | +0.2525 | 0.5950 | 0.0917 | -0.0154 | 0.3967 |
| **__OVERALL__** | **1598** | **0.3847** | **0.2680** | **+0.1167** | **0.5192** | **0.0510** | **-0.0069** | **0.2487** |

**Reading aid (Cohen's κ benchmarks)**: < 0 = worse than chance; 0.0–0.20 = slight; 0.21–0.40 = fair; 0.41–0.60 = moderate; 0.61–0.80 = substantial; > 0.80 = almost perfect (Landis & Koch 1977).

---

## 4. Position bias check

`winrate_unisrm` clustered around a constant value across all TTS models would indicate the judge is not actually distinguishing TTS quality (e.g. always picking position-2).

| | min | max | spread (max − min) | std |
|---|---:|---:|---:|---:|
| `winrate_gemini` | 0.1500 | 0.6407 | 0.4907 | 0.1481 |
| `winrate_unisrm` | 0.0200 | 0.5150 | 0.4950 | 0.1677 |

- A spread close to 0 on `unisrm` despite real quality differences across the 8 TTS models is strong evidence of position bias.

---

## 5. Model-level ranking agreement

Ranks 1–8 = best→worst by winrate (over baseline). Ties get average ranks.

| TTS model | rank_gemini | rank_unisrm | Δrank |
|---|---:|---:|---:|
| `HumeAI` | 3.0 | 3.0 | 0.0 |
| `Qwen2.5-Omni-7B` | 7.0 | 4.0 | 3.0 |
| `Sesame1B` | 8.0 | 8.0 | 0.0 |
| `deepgram` | 6.0 | 5.0 | 1.0 |
| `eleven_multilingual_v2` | 4.0 | 6.0 | 2.0 |
| `gpt-4o-audio-preview-2024-12-17` | 1.0 | 2.0 | 1.0 |
| `gpt-4o-mini-tts` | 2.0 | 1.0 | 1.0 |
| `orpheus-tts-0.1-finetune-prod` | 5.0 | 7.0 | 2.0 |

- **Spearman ρ** = `0.7619`  (p = `0.0280`)
- **Kendall's W** = `0.8810`

---

## 6. Bottom line

- **Data validity is solid**: failure rate = 0.1% (< 5%).
- **No position bias**: `unisrm` winrate spread is 0.495 (close to `gemini`'s 0.491).
- **Model-level ranking is statistically significant**: Spearman ρ = `0.762`, p = `0.028` (< 0.05); the two judges agree on how to rank the 8 TTS systems.
- **Sample-level agreement is slight (close to chance)**: Cohen's κ = `0.051` (< 0.20). Two reasonable judges often disagree at the per-sample level even when their aggregate rankings line up — this metric is not by itself disqualifying.

**Conclusion**: `unisrm` is a **viable leaderboard replacement** for `gemini` (low failure rate, no position bias, statistically significant ranking agreement). For fine-grained per-sample analysis (κ ≥ 0.40), the closed baseline is still preferable.

---

## Source files

- Per-TTS metrics CSV: `judges/results/comparison/summary_gemini_vs_unisrm.csv`
- Per-TTS metrics JSON: `judges/results/comparison/per_model_gemini_vs_unisrm.json`
- Raw judge predictions: `judges/results/unisrm/<TTS>/predictions.jsonl`
- Per-shard run logs: `judges/results/unisrm/<TTS>/shard_*/run.log`

Reproduce:

```bash
python judges/compare.py \
  --judge-a-root judges/results/gemini --judge-a-name gemini \
  --judge-b-root judges/results/unisrm --judge-b-name unisrm \
  --out-dir judges/results/comparison
python judges/generate_report.py \
  --judge-a-root judges/results/gemini --judge-a-name gemini \
  --judge-b-root judges/results/unisrm --judge-b-name unisrm \
  --out-dir judges/results/comparison
```
