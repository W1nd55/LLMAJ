# Judge Comparison Report: `gemini` vs `qwen2-audio`

Sample-level and model-level agreement between two judges on EmergentTTS-Eval. Aligned by `unique_id_eval` across 8 TTS models.

---

## How to read this report

This report answers two questions about candidate judge `qwen2-audio`: (a) **does it produce usable output** at all, and (b) **how well do its judgements agree with the baseline `gemini`?** Agreement is measured at two granularities — per individual sample (does each judgement match?) and per TTS model (do the two judges rank the 8 TTS systems the same way?). Below, every metric used in the tables is defined alongside concrete thresholds for deciding whether the value is **good enough** to call the candidate a viable replacement.

### Definitions

- **A sample** = one (text, baseline_audio, comparison_TTS_audio) triple. The judge outputs `{score_1, score_2, winner}`. EmergentTTS randomizes which audio goes to position 1 vs 2; `predicted_speech_index` records this so we can canonicalize.
- **Canonical winner**: 1 = baseline wins, 2 = comparison TTS wins, 0 = tie. After canonicalization the judge-side position randomization is removed, so the only thing being compared is the judgement.
- **A judge produces a parseable verdict** when the model output yields a JSON dict matching EmergentTTS's schema with `winner ∈ {0,1,2}`. Otherwise the framework records `winner = -1` and the sample is unusable.
- **Aligned samples** = both judges produced parseable verdicts on the same `unique_id_eval` AND chose the same audio-to-position assignment. Only aligned samples can be compared.

### Sample-level metrics

| Metric | What it measures | Good enough when |
|---|---|---|
| **Failure rate** | Fraction of samples for which the candidate produced no parseable verdict. | < 5% acceptable; 5-15% marginal; > 15% disqualifying for benchmark replacement. |
| **winrate_gemini** / **winrate_qwen2-audio** | Among aligned samples for one TTS model, the fraction where the judge picked the comparison TTS over the baseline (ties = 0.5). Range 0-1. | Useful as a quality estimate per TTS — not a judge-quality metric on its own. Compare to baseline winrate to see if the judges agree on the TTS's quality. |
| **winrate_diff** | `winrate_gemini − winrate_qwen2-audio` per TTS. | Sanity check on how generous one judge is vs the other for a given TTS. Large absolute values are not necessarily bad as long as the *ranking* is preserved. |
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

- **Leaderboard-replacement** (you only need `qwen2-audio` to *rank* TTS systems the same way `gemini` does, e.g. for ML benchmark reporting): require **failure rate < 5%**, **winrate spread comparable to baseline**, and **Spearman ρ > 0.7 with p < 0.05**. Sample-level κ may stay low.
- **Per-sample replacement** (you need `qwen2-audio` to render the same verdict as `gemini` on individual examples, e.g. for fine-grained TTS error analysis or reward-model training): additionally require **Cohen's κ ≥ 0.40** (moderate) and **Pearson r ≥ 0.30** on at least the categories of interest.

---

## 1. Headline numbers

- **Overall sample-level accuracy** (3-class winner): `0.4776`
- **Overall Cohen's kappa**: `0.0762`  (≥0.20 = slight, ≥0.40 = moderate)
- **Overall weighted kappa (quadratic)**: `0.0804`
- **Overall winrate gap** (`gemini` − `qwen2-audio`): `-0.1249`
- **Model-level Spearman ρ** (8-TTS ranking agreement): `0.5774` (p = `0.1340`)
- **Model-level Kendall's W**: `0.5000`
- **`qwen2-audio` parsing failure rate**: `400/1600` = `25.0%`

---

## 2. Data validity (qwen2-audio parsing failure)

Sample is a *failure* when the judge produces no valid `{score_1, score_2, winner}` JSON.

| TTS model | failure rate (qwen2-audio) |
|---|---:|
| `HumeAI` | 50/200 = **25.0%** |
| `Qwen2.5-Omni-7B` | 50/200 = **25.0%** |
| `Sesame1B` | 50/200 = **25.0%** |
| `deepgram` | 50/200 = **25.0%** |
| `eleven_multilingual_v2` | 50/200 = **25.0%** |
| `gpt-4o-audio-preview-2024-12-17` | 50/200 = **25.0%** |
| `gpt-4o-mini-tts` | 50/200 = **25.0%** |
| `orpheus-tts-0.1-finetune-prod` | 50/200 = **25.0%** |
| **Total** | **400/1600 = 25.0%** |

### 2.1 Failure rate by category (across all TTS models)

| Category | failure rate (qwen2-audio) | aligned & valid | accuracy vs `gemini` | Cohen's κ |
|---|---:|---:|---:|---:|
| Emotions | 16/264 = **6.1%** | 247 | 0.6397 | 0.2858 |
| Foreign Words | 24/232 = **10.3%** | 208 | 0.5048 | 0.1460 |
| Paralinguistics | 16/328 = **4.9%** | 312 | 0.6122 | 0.2673 |
| Pronunciation | 112/208 = **53.8%** | 96 | 0.4375 | -0.0269 |
| Questions | 224/264 = **84.8%** | 40 | 0.2000 | 0.0588 |
| Syntactic Complexity | 8/304 = **2.6%** | 296 | 0.2601 | -0.2300 |

### 2.2 Failure mode classification

Heuristic categorization of 400 failed samples:

| Mode | count | share |
|---|---:|---:|
| `preamble_no_json` | 216 | 54.0% |
| `refusal` | 120 | 30.0% |
| `other` | 48 | 12.0% |
| `list_returned_by_parser` | 16 | 4.0% |

- `refusal`: model says "I'm sorry", "please provide", or "unable to analyze" → no judgement attempted.
- `preamble_no_json`: model writes free-form analysis but never produces `{...}`; runs out of `max_new_tokens`.
- `list_returned_by_parser`: model wrote valid JSON, but `json_repair.loads` returned a list — schema check rejects it. (Fixable parser bug.)
- `other`: rare malformations (truncated mid-token, score out of range, etc.).

---

## 3. Sample-level agreement (per TTS model)

| TTS model | n_aligned | winrate_gemini | winrate_qwen2-audio | winrate_diff | accuracy | Cohen's κ | weighted κ | Pearson r (other_score) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `HumeAI` | 150 | 0.4400 | 0.5133 | -0.0733 | 0.4867 | 0.0967 | 0.1296 | -0.0631 |
| `Qwen2.5-Omni-7B` | 150 | 0.2900 | 0.5133 | -0.2233 | 0.4533 | 0.0391 | -0.0297 | -0.0193 |
| `Sesame1B` | 150 | 0.1167 | 0.5133 | -0.3967 | 0.4733 | 0.0129 | 0.0310 | 0.0136 |
| `deepgram` | 150 | 0.2767 | 0.5133 | -0.2367 | 0.5267 | 0.1587 | 0.1951 | -0.0010 |
| `eleven_multilingual_v2` | 150 | 0.3767 | 0.5133 | -0.1367 | 0.5133 | 0.1609 | 0.1096 | 0.0106 |
| `gpt-4o-audio-preview-2024-12-17` | 149 | 0.6846 | 0.5168 | +0.1678 | 0.4564 | 0.0191 | 0.0768 | -0.0443 |
| `gpt-4o-mini-tts` | 150 | 0.5967 | 0.5133 | +0.0833 | 0.4333 | 0.0462 | 0.0506 | -0.0525 |
| `orpheus-tts-0.1-finetune-prod` | 150 | 0.3300 | 0.5133 | -0.1833 | 0.5333 | 0.1585 | 0.0591 | 0.0141 |
| **__OVERALL__** | **1199** | **0.3889** | **0.5138** | **-0.1249** | **0.4776** | **0.0762** | **0.0804** | **-0.0223** |

**Reading aid (Cohen's κ benchmarks)**: < 0 = worse than chance; 0.0–0.20 = slight; 0.21–0.40 = fair; 0.41–0.60 = moderate; 0.61–0.80 = substantial; > 0.80 = almost perfect (Landis & Koch 1977).

---

## 4. Position bias check

`winrate_qwen2-audio` clustered around a constant value across all TTS models would indicate the judge is not actually distinguishing TTS quality (e.g. always picking position-2).

| | min | max | spread (max − min) | std |
|---|---:|---:|---:|---:|
| `winrate_gemini` | 0.1167 | 0.6846 | 0.5679 | 0.1708 |
| `winrate_qwen2-audio` | 0.5133 | 0.5168 | 0.0035 | 0.0012 |

- A spread close to 0 on `qwen2-audio` despite real quality differences across the 8 TTS models is strong evidence of position bias.

---

## 5. Model-level ranking agreement

Ranks 1–8 = best→worst by winrate (over baseline). Ties get average ranks.

| TTS model | rank_gemini | rank_qwen2-audio | Δrank |
|---|---:|---:|---:|
| `HumeAI` | 3.0 | 5.0 | 2.0 |
| `Qwen2.5-Omni-7B` | 6.0 | 5.0 | 1.0 |
| `Sesame1B` | 8.0 | 5.0 | 3.0 |
| `deepgram` | 7.0 | 5.0 | 2.0 |
| `eleven_multilingual_v2` | 4.0 | 5.0 | 1.0 |
| `gpt-4o-audio-preview-2024-12-17` | 1.0 | 1.0 | 0.0 |
| `gpt-4o-mini-tts` | 2.0 | 5.0 | 3.0 |
| `orpheus-tts-0.1-finetune-prod` | 5.0 | 5.0 | 0.0 |

- **Spearman ρ** = `0.5774`  (p = `0.1340`)
- **Kendall's W** = `0.5000`

---

## 6. Bottom line

- **Data loss is severe**: 25.0% of samples have no parseable verdict (≥ 15%).
- **Position bias detected**: `qwen2-audio` winrate spread is 0.004 vs `gemini`'s 0.568; the candidate cannot discriminate TTS quality.
- **Model-level ranking is not statistically significant**: Spearman ρ = `0.577`, p = `0.134` (≥ 0.05) — with n=8 TTS systems we cannot distinguish this from chance.
- **Sample-level agreement is slight (close to chance)**: Cohen's κ = `0.076` (< 0.20). Two reasonable judges often disagree at the per-sample level even when their aggregate rankings line up — this metric is not by itself disqualifying.

**Conclusion**: `qwen2-audio` **cannot** safely replace `gemini` under either criterion.

---

## Source files

- Per-TTS metrics CSV: `judges/results/comparison/summary_gemini_vs_qwen2-audio.csv`
- Per-TTS metrics JSON: `judges/results/comparison/per_model_gemini_vs_qwen2-audio.json`
- Raw judge predictions: `judges/results/qwen2-audio/<TTS>/predictions.jsonl`
- Per-shard run logs: `judges/results/qwen2-audio/<TTS>/shard_*/run.log`

Reproduce:

```bash
python judges/compare.py \
  --judge-a-root judges/results/gemini --judge-a-name gemini \
  --judge-b-root judges/results/qwen2-audio --judge-b-name qwen2-audio \
  --out-dir judges/results/comparison
python judges/generate_report.py \
  --judge-a-root judges/results/gemini --judge-a-name gemini \
  --judge-b-root judges/results/qwen2-audio --judge-b-name qwen2-audio \
  --out-dir judges/results/comparison
```
