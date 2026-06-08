# `judges/results/comparison/` — How to read these files

This directory contains the **head-to-head comparisons** of two open-source candidate
judges (Qwen2-Audio-7B and UniSRM) against the closed-source baseline judge
**Gemini-2.5-Pro** on the [EmergentTTS-Eval](https://huggingface.co/datasets/bosonai/EmergentTTS-Eval)
benchmark.

> **The question we set out to answer:** *Can an open-source audio model replace
> Gemini-2.5-Pro as the judge for EmergentTTS-Eval?*

Each comparison is run on **the same 1,600 (sample, TTS-model) pairs**: 200 samples
× 8 core TTS systems × 6 categories.

---

## Read in this order

| If you want… | Open this |
|---|---|
| The 30-second answer | **`SUMMARY.md`** |
| Full breakdown of one candidate | `REPORT_gemini_vs_unisrm.md` (the viable candidate) or `REPORT_gemini_vs_qwen2-audio.md` |
| Headline numbers in a spreadsheet | `summary_gemini_vs_<judge>.csv` |
| Per-TTS / per-category metrics in JSON | `per_model_gemini_vs_<judge>.json` |

---

## TL;DR

| | Qwen2-Audio-7B | **UniSRM** |
|---|---|---|
| Failure rate (no parseable verdict) | **25.1%** ⛔ | **0.1%** ✅ |
| Model-level Spearman ρ vs Gemini | 0.58 (p=0.13, n.s.) | **0.76 (p=0.028, sig.)** |
| Position bias | Severe (winrate ≈ 0.51 for every TTS) | None |
| Sample-level Cohen's κ | 0.08 | 0.05 |
| Verdict | **Not viable** | **Viable for ranking, not for fine-grained per-sample analysis** |

**Bottom line:** if your goal is to *rank* TTS systems on EmergentTTS-Eval, UniSRM
is a usable open-source replacement for Gemini. If you need Gemini-grade *per-sample*
agreement (e.g. for fine-grained error analysis), the closed baseline still wins.

See `SUMMARY.md` for the full leaderboard and `REPORT_gemini_vs_unisrm.md` for
the deep dive.

---

## File-by-file

### `SUMMARY.md` — Leaderboard
Side-by-side scoreboard of every candidate vs the baseline. Has:

1. Headline-metric table (failure rate, accuracy, κ, Spearman ρ, Kendall W, …).
2. Per-TTS winrate from each judge (so you can see *which* TTS each judge favours).
3. Per-category failure-rate and accuracy breakdown (so you can see where each
   candidate breaks down — e.g. Qwen2-Audio fails on 84.8% of `Questions`).
4. Auto-generated verdict picking the winning candidate by counting "best" wins
   across the 7 metrics.

**Start here.**

### `REPORT_gemini_vs_<judge>.md` — Full-diagnostic report per candidate
For each candidate judge, a comprehensive Markdown report covering:

- **§1 Headline numbers** — same as `SUMMARY.md` but only for this candidate.
- **§2 Data validity** — failure rate by category, the actual JSON-parsing error
  modes (preamble before JSON, refusal, schema mismatch, etc.).
- **§3 Sample-level agreement** — accuracy and Cohen's κ per TTS and per category.
- **§4 Score correlation** — Pearson r between the two judges' 0–3 scalar scores.
- **§5 Position-bias diagnostic** — does the candidate prefer "second audio"
  regardless of content? (winrate spread close to 0 = severe bias.)
- **§6 Model-level ranking agreement** — Spearman ρ on the 8-TTS rank order
  with significance test, plus Kendall's W.
- **§7 Verdict & glossary** — pass/fail decision with explanations of every
  metric and what threshold counts as "good enough".

### `summary_gemini_vs_<judge>.csv` — Headline tables
One row per TTS model + an `__OVERALL__` row. Columns:

| Column | Meaning |
|---|---|
| `n_aligned` | samples where both judges produced a valid winner (out of 200) |
| `winrate_a`, `winrate_b` | fraction of samples where each judge picked the comparison TTS over the baseline |
| `winrate_diff` | absolute difference (closer to 0 = more aligned in aggregate) |
| `accuracy` | fraction of aligned samples where both picked the same winner (3-class: TTS / baseline / tie) |
| `cohen_kappa` | accuracy corrected for chance agreement; range −1 → 1 (0 = chance) |
| `weighted_kappa` | κ that penalises a "TIE↔BASELINE" disagreement less than a "TTS↔BASELINE" disagreement |
| `pearson_r_other_score` | correlation of the two judges' 0–3 score for the comparison TTS audio |
| `pearson_r_baseline_score` | same, but for the baseline audio's score |
| `kendall_w` | concordance of the two judges' winner labels (1 = perfect agreement) |

Use this when you want to load the table into pandas, plot per-TTS bars, etc.

### `per_model_gemini_vs_<judge>.json` — Structured per-model + ranking metrics
Same per-TTS metrics as the CSV, plus the **model-level ranking diagnostics**
that the CSV does not have:

- `ranking_spearman_rho` — does the candidate rank the 8 TTS systems in the same
  order as Gemini? (1 = identical, −1 = reversed)
- `ranking_spearman_p` — significance of that rank correlation (8 TTS gives few
  d.o.f., so p < 0.05 already requires ρ ≳ 0.74).
- `ranking_kendall_w` — concordance coefficient of the two rankings (chance ≈ 0.5).

These three numbers are the **strongest evidence** for "leaderboard usable" or
not. UniSRM has ρ = 0.76, p = 0.028, W = 0.88. Qwen2-Audio has ρ = 0.58, p = 0.13,
W = 0.50.

---

## Glossary cheat-sheet

| Metric | Range | Read as |
|---|---|---|
| **Failure rate** | 0–100% | Below 5% = fine; above 20% = data-loss disaster. |
| **Accuracy** (3-class) | 0–1 | Chance ≈ 0.4 (because most winners are "TTS or baseline" not "tie"). |
| **Cohen's κ** | −1 → 1 | <0.2 slight, 0.2–0.4 fair, 0.4–0.6 moderate, >0.6 substantial (Landis & Koch 1977). |
| **Pearson r** | −1 → 1 | >0.3 = the two judges' raw scores move together. |
| **Spearman ρ** (rank) | −1 → 1 | >0.7 with p<0.05 on 8 items = ranking-grade agreement. |
| **Kendall's W** | 0–1 | Concordance over multiple raters; >0.7 = strong consensus. |
| **Winrate spread** | 0–1 | std-dev of per-TTS winrates; ≈0 = position bias (judge always prefers same slot). |

---

## Caveats

1. **Sample budget = 200/TTS.** Bootstrap CIs on per-TTS κ are wide (±0.07).
   The model-level ranking with n = 8 TTS is also low-power: Spearman ρ ≈ 0.74
   is the threshold for p < 0.05.
2. **Position bias is a pre-condition, not a metric.** Qwen2-Audio's
   winrate-spread of 0.0034 means it is essentially a constant function of the
   prompt — accuracy is roughly chance-shaped after that.
3. **Failure handling.** "Failure" = the judge produced output we could not parse
   into the 6-key JSON schema (winner, score_1, score_2, three reasoning fields).
   `n_aligned` excludes those rows; if a judge fails systematically on one
   category, that category's per-row metrics are computed on the surviving rows
   and are over-optimistic.
4. **No human gold.** Gemini is the *baseline*, not the truth. "UniSRM agrees
   with Gemini" ≠ "UniSRM is correct". To convert this into a true accuracy
   metric you'd need a human-rated subset.
5. **English-centric.** EmergentTTS-Eval has a `Foreign Words` category but the
   carrier text is mostly English; non-English judging quality is not directly
   measured here.

---

## How to regenerate these files

The orchestration is fully scripted from the repo root:

```bash
# 1. Run the candidate judge over all 8 TTS models (200 samples each)
JUDGE_NAME=unisrm \
  JUDGE_CFG=configs/judge_unisrm.yaml \
  OUT_ROOT=judges/results/unisrm \
  bash scripts/run_judge_4gpu.sh

# 2. Compute the comparison vs the gemini baseline → CSV + JSON
python -m judges.compare \
  --judge-a-root judges/results/gemini \
  --judge-b-root judges/results/unisrm \
  --judge-a-name gemini --judge-b-name unisrm \
  --out-dir judges/results/comparison

# 3. Render the human-readable Markdown report
python -m judges.generate_report \
  --judge-a-root judges/results/gemini \
  --judge-b-root judges/results/unisrm \
  --judge-a-name gemini --judge-b-name unisrm \
  --out judges/results/comparison/REPORT_gemini_vs_unisrm.md

# 4. Re-render the side-by-side leaderboard
python -m judges.generate_leaderboard \
  --baseline-root judges/results/gemini --baseline-name gemini \
  --candidate-roots judges/results/qwen2-audio judges/results/unisrm \
  --candidate-names qwen2-audio unisrm \
  --comparison-dir judges/results/comparison \
  --out judges/results/comparison/SUMMARY.md
```

For a **manual A/B audition** of where two judges disagree — listening to the
audio pairs and reading both judges' written reasoning side-by-side — see
[`judges/inspect_pairs.py`](../../inspect_pairs.py) (documented in the root
[`README.md`](../../../README.md#listening-to-pairs-of-audios-with-both-judges-verdicts)).
