"""Generate a self-contained markdown report comparing two judges.

Consumes:
    * <out_dir>/per_model_<judge_a>_vs_<judge_b>.json  (from compare.py)
    * <out_dir>/summary_<judge_a>_vs_<judge_b>.csv     (from compare.py)
    * <judge_b_root>/<tts_model>/predictions.jsonl     (raw, for failure breakdown)

Produces:
    * <out_dir>/REPORT_<judge_a>_vs_<judge_b>.md
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, cohen_kappa_score

from compare import (
    list_model_dirs,
    load_jsonl,
    normalize_records,
)


def per_category_metrics(
    judge_a_root: str, judge_b_root: str, tts_models: list[str]
) -> dict[str, dict]:
    """For each category: aligned count, fail rate (judge_b), accuracy, kappa."""
    by_cat = defaultdict(lambda: {
        "n_total_b": 0,
        "n_fail_b": 0,
        "n_total_a": 0,
        "n_fail_a": 0,
        "winners_a": [],
        "winners_b": [],
    })
    for m in tts_models:
        recs_a = load_jsonl(os.path.join(judge_a_root, m, "predictions.jsonl"))
        recs_b = load_jsonl(os.path.join(judge_b_root, m, "predictions.jsonl"))
        a_by_uid = normalize_records(recs_a)
        b_by_uid = normalize_records(recs_b)
        common = set(a_by_uid) & set(b_by_uid)

        # all-record stats per judge for failure rate (use union-of-uids per judge)
        for r in recs_b:
            cat = r.get("category", "Unknown")
            by_cat[cat]["n_total_b"] += 1
            if r["judger_output_win_rate_based"].get("winner", -1) == -1:
                by_cat[cat]["n_fail_b"] += 1
        for r in recs_a:
            uid = r.get("unique_id_eval")
            if uid not in common:
                continue
            cat = r.get("category", "Unknown")
            by_cat[cat]["n_total_a"] += 1
            if r["judger_output_win_rate_based"].get("winner", -1) == -1:
                by_cat[cat]["n_fail_a"] += 1

        for uid in common:
            a = a_by_uid[uid]
            b = b_by_uid[uid]
            if a["predicted_speech_index"] != b["predicted_speech_index"]:
                continue
            if a["canonical_winner"] is None or b["canonical_winner"] is None:
                continue
            cat = a["category"]
            by_cat[cat]["winners_a"].append(a["canonical_winner"])
            by_cat[cat]["winners_b"].append(b["canonical_winner"])

    results = {}
    for cat, d in by_cat.items():
        wa = np.asarray(d["winners_a"], dtype=int)
        wb = np.asarray(d["winners_b"], dtype=int)
        n_aligned = len(wa)
        if n_aligned > 0 and len(set(wa.tolist() + wb.tolist())) > 1:
            acc = float(accuracy_score(wa, wb))
            kap = float(cohen_kappa_score(wa, wb, labels=[0, 1, 2]))
        else:
            acc = float("nan")
            kap = float("nan")
        results[cat] = {
            "n_total_b": d["n_total_b"],
            "n_fail_b": d["n_fail_b"],
            "fail_rate_b": d["n_fail_b"] / d["n_total_b"] if d["n_total_b"] else float("nan"),
            "n_aligned": n_aligned,
            "accuracy": acc,
            "cohen_kappa": kap,
        }
    return results


def failure_modes_breakdown(judge_b_root: str, tts_models: list[str]) -> dict[str, int]:
    """Heuristic categorization of judge_b failure modes from raw model output."""
    import re

    modes = {
        "preamble_no_json": 0,
        "refusal": 0,
        "list_returned_by_parser": 0,
        "other": 0,
    }
    import json_repair as jr

    for m in tts_models:
        recs = load_jsonl(os.path.join(judge_b_root, m, "predictions.jsonl"))
        for r in recs:
            if r["judger_output_win_rate_based"].get("winner", -1) != -1:
                continue
            err_str = r["judger_output_win_rate_based"].get("reasoning_system_1", "")
            raw = err_str.split("parsed from", 1)[-1].strip() if "parsed from" in err_str else err_str

            # parser bug check first
            try:
                parsed = jr.loads(raw)
                if isinstance(parsed, list) and any(
                    isinstance(x, dict) and "winner" in x for x in parsed
                ):
                    modes["list_returned_by_parser"] += 1
                    continue
            except Exception:
                pass

            head = raw[:300].lower()
            if (
                "i'm sorry" in head
                or "i am sorry" in head
                or "unable to" in head
                or "cannot analyze" in head
                or "i'm ready to assist" in head
                or "please provide" in head
            ):
                modes["refusal"] += 1
            elif re.search(r"\b(score|winner|TTS system [12]|analysis)\b", raw, re.IGNORECASE) and "{" not in raw:
                modes["preamble_no_json"] += 1
            else:
                modes["other"] += 1
    return modes


def fmt_pct(x: float) -> str:
    return f"{x*100:.1f}%" if not np.isnan(x) else "—"


def fmt_num(x: float, ndigits: int = 4) -> str:
    return f"{x:.{ndigits}f}" if not np.isnan(x) else "—"


def generate_report(
    judge_a_root: str,
    judge_b_root: str,
    judge_a_name: str,
    judge_b_name: str,
    out_dir: str,
) -> str:
    json_path = os.path.join(out_dir, f"per_model_{judge_a_name}_vs_{judge_b_name}.json")
    csv_path = os.path.join(out_dir, f"summary_{judge_a_name}_vs_{judge_b_name}.csv")
    if not os.path.isfile(json_path) or not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Missing {json_path} or {csv_path}. Run judges/compare.py first."
        )

    summary = json.load(open(json_path))
    df = pd.read_csv(csv_path)
    tts_models = summary["shared_tts_models"]

    # raw-data derived stats
    fail_per_tts = []
    for m in tts_models:
        recs = load_jsonl(os.path.join(judge_b_root, m, "predictions.jsonl"))
        n = len(recs)
        f = sum(1 for r in recs if r["judger_output_win_rate_based"].get("winner", -1) == -1)
        fail_per_tts.append((m, f, n))

    cat_stats = per_category_metrics(judge_a_root, judge_b_root, tts_models)
    fail_modes = failure_modes_breakdown(judge_b_root, tts_models)
    total_fail = sum(f for _, f, _ in fail_per_tts)

    overall = df[df["tts_model"] == "__OVERALL__"]
    overall_row = overall.iloc[0] if len(overall) else None

    lines: list[str] = []
    lines.append(f"# Judge Comparison Report: `{judge_a_name}` vs `{judge_b_name}`")
    lines.append("")
    lines.append(
        "Sample-level and model-level agreement between two judges on EmergentTTS-Eval. "
        f"Aligned by `unique_id_eval` across {len(tts_models)} TTS models."
    )
    lines.append("")
    lines.append("---")

    # 0. Metric glossary
    lines.append("")
    lines.append("## How to read this report")
    lines.append("")
    lines.append(
        "This report answers two questions about candidate judge "
        f"`{judge_b_name}`: (a) **does it produce usable output** at all, and "
        f"(b) **how well do its judgements agree with the baseline `{judge_a_name}`?**"
        " Agreement is measured at two granularities — per individual sample (does each judgement match?) "
        "and per TTS model (do the two judges rank the 8 TTS systems the same way?). "
        "Below, every metric used in the tables is defined alongside concrete thresholds for "
        "deciding whether the value is **good enough** to call the candidate a viable replacement."
    )
    lines.append("")
    lines.append("### Definitions")
    lines.append("")
    lines.append("- **A sample** = one (text, baseline_audio, comparison_TTS_audio) triple. The judge outputs `{score_1, score_2, winner}`. EmergentTTS randomizes which audio goes to position 1 vs 2; `predicted_speech_index` records this so we can canonicalize.")
    lines.append("- **Canonical winner**: 1 = baseline wins, 2 = comparison TTS wins, 0 = tie. After canonicalization the judge-side position randomization is removed, so the only thing being compared is the judgement.")
    lines.append("- **A judge produces a parseable verdict** when the model output yields a JSON dict matching EmergentTTS's schema with `winner ∈ {0,1,2}`. Otherwise the framework records `winner = -1` and the sample is unusable.")
    lines.append("- **Aligned samples** = both judges produced parseable verdicts on the same `unique_id_eval` AND chose the same audio-to-position assignment. Only aligned samples can be compared.")
    lines.append("")
    lines.append("### Sample-level metrics")
    lines.append("")
    lines.append("| Metric | What it measures | Good enough when |")
    lines.append("|---|---|---|")
    lines.append("| **Failure rate** | Fraction of samples for which the candidate produced no parseable verdict. | < 5% acceptable; 5-15% marginal; > 15% disqualifying for benchmark replacement. |")
    lines.append(f"| **winrate_{judge_a_name}** / **winrate_{judge_b_name}** | Among aligned samples for one TTS model, the fraction where the judge picked the comparison TTS over the baseline (ties = 0.5). Range 0-1. | Useful as a quality estimate per TTS — not a judge-quality metric on its own. Compare to baseline winrate to see if the judges agree on the TTS's quality. |")
    lines.append(f"| **winrate_diff** | `winrate_{judge_a_name} − winrate_{judge_b_name}` per TTS. | Sanity check on how generous one judge is vs the other for a given TTS. Large absolute values are not necessarily bad as long as the *ranking* is preserved. |")
    lines.append("| **accuracy** | Per-TTS fraction of aligned samples where both judges output the same canonical winner ∈ {0,1,2}. | > 0.50 is non-trivial (tie class is rare so chance ≈ 0.5); > 0.65 is good; > 0.75 is strong. |")
    lines.append("| **Cohen's κ** | Inter-rater agreement above chance on canonical winners. Corrects for both judges happening to pick the same class by accident. | Landis & Koch (1977): < 0 worse than chance; **0.0-0.20 slight; 0.21-0.40 fair; 0.41-0.60 moderate**; 0.61-0.80 substantial; > 0.80 almost perfect. For benchmark replacement aim for ≥ 0.40 (moderate). |")
    lines.append("| **weighted κ (quadratic)** | Same as κ but penalises larger disagreements (e.g. winner=1 vs winner=2 is harsher than winner=1 vs winner=0). Treats the 3 classes as ordinal. | Same Landis & Koch thresholds. Negative values indicate systematic disagreement on the win/lose axis (worse than κ would suggest). |")
    lines.append("| **Pearson r (other_score)** | Per-TTS correlation of the comparison-TTS's score (0-3) between the two judges. | 0 = no linear relationship; 0.3 weak; 0.5 moderate; 0.7+ strong. > 0.30 is the minimum to claim the candidate's *score signal* tracks the baseline. |")
    lines.append("| **Pearson r (baseline_score)** | Same but for the baseline audio's score. | Often noisier (less variance), so 0.10-0.20 can already be informative. |")
    lines.append("")
    lines.append("### Model-level metrics (across all 8 TTS systems)")
    lines.append("")
    lines.append("| Metric | What it measures | Good enough when |")
    lines.append("|---|---|---|")
    lines.append("| **Spearman ρ** | Rank correlation of the 8-TTS winrate vectors of the two judges. Tells you whether they would draw the same leaderboard. Range -1 to 1. | ρ > 0.7 is strong; ρ > 0.9 near-perfect. Always check the **p-value**: with only n=8 TTS systems, ρ ≥ 0.74 is needed for p < 0.05 (two-sided). |")
    lines.append("| **Spearman p-value** | Probability that the observed ρ arose by chance under no agreement. | **p < 0.05** = the rank agreement is statistically significant. |")
    lines.append("| **Kendall's W** | Concordance among judges (here, m=2 raters); equivalent to Spearman ρ for n=2. Range 0-1. | W > 0.7 = good agreement; W ≈ 0.5 = no signal beyond chance. |")
    lines.append("| **Winrate spread** (max − min across TTS) | How widely the candidate's winrates vary across the 8 TTS systems. Large = the judge distinguishes good vs bad TTS. Small = position bias / inability to discriminate. | Spread should be **comparable to the baseline's spread**. A spread < 0.05 while the baseline is > 0.20 is strong evidence of position bias. |")
    lines.append("")
    lines.append("### How to decide *enough*")
    lines.append("")
    lines.append("Pass / fail is task-dependent. Two common bars:")
    lines.append("")
    lines.append(f"- **Leaderboard-replacement** (you only need `{judge_b_name}` to *rank* TTS systems the same way `{judge_a_name}` does, e.g. for ML benchmark reporting): require **failure rate < 5%**, **winrate spread comparable to baseline**, and **Spearman ρ > 0.7 with p < 0.05**. Sample-level κ may stay low.")
    lines.append(f"- **Per-sample replacement** (you need `{judge_b_name}` to render the same verdict as `{judge_a_name}` on individual examples, e.g. for fine-grained TTS error analysis or reward-model training): additionally require **Cohen's κ ≥ 0.40** (moderate) and **Pearson r ≥ 0.30** on at least the categories of interest.")
    lines.append("")
    lines.append("---")

    # 1. Headline numbers
    lines.append("")
    lines.append("## 1. Headline numbers")
    lines.append("")
    if overall_row is not None:
        lines.append(f"- **Overall sample-level accuracy** (3-class winner): `{overall_row['accuracy']:.4f}`")
        lines.append(f"- **Overall Cohen's kappa**: `{overall_row['cohen_kappa']:.4f}`  (≥0.20 = slight, ≥0.40 = moderate)")
        lines.append(f"- **Overall weighted kappa (quadratic)**: `{overall_row['weighted_kappa']:.4f}`")
        lines.append(f"- **Overall winrate gap** (`{judge_a_name}` − `{judge_b_name}`): `{overall_row['winrate_diff']:+.4f}`")
    lines.append(f"- **Model-level Spearman ρ** (8-TTS ranking agreement): `{summary['ranking_spearman_rho']:.4f}` (p = `{summary['ranking_spearman_p']:.4f}`)")
    lines.append(f"- **Model-level Kendall's W**: `{summary['ranking_kendall_w']:.4f}`")
    lines.append(f"- **`{judge_b_name}` parsing failure rate**: `{total_fail}/{sum(n for _,_,n in fail_per_tts)}` = `{fmt_pct(total_fail/sum(n for _,_,n in fail_per_tts))}`")
    lines.append("")
    lines.append("---")

    # 2. Per-TTS failure rate
    lines.append("")
    lines.append(f"## 2. Data validity ({judge_b_name} parsing failure)")
    lines.append("")
    lines.append("Sample is a *failure* when the judge produces no valid `{score_1, score_2, winner}` JSON.")
    lines.append("")
    lines.append(f"| TTS model | failure rate ({judge_b_name}) |")
    lines.append("|---|---:|")
    for m, f, n in fail_per_tts:
        lines.append(f"| `{m}` | {f}/{n} = **{fmt_pct(f/n)}** |")
    lines.append(f"| **Total** | **{total_fail}/{sum(n for _,_,n in fail_per_tts)} = {fmt_pct(total_fail/sum(n for _,_,n in fail_per_tts))}** |")
    lines.append("")

    # 3. Failure rate by category
    lines.append(f"### 2.1 Failure rate by category (across all TTS models)")
    lines.append("")
    lines.append(f"| Category | failure rate ({judge_b_name}) | aligned & valid | accuracy vs `{judge_a_name}` | Cohen's κ |")
    lines.append("|---|---:|---:|---:|---:|")
    for cat in sorted(cat_stats.keys()):
        c = cat_stats[cat]
        lines.append(
            f"| {cat} | {c['n_fail_b']}/{c['n_total_b']} = **{fmt_pct(c['fail_rate_b'])}** "
            f"| {c['n_aligned']} | {fmt_num(c['accuracy'], 4)} | {fmt_num(c['cohen_kappa'], 4)} |"
        )
    lines.append("")

    # 4. Failure mode classification
    lines.append("### 2.2 Failure mode classification")
    lines.append("")
    lines.append(f"Heuristic categorization of {total_fail} failed samples:")
    lines.append("")
    lines.append("| Mode | count | share |")
    lines.append("|---|---:|---:|")
    for mode, count in sorted(fail_modes.items(), key=lambda kv: -kv[1]):
        share = count / total_fail if total_fail else float("nan")
        lines.append(f"| `{mode}` | {count} | {fmt_pct(share)} |")
    lines.append("")
    lines.append("- `refusal`: model says \"I'm sorry\", \"please provide\", or \"unable to analyze\" → no judgement attempted.")
    lines.append("- `preamble_no_json`: model writes free-form analysis but never produces `{...}`; runs out of `max_new_tokens`.")
    lines.append("- `list_returned_by_parser`: model wrote valid JSON, but `json_repair.loads` returned a list — schema check rejects it. (Fixable parser bug.)")
    lines.append("- `other`: rare malformations (truncated mid-token, score out of range, etc.).")
    lines.append("")
    lines.append("---")

    # 5. Per-TTS sample-level metrics
    lines.append("")
    lines.append("## 3. Sample-level agreement (per TTS model)")
    lines.append("")
    rows = df[df["tts_model"] != "__OVERALL__"]
    lines.append(
        f"| TTS model | n_aligned | winrate_{judge_a_name} | winrate_{judge_b_name} | winrate_diff | accuracy | Cohen's κ | weighted κ | Pearson r (other_score) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in rows.iterrows():
        lines.append(
            f"| `{r['tts_model']}` | {int(r['n_aligned'])} | "
            f"{r[f'winrate_{judge_a_name}']:.4f} | {r[f'winrate_{judge_b_name}']:.4f} | "
            f"{r['winrate_diff']:+.4f} | {r['accuracy']:.4f} | {r['cohen_kappa']:.4f} | "
            f"{r['weighted_kappa']:.4f} | {r['pearson_r_other_score']:.4f} |"
        )
    if overall_row is not None:
        lines.append(
            f"| **__OVERALL__** | **{int(overall_row['n_aligned'])}** | "
            f"**{overall_row[f'winrate_{judge_a_name}']:.4f}** | **{overall_row[f'winrate_{judge_b_name}']:.4f}** | "
            f"**{overall_row['winrate_diff']:+.4f}** | **{overall_row['accuracy']:.4f}** | **{overall_row['cohen_kappa']:.4f}** | "
            f"**{overall_row['weighted_kappa']:.4f}** | **{overall_row['pearson_r_other_score']:.4f}** |"
        )
    lines.append("")
    lines.append("**Reading aid (Cohen's κ benchmarks)**: < 0 = worse than chance; 0.0–0.20 = slight; 0.21–0.40 = fair; 0.41–0.60 = moderate; 0.61–0.80 = substantial; > 0.80 = almost perfect (Landis & Koch 1977).")
    lines.append("")
    lines.append("---")

    # 6. Position bias
    lines.append("")
    lines.append("## 4. Position bias check")
    lines.append("")
    lines.append(
        f"`winrate_{judge_b_name}` clustered around a constant value across all TTS models would indicate "
        f"the judge is not actually distinguishing TTS quality (e.g. always picking position-2)."
    )
    lines.append("")
    wr_b_values = rows[f"winrate_{judge_b_name}"].values
    wr_a_values = rows[f"winrate_{judge_a_name}"].values
    lines.append(f"| | min | max | spread (max − min) | std |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(f"| `winrate_{judge_a_name}` | {wr_a_values.min():.4f} | {wr_a_values.max():.4f} | {wr_a_values.max()-wr_a_values.min():.4f} | {wr_a_values.std():.4f} |")
    lines.append(f"| `winrate_{judge_b_name}` | {wr_b_values.min():.4f} | {wr_b_values.max():.4f} | {wr_b_values.max()-wr_b_values.min():.4f} | {wr_b_values.std():.4f} |")
    lines.append("")
    lines.append(
        f"- A spread close to 0 on `{judge_b_name}` despite real quality differences across the {len(tts_models)} TTS models is strong evidence of position bias."
    )
    lines.append("")
    lines.append("---")

    # 7. Model-level ranking
    lines.append("")
    lines.append("## 5. Model-level ranking agreement")
    lines.append("")
    a_rank = pd.Series(wr_a_values, index=rows["tts_model"].values).rank(ascending=False, method="average")
    b_rank = pd.Series(wr_b_values, index=rows["tts_model"].values).rank(ascending=False, method="average")
    lines.append(
        f"Ranks 1–{len(tts_models)} = best→worst by winrate (over baseline). Ties get average ranks."
    )
    lines.append("")
    lines.append(f"| TTS model | rank_{judge_a_name} | rank_{judge_b_name} | Δrank |")
    lines.append("|---|---:|---:|---:|")
    for m in tts_models:
        ra, rb = a_rank[m], b_rank[m]
        lines.append(f"| `{m}` | {ra:.1f} | {rb:.1f} | {abs(ra-rb):.1f} |")
    lines.append("")
    lines.append(f"- **Spearman ρ** = `{summary['ranking_spearman_rho']:.4f}`  (p = `{summary['ranking_spearman_p']:.4f}`)")
    lines.append(f"- **Kendall's W** = `{summary['ranking_kendall_w']:.4f}`")
    lines.append("")
    lines.append("---")

    # 8. Conclusion
    lines.append("")
    lines.append("## 6. Bottom line")
    lines.append("")
    spread_b = wr_b_values.max() - wr_b_values.min()
    spread_a = wr_a_values.max() - wr_a_values.min()
    overall_kappa = float(overall_row["cohen_kappa"]) if overall_row is not None else float("nan")
    overall_acc = float(overall_row["accuracy"]) if overall_row is not None else float("nan")
    fail_rate = total_fail / sum(n for _, _, n in fail_per_tts)

    spearman_rho = float(summary.get("ranking_spearman_rho", float("nan")))
    spearman_p = float(summary.get("ranking_spearman_p", float("nan")))

    leaderboard_pass = (
        fail_rate < 0.05
        and not (spread_b < 0.05 and spread_a > 0.20)
        and not np.isnan(spearman_p)
        and spearman_p < 0.05
        and spearman_rho > 0.70
    )
    sample_pass = (
        not np.isnan(overall_kappa) and overall_kappa >= 0.40
    )

    findings = []
    if fail_rate < 0.05:
        findings.append(f"**Data validity is solid**: failure rate = {fmt_pct(fail_rate)} (< 5%).")
    elif fail_rate < 0.15:
        findings.append(f"**Data validity is marginal**: failure rate = {fmt_pct(fail_rate)} (5-15%).")
    else:
        findings.append(f"**Data loss is severe**: {fmt_pct(fail_rate)} of samples have no parseable verdict (≥ 15%).")

    if spread_b < 0.05 and spread_a > 0.20:
        findings.append(
            f"**Position bias detected**: `{judge_b_name}` winrate spread is {spread_b:.3f} vs `{judge_a_name}`'s {spread_a:.3f}; the candidate cannot discriminate TTS quality."
        )
    elif spread_b >= max(spread_a * 0.6, 0.20):
        findings.append(
            f"**No position bias**: `{judge_b_name}` winrate spread is {spread_b:.3f} (close to `{judge_a_name}`'s {spread_a:.3f})."
        )
    else:
        findings.append(
            f"**Mild discrimination weakness**: `{judge_b_name}` winrate spread is {spread_b:.3f} vs `{judge_a_name}`'s {spread_a:.3f}."
        )

    if not np.isnan(spearman_p) and spearman_p < 0.05 and spearman_rho > 0.70:
        findings.append(
            f"**Model-level ranking is statistically significant**: Spearman ρ = `{spearman_rho:.3f}`, p = `{spearman_p:.3f}` (< 0.05); the two judges agree on how to rank the {len(tts_models)} TTS systems."
        )
    elif not np.isnan(spearman_p) and spearman_p < 0.05:
        findings.append(
            f"**Ranking weakly significant**: Spearman ρ = `{spearman_rho:.3f}`, p = `{spearman_p:.3f}` (< 0.05) but ρ < 0.70."
        )
    else:
        findings.append(
            f"**Model-level ranking is not statistically significant**: Spearman ρ = `{spearman_rho:.3f}`, p = `{spearman_p:.3f}` (≥ 0.05) — with n={len(tts_models)} TTS systems we cannot distinguish this from chance."
        )

    if not np.isnan(overall_kappa):
        if overall_kappa >= 0.40:
            findings.append(f"**Sample-level agreement is moderate or better**: Cohen's κ = `{overall_kappa:.3f}` (≥ 0.40).")
        elif overall_kappa >= 0.20:
            findings.append(f"**Sample-level agreement is fair**: Cohen's κ = `{overall_kappa:.3f}` (0.20-0.40).")
        else:
            findings.append(f"**Sample-level agreement is slight (close to chance)**: Cohen's κ = `{overall_kappa:.3f}` (< 0.20). Two reasonable judges often disagree at the per-sample level even when their aggregate rankings line up — this metric is not by itself disqualifying.")

    for b in findings:
        lines.append(f"- {b}")
    lines.append("")

    if leaderboard_pass and sample_pass:
        verdict = (
            f"**Conclusion**: `{judge_b_name}` is a **viable replacement** for `{judge_a_name}` "
            f"both for ranking TTS systems AND for per-sample analysis."
        )
    elif leaderboard_pass:
        verdict = (
            f"**Conclusion**: `{judge_b_name}` is a **viable leaderboard replacement** for `{judge_a_name}` "
            f"(low failure rate, no position bias, statistically significant ranking agreement). "
            f"For fine-grained per-sample analysis (κ ≥ 0.40), the closed baseline is still preferable."
        )
    else:
        verdict = (
            f"**Conclusion**: `{judge_b_name}` **cannot** safely replace `{judge_a_name}` "
            f"under either criterion."
        )
    lines.append(verdict)
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Source files")
    lines.append("")
    lines.append(f"- Per-TTS metrics CSV: `{os.path.relpath(csv_path)}`")
    lines.append(f"- Per-TTS metrics JSON: `{os.path.relpath(json_path)}`")
    lines.append(f"- Raw judge predictions: `{os.path.relpath(judge_b_root)}/<TTS>/predictions.jsonl`")
    lines.append(f"- Per-shard run logs: `{os.path.relpath(judge_b_root)}/<TTS>/shard_*/run.log`")
    lines.append("")
    lines.append("Reproduce:")
    lines.append("")
    lines.append("```bash")
    lines.append(
        f"python judges/compare.py \\\n"
        f"  --judge-a-root {os.path.relpath(judge_a_root)} --judge-a-name {judge_a_name} \\\n"
        f"  --judge-b-root {os.path.relpath(judge_b_root)} --judge-b-name {judge_b_name} \\\n"
        f"  --out-dir {os.path.relpath(out_dir)}\n"
        f"python judges/generate_report.py \\\n"
        f"  --judge-a-root {os.path.relpath(judge_a_root)} --judge-a-name {judge_a_name} \\\n"
        f"  --judge-b-root {os.path.relpath(judge_b_root)} --judge-b-name {judge_b_name} \\\n"
        f"  --out-dir {os.path.relpath(out_dir)}"
    )
    lines.append("```")
    lines.append("")

    md_path = os.path.join(out_dir, f"REPORT_{judge_a_name}_vs_{judge_b_name}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[saved] {md_path}")
    return md_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-a-root", required=True)
    ap.add_argument("--judge-b-root", required=True)
    ap.add_argument("--judge-a-name", required=True)
    ap.add_argument("--judge-b-name", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    generate_report(
        args.judge_a_root,
        args.judge_b_root,
        args.judge_a_name,
        args.judge_b_name,
        args.out_dir,
    )


if __name__ == "__main__":
    main()
