"""Generate a side-by-side leaderboard comparing N candidate judges against a baseline.

Reads the per_model JSON + summary CSV for each judge that has been compared
against the baseline, then writes a single SUMMARY.md ranking the candidates.

Usage::

    python judges/generate_leaderboard.py \
        --baseline gemini \
        --candidates qwen2-audio unisrm \
        --comparison-dir judges/results/comparison \
        --judges-root judges/results
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from compare import list_model_dirs, load_jsonl, normalize_records


def collect_per_judge(comparison_dir: str, baseline: str, candidates: list[str]) -> dict:
    """Load per-model JSON for each candidate compared against baseline."""
    out = {}
    for cand in candidates:
        json_path = os.path.join(comparison_dir, f"per_model_{baseline}_vs_{cand}.json")
        csv_path = os.path.join(comparison_dir, f"summary_{baseline}_vs_{cand}.csv")
        if not (os.path.isfile(json_path) and os.path.isfile(csv_path)):
            print(f"[skip] {cand}: missing {json_path} or {csv_path}; run compare.py first")
            continue
        out[cand] = {
            "json": json.load(open(json_path)),
            "csv": pd.read_csv(csv_path),
        }
    return out


def per_category_metrics(judge_a_root: str, judge_b_root: str, tts_models: list[str]):
    from sklearn.metrics import accuracy_score, cohen_kappa_score
    by_cat = defaultdict(lambda: {"n_total_b": 0, "n_fail_b": 0, "winners_a": [], "winners_b": []})
    for m in tts_models:
        recs_a = load_jsonl(os.path.join(judge_a_root, m, "predictions.jsonl"))
        recs_b = load_jsonl(os.path.join(judge_b_root, m, "predictions.jsonl"))
        a_by_uid = normalize_records(recs_a)
        b_by_uid = normalize_records(recs_b)
        common = set(a_by_uid) & set(b_by_uid)
        for r in recs_b:
            cat = r.get("category", "Unknown")
            by_cat[cat]["n_total_b"] += 1
            if r["judger_output_win_rate_based"].get("winner", -1) == -1:
                by_cat[cat]["n_fail_b"] += 1
        for uid in common:
            a, b = a_by_uid[uid], b_by_uid[uid]
            if a["predicted_speech_index"] != b["predicted_speech_index"]:
                continue
            if a["canonical_winner"] is None or b["canonical_winner"] is None:
                continue
            cat = a["category"]
            by_cat[cat]["winners_a"].append(a["canonical_winner"])
            by_cat[cat]["winners_b"].append(b["canonical_winner"])
    out = {}
    for cat, d in by_cat.items():
        wa = np.asarray(d["winners_a"]); wb = np.asarray(d["winners_b"])
        if len(wa) and len(set(wa.tolist() + wb.tolist())) > 1:
            acc = float(accuracy_score(wa, wb))
            kap = float(cohen_kappa_score(wa, wb, labels=[0, 1, 2]))
        else:
            acc, kap = float("nan"), float("nan")
        out[cat] = {
            "fail_rate": d["n_fail_b"] / d["n_total_b"] if d["n_total_b"] else float("nan"),
            "accuracy": acc, "cohen_kappa": kap,
        }
    return out


def fmt_num(x, ndigits=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{ndigits}f}"


def fmt_pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x*100:.1f}%"


def best_marker(values: list, prefer_higher: bool = True) -> list[str]:
    """Return ` **bold**` for the best value, '' otherwise."""
    arr = np.array([float("nan") if v is None or np.isnan(v) else v for v in values], dtype=float)
    if np.all(np.isnan(arr)):
        return [""] * len(values)
    best = np.nanargmax(arr) if prefer_higher else np.nanargmin(arr)
    return ["**" if i == best else "" for i in range(len(values))]


def generate(
    baseline: str,
    candidates: list[str],
    comparison_dir: str,
    judges_root: str,
    out_path: str | None = None,
) -> str:
    data = collect_per_judge(comparison_dir, baseline, candidates)
    if not data:
        raise RuntimeError("No candidate data loaded.")

    candidates = list(data.keys())
    tts_models = data[candidates[0]]["json"]["shared_tts_models"]

    cat_metrics = {
        c: per_category_metrics(
            os.path.join(judges_root, baseline),
            os.path.join(judges_root, c),
            tts_models,
        )
        for c in candidates
    }

    lines: list[str] = []
    lines.append(f"# Judge Leaderboard vs `{baseline}`")
    lines.append("")
    lines.append(
        f"Side-by-side comparison of {len(candidates)} candidate judges "
        f"against the baseline `{baseline}` across {len(tts_models)} TTS models on EmergentTTS-Eval."
    )
    lines.append("")
    lines.append("**Candidates evaluated**: " + ", ".join(f"`{c}`" for c in candidates))
    lines.append("")
    lines.append("---")

    lines.append("")
    lines.append("## 1. Headline scoreboard")
    lines.append("")
    higher_is_better = {
        "Aligned samples (out of 1600)": True,
        "Sample accuracy (3-class)": True,
        "Sample Cohen's κ": True,
        "Sample Pearson r (other_score)": True,
        "Model Spearman ρ": True,
        "Model Kendall W": True,
        "Winrate spread": True,
    }
    rows: dict[str, list] = {k: [] for k in higher_is_better}
    rows["Sample Spearman p-value"] = []  # lower is better
    rows["Failure rate"] = []  # lower is better

    for c in candidates:
        s = data[c]["json"]
        df = data[c]["csv"]
        ov = df[df["tts_model"] == "__OVERALL__"].iloc[0]
        per_model = s["per_model"]
        wr_b = [per_model[m]["winrate_b"] for m in tts_models if not np.isnan(per_model[m]["winrate_b"])]

        rows["Aligned samples (out of 1600)"].append(int(ov["n_aligned"]))
        rows["Failure rate"].append(1 - int(ov["n_aligned"]) / (200 * len(tts_models)))
        rows["Sample accuracy (3-class)"].append(float(ov["accuracy"]))
        rows["Sample Cohen's κ"].append(float(ov["cohen_kappa"]))
        rows["Sample Pearson r (other_score)"].append(float(ov["pearson_r_other_score"]))
        rows["Model Spearman ρ"].append(float(s["ranking_spearman_rho"]))
        rows["Sample Spearman p-value"].append(float(s["ranking_spearman_p"]))
        rows["Model Kendall W"].append(float(s["ranking_kendall_w"]))
        rows["Winrate spread"].append((max(wr_b) - min(wr_b)) if wr_b else float("nan"))

    headers = "| Metric | " + " | ".join(f"`{c}`" for c in candidates) + " | Better when |"
    sep = "|---|" + "---:|" * len(candidates) + "---|"
    lines.append(headers)
    lines.append(sep)

    metric_order = [
        ("Failure rate", False, "fmt_pct"),
        ("Aligned samples (out of 1600)", True, "int"),
        ("Sample accuracy (3-class)", True, "fmt"),
        ("Sample Cohen's κ", True, "fmt"),
        ("Sample Pearson r (other_score)", True, "fmt"),
        ("Model Spearman ρ", True, "fmt"),
        ("Sample Spearman p-value", False, "fmt"),
        ("Model Kendall W", True, "fmt"),
        ("Winrate spread", True, "fmt"),
    ]
    for label, higher, kind in metric_order:
        vals = rows[label]
        marks = best_marker(vals, prefer_higher=higher)
        formatted = []
        for v, m in zip(vals, marks):
            if kind == "fmt_pct":
                s = fmt_pct(v)
            elif kind == "int":
                s = str(v)
            else:
                s = fmt_num(v)
            formatted.append(f"{m}{s}{m}" if m else s)
        direction = "higher" if higher else "lower"
        lines.append(f"| {label} | " + " | ".join(formatted) + f" | {direction} |")

    lines.append("")
    lines.append("- **Failure rate** = fraction of samples where the candidate produced no parseable verdict.")
    lines.append("- **Aligned samples** = both judges produced valid winners; smaller means more data lost.")
    lines.append("- **Sample κ < 0.20 = slight, 0.21–0.40 = fair, 0.41–0.60 = moderate** (Landis & Koch 1977).")
    lines.append("- **Spearman p-value < 0.05** means the model-level rank agreement is statistically significant.")
    lines.append("- **Winrate spread close to 0** indicates position bias (same winrate for every TTS).")
    lines.append("")
    lines.append("---")

    lines.append("")
    lines.append("## 2. Per-TTS winrate (baseline vs candidates)")
    lines.append("")
    df0 = data[candidates[0]]["csv"]
    a_col = f"winrate_{baseline}"
    a_winrates = df0[df0["tts_model"] != "__OVERALL__"].set_index("tts_model")[a_col]
    cols_header = (
        "| TTS model | "
        + f"`{baseline}` | "
        + " | ".join(f"`{c}`" for c in candidates)
        + " |"
    )
    lines.append(cols_header)
    lines.append("|---|" + "---:|" * (1 + len(candidates)))
    for m in tts_models:
        row = [f"`{m}`", fmt_num(a_winrates.get(m, float("nan")))]
        for c in candidates:
            wr_b = data[c]["json"]["per_model"][m]["winrate_b"]
            row.append(fmt_num(wr_b))
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append(f"### 2.1 Per-TTS ranks (best=1)")
    lines.append("")
    rank_a = pd.Series(a_winrates).rank(ascending=False, method="average")
    cols_header = "| TTS model | " + f"rank_{baseline} | " + " | ".join(f"rank_{c}" for c in candidates) + " |"
    lines.append(cols_header)
    lines.append("|---|" + "---:|" * (1 + len(candidates)))
    for m in tts_models:
        row = [f"`{m}`", f"{rank_a[m]:.1f}"]
        for c in candidates:
            wr_c = pd.Series(
                {mm: data[c]["json"]["per_model"][mm]["winrate_b"] for mm in tts_models}
            ).rank(ascending=False, method="average")
            row.append(f"{wr_c[m]:.1f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("---")

    lines.append("")
    lines.append("## 3. Per-category breakdown")
    lines.append("")
    cats = sorted({c for cm in cat_metrics.values() for c in cm.keys()})
    lines.append("### 3.1 Failure rate per category")
    lines.append("")
    lines.append("| Category | " + " | ".join(f"`{c}`" for c in candidates) + " |")
    lines.append("|---|" + "---:|" * len(candidates))
    for cat in cats:
        vals = [cat_metrics[c].get(cat, {}).get("fail_rate", float("nan")) for c in candidates]
        marks = best_marker(vals, prefer_higher=False)
        cells = [f"{m}{fmt_pct(v)}{m}" if m else fmt_pct(v) for v, m in zip(vals, marks)]
        lines.append(f"| {cat} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("### 3.2 Sample accuracy per category (vs baseline)")
    lines.append("")
    lines.append("| Category | " + " | ".join(f"`{c}`" for c in candidates) + " |")
    lines.append("|---|" + "---:|" * len(candidates))
    for cat in cats:
        vals = [cat_metrics[c].get(cat, {}).get("accuracy", float("nan")) for c in candidates]
        marks = best_marker(vals, prefer_higher=True)
        cells = [f"{m}{fmt_num(v)}{m}" if m else fmt_num(v) for v, m in zip(vals, marks)]
        lines.append(f"| {cat} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("### 3.3 Cohen's κ per category (vs baseline)")
    lines.append("")
    lines.append("| Category | " + " | ".join(f"`{c}`" for c in candidates) + " |")
    lines.append("|---|" + "---:|" * len(candidates))
    for cat in cats:
        vals = [cat_metrics[c].get(cat, {}).get("cohen_kappa", float("nan")) for c in candidates]
        marks = best_marker(vals, prefer_higher=True)
        cells = [f"{m}{fmt_num(v)}{m}" if m else fmt_num(v) for v, m in zip(vals, marks)]
        lines.append(f"| {cat} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("---")

    lines.append("")
    lines.append("## 4. Verdict")
    lines.append("")
    scores: dict[str, int] = {c: 0 for c in candidates}
    weights = {
        "Failure rate": (False, 3),
        "Model Spearman ρ": (True, 3),
        "Model Kendall W": (True, 2),
        "Winrate spread": (True, 2),
        "Sample Pearson r (other_score)": (True, 1),
        "Sample accuracy (3-class)": (True, 1),
        "Sample Cohen's κ": (True, 1),
    }
    for metric, (higher, weight) in weights.items():
        vals = rows[metric]
        if len(vals) <= 1:
            continue
        idx = (np.nanargmax if higher else np.nanargmin)(np.array(vals, dtype=float))
        scores[candidates[idx]] += weight

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    lines.append(f"Weighted-best aggregation across {len(weights)} metrics:")
    lines.append("")
    for c, s in ranked:
        lines.append(f"- `{c}`: **{s}** wins")
    lines.append("")
    best = ranked[0][0]
    lines.append(
        f"**Best candidate replacement for `{baseline}`**: `{best}`."
    )
    lines.append("")
    best_p = data[best]["json"]["ranking_spearman_p"]
    best_rho = data[best]["json"]["ranking_spearman_rho"]
    fail = 1 - int(data[best]["csv"][data[best]["csv"]["tts_model"] == "__OVERALL__"]["n_aligned"].iloc[0]) / (200 * len(tts_models))
    if best_p < 0.05 and fail < 0.05:
        verdict = (
            f"`{best}` shows statistically significant model-level ranking agreement "
            f"with `{baseline}` (Spearman ρ = `{best_rho:.3f}`, p = `{best_p:.3f}`) and a "
            f"low failure rate of `{fail*100:.1f}%`. It is a viable replacement when the goal "
            f"is to rank TTS systems on the EmergentTTS-Eval benchmark; sample-level disagreement "
            f"with the baseline remains substantial, so for fine-grained per-sample analyses the "
            f"closed baseline is still preferable."
        )
    else:
        verdict = (
            f"`{best}` is the strongest candidate but does not yet meet both criteria "
            f"(p < 0.05 AND failure rate < 5%). Sample-level agreement is limited."
        )
    lines.append(verdict)
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Source files")
    lines.append("")
    for c in candidates:
        lines.append(f"- `{baseline}` vs `{c}`:")
        lines.append(f"  - CSV: `{comparison_dir}/summary_{baseline}_vs_{c}.csv`")
        lines.append(f"  - JSON: `{comparison_dir}/per_model_{baseline}_vs_{c}.json`")
        lines.append(f"  - Full report: `{comparison_dir}/REPORT_{baseline}_vs_{c}.md`")

    if out_path is None:
        out_path = os.path.join(comparison_dir, "SUMMARY.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[saved] {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--comparison-dir", required=True)
    ap.add_argument("--judges-root", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    generate(args.baseline, args.candidates, args.comparison_dir, args.judges_root, args.out)


if __name__ == "__main__":
    main()
