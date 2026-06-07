"""Compare two judges' predictions on EmergentTTS-Eval.

Given two directories laid out as::

    <judge_root>/
        <tts_model_a>/predictions.jsonl
        <tts_model_b>/predictions.jsonl
        ...

this module aligns predictions by ``unique_id_eval`` and computes:

Item-level winner agreement (3-way: 0 = tie, 1 = system_1 wins, 2 = system_2 wins):
    * accuracy
    * Cohen's kappa (nominal)
    * Weighted Cohen's kappa (quadratic, ordinal)

Item-level score correlation (per-system audio quality scores 0..3):
    * Pearson r (other_score)
    * Pearson r (baseline_score)

TTS-model-level ranking agreement (uses win_rate per TTS model):
    * Spearman rho across all TTS models
    * Kendall's W (treats both judges as raters)

Outputs:
    * <out_dir>/per_model_<judge_a>_vs_<judge_b>.json
    * <out_dir>/summary_<judge_a>_vs_<judge_b>.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.metrics import cohen_kappa_score


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def list_model_dirs(judge_root: str) -> list[str]:
    """List immediate subdirectories that contain a predictions.jsonl."""
    if not os.path.isdir(judge_root):
        raise FileNotFoundError(f"Judge results dir not found: {judge_root}")
    out = []
    for name in sorted(os.listdir(judge_root)):
        sub = os.path.join(judge_root, name)
        if os.path.isdir(sub) and os.path.isfile(os.path.join(sub, "predictions.jsonl")):
            out.append(name)
    return out


def map_winner_to_canonical(judger_winner: int, predicted_speech_index: int) -> int | None:
    """Map judger winner field into a canonical 3-class label.

    Returns:
        0 = tie
        1 = baseline wins
        2 = comparison TTS wins
        None = unparseable (-1)
    """
    if judger_winner == -1:
        return None
    if judger_winner == 0:
        return 0
    if (predicted_speech_index == 1 and judger_winner == 1) or (
        predicted_speech_index == 2 and judger_winner == 2
    ):
        return 2
    return 1


def extract_scores_canonical(
    judger_output: dict, predicted_speech_index: int
) -> tuple[float | None, float | None]:
    """Return (baseline_score, comparison_score) using canonical orientation."""
    s1 = judger_output.get("score_1")
    s2 = judger_output.get("score_2")
    if predicted_speech_index == 1:
        return s2, s1
    return s1, s2


def kendalls_w(ratings: np.ndarray) -> float:
    """Kendall's W. ratings shape: (n_items, m_raters)."""
    n, m = ratings.shape
    ranks = np.apply_along_axis(rankdata, 0, ratings)
    R = np.sum(ranks, axis=1)
    S = np.sum((R - R.mean()) ** 2)
    if m == 0 or n <= 1:
        return float("nan")
    return 12 * S / (m**2 * (n**3 - n))


def compute_winrate(samples: list[dict]) -> float:
    """Win-rate of comparison TTS over baseline using canonical labels.

    Ties contribute 0.5; unparseable items skipped.
    """
    counted = 0
    score = 0.0
    for s in samples:
        if s["canonical_winner"] is None:
            continue
        if s["canonical_winner"] == 2:
            score += 1.0
        elif s["canonical_winner"] == 0:
            score += 0.5
        counted += 1
    return score / counted if counted else float("nan")


def safe_pearson(x: list[float], y: list[float]) -> float:
    """Pearson r returning nan when degenerate."""
    if len(x) < 2:
        return float("nan")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    r, _ = pearsonr(x, y)
    return r


def normalize_records(records: list[dict]) -> dict[int, dict]:
    """Index records by unique_id_eval and add canonical_winner / canonical_scores."""
    out: dict[int, dict] = {}
    for r in records:
        uid = r.get("unique_id_eval")
        if uid is None:
            continue
        psi = r.get("predicted_speech_index")
        judger_out = r.get("judger_output_win_rate_based") or {}
        winner = judger_out.get("winner", -1)
        canonical = map_winner_to_canonical(winner, psi) if psi is not None else None
        baseline_score, other_score = (
            extract_scores_canonical(judger_out, psi) if psi is not None else (None, None)
        )
        out[uid] = {
            "raw": r,
            "predicted_speech_index": psi,
            "canonical_winner": canonical,
            "baseline_score": baseline_score,
            "other_score": other_score,
            "category": r.get("category"),
            "evolution_depth": r.get("evolution_depth"),
        }
    return out


def compare_one_model(
    samples_a: dict[int, dict],
    samples_b: dict[int, dict],
) -> dict:
    """Compare two judges on a single TTS model."""
    common_ids = sorted(set(samples_a.keys()) & set(samples_b.keys()))

    aligned = []
    for uid in common_ids:
        a = samples_a[uid]
        b = samples_b[uid]
        if a["predicted_speech_index"] != b["predicted_speech_index"]:
            continue
        if a["canonical_winner"] is None or b["canonical_winner"] is None:
            continue
        aligned.append((uid, a, b))

    n_total = len(common_ids)
    n_aligned = len(aligned)

    if n_aligned == 0:
        return {
            "n_common": n_total,
            "n_aligned": 0,
            "winrate_a": float("nan"),
            "winrate_b": float("nan"),
            "accuracy": float("nan"),
            "cohen_kappa": float("nan"),
            "weighted_kappa": float("nan"),
            "pearson_r_other_score": float("nan"),
            "pearson_r_baseline_score": float("nan"),
            "kendall_w": float("nan"),
        }

    winners_a = [a["canonical_winner"] for _, a, _ in aligned]
    winners_b = [b["canonical_winner"] for _, _, b in aligned]
    winners_a_arr = np.asarray(winners_a, dtype=int)
    winners_b_arr = np.asarray(winners_b, dtype=int)

    accuracy = float((winners_a_arr == winners_b_arr).mean())

    cohen = cohen_kappa_score(winners_a_arr, winners_b_arr) if len(set(winners_a_arr.tolist() + winners_b_arr.tolist())) > 1 else float("nan")
    weighted = (
        cohen_kappa_score(winners_a_arr, winners_b_arr, weights="quadratic")
        if len(set(winners_a_arr.tolist() + winners_b_arr.tolist())) > 1
        else float("nan")
    )

    other_a, other_b, base_a, base_b = [], [], [], []
    for _, a, b in aligned:
        if a["other_score"] is not None and b["other_score"] is not None:
            other_a.append(float(a["other_score"]))
            other_b.append(float(b["other_score"]))
        if a["baseline_score"] is not None and b["baseline_score"] is not None:
            base_a.append(float(a["baseline_score"]))
            base_b.append(float(b["baseline_score"]))

    pearson_other = safe_pearson(other_a, other_b)
    pearson_base = safe_pearson(base_a, base_b)

    kendall_W = kendalls_w(np.vstack([winners_a_arr, winners_b_arr]).T)

    winrate_a = compute_winrate([a for _, a, _ in aligned])
    winrate_b = compute_winrate([b for _, _, b in aligned])

    return {
        "n_common": n_total,
        "n_aligned": n_aligned,
        "winrate_a": winrate_a,
        "winrate_b": winrate_b,
        "winrate_diff": winrate_a - winrate_b,
        "accuracy": accuracy,
        "cohen_kappa": float(cohen) if not np.isnan(cohen) else float("nan"),
        "weighted_kappa": float(weighted) if not np.isnan(weighted) else float("nan"),
        "pearson_r_other_score": pearson_other,
        "pearson_r_baseline_score": pearson_base,
        "kendall_w": kendall_W,
    }


def compare_judges(
    judge_a_root: str,
    judge_b_root: str,
    judge_a_name: str,
    judge_b_name: str,
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    models_a = set(list_model_dirs(judge_a_root))
    models_b = set(list_model_dirs(judge_b_root))
    shared_models = sorted(models_a & models_b)
    only_a = sorted(models_a - models_b)
    only_b = sorted(models_b - models_a)

    if only_a:
        print(f"[warn] models present only in {judge_a_name}: {only_a}")
    if only_b:
        print(f"[warn] models present only in {judge_b_name}: {only_b}")
    if not shared_models:
        raise RuntimeError("No shared TTS models between the two judge roots.")

    per_model_results: dict[str, dict] = {}
    for tts_model in shared_models:
        path_a = os.path.join(judge_a_root, tts_model, "predictions.jsonl")
        path_b = os.path.join(judge_b_root, tts_model, "predictions.jsonl")
        recs_a = normalize_records(load_jsonl(path_a))
        recs_b = normalize_records(load_jsonl(path_b))
        per_model_results[tts_model] = compare_one_model(recs_a, recs_b)
        print(
            f"[{tts_model}] aligned={per_model_results[tts_model]['n_aligned']:>5}  "
            f"acc={per_model_results[tts_model]['accuracy']:.3f}  "
            f"kappa={per_model_results[tts_model]['cohen_kappa']:.3f}  "
            f"wkappa={per_model_results[tts_model]['weighted_kappa']:.3f}  "
            f"WR_a={per_model_results[tts_model]['winrate_a']:.3f}  "
            f"WR_b={per_model_results[tts_model]['winrate_b']:.3f}"
        )

    winrates_a = np.array([per_model_results[m]["winrate_a"] for m in shared_models])
    winrates_b = np.array([per_model_results[m]["winrate_b"] for m in shared_models])

    if len(shared_models) >= 2 and not np.isnan(winrates_a).any() and not np.isnan(winrates_b).any():
        spearman_rho, spearman_p = spearmanr(winrates_a, winrates_b)
        ranking_kendall_W = kendalls_w(np.vstack([winrates_a, winrates_b]).T)
    else:
        spearman_rho, spearman_p, ranking_kendall_W = float("nan"), float("nan"), float("nan")

    summary = {
        "judge_a": judge_a_name,
        "judge_b": judge_b_name,
        "shared_tts_models": shared_models,
        "ranking_spearman_rho": float(spearman_rho) if not np.isnan(spearman_rho) else float("nan"),
        "ranking_spearman_p": float(spearman_p) if not np.isnan(spearman_p) else float("nan"),
        "ranking_kendall_w": float(ranking_kendall_W) if not np.isnan(ranking_kendall_W) else float("nan"),
        "per_model": per_model_results,
    }

    per_model_path = os.path.join(out_dir, f"per_model_{judge_a_name}_vs_{judge_b_name}.json")
    with open(per_model_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[saved] {per_model_path}")

    rows = []
    for m in shared_models:
        d = per_model_results[m]
        rows.append({
            "tts_model": m,
            "n_aligned": d["n_aligned"],
            "winrate_" + judge_a_name: d["winrate_a"],
            "winrate_" + judge_b_name: d["winrate_b"],
            "winrate_diff": d.get("winrate_diff"),
            "accuracy": d["accuracy"],
            "cohen_kappa": d["cohen_kappa"],
            "weighted_kappa": d["weighted_kappa"],
            "pearson_r_other_score": d["pearson_r_other_score"],
            "pearson_r_baseline_score": d["pearson_r_baseline_score"],
            "kendall_w": d["kendall_w"],
        })
    rows.append({
        "tts_model": "__OVERALL__",
        "n_aligned": int(np.nansum([r["n_aligned"] for r in rows])),
        "winrate_" + judge_a_name: float(np.nanmean(winrates_a)),
        "winrate_" + judge_b_name: float(np.nanmean(winrates_b)),
        "winrate_diff": float(np.nanmean([r["winrate_diff"] for r in rows if r.get("winrate_diff") is not None])),
        "accuracy": float(np.nanmean([r["accuracy"] for r in rows[:-1]])),
        "cohen_kappa": float(np.nanmean([r["cohen_kappa"] for r in rows[:-1]])),
        "weighted_kappa": float(np.nanmean([r["weighted_kappa"] for r in rows[:-1]])),
        "pearson_r_other_score": float(np.nanmean([r["pearson_r_other_score"] for r in rows[:-1]])),
        "pearson_r_baseline_score": float(np.nanmean([r["pearson_r_baseline_score"] for r in rows[:-1]])),
        "kendall_w": float(np.nanmean([r["kendall_w"] for r in rows[:-1]])),
    })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, f"summary_{judge_a_name}_vs_{judge_b_name}.csv")
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"[saved] {csv_path}")

    print()
    print(f"=== Ranking agreement across {len(shared_models)} TTS models ===")
    print(f"Spearman rho = {spearman_rho:.4f}  (p = {spearman_p:.4f})")
    print(f"Kendall  W   = {ranking_kendall_W:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judge-a-root", required=True, help="Directory containing <tts_model>/predictions.jsonl for judge A")
    parser.add_argument("--judge-b-root", required=True, help="Same for judge B")
    parser.add_argument("--judge-a-name", default="A", help="Pretty name for judge A (used in output filenames)")
    parser.add_argument("--judge-b-name", default="B", help="Pretty name for judge B")
    parser.add_argument("--out-dir", required=True, help="Where to write the comparison summaries")
    args = parser.parse_args()

    compare_judges(
        judge_a_root=args.judge_a_root,
        judge_b_root=args.judge_b_root,
        judge_a_name=args.judge_a_name,
        judge_b_name=args.judge_b_name,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
