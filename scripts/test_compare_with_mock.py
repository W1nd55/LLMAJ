"""Smoke test compare.py with synthetic Gemini/Qwen2-Audio judgments."""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import tempfile

import numpy as np

CORE_MODELS = [
    "gpt-4o-audio-preview-2024-12-17",
    "HumeAI",
    "eleven_multilingual_v2",
    "deepgram",
    "orpheus-tts-0.1-finetune-prod",
    "Sesame1B",
    "Qwen2.5-Omni-7B",
    "gpt-4o-mini-tts",
]


def make_record(uid: int, judger_name: str, true_winner: int, noise: float = 0.0) -> dict:
    psi = 1 + (uid % 2)
    if random.random() < noise:
        winner_canonical = random.choice([0, 1, 2])
    else:
        winner_canonical = true_winner

    if winner_canonical == 0:
        winner = 0
    elif winner_canonical == 2:
        winner = psi
    else:
        winner = 1 if psi == 2 else 2

    base_score = random.randint(0, 3)
    other_score = base_score + (1 if winner_canonical == 2 else (-1 if winner_canonical == 1 else 0))
    other_score = max(0, min(3, other_score))
    if psi == 1:
        score_1 = other_score
        score_2 = base_score
    else:
        score_1 = base_score
        score_2 = other_score

    return {
        "unique_id_eval": uid,
        "category": ["Emotions", "Questions", "Foreign Words"][uid % 3],
        "evolution_depth": uid % 4,
        "language": "en",
        "predicted_speech_index": psi,
        "judger_model": judger_name,
        "judger_output_win_rate_based": {
            "winner": winner,
            "score_1": score_1,
            "score_2": score_2,
            "reasoning_system_1": "...",
            "reasoning_system_2": "...",
            "system_comparison": "...",
        },
        "baseline_audio_path": f"/fake/baseline/{uid}.wav",
        "audio_out_path": f"/fake/{judger_name}/{uid}.wav",
    }


def write_jsonl(path: str, records: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    rng = np.random.default_rng(42)
    random.seed(42)

    tmp = tempfile.mkdtemp(prefix="llmaj_compare_test_")
    print(f"[tmp] {tmp}")
    gem_root = os.path.join(tmp, "gemini")
    qwen_root = os.path.join(tmp, "qwen2-audio")
    out_dir = os.path.join(tmp, "comparison")

    n_per_model = 100
    for tts_model in CORE_MODELS:
        true_pref = rng.choice([0, 1, 2], size=n_per_model, p=[0.1, 0.4, 0.5])
        gemini_recs = [make_record(uid, "gemini-2.5-pro-preview", int(t), noise=0.05) for uid, t in enumerate(true_pref)]
        qwen_recs = [make_record(uid, "qwen2-audio-7b", int(t), noise=0.30) for uid, t in enumerate(true_pref)]
        write_jsonl(os.path.join(gem_root, tts_model, "predictions.jsonl"), gemini_recs)
        write_jsonl(os.path.join(qwen_root, tts_model, "predictions.jsonl"), qwen_recs)

    cmd = [
        "python", "judges/compare.py",
        "--judge-a-root", gem_root,
        "--judge-b-root", qwen_root,
        "--judge-a-name", "gemini",
        "--judge-b-name", "qwen2-audio",
        "--out-dir", out_dir,
    ]
    print("[run]", " ".join(cmd))
    res = subprocess.run(cmd, check=False)
    print(f"\nReturn code: {res.returncode}")

    summary_csv = os.path.join(out_dir, "summary_gemini_vs_qwen2-audio.csv")
    if os.path.isfile(summary_csv):
        print("\n----- summary CSV -----")
        with open(summary_csv) as f:
            print(f.read())

    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
