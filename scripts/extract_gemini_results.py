"""Copy Gemini judgments from the Google Drive download into judges/results/gemini/.

The Drive folder layout (after gdown extraction) is::

    drive_data/EmergentTTS-Eval_Predictions/
        <tts_model>/
            emergent-tts-eval_*_evaluation-predictions.jsonl
            emergent-tts-eval_output-audios/
                0.wav, 1.wav, ..., 1644.wav

This script symlinks (or copies) each prediction JSONL to::

    judges/results/gemini/<tts_model>/predictions.jsonl

so that compare.py can consume both judge roots with a uniform layout.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

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

# Mirrors EmergentTTS-Eval-public/human_alignment/run_survey_evals.get_model_filenames().
# We pin the exact prediction JSONL to use for each TTS model so behaviour matches the
# paper's leaderboard (strong-prompting / specific voice).
PREFERRED_PREDICTION_FILENAMES = {
    "gpt-4o-audio-preview-2024-12-17": "emergent-tts-eval_strong-prompting_ballad_evaluation-predictions.jsonl",
    "HumeAI": "emergent-tts-eval_strong-prompting_evaluation-predictions.jsonl",
    "eleven_multilingual_v2": "emergent-tts-eval_nPczCjzI2devNBz1zQrb_evaluation-predictions.jsonl",
    "deepgram": "emergent-tts-eval_thalia-en_evaluation-predictions.jsonl",
    "orpheus-tts-0.1-finetune-prod": "emergent-tts-eval_tara_evaluation-predictions.jsonl",
    "Sesame1B": "emergent-tts-eval_evaluation-predictions.jsonl",
    "Qwen2.5-Omni-7B": "emergent-tts-eval_strong-prompting_Chelsie_evaluation-predictions.jsonl",
    "gpt-4o-mini-tts": "emergent-tts-eval_strong-prompting_alloy_evaluation-predictions.jsonl",
}


def find_predictions_jsonl(model_dir: str, model_name: str | None = None) -> str | None:
    if model_name and model_name in PREFERRED_PREDICTION_FILENAMES:
        preferred = os.path.join(model_dir, PREFERRED_PREDICTION_FILENAMES[model_name])
        if os.path.isfile(preferred):
            return preferred
        print(f"[warn] {model_name}: preferred file {PREFERRED_PREDICTION_FILENAMES[model_name]!r} not present; falling back to glob")

    candidates = sorted(glob.glob(os.path.join(model_dir, "*evaluation-predictions.jsonl")))
    if not candidates:
        return None
    if len(candidates) > 1:
        print(f"[warn] {model_dir}: multiple prediction files found; using {candidates[0]}")
    return candidates[0]


def filter_records(records: list[dict], expected_judger_substr: str | None) -> list[dict]:
    """Optionally drop records whose judger_model does not match the expected substring."""
    if not expected_judger_substr:
        return records
    return [r for r in records if expected_judger_substr in r.get("judger_model", "")]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--drive-root",
        default="drive_data/EmergentTTS-Eval_Predictions",
        help="Root of the gdown-downloaded Drive folder (with one subdir per TTS model)",
    )
    parser.add_argument(
        "--out-root",
        default="judges/results/gemini",
        help="Where to materialize the gemini judge results (one subdir per TTS model)",
    )
    parser.add_argument(
        "--mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="symlink (cheap) or copy (portable) the prediction JSONLs",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=CORE_MODELS,
        help="TTS model folder names to extract (default: 8 core models from the paper)",
    )
    parser.add_argument(
        "--expected-judger-substr",
        default="",
        help="Substring required in judger_model field to keep a record. Default empty (no filter), since the published Drive JSONLs lack the judger_model field but are known to be Gemini 2.5 Pro.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.drive_root):
        raise FileNotFoundError(
            f"Drive root not found: {args.drive_root}. Wait for gdown to finish, "
            "or pass --drive-root to point to your downloaded folder."
        )

    os.makedirs(args.out_root, exist_ok=True)
    summary = {}

    for tts_model in args.models:
        src_dir = os.path.join(args.drive_root, tts_model)
        if not os.path.isdir(src_dir):
            print(f"[skip] {tts_model}: source dir not found ({src_dir})")
            summary[tts_model] = {"status": "missing_source"}
            continue

        src_jsonl = find_predictions_jsonl(src_dir, model_name=tts_model)
        if not src_jsonl:
            print(f"[skip] {tts_model}: no prediction JSONL inside {src_dir}")
            summary[tts_model] = {"status": "missing_jsonl"}
            continue

        dst_dir = os.path.join(args.out_root, tts_model)
        os.makedirs(dst_dir, exist_ok=True)
        dst_jsonl = os.path.join(dst_dir, "predictions.jsonl")

        if args.expected_judger_substr:
            with open(src_jsonl, "r", encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]
            filtered = filter_records(records, args.expected_judger_substr)
            with open(dst_jsonl, "w", encoding="utf-8") as f:
                for r in filtered:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_kept = len(filtered)
            n_total = len(records)
            print(f"[ok ] {tts_model}: kept {n_kept}/{n_total} records (judger contains '{args.expected_judger_substr}')")
            summary[tts_model] = {"status": "ok", "kept": n_kept, "total": n_total, "src": src_jsonl}
        else:
            if os.path.lexists(dst_jsonl):
                os.remove(dst_jsonl)
            if args.mode == "symlink":
                os.symlink(os.path.abspath(src_jsonl), dst_jsonl)
            else:
                shutil.copyfile(src_jsonl, dst_jsonl)
            print(f"[ok ] {tts_model}: {args.mode} -> {dst_jsonl}")
            summary[tts_model] = {"status": "ok", "src": src_jsonl, "mode": args.mode}

    summary_path = os.path.join(args.out_root, "_extract_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[saved] {summary_path}")


if __name__ == "__main__":
    main()
