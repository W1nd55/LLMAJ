"""Walk the EmergentTTS-Eval Drive folder and download predictions + audios
for the 8 core TTS models, bypassing gdown's 50-file/folder limit by using
the embeddedfolderview endpoint."""

from __future__ import annotations

import argparse
import os
import sys

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURR_DIR)
from drive_folder_downloader import (  # noqa: E402  (sibling file)
    download_folder,
    download_one,
    list_public_folder,
)
import requests  # noqa: E402

PARENT_FOLDER_ID = "1SGEGaUai2UqOMbwXx447yZeY-6gCU0F_"

CORE_MODELS = [
    "gpt-4o-mini-tts",
    "gpt-4o-audio-preview-2024-12-17",
    "HumeAI",
    "eleven_multilingual_v2",
    "deepgram",
    "orpheus-tts-0.1-finetune-prod",
    "Sesame1B",
    "Qwen2.5-Omni-7B",
]

# Each TTS model has potentially several variants in the Drive (alloy/ballad,
# with/without strong-prompting). Pick the variant that matches the paper's
# leaderboard (strong-prompting + voice). The audio subfolder name mirrors the
# JSONL name: `*_<variant>_evaluation-predictions.jsonl` ↔ `*_<variant>_output-audios`.
PREFERRED_VARIANT_SUFFIX = {
    "gpt-4o-mini-tts": "strong-prompting_alloy",
    "gpt-4o-audio-preview-2024-12-17": "strong-prompting_ballad",
    "HumeAI": "strong-prompting",
    "eleven_multilingual_v2": "nPczCjzI2devNBz1zQrb",
    "deepgram": "thalia-en",
    "orpheus-tts-0.1-finetune-prod": "tara",
    "Sesame1B": "",
    "Qwen2.5-Omni-7B": "strong-prompting_Chelsie",
}


def expected_subfolder_name(variant_suffix: str) -> str:
    if variant_suffix:
        return f"emergent-tts-eval_{variant_suffix}_output-audios"
    return "emergent-tts-eval_output-audios"


def expected_jsonl_name(variant_suffix: str) -> str:
    if variant_suffix:
        return f"emergent-tts-eval_{variant_suffix}_evaluation-predictions.jsonl"
    return "emergent-tts-eval_evaluation-predictions.jsonl"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out-root",
        default="drive_data/EmergentTTS-Eval_Predictions",
        help="Local root mirroring the Drive folder",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=CORE_MODELS,
        help="Subset of model folders to download",
    )
    parser.add_argument(
        "--predictions-only",
        action="store_true",
        help="Skip output-audios subfolders (~1.5GB each); just grab the JSONL/JSON metadata",
    )
    parser.add_argument("--parallel", type=int, default=4, help="Concurrent downloads per folder")
    args = parser.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    parent_items = list_public_folder(PARENT_FOLDER_ID, session=session)
    parent_lookup = {item.name: item for item in parent_items}

    summary = {}
    for model_name in args.models:
        if model_name not in parent_lookup:
            print(f"[skip] {model_name}: folder not found in Drive parent")
            summary[model_name] = "missing"
            continue

        model_item = parent_lookup[model_name]
        model_dir = os.path.join(args.out_root, model_name)
        os.makedirs(model_dir, exist_ok=True)

        variant = PREFERRED_VARIANT_SUFFIX.get(model_name, "")
        wanted_jsonl = expected_jsonl_name(variant) if model_name in PREFERRED_VARIANT_SUFFIX else None
        wanted_audio_folder = (
            expected_subfolder_name(variant) if model_name in PREFERRED_VARIANT_SUFFIX else None
        )

        children = list_public_folder(model_item.id, session=session)
        print(f"\n=== {model_name} ({len(children)} immediate children, preferred variant={variant or '(default)'}) ===")

        for child in children:
            if child.is_folder:
                if args.predictions_only:
                    print(f"[skip] {model_name}/{child.name} (predictions-only mode)")
                    continue
                if wanted_audio_folder is not None and child.name != wanted_audio_folder:
                    print(f"[skip] {model_name}/{child.name} (not the preferred audio variant)")
                    continue
                sub_out = os.path.join(model_dir, child.name)
                done, total = download_folder(child.id, sub_out, parallel=args.parallel, only_extensions=(".wav",))
                print(f"[audios] {model_name}/{child.name}: {done}/{total}")
            else:
                if wanted_jsonl is not None and child.name.endswith(".jsonl") and child.name != wanted_jsonl:
                    print(f"[skip] {model_name}/{child.name} (not the preferred JSONL variant)")
                    continue
                dest = os.path.join(model_dir, child.name)
                if os.path.isfile(dest) and os.path.getsize(dest) > 0:
                    print(f"[have ] {model_name}/{child.name}")
                    continue
                ok = download_one(child.id, dest, session=session)
                print(f"[file ] {model_name}/{child.name}: {'ok' if ok else 'FAIL'}")
        summary[model_name] = "ok"

    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
