"""Quick benchmark: how long does Qwen2-Audio-7B take per pairwise audio judgement?

Loads the model once and measures generation latency on 3 real samples taken
from EmergentTTS-Eval. Reports throughput so we can decide on subset size.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "EmergentTTS-Eval-public"))

from utils_eval import get_audio_array_from_path  # noqa: E402
from prompts import EmergentTextSpeech  # noqa: E402
from judges import get_judge  # noqa: E402


def main():
    config_path = os.path.join(REPO_ROOT, "configs/judge_qwen2_audio.yaml")
    judge = get_judge(config_path)

    eval_data_path = os.path.join(REPO_ROOT, "EmergentTTS-Eval-public/data/emergent_tts_eval_data.jsonl")
    with open(eval_data_path) as f:
        samples = [json.loads(l) for l in f][:3]

    baseline_dir = os.path.join(REPO_ROOT, "EmergentTTS-Eval-public/data/baseline_audios")
    other_dir = os.path.join(
        REPO_ROOT,
        "drive_data/EmergentTTS-Eval_Predictions/gpt-4o-mini-tts/emergent-tts-eval_strong-prompting_alloy_output-audios",
    )

    prompting = EmergentTextSpeech()
    gen_cfg = judge.default_generation_config
    gen_cfg["max_new_tokens"] = 768

    times = []
    for i, s in enumerate(samples):
        baseline_audio = get_audio_array_from_path(os.path.join(baseline_dir, f"{i}.wav"))
        other_audio = get_audio_array_from_path(os.path.join(other_dir, f"{i}.wav"))

        sys_msg, user_msg, post_audio_msg = prompting.get_win_rate_prompts(
            text_to_synthesize=s["text_to_synthesize"], category=s["category"]
        )

        t0 = time.time()
        resp = judge.generate_w_audio_comparison(
            model_name="qwen2-audio",
            system_message=sys_msg,
            user_message=user_msg,
            audio_array_1=baseline_audio,
            post_audio_1_message=post_audio_msg,
            audio_array_2=other_audio,
            **gen_cfg,
        )
        dt = time.time() - t0
        times.append(dt)
        print(f"[sample {i}] {dt:.1f}s | category={s['category']} | resp_chars={len(resp)}")
        print(f"        first 300 chars: {resp[:300]!r}")

    avg = sum(times) / len(times)
    print()
    print(f"avg latency = {avg:.1f}s/sample")
    for n_samples_per_model in [50, 100, 200, 500, 1645]:
        total = n_samples_per_model * 8
        secs = total * avg
        print(f"  {n_samples_per_model:>4} samples/model ({total:>5} total) -> {secs/60:.1f} min ({secs/3600:.1f} h)")


if __name__ == "__main__":
    main()
