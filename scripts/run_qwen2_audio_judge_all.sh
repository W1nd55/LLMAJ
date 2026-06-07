#!/usr/bin/env bash
# Run the Qwen2-Audio judge over the 8 core TTS models, reusing the
# pre-generated audio files downloaded from the EmergentTTS-Eval Drive folder.
#
# This emits judges/results/qwen2-audio/<tts_model>/predictions.jsonl, which
# the comparison module can pair against judges/results/gemini/<tts_model>/.
#
# Layout assumptions (created by scripts/extract_gemini_results.py + gdown):
#   drive_data/EmergentTTS-Eval_Predictions/<tts_model>/emergent-tts-eval_output-audios/
#   EmergentTTS-Eval-public/data/baseline_audios/

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/EmergentTTS-Eval-public"

DRIVE_ROOT="${DRIVE_ROOT:-$REPO_ROOT/drive_data/EmergentTTS-Eval_Predictions}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/judges/results/qwen2-audio}"
JUDGE_CFG="${JUDGE_CFG:-$REPO_ROOT/configs/judge_qwen2_audio.yaml}"
BASELINE_DIR="${BASELINE_DIR:-data/baseline_audios}"
NUM_SAMPLES="${NUM_SAMPLES:-}"

# Core 8 TTS models mirroring run_survey_evals.MODEL_KEYS.
# model_name_or_path passed to evaluation_runner.py is what determines the
# api/hf client, but we set --fetch_audios_from_path so no audio is generated.
MODELS=(
  "gpt-4o-audio-preview-2024-12-17"
  "HumeAI"
  "eleven_multilingual_v2"
  "deepgram"
  "orpheus-tts-0.1-finetune-prod"
  "Sesame1B"
  "Qwen2.5-Omni-7B"
  "gpt-4o-mini-tts"
)

# Each model has a preferred audio subfolder name + a model_name_or_path that
# evaluation_runner.py uses to instantiate a (no-op for audio gen) client.
# We pass dummy api keys to avoid OpenAIClient/etc. constructor failures.
declare -A AUDIO_SUBDIR=(
  ["gpt-4o-mini-tts"]="emergent-tts-eval_strong-prompting_alloy_output-audios"
  ["gpt-4o-audio-preview-2024-12-17"]="emergent-tts-eval_strong-prompting_ballad_output-audios"
  ["HumeAI"]="emergent-tts-eval_strong-prompting_output-audios"
  ["eleven_multilingual_v2"]="emergent-tts-eval_nPczCjzI2devNBz1zQrb_output-audios"
  ["deepgram"]="emergent-tts-eval_thalia-en_output-audios"
  ["orpheus-tts-0.1-finetune-prod"]="emergent-tts-eval_tara_output-audios"
  ["Sesame1B"]="emergent-tts-eval_output-audios"
  ["Qwen2.5-Omni-7B"]="emergent-tts-eval_strong-prompting_Chelsie_output-audios"
)
declare -A MODEL_NAME_OR_PATH=(
  ["gpt-4o-audio-preview-2024-12-17"]="gpt-4o-mini-tts"
  ["HumeAI"]="Hume"
  ["eleven_multilingual_v2"]="eleven_multilingual_v2"
  ["deepgram"]="deepgram"
  ["orpheus-tts-0.1-finetune-prod"]="canopylabs/orpheus-tts-0.1-finetune-prod"
  ["Sesame1B"]="Sesame1B"
  ["Qwen2.5-Omni-7B"]="Qwen/Qwen2.5-Omni-7B"
  ["gpt-4o-mini-tts"]="gpt-4o-mini-tts"
)

export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export ELEVENLABS_API_KEY="${ELEVENLABS_API_KEY:-dummy}"
export DEEPGRAM_API_KEY="${DEEPGRAM_API_KEY:-dummy}"
export HUME_API_KEY="${HUME_API_KEY:-dummy}"

mkdir -p "$OUT_ROOT"

EXTRA_ARGS=()
if [[ -n "$NUM_SAMPLES" ]]; then
  EXTRA_ARGS+=(--num_samples "$NUM_SAMPLES")
fi

for tts_model in "${MODELS[@]}"; do
  audio_subdir="${AUDIO_SUBDIR[$tts_model]}"
  audio_dir="$DRIVE_ROOT/$tts_model/$audio_subdir"
  if [[ ! -d "$audio_dir" ]]; then
    echo "[skip] $tts_model: audio dir not found at $audio_dir"
    continue
  fi

  out_dir="$OUT_ROOT/$tts_model"
  mkdir -p "$out_dir"

  echo
  echo "==== running qwen2-audio judge over $tts_model ===="
  python -u evaluation_runner.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH[$tts_model]}" \
    --judge_config "$JUDGE_CFG" \
    --tts_judger_evaluate_function win_rate \
    --baseline_audios_path "$BASELINE_DIR" \
    --fetch_audios_from_path "$audio_dir" \
    --output_dir "$out_dir" \
    "${EXTRA_ARGS[@]}"

  raw_jsonl=$(ls "$out_dir"/*evaluation-predictions.jsonl 2>/dev/null | head -n1 || true)
  if [[ -n "$raw_jsonl" ]]; then
    cp -f "$raw_jsonl" "$out_dir/predictions.jsonl"
    echo "[ok ] $tts_model -> $out_dir/predictions.jsonl"
  else
    echo "[warn] $tts_model: no predictions JSONL written under $out_dir"
  fi
done

echo
echo "All done. Compare with:"
echo "  python judges/compare.py \\"
echo "    --judge-a-root judges/results/gemini --judge-a-name gemini \\"
echo "    --judge-b-root judges/results/qwen2-audio --judge-b-name qwen2-audio \\"
echo "    --out-dir judges/results/comparison"
