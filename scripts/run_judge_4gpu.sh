#!/usr/bin/env bash
# Run a registered local judge over the 8 core TTS models with hybrid TP+DP
# parallelism: NUM_SHARDS DP workers, each pinned to a TP group of GPUs.
#
# Usage:
#   JUDGE_NAME=unisrm JUDGE_CFG=configs/judge_unisrm.yaml \
#     OUT_ROOT=judges/results/unisrm \
#     bash scripts/run_judge_4gpu.sh
#
#   JUDGE_NAME=qwen2-audio JUDGE_CFG=configs/judge_qwen2_audio.yaml \
#     OUT_ROOT=judges/results/qwen2-audio \
#     bash scripts/run_judge_4gpu.sh
#
# Env knobs:
#   NUM_SHARDS=2, TP_GROUPS="0,1;2,3" -> 2 workers, each spreading the 7B model
#   across 2 A10G via device_map="auto".
#   NUM_SAMPLES=200 (per shard sample budget per TTS model).

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

JUDGE_NAME="${JUDGE_NAME:?must set JUDGE_NAME (registered judge identifier, e.g. unisrm)}"
JUDGE_CFG="${JUDGE_CFG:?must set JUDGE_CFG (path to YAML, relative to repo root or absolute)}"
OUT_ROOT="${OUT_ROOT:-judges/results/$JUDGE_NAME}"

# Normalize to absolute paths BEFORE cd into EmergentTTS-Eval-public
[[ "$JUDGE_CFG" != /* ]] && JUDGE_CFG="$REPO_ROOT/$JUDGE_CFG"
[[ "$OUT_ROOT" != /* ]] && OUT_ROOT="$REPO_ROOT/$OUT_ROOT"

cd "$REPO_ROOT/EmergentTTS-Eval-public"

DRIVE_ROOT="${DRIVE_ROOT:-$REPO_ROOT/drive_data/EmergentTTS-Eval_Predictions}"
BASELINE_DIR="${BASELINE_DIR:-data/baseline_audios}"
NUM_SAMPLES="${NUM_SAMPLES:-200}"
NUM_SHARDS="${NUM_SHARDS:-2}"
TP_GROUPS="${TP_GROUPS:-0,1;2,3}"
MODELS_OVERRIDE="${MODELS:-}"

IFS=';' read -r -a TP_GROUP_LIST <<< "$TP_GROUPS"
if [[ ${#TP_GROUP_LIST[@]} -ne $NUM_SHARDS ]]; then
  echo "[fatal] TP_GROUPS group count (${#TP_GROUP_LIST[@]}) must equal NUM_SHARDS ($NUM_SHARDS). Got TP_GROUPS=$TP_GROUPS, NUM_SHARDS=$NUM_SHARDS" >&2
  exit 1
fi

DEFAULT_MODELS=(
  "gpt-4o-mini-tts"
  "gpt-4o-audio-preview-2024-12-17"
  "HumeAI"
  "eleven_multilingual_v2"
  "deepgram"
  "orpheus-tts-0.1-finetune-prod"
  "Sesame1B"
  "Qwen2.5-Omni-7B"
)

if [[ -n "$MODELS_OVERRIDE" ]]; then
  IFS=' ' read -r -a MODELS <<< "$MODELS_OVERRIDE"
else
  MODELS=("${DEFAULT_MODELS[@]}")
fi

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

PLACEHOLDER_MODEL_NAME_OR_PATH="gpt-4o-mini-tts"

export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export ELEVENLABS_API_KEY="${ELEVENLABS_API_KEY:-dummy}"
export DEEPGRAM_API_KEY="${DEEPGRAM_API_KEY:-dummy}"
export HUME_API_KEY="${HUME_API_KEY:-dummy}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$OUT_ROOT"

EXTRA_ARGS=()
if [[ -n "$NUM_SAMPLES" ]]; then
  EXTRA_ARGS+=(--num_samples "$NUM_SAMPLES")
fi

for tts_model in "${MODELS[@]}"; do
  audio_subdir="${AUDIO_SUBDIR[$tts_model]:-}"
  if [[ -z "$audio_subdir" ]]; then
    echo "[skip] $tts_model: no AUDIO_SUBDIR mapping"
    continue
  fi
  audio_dir="$DRIVE_ROOT/$tts_model/$audio_subdir"
  if [[ ! -d "$audio_dir" ]]; then
    echo "[skip] $tts_model: audio dir not found at $audio_dir"
    continue
  fi

  out_dir="$OUT_ROOT/$tts_model"
  mkdir -p "$out_dir"

  echo
  echo "==== running $JUDGE_NAME judge over $tts_model (NUM_SHARDS=$NUM_SHARDS, TP_GROUPS=$TP_GROUPS) ===="
  echo "audio_dir=$audio_dir"
  echo "out_dir=$out_dir"

  pids=()
  for ((s = 0; s < NUM_SHARDS; s++)); do
    tp_group="${TP_GROUP_LIST[$s]}"
    shard_out="$out_dir/shard_$s"
    mkdir -p "$shard_out"
    log="$shard_out/run.log"

    CUDA_VISIBLE_DEVICES="$tp_group" \
      python -u evaluation_runner.py \
        --model_name_or_path "$PLACEHOLDER_MODEL_NAME_OR_PATH" \
        --judge_config "$JUDGE_CFG" \
        --judge_model_provider "$JUDGE_NAME" \
        --tts_judger_evaluate_function win_rate \
        --baseline_audios_path "$BASELINE_DIR" \
        --fetch_audios_from_path "$audio_dir" \
        --output_dir "$shard_out" \
        --shard_id "$s" \
        --num_shards "$NUM_SHARDS" \
        --api_num_threads 1 \
        "${EXTRA_ARGS[@]}" \
        > "$log" 2>&1 &
    pids+=($!)
    echo "  [launched] shard $s on GPUs [$tp_group], pid=${pids[$s]}, log=$log"
  done

  fail=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      echo "[fail] shard pid $pid exited non-zero"
      fail=1
    fi
  done

  if (( fail == 1 )); then
    echo "[error] $tts_model: at least one shard failed; check $out_dir/shard_*/run.log"
    continue
  fi

  shopt -s nullglob
  shard_jsonls=("$out_dir"/shard_*/*evaluation-predictions.jsonl)
  shopt -u nullglob
  if [[ ${#shard_jsonls[@]} -ne $NUM_SHARDS ]]; then
    echo "[warn] $tts_model: expected $NUM_SHARDS shard JSONLs, found ${#shard_jsonls[@]}"
  fi
  cat "${shard_jsonls[@]}" > "$out_dir/predictions.jsonl"
  n_lines=$(wc -l < "$out_dir/predictions.jsonl")
  echo "[ok ] $tts_model -> $out_dir/predictions.jsonl ($n_lines records)"
done

echo
echo "All done. Compare with:"
echo "  python judges/compare.py \\"
echo "    --judge-a-root judges/results/gemini --judge-a-name gemini \\"
echo "    --judge-b-root $OUT_ROOT --judge-b-name $JUDGE_NAME \\"
echo "    --out-dir judges/results/comparison"
