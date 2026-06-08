# LLMAJ — TTS Judge Comparison Framework

> **The question:** *Can an open-source audio model replace closed-source
> Gemini-2.5-Pro as the judge for the [EmergentTTS-Eval](https://huggingface.co/datasets/bosonai/EmergentTTS-Eval)
> benchmark?*

This repo wraps the EmergentTTS-Eval inference pipeline so you can plug in a new
candidate judge, run it across the 8 core TTS systems on 4 GPUs, and compare
its verdicts head-to-head with the Gemini baseline at both the **sample level**
(accuracy, Cohen's κ, Pearson r) and the **model-ranking level**
(Spearman ρ, Kendall's W).

**Latest result** ([`judges/results/comparison/SUMMARY.md`](judges/results/comparison/SUMMARY.md)):

| Candidate | Failure rate | Spearman ρ vs Gemini | Verdict |
|---|---:|---:|---|
| Qwen2-Audio-7B | 25.1% | 0.58 (p=0.13) | Not viable — severe parsing failures + position bias |
| **UniSRM** | **0.1%** | **0.76 (p=0.028)** | **Viable for ranking; not a 1:1 sample-level replacement** |

---

## Repo layout

```
LLMAJ/
├── EmergentTTS-Eval-public/     # Submodule fork of bosonai/EmergentTTS-Eval (MOS model, audios, inference loop)
├── configs/
│   ├── judge_qwen2_audio.yaml   # Per-judge YAML: model_path, dtype, generation knobs
│   └── judge_unisrm.yaml
├── judges/
│   ├── base.py                  # BaseJudge interface (load_model, generate_w_audio_comparison)
│   ├── registry.py              # name → Judge class lookup
│   ├── models/                  # Per-judge implementations
│   │   ├── qwen2_audio.py
│   │   └── unisrm.py
│   ├── compare.py               # Sample-level + model-level metrics
│   ├── generate_report.py       # Render full Markdown diagnostic per candidate
│   ├── generate_leaderboard.py  # Render side-by-side SUMMARY.md across candidates
│   ├── inspect_pairs.py         # Pair-level audio + judgement HTML inspector  ← see below
│   └── results/
│       ├── gemini/              # Per-TTS predictions.jsonl from the closed baseline
│       ├── qwen2-audio/         # Same, candidate 1 (gitignored — too big)
│       ├── unisrm/              # Same, candidate 2 (gitignored)
│       └── comparison/          # ★ Pushed: reports, CSVs, JSON ★ (see its README)
├── scripts/
│   ├── download_drive_predictions.py   # Pulls ~12 GB of TTS audio + predictions
│   ├── extract_gemini_results.py       # Symlinks Gemini's predictions in place
│   └── run_judge_4gpu.sh               # Hybrid TP+DP runner for any registered judge
└── models/                              # Local model weights (gitignored)
```

---

## Quick start

### 0. Environment

```bash
conda activate tts        # has torch, transformers, accelerate, pydub, json_repair, etc.
git submodule update --init --recursive
```

### 1. Bootstrap data (~12 GB once)

```bash
python scripts/download_drive_predictions.py   # Pull TTS predictions + audio
python scripts/extract_gemini_results.py       # Symlink Gemini's predictions under judges/results/gemini/
```

### 2. Run a candidate judge over the 8 TTS models

```bash
JUDGE_NAME=unisrm \
  JUDGE_CFG=configs/judge_unisrm.yaml \
  OUT_ROOT=judges/results/unisrm \
  bash scripts/run_judge_4gpu.sh
```

**Hybrid parallelism**: the script launches `NUM_SHARDS` (default 2) data-parallel
workers, each pinned to a TP group of GPUs (default `0,1` and `2,3`) so a 7B
model can split across 2× A10G via `device_map="auto"` and avoid OOM. Each shard
produces `judges/results/<judge>/<TTS>/shard_N/predictions.jsonl`; they get
concatenated into `<TTS>/predictions.jsonl` at the end.

Defaults you can override via env vars:

| Var | Default | Meaning |
|---|---|---|
| `NUM_SHARDS` | `2` | DP workers (each is a separate Python process) |
| `TP_GROUPS` | `0,1;2,3` | `;`-separated CUDA_VISIBLE_DEVICES per worker |
| `NUM_SAMPLES` | `200` | Per-TTS sample budget (tot. = `NUM_SAMPLES × 8 TTS`) |

### 3. Compare against Gemini and render reports

```bash
python -m judges.compare \
  --judge-a-root judges/results/gemini \
  --judge-b-root judges/results/unisrm \
  --judge-a-name gemini --judge-b-name unisrm \
  --out-dir judges/results/comparison

python -m judges.generate_report \
  --judge-a-root judges/results/gemini \
  --judge-b-root judges/results/unisrm \
  --judge-a-name gemini --judge-b-name unisrm \
  --out judges/results/comparison/REPORT_gemini_vs_unisrm.md

python -m judges.generate_leaderboard \
  --baseline-root judges/results/gemini --baseline-name gemini \
  --candidate-roots judges/results/qwen2-audio judges/results/unisrm \
  --candidate-names qwen2-audio unisrm \
  --comparison-dir judges/results/comparison \
  --out judges/results/comparison/SUMMARY.md
```

---

## Listening to pairs of audios with both judges' verdicts

`judges/inspect_pairs.py` generates a static HTML page that shows, for each
sample:

- the target text + category badge,
- the **two audios** (baseline + comparison TTS) as `<audio>` players,
- both judges' **winner / scores / per-audio reasoning / comparison summary**
  side-by-side,
- a green/red `AGREE`/`DISAGREE` badge,
- live filter dropdowns for **Agreement / TTS model / Category**.

Use this when you want to *audition* a handful of disagreements yourself and
decide whose taste you trust.

### Generate a page

```bash
python judges/inspect_pairs.py \
  --judge-a-root judges/results/gemini  --judge-a-name "Gemini-2.5-Pro" \
  --judge-b-root judges/results/unisrm  --judge-b-name "UniSRM" \
  --filter disagree \
  --num-samples 60 \
  --seed 7 \
  --out judges/results/comparison/inspect_gemini_vs_unisrm_disagree.html
```

| Flag | Default | Meaning |
|---|---|---|
| `--filter` | `all` | `disagree` = only show samples where the two judges chose different winners (most informative). `agree` = only show consensus rows (sanity check). |
| `--num-samples` | `30` | How many samples to embed (random sample of the filtered pool). |
| `--tts-model` | _(none)_ | Restrict to one TTS, e.g. `Sesame1B`. |
| `--category` | _(none)_ | Restrict to one EmergentTTS category, e.g. `Questions`. |
| `--seed` | `42` | Reproducible random sub-sampling. |

### View it

The HTML uses absolute server paths like `/drive_data/.../<uid>.wav` so the audio
fetches resolve against an HTTP server rooted at the repo. Two ways to view:

**Local (on the machine that has the audio data):**

```bash
cd /path/to/LLMAJ
python -m http.server 8765
# open http://localhost:8765/judges/results/comparison/inspect_gemini_vs_unisrm_disagree.html
```

**Remote machine, view from your laptop** (SSH port-forward):

```bash
# on the remote (linux) box:
cd /path/to/LLMAJ && python -m http.server 8765

# on your laptop, in a new terminal:
ssh -N -L 8765:localhost:8765 user@remote-host
# then open http://localhost:8765/judges/results/comparison/inspect_gemini_vs_unisrm_disagree.html
```

Audio playback requires the `drive_data/` and `EmergentTTS-Eval-public/data/`
directories from step 1 — without them the page renders fine but the audio
buttons return 404.

### Recommended workflow for "do I prefer Gemini or UniSRM?"

1. Generate the **disagree** view (those rows are where the choice actually matters).
2. Pick a `--tts-model` you have an opinion on — e.g. `Sesame1B` (low quality, lots
   of judge disagreement) or `gpt-4o-audio-preview-2024-12-17` (high quality,
   subtle differences).
3. For 10–20 samples: listen to A and B, decide your own winner, then expand
   each judge's reasoning. Tally how often you agree with each.
4. Cross-check categories: `Questions` and `Pronunciation` are where the two
   judges diverge most.

---

## Adding a new candidate judge

1. **YAML config** in `configs/judge_<name>.yaml`:
   ```yaml
   judge:
     name: <name>
     model_path: /abs/path/to/checkpoint
     device: cuda
     dtype: bfloat16
     generation:
       temperature: 0
       max_new_tokens: 2048
       top_p: 0.9
   ```
2. **Adapter** in `judges/models/<name>.py` subclassing `BaseJudge`. Implement:
   - `load_model(self)` — populate `self.model` and `self.processor`.
   - `_build_conversation(...)` — format the system + user prompt + audio inputs
     into whatever your model expects.
   - `generate_w_audio_comparison(...)` — return a string that `inference.py`'s
     `json_repair.loads` can parse into the EmergentTTS 6-key schema:
     `{winner, score_1, score_2, reasoning_system_1, reasoning_system_2, system_comparison}`.
3. **Register** the class in `judges/registry.py` so `JUDGE_NAME=<name>` resolves.
4. Run steps 2 and 3 of the Quick Start. The runner script and `compare.py` are
   judge-agnostic.

See `judges/models/qwen2_audio.py` (ChatML/HF Transformers reference) and
`judges/models/unisrm.py` (Qwen2.5-Omni-Thinker reference, with prompt + score
schema translation between native UniSRM output and EmergentTTS-Eval's expected
JSON) as templates.

---

## What's in the repo vs gitignored

| Pushed | Ignored |
|---|---|
| All code (`judges/`, `scripts/`, `configs/`) | `models/` (local weights, ~17 GB) |
| `judges/results/comparison/*.{md,csv,json}` | `judges/results/<judge>/` per-TTS predictions |
| | `drive_data/`, `EmergentTTS-Eval-public/data/` (~12 GB) |

So a fresh clone has every report and every config, but to actually re-run a
judge you need to bootstrap the data + models locally.

---

## Results — go straight to the comparison

- [`judges/results/comparison/SUMMARY.md`](judges/results/comparison/SUMMARY.md) — leaderboard
- [`judges/results/comparison/REPORT_gemini_vs_unisrm.md`](judges/results/comparison/REPORT_gemini_vs_unisrm.md) — UniSRM full report
- [`judges/results/comparison/REPORT_gemini_vs_qwen2-audio.md`](judges/results/comparison/REPORT_gemini_vs_qwen2-audio.md) — Qwen2-Audio full report
- [`judges/results/comparison/README.md`](judges/results/comparison/README.md) — explanation of every metric and file
