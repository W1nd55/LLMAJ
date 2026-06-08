"""Generate a static HTML inspector for listening to TTS pairs and comparing judges.

For each sample it shows:
  * The target text and category
  * Both audios (baseline + comparison TTS) as <audio> tags so you can listen
  * Side-by-side judgements from judge_a and judge_b: winner, scores, reasoning
  * Whether the judges agreed on the winner

Filtering:
  --filter agree    only samples where both judges chose the same winner
  --filter disagree only samples where judges chose different winners (most informative)
  --filter all      no filter
  --tts-model       restrict to one TTS model (e.g. Sesame1B)
  --category        restrict to one category (Questions / Pronunciation / ...)

Serving:
  The HTML uses absolute server paths starting with '/' so audio fetches resolve
  via an HTTP server rooted at the repo. After generation::

      cd /home/ubuntu/Project/LLMAJ
      python -m http.server 8000

  Then open http://localhost:8000/<HTML_PATH>
"""
from __future__ import annotations

import argparse
import html
import json
import os
import random
from collections import defaultdict

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def absolute_to_server(path: str) -> str:
    """Convert a /home/ubuntu/Project/LLMAJ/foo/bar.wav path to /foo/bar.wav."""
    if not path:
        return ""
    if path.startswith(REPO_ROOT):
        return path[len(REPO_ROOT):] or "/"
    return path  # keep as-is; user will need to fix


def canonical_winner(judger_winner: int, predicted_speech_index: int) -> int | None:
    if judger_winner == -1:
        return None
    if judger_winner == 0:
        return 0
    if (predicted_speech_index == 1 and judger_winner == 1) or (
        predicted_speech_index == 2 and judger_winner == 2
    ):
        return 2  # comparison TTS wins
    return 1  # baseline wins


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_records(
    judge_a_root: str,
    judge_b_root: str,
    tts_models: list[str],
) -> list[dict]:
    out = []
    for tts in tts_models:
        path_a = os.path.join(judge_a_root, tts, "predictions.jsonl")
        path_b = os.path.join(judge_b_root, tts, "predictions.jsonl")
        if not (os.path.isfile(path_a) and os.path.isfile(path_b)):
            continue
        a_by_uid = {r["unique_id_eval"]: r for r in load_jsonl(path_a)}
        b_by_uid = {r["unique_id_eval"]: r for r in load_jsonl(path_b)}
        for uid in sorted(set(a_by_uid) & set(b_by_uid)):
            a, b = a_by_uid[uid], b_by_uid[uid]
            if a.get("predicted_speech_index") != b.get("predicted_speech_index"):
                continue
            wa = a["judger_output_win_rate_based"].get("winner", -1)
            wb = b["judger_output_win_rate_based"].get("winner", -1)
            ca = canonical_winner(wa, a["predicted_speech_index"])
            cb = canonical_winner(wb, b["predicted_speech_index"])
            baseline_path = a.get("baseline_audio_path") or b.get("baseline_audio_path") or ""
            comparison_path = a.get("audio_out_path") or b.get("audio_out_path") or ""
            out.append({
                "tts_model": tts,
                "uid": uid,
                "category": a.get("category"),
                "text": a.get("text_to_synthesize"),
                "predicted_speech_index": a.get("predicted_speech_index"),
                "baseline_audio": absolute_to_server(baseline_path),
                "comparison_audio": absolute_to_server(comparison_path),
                "judge_a": {
                    "raw_winner": wa,
                    "canonical": ca,
                    "score_1": a["judger_output_win_rate_based"].get("score_1"),
                    "score_2": a["judger_output_win_rate_based"].get("score_2"),
                    "reasoning_1": a["judger_output_win_rate_based"].get("reasoning_system_1", ""),
                    "reasoning_2": a["judger_output_win_rate_based"].get("reasoning_system_2", ""),
                    "comparison": a["judger_output_win_rate_based"].get("system_comparison", ""),
                },
                "judge_b": {
                    "raw_winner": wb,
                    "canonical": cb,
                    "score_1": b["judger_output_win_rate_based"].get("score_1"),
                    "score_2": b["judger_output_win_rate_based"].get("score_2"),
                    "reasoning_1": b["judger_output_win_rate_based"].get("reasoning_system_1", ""),
                    "reasoning_2": b["judger_output_win_rate_based"].get("reasoning_system_2", ""),
                    "comparison": b["judger_output_win_rate_based"].get("system_comparison", ""),
                },
                "agree": ca is not None and cb is not None and ca == cb,
                "both_valid": ca is not None and cb is not None,
            })
    return out


def filter_records(
    records: list[dict],
    mode: str,
    tts_model: str | None,
    category: str | None,
) -> list[dict]:
    out = records
    if tts_model:
        out = [r for r in out if r["tts_model"] == tts_model]
    if category:
        out = [r for r in out if r["category"] == category]
    out = [r for r in out if r["both_valid"]]
    if mode == "agree":
        out = [r for r in out if r["agree"]]
    elif mode == "disagree":
        out = [r for r in out if not r["agree"]]
    return out


def label_winner(canonical: int | None) -> str:
    return {None: "—", 0: "TIE", 1: "BASELINE", 2: "TTS"}[canonical]


def render_card(r: dict, judge_a_name: str, judge_b_name: str) -> str:
    """Render one sample as an HTML card."""
    a = r["judge_a"]
    b = r["judge_b"]
    agree_color = "#5cb85c" if r["agree"] else "#d9534f"
    agree_text = "AGREE" if r["agree"] else "DISAGREE"

    psi = r["predicted_speech_index"]
    a1_label = "Speech A (TTS)" if psi == 1 else "Speech A (baseline)"
    a2_label = "Speech B (baseline)" if psi == 1 else "Speech B (TTS)"

    text_safe = html.escape(r["text"] or "")
    cat_safe = html.escape(r["category"] or "")
    tts_safe = html.escape(r["tts_model"] or "")
    a_r1 = html.escape(a["reasoning_1"] or "")
    a_r2 = html.escape(a["reasoning_2"] or "")
    a_cmp = html.escape(a["comparison"] or "")
    b_r1 = html.escape(b["reasoning_1"] or "")
    b_r2 = html.escape(b["reasoning_2"] or "")
    b_cmp = html.escape(b["comparison"] or "")

    s1a = a.get("score_1"); s2a = a.get("score_2")
    s1b = b.get("score_1"); s2b = b.get("score_2")
    s1a_str = f"{s1a:.2f}" if isinstance(s1a, (int, float)) else "—"
    s2a_str = f"{s2a:.2f}" if isinstance(s2a, (int, float)) else "—"
    s1b_str = f"{s1b:.2f}" if isinstance(s1b, (int, float)) else "—"
    s2b_str = f"{s2b:.2f}" if isinstance(s2b, (int, float)) else "—"

    return f"""
<div class="card" data-agree="{1 if r['agree'] else 0}" data-tts="{tts_safe}" data-cat="{cat_safe}">
  <div class="card-header">
    <span class="badge tts">{tts_safe}</span>
    <span class="badge cat">{cat_safe}</span>
    <span class="badge uid">uid={r['uid']}</span>
    <span class="badge agree" style="background:{agree_color}">{agree_text}</span>
  </div>
  <div class="text">{text_safe}</div>

  <div class="audios">
    <div class="audio-block">
      <div class="audio-label">{a1_label}</div>
      <audio controls preload="none" src="{html.escape(r['baseline_audio'] if psi==2 else r['comparison_audio'])}"></audio>
    </div>
    <div class="audio-block">
      <div class="audio-label">{a2_label}</div>
      <audio controls preload="none" src="{html.escape(r['baseline_audio'] if psi==1 else r['comparison_audio'])}"></audio>
    </div>
  </div>

  <div class="judges">
    <div class="judge">
      <div class="judge-name">{html.escape(judge_a_name)}</div>
      <div class="verdict">winner = <b>{label_winner(a['canonical'])}</b>
        &nbsp;|&nbsp; score_1 = {s1a_str} &nbsp;|&nbsp; score_2 = {s2a_str}</div>
      <details><summary>Reasoning [Speech A]</summary><pre>{a_r1}</pre></details>
      <details><summary>Reasoning [Speech B]</summary><pre>{a_r2}</pre></details>
      <details><summary>Comparison summary</summary><pre>{a_cmp}</pre></details>
    </div>
    <div class="judge">
      <div class="judge-name">{html.escape(judge_b_name)}</div>
      <div class="verdict">winner = <b>{label_winner(b['canonical'])}</b>
        &nbsp;|&nbsp; score_1 = {s1b_str} &nbsp;|&nbsp; score_2 = {s2b_str}</div>
      <details><summary>Reasoning [Speech A]</summary><pre>{b_r1}</pre></details>
      <details><summary>Reasoning [Speech B]</summary><pre>{b_r2}</pre></details>
      <details><summary>Comparison summary</summary><pre>{b_cmp}</pre></details>
    </div>
  </div>
</div>
"""


CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, system-ui, sans-serif; margin: 0; padding: 1rem 2rem; max-width: 1400px; margin: 0 auto; background: #fafafa; color: #222; }
h1 { margin-top: 0; }
.controls { position: sticky; top: 0; background: #fafafa; padding: 0.75rem 0; border-bottom: 1px solid #ddd; z-index: 10; display: flex; flex-wrap: wrap; gap: 1rem; align-items: center; }
.controls label { font-size: 0.9em; }
.controls select { padding: 0.25rem 0.5rem; }
.summary { font-size: 0.9em; color: #666; }
.card { border: 1px solid #ddd; border-radius: 8px; padding: 1rem 1.25rem; margin: 1rem 0; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
.card-header { display: flex; gap: 0.5rem; flex-wrap: wrap; align-items: center; margin-bottom: 0.5rem; }
.badge { padding: 2px 8px; border-radius: 12px; font-size: 0.8em; color: white; }
.badge.tts { background: #5b9bd5; }
.badge.cat { background: #70ad47; }
.badge.uid { background: #777; }
.badge.agree { font-weight: bold; }
.text { font-size: 1.05em; line-height: 1.4; padding: 0.5rem 0.75rem; background: #f3f3f3; border-left: 3px solid #5b9bd5; margin-bottom: 0.75rem; border-radius: 4px; }
.audios { display: flex; gap: 1rem; margin-bottom: 0.75rem; }
.audio-block { flex: 1; }
.audio-label { font-size: 0.85em; color: #666; margin-bottom: 0.25rem; }
.audio-block audio { width: 100%; }
.judges { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
.judge { border: 1px solid #eee; padding: 0.75rem; border-radius: 6px; background: #fbfbfb; }
.judge-name { font-weight: bold; color: #333; margin-bottom: 0.25rem; font-family: ui-monospace, monospace; }
.verdict { font-size: 0.95em; margin-bottom: 0.5rem; }
.judge details { margin-top: 0.25rem; font-size: 0.88em; }
.judge details summary { cursor: pointer; color: #555; }
.judge pre { white-space: pre-wrap; word-break: break-word; max-height: 250px; overflow-y: auto; background: #f7f7f7; padding: 0.5rem; border-radius: 4px; font-family: ui-monospace, monospace; font-size: 0.85em; }
"""


JS = """
function applyFilters() {
  const agreeFilter = document.getElementById('agree-filter').value;
  const ttsFilter = document.getElementById('tts-filter').value;
  const catFilter = document.getElementById('cat-filter').value;
  let shown = 0;
  document.querySelectorAll('.card').forEach(c => {
    const agree = c.dataset.agree === '1';
    const tts = c.dataset.tts;
    const cat = c.dataset.cat;
    let ok = true;
    if (agreeFilter === 'agree' && !agree) ok = false;
    if (agreeFilter === 'disagree' && agree) ok = false;
    if (ttsFilter !== 'all' && tts !== ttsFilter) ok = false;
    if (catFilter !== 'all' && cat !== catFilter) ok = false;
    c.style.display = ok ? '' : 'none';
    if (ok) shown++;
  });
  document.getElementById('shown-count').textContent = shown;
}
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('agree-filter').addEventListener('change', applyFilters);
  document.getElementById('tts-filter').addEventListener('change', applyFilters);
  document.getElementById('cat-filter').addEventListener('change', applyFilters);
  applyFilters();
});
"""


def render_html(records: list[dict], judge_a_name: str, judge_b_name: str) -> str:
    tts_options = sorted({r["tts_model"] for r in records})
    cat_options = sorted({r["category"] for r in records})
    n_total = len(records)
    n_agree = sum(1 for r in records if r["agree"])
    n_disagree = n_total - n_agree

    cards_html = "\n".join(render_card(r, judge_a_name, judge_b_name) for r in records)
    tts_select = "\n".join([f'<option value="{html.escape(t)}">{html.escape(t)}</option>' for t in tts_options])
    cat_select = "\n".join([f'<option value="{html.escape(c)}">{html.escape(c)}</option>' for c in cat_options])

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Inspect: {html.escape(judge_a_name)} vs {html.escape(judge_b_name)}</title>
<style>{CSS}</style>
</head>
<body>
<h1>Pair inspector: <code>{html.escape(judge_a_name)}</code> vs <code>{html.escape(judge_b_name)}</code></h1>
<p class="summary">{n_total} samples loaded · {n_agree} agree · {n_disagree} disagree.
Click any audio to listen. Expand reasoning sections to see what each judge wrote.</p>

<div class="controls">
  <label>Agreement:
    <select id="agree-filter">
      <option value="all">all</option>
      <option value="disagree">disagree (most informative)</option>
      <option value="agree">agree</option>
    </select>
  </label>
  <label>TTS model:
    <select id="tts-filter">
      <option value="all">all</option>
      {tts_select}
    </select>
  </label>
  <label>Category:
    <select id="cat-filter">
      <option value="all">all</option>
      {cat_select}
    </select>
  </label>
  <span class="summary">Showing <span id="shown-count">{n_total}</span> / {n_total}.</span>
</div>

{cards_html}

<script>{JS}</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-a-root", required=True)
    ap.add_argument("--judge-b-root", required=True)
    ap.add_argument("--judge-a-name", required=True)
    ap.add_argument("--judge-b-name", required=True)
    ap.add_argument("--num-samples", type=int, default=30)
    ap.add_argument("--filter", choices=["agree", "disagree", "all"], default="all")
    ap.add_argument("--tts-model", default=None)
    ap.add_argument("--category", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, help="Output HTML path (relative to repo root for serving)")
    args = ap.parse_args()

    a_models = [d for d in os.listdir(args.judge_a_root) if os.path.isdir(os.path.join(args.judge_a_root, d))]
    b_models = [d for d in os.listdir(args.judge_b_root) if os.path.isdir(os.path.join(args.judge_b_root, d))]
    shared = sorted(set(a_models) & set(b_models))

    records = build_records(args.judge_a_root, args.judge_b_root, shared)
    print(f"Loaded {len(records)} aligned samples across {len(shared)} TTS models.")

    records = filter_records(records, args.filter, args.tts_model, args.category)
    print(f"After filtering ({args.filter}, tts={args.tts_model}, cat={args.category}): {len(records)} samples.")

    rng = random.Random(args.seed)
    rng.shuffle(records)
    records = records[: args.num_samples]
    records.sort(key=lambda r: (r["tts_model"], r["category"], r["uid"]))

    html_str = render_html(records, args.judge_a_name, args.judge_b_name)
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html_str)

    rel_to_repo = os.path.relpath(args.out, REPO_ROOT)
    print(f"\n[saved] {args.out}")
    print(f"\nTo browse via local HTTP server:")
    print(f"  cd {REPO_ROOT}")
    print(f"  python -m http.server 8765")
    print(f"  open http://localhost:8765/{rel_to_repo}")


if __name__ == "__main__":
    main()
