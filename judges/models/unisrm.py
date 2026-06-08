"""UniSRM judge: Qwen2.5-Omni-Thinker fine-tuned for pairwise speech reward.

UniSRM (Wang et al., 2026) is a unified speech reward model trained on top of
Qwen2.5-Omni-7B-Thinker. It supports A/B speech comparison via a fixed
``<think>...</think><answer>Speech A is better</answer>`` schema, scoring
three dimensions (Text Fidelity, Scenario Style Match, Naturalness) on 0-10.

This adapter:
  * Uses UniSRM's NATIVE scenario prompt (the model was trained on this format
    -- forcing it into EmergentTTS's strict 6-key JSON schema would defeat the
    purpose).
  * Parses ``text_to_synthesize`` and ``text_category`` out of EmergentTTS's
    user_message (since the framework only passes the rendered prompt, not the
    structured fields).
  * Maps UniSRM's output back into the 6-key JSON schema EmergentTTS expects:
        winner (1/2/0), score_1, score_2, reasoning_system_*, system_comparison.

References:
  * https://github.com/lavendery/UniSRM
  * https://huggingface.co/lavendery/UniSRM
"""
from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

from judges.base import BaseJudge
from judges.registry import register_judge


SYSTEM_PROMPT_SCENARIO = """
You are an expert judge for SCENARIO-AWARE speech evaluation.

Inputs:
- Scene Context: Scenario Description, Paragraph Context, Target Emotion.
- Target text: the exact sentence that should be spoken.
- Two audios: [Speech A] and [Speech B] for the SAME target text.

Your task:
1) Evaluate Speech A and Speech B as realizations of the target text under the given context.
2) For EACH speech, give scores in [0, 10] and 1-2 sentence explanations on THREE dimensions:
   (1) Text Fidelity & Intelligibility
   (2) Scenario Style Match  [CRITICAL]
   (3) Naturalness & Audio Quality
3) Compute Total_A and Total_B as the sum of the three scores (they MUST be different).
4) In <answer>, decide which speech is better overall.

Dimension hints:
- Text Fidelity & Intelligibility: matches the target text? clear and understandable?
- Scenario Style Match: does emotion and speaking style fit the target emotion + context?
- Naturalness & Audio Quality: human-like, stable, and comfortable to listen to?

Hard constraints:
- Output ONLY <think> and <answer>, nothing else.
- In <think>, you MUST include both [Speech A] and [Speech B] sections with scores and explanations,
  and a [Comparison summary] with 2-4 sentences explaining the key differences and the winner.
- In <answer>, output EXACTLY:
  "Speech A is better" OR "Speech B is better".

Output format:
<think>
[Speech A]
1) Text Fidelity & Intelligibility: score=a1/10; explanation: ...
2) Scenario Style Match:           score=a2/10; explanation: ...
3) Naturalness & Audio Quality:    score=a3/10; explanation: ...
Total_A = a1+a2+a3 = A_total

[Speech B]
1) Text Fidelity & Intelligibility: score=b1/10; explanation: ...
2) Scenario Style Match:           score=b2/10; explanation: ...
3) Naturalness & Audio Quality:    score=b3/10; explanation: ...
Total_B = b1+b2+b3 = B_total

[Comparison summary]
- 2-4 sentences highlighting the main differences and why the winner is better.
</think>
<answer>Speech A is better</answer>
"""

CATEGORY_TO_SCENARIO = {
    "Questions": (
        "Inquisitive",
        "TTS rendering of text containing questions and statements; proper interrogative "
        "intonation and clear differentiation of declarative vs interrogative phrasing are critical.",
    ),
    "Emotions": (
        "Variable (context-dependent)",
        "Expressive narration with quoted dialogue; the speaker should clearly differentiate "
        "narrative tone from emotional dialogue and modulate emotion across quotes.",
    ),
    "Paralinguistics": (
        "Variable (context-dependent)",
        "Speech containing paralinguistic cues such as interjections, vowel elongation, "
        "syllable stress, stutters, and pacing markers; cues must be rendered naturally.",
    ),
    "Syntactic Complexity": (
        "Neutral",
        "Reading complex sentence structures; appropriate prosody (pauses, phrasing, stress) "
        "must clarify the syntactic structure and intended meaning.",
    ),
    "Foreign Words": (
        "Neutral",
        "Bilingual or code-switched text containing foreign words; the speaker should pronounce "
        "foreign words with correct or accepted anglicized pronunciation and seamless code-switching.",
    ),
    "Pronunciation": (
        "Neutral",
        "Reading text with complex pronunciation: numbers, dates, URLs, equations, acronyms, "
        "tongue twisters; correctness and clarity of these elements is the primary concern.",
    ),
}


def _extract_field_from_user_message(msg: str, field_name: str) -> str:
    """Extract a labeled section from EmergentTTS's rendered user_message.

    The user_message contains sections like::

        **text_to_synthesize**
        Some text here

        **text_category**
        Questions

    """
    pattern = rf"\*\*{re.escape(field_name)}\*\*\s*\n\s*(.+?)(?=\n\s*\n|\Z)"
    m = re.search(pattern, msg, flags=re.DOTALL)
    return m.group(1).strip() if m else ""


def _build_unisrm_user_prompt(text: str, category: str) -> str:
    emotion, scenario = CATEGORY_TO_SCENARIO.get(category, ("Neutral", category))
    return (
        "[Scene Context]\n"
        f"- Scenario Description: {scenario}\n"
        f"- Paragraph Context: TTS evaluation under the EmergentTTS-Eval '{category}' category.\n"
        f"- Target Emotion: {emotion}\n\n"
        "[Target text to be spoken exactly]\n"
        f"{text}\n\n"
        "Please judge Speech A and Speech B based on the scene context, target emotion, "
        "and target text. Follow the system instructions: score the three dimensions for "
        "both speeches with explanations, compute Total_A and Total_B (they must be different), "
        "and output your reasoning and final choice strictly in the required <think> and <answer> format."
    )


_RE_ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
_RE_TOTAL_A = re.compile(r"Total[_\s]*A\s*=.*?=\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_RE_TOTAL_B = re.compile(r"Total[_\s]*B\s*=.*?=\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_RE_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_RE_SECTION_A = re.compile(r"\[Speech\s*A\][^\[]*", re.IGNORECASE)
_RE_SECTION_B = re.compile(r"\[Speech\s*B\][^\[]*", re.IGNORECASE)
_RE_SECTION_COMP = re.compile(r"\[Comparison summary\][^\[]*", re.IGNORECASE)


def _parse_unisrm_response(reply: str) -> dict:
    """Convert UniSRM's <think>/<answer> output into EmergentTTS schema.

    Returns a dict with keys: reasoning_system_1, reasoning_system_2,
    system_comparison, score_1, score_2, winner. Raises ValueError on
    unrecoverable malformations.
    """
    answer_m = _RE_ANSWER.search(reply)
    if not answer_m:
        raise ValueError("missing <answer> block")
    answer_text = answer_m.group(1)
    if "Speech A is better" in answer_text:
        winner = 1
    elif "Speech B is better" in answer_text:
        winner = 2
    else:
        raise ValueError(f"unrecognized verdict in <answer>: {answer_text!r}")

    think_m = _RE_THINK.search(reply)
    think_text = think_m.group(1) if think_m else reply

    a_total_m = _RE_TOTAL_A.search(think_text)
    b_total_m = _RE_TOTAL_B.search(think_text)
    score_1 = float(a_total_m.group(1)) if a_total_m else 0.0
    score_2 = float(b_total_m.group(1)) if b_total_m else 0.0

    score_1 = max(0.0, min(score_1, 30.0)) / 10.0
    score_2 = max(0.0, min(score_2, 30.0)) / 10.0

    sec_a = _RE_SECTION_A.search(think_text)
    sec_b = _RE_SECTION_B.search(think_text)
    sec_comp = _RE_SECTION_COMP.search(think_text)
    reasoning_1 = sec_a.group(0).strip() if sec_a else ""
    reasoning_2 = sec_b.group(0).strip() if sec_b else ""
    comparison = sec_comp.group(0).strip() if sec_comp else think_text.strip()[:1000]

    return {
        "reasoning_system_1": reasoning_1 or "(no [Speech A] section parsed)",
        "reasoning_system_2": reasoning_2 or "(no [Speech B] section parsed)",
        "system_comparison": comparison or "(no [Comparison summary] section parsed)",
        "score_1": score_1,
        "score_2": score_2,
        "winner": winner,
    }


@register_judge("unisrm")
class UniSRMJudge(BaseJudge):
    """Pairwise speech judge using UniSRM (Qwen2.5-Omni-Thinker fine-tune)."""

    def __init__(self, config: dict):
        self.model_path: str = config["model_path"]
        self.device: str = config.get("device", "cuda")
        self.dtype_str: str = config.get("dtype", "bfloat16")
        self._gen_config: dict = config.get("generation", {})
        self.model = None
        self.processor = None

    @property
    def _torch_dtype(self):
        import torch

        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return mapping.get(self.dtype_str, torch.bfloat16)

    def load_model(self) -> None:
        from transformers import (
            Qwen2_5OmniProcessor,
            Qwen2_5OmniThinkerForConditionalGeneration,
        )

        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_path)
        load_kwargs = dict(
            torch_dtype=self._torch_dtype,
            device_map=self.device if self.device != "cuda" else "auto",
        )
        try:
            self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                self.model_path,
                attn_implementation="flash_attention_2",
                **load_kwargs,
            )
            print("[unisrm] loaded with flash_attention_2")
        except (ImportError, ValueError, RuntimeError) as e:
            print(f"[unisrm] flash_attention_2 unavailable ({type(e).__name__}: {e}); falling back to sdpa")
            self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                self.model_path,
                attn_implementation="sdpa",
                **load_kwargs,
            )
        self.model.eval()

    @property
    def default_generation_config(self) -> dict:
        return {
            "temperature": self._gen_config.get("temperature", 0),
            "max_new_tokens": self._gen_config.get("max_new_tokens", 2048),
            "top_p": self._gen_config.get("top_p", 0.9),
        }

    def _resample_if_needed(self, audio_array: np.ndarray) -> np.ndarray:
        target_sr = getattr(self.processor.feature_extractor, "sampling_rate", 16000)
        source_sr = 16000
        if target_sr != source_sr:
            import librosa

            audio_array = librosa.resample(
                audio_array, orig_sr=source_sr, target_sr=target_sr
            )
        return audio_array

    def _build_conversation(
        self,
        target_text: str,
        category: str,
    ) -> list[dict]:
        unisrm_user = _build_unisrm_user_prompt(target_text, category)
        return [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_SCENARIO.strip()}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "\n" + unisrm_user},
                    {"type": "text", "text": "\n[Speech A]: "},
                    {"type": "audio", "audio": "_AUDIO_A_"},
                    {"type": "text", "text": "\n[Speech B]: "},
                    {"type": "audio", "audio": "_AUDIO_B_"},
                ],
            },
        ]

    def generate_w_audio_comparison(
        self,
        model_name: str,
        system_message: str,
        user_message: str,
        audio_array_1: np.ndarray,
        post_audio_1_message: str,
        audio_array_2: np.ndarray,
        **generation_config: Any,
    ) -> str:
        import torch

        target_text = _extract_field_from_user_message(user_message, "text_to_synthesize")
        category = _extract_field_from_user_message(user_message, "text_category")
        if not target_text:
            target_text = "(unknown)"
        if not category:
            category = "Neutral"

        conversation = self._build_conversation(target_text, category)

        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        audios = [
            self._resample_if_needed(audio_array_1),
            self._resample_if_needed(audio_array_2),
        ]

        inputs = self.processor(
            text=text,
            audio=audios,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        temperature = generation_config.get("temperature", 0)
        max_new_tokens = generation_config.get("max_new_tokens", 2048)
        top_p = generation_config.get("top_p", 0.9)

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "use_audio_in_video": False,
            "eos_token_id": self.model.config.eos_token_id,
            "pad_token_id": self.model.config.pad_token_id,
        }
        if temperature > 0:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generate_kwargs)
        output_ids = output_ids[:, inputs.input_ids.size(1):]

        reply = self.processor.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        del inputs, output_ids
        torch.cuda.empty_cache()

        try:
            parsed = _parse_unisrm_response(reply)
            return json.dumps(parsed, ensure_ascii=False)
        except Exception as e:
            return json.dumps(
                {
                    "reasoning_system_1": f"UniSRM parse error: {type(e).__name__}: {e}. Raw reply head: {reply[:500]}",
                    "reasoning_system_2": "(parse failed)",
                    "system_comparison": "(parse failed)",
                    "score_1": 0,
                    "score_2": 0,
                    "winner": -1,
                },
                ensure_ascii=False,
            )
