from __future__ import annotations

from typing import Any

import numpy as np

from judges.base import BaseJudge
from judges.registry import register_judge

AUDIO_PLACEHOLDER = "audio_placeholder"


@register_judge("qwen2-audio")
class Qwen2AudioJudge(BaseJudge):
    """Judge implementation using Qwen2-Audio-7B-Instruct (local transformers inference)."""

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
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=self._torch_dtype,
            device_map=self.device if self.device != "cuda" else "auto",
        )
        self.model.eval()

    @property
    def default_generation_config(self) -> dict:
        return {
            "temperature": self._gen_config.get("temperature", 0),
            "max_new_tokens": self._gen_config.get("max_new_tokens", 16384),
            "top_p": self._gen_config.get("top_p", 0.9),
        }

    def _resample_if_needed(self, audio_array: np.ndarray) -> np.ndarray:
        """Resample audio to the processor's expected sample rate if needed.

        EmergentTTS pipeline provides 16kHz audio; Qwen2-Audio expects 16kHz as well,
        but we check to be safe.
        """
        target_sr = self.processor.feature_extractor.sampling_rate
        source_sr = 16000
        if target_sr != source_sr:
            import librosa

            audio_array = librosa.resample(
                audio_array, orig_sr=source_sr, target_sr=target_sr
            )
        return audio_array

    def _build_conversation(
        self,
        system_message: str,
        user_message: str,
        post_audio_1_message: str,
    ) -> list[dict]:
        """Build ChatML conversation with audio placeholders."""
        conversation = []
        if system_message:
            conversation.append({"role": "system", "content": system_message})

        user_content = [
            {"type": "text", "text": user_message},
            {"type": "audio", "audio_url": AUDIO_PLACEHOLDER},
            {"type": "text", "text": post_audio_1_message},
            {"type": "audio", "audio_url": AUDIO_PLACEHOLDER},
        ]
        conversation.append({"role": "user", "content": user_content})
        return conversation

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

        conversation = self._build_conversation(
            system_message, user_message, post_audio_1_message
        )

        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        audios = [
            self._resample_if_needed(audio_array_1),
            self._resample_if_needed(audio_array_2),
        ]

        inputs = self.processor(
            text=text, audios=audios, return_tensors="pt", padding=True
        )
        inputs = inputs.to(self.model.device)

        temperature = generation_config.get("temperature", 0)
        max_new_tokens = generation_config.get("max_new_tokens", 16384)
        top_p = generation_config.get("top_p", 0.9)

        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generate_kwargs)
        output_ids = output_ids[:, inputs.input_ids.size(1):]

        response = self.processor.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return response
