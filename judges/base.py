from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseJudge(ABC):
    """Abstract base class for all judge model plugins.

    Subclasses must implement:
      - load_model(): load weights/processor onto device
      - generate_w_audio_comparison(): pairwise audio comparison returning JSON text
      - default_generation_config: property returning default generation kwargs
    """

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights to device. Called once at startup."""
        ...

    @abstractmethod
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
        """Perform pairwise audio comparison and return a JSON string.

        Expected JSON schema:
            {
                "reasoning_system_1": str,
                "reasoning_system_2": str,
                "system_comparison": str,
                "score_1": int (0-3),
                "score_2": int (0-3),
                "winner": int (0=tie, 1=system1, 2=system2)
            }
        """
        ...

    @property
    @abstractmethod
    def default_generation_config(self) -> dict:
        """Return default generation config (temperature, max_new_tokens, top_p, etc.)."""
        ...
