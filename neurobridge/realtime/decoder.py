from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from loguru import logger

from ..config import DatasetConfig, RealtimeConfig
from ..data_pipeline import PhonemeInventory


class RealtimeDecoder:
    """Maintains a rolling buffer of ECoG frames and emits smoothed phoneme predictions."""

    def __init__(
        self,
        model_path: Path | str,
        dataset_cfg: DatasetConfig,
        realtime_cfg: RealtimeConfig,
        inventory: PhonemeInventory,
    ) -> None:
        self.dataset_cfg = dataset_cfg
        self.realtime_cfg = realtime_cfg
        self.inventory = inventory
        self.model = tf.keras.models.load_model(model_path)
        self.buffer = deque(maxlen=dataset_cfg.window_length)
        self._prob_history = deque(maxlen=realtime_cfg.smoothing_window)
        self._frames_since_emit = 1e9
        logger.info("Realtime decoder loaded from %s", model_path)

    def reset(self) -> None:
        self.buffer.clear()
        self._prob_history.clear()
        self._frames_since_emit = 1e9

    def push_sample(self, feature_vector: np.ndarray) -> None:
        if feature_vector.shape[-1] != self.dataset_cfg.num_features:
            raise ValueError(
                f"Expected feature vector with {self.dataset_cfg.num_features} channels, "
                f"got {feature_vector.shape[-1]}"
            )
        self.buffer.append(feature_vector.astype(np.float32))
        self._frames_since_emit += 1

    def ready(self) -> bool:
        return len(self.buffer) == self.dataset_cfg.window_length

    def _predict_window(self) -> np.ndarray:
        window = np.stack(self.buffer)[None, ...]
        prediction = self.model.predict(window, verbose=0)
        # Use most recent timestep probability vector.
        return prediction[0, -1, :]

    def infer(self) -> Optional[dict]:
        if not self.ready():
            return None
        probs = self._predict_window()
        self._prob_history.append(probs)
        smoothed = np.mean(self._prob_history, axis=0)
        top_idx = int(np.argmax(smoothed))
        top_prob = float(smoothed[top_idx])
        if top_prob < self.realtime_cfg.emit_threshold:
            return None
        if self._frames_since_emit * self.realtime_cfg.frame_ms < self.realtime_cfg.debounce_ms:
            return None
        self._frames_since_emit = 0
        phoneme = self.inventory.decode(top_idx)
        return {
            "phoneme": phoneme,
            "confidence": top_prob,
            "probabilities": smoothed,
        }
