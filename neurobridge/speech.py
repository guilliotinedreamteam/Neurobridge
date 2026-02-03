from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from loguru import logger
from scipy.io import wavfile

from .config import SpeechConfig
from .data_pipeline import PhonemeInventory


FORMANT_FREQUENCIES = {
    "aa": 730,
    "ae": 660,
    "ah": 640,
    "ao": 570,
    "aw": 610,
    "ay": 660,
    "eh": 530,
    "er": 490,
    "ey": 530,
    "ih": 440,
    "iy": 270,
    "ow": 360,
    "oy": 400,
    "uh": 440,
    "uw": 300,
    "sil": 0,
}

FRICATIVES = {"s", "sh", "f", "th", "z", "v"}


class PhonemeSynthesizer:
    """Procedurally generates rough audio for debugging predicted phoneme sequences."""

    def __init__(self, inventory: PhonemeInventory, config: SpeechConfig) -> None:
        self.inventory = inventory
        self.config = config

    def _phoneme_duration_samples(self) -> int:
        return int(self.config.sample_rate * (self.config.phoneme_duration_ms / 1000.0))

    def _release_samples(self) -> int:
        return max(1, int(self.config.sample_rate * (self.config.release_ms / 1000.0)))

    def _sine_wave(self, frequency: float, samples: int) -> np.ndarray:
        if frequency <= 0:
            return np.zeros(samples, dtype=np.float32)
        t = np.linspace(0, samples / self.config.sample_rate, samples, endpoint=False)
        return np.sin(2.0 * np.pi * frequency * t).astype(np.float32)

    def _noise_burst(self, samples: int) -> np.ndarray:
        return np.random.randn(samples).astype(np.float32) * 0.05

    def _phoneme_to_wave(self, phoneme: str) -> np.ndarray:
        samples = self._phoneme_duration_samples()
        base_freq = FORMANT_FREQUENCIES.get(phoneme, 220.0)
        if phoneme in FRICATIVES:
            wave = self._noise_burst(samples)
        else:
            wave = self._sine_wave(base_freq, samples)
        # Apply simple attack / release envelope.
        attack = np.linspace(0, 1, samples // 4, dtype=np.float32)
        sustain = np.ones(samples - len(attack) - self._release_samples(), dtype=np.float32)
        release = np.linspace(1, 0, self._release_samples(), dtype=np.float32)
        envelope = np.concatenate([attack, sustain, release])
        envelope = envelope[:samples]
        wave = wave[:samples] * envelope * self.config.base_amplitude
        return wave

    def synthesize(self, phoneme_ids: Sequence[int]) -> np.ndarray:
        waves = []
        for phoneme_id in phoneme_ids:
            phoneme = self.inventory.decode(int(phoneme_id))
            waves.append(self._phoneme_to_wave(phoneme))
        if not waves:
            return np.array([], dtype=np.float32)
        audio = np.concatenate(waves)
        audio = np.clip(audio, -1.0, 1.0)
        return audio.astype(np.float32)

    def save_wav(self, audio: np.ndarray, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(destination, self.config.sample_rate, audio)
        logger.info("Wrote synthesized audio to %s", destination)
        return destination
