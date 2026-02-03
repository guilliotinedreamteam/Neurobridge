"""NeuroBridge Python package providing ECoG processing, modeling, and inference helpers."""

from .config import (
    DatasetConfig,
    ModelConfig,
    NeuroBridgeConfig,
    RealtimeConfig,
    SpeechConfig,
    TrainingConfig,
)
from .data_pipeline import ECoGDatasetBuilder, PhonemeInventory
from .models import build_offline_decoder, build_realtime_decoder, initialize_realtime_from_offline
from .realtime import RealtimeDecoder
from .speech import PhonemeSynthesizer
from .training import train_and_evaluate

__all__ = [
    "DatasetConfig",
    "ModelConfig",
    "TrainingConfig",
    "RealtimeConfig",
    "SpeechConfig",
    "NeuroBridgeConfig",
    "PhonemeInventory",
    "ECoGDatasetBuilder",
    "build_offline_decoder",
    "build_realtime_decoder",
    "initialize_realtime_from_offline",
    "train_and_evaluate",
    "RealtimeDecoder",
    "PhonemeSynthesizer",
]
