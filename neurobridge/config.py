from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml


def _default_phonemes() -> List[str]:
    """Return a conservative ARPAbet-inspired inventory including silence."""
    return [
        "sil",
        "aa",
        "ae",
        "ah",
        "ao",
        "aw",
        "ay",
        "b",
        "ch",
        "d",
        "dh",
        "eh",
        "er",
        "ey",
        "f",
        "g",
        "hh",
        "ih",
        "iy",
        "jh",
        "k",
        "l",
        "m",
        "n",
        "ng",
        "ow",
        "oy",
        "p",
        "r",
        "s",
        "sh",
        "t",
        "th",
        "uh",
        "uw",
        "v",
        "w",
        "y",
        "z",
    ]


@dataclass
class DatasetConfig:
    """Describes how raw ECoG signals and phoneme labels are stored on disk."""

    ecog_dir: Path
    labels_dir: Path
    file_extension: str = ".mat"
    label_extension: str = ".csv"
    mat_key: Optional[str] = None
    sampling_rate_hz: int = 1000
    target_rate_hz: int = 200
    window_duration_ms: int = 400
    stride_ms: int = 40
    num_features: int = 128
    phonemes: List[str] = field(default_factory=_default_phonemes)
    label_columns: Dict[str, str] = field(
        default_factory=lambda: {"start": "start_sec", "end": "end_sec", "label": "phoneme"}
    )
    alignment_tolerance_ms: int = 20
    max_files: Optional[int] = None
    min_signal_duration_s: float = 2.0
    random_seed: int = 13

    def __post_init__(self) -> None:
        self.ecog_dir = Path(self.ecog_dir)
        self.labels_dir = Path(self.labels_dir)
        if not self.phonemes:
            self.phonemes = _default_phonemes()
        if "sil" not in self.phonemes:
            self.phonemes.insert(0, "sil")

    @property
    def num_classes(self) -> int:
        return len(self.phonemes)

    @property
    def window_length(self) -> int:
        """Number of samples per training window at the down-sampled rate."""
        return int(self.target_rate_hz * (self.window_duration_ms / 1000.0))

    @property
    def stride_length(self) -> int:
        return max(1, int(self.target_rate_hz * (self.stride_ms / 1000.0)))


@dataclass
class ModelConfig:
    """Defines architectural choices for the offline and real-time decoders."""

    architecture: str = "rnn"  # "rnn" or "transformer"
    rnn_units: Sequence[int] = (256, 256)
    realtime_units: Sequence[int] = (192, 128)
    conv_filters: Sequence[int] = (64, 128)
    conv_kernel_size: int = 5
    dense_units: int = 256
    dropout_rate: float = 0.3
    layer_norm: bool = True
    # Transformer specific
    num_heads: int = 4
    transformer_layers: int = 2
    ff_dim: int = 512


@dataclass
class TrainingConfig:
    """Training hyper-parameters and artifact locations."""

    batch_size: int = 16
    max_epochs: int = 50
    learning_rate: float = 1e-3
    val_split: float = 0.1
    test_split: float = 0.1
    cache_dataset: bool = True
    steps_per_epoch: Optional[int] = None
    log_dir: Path = Path("runs/neurobridge")
    checkpoint_dir: Path = Path("artifacts/neurobridge")
    metrics_path: Path = Path("artifacts/neurobridge/metrics.json")

    def __post_init__(self) -> None:
        self.log_dir = Path(self.log_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.metrics_path = Path(self.metrics_path)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class RealtimeConfig:
    """Streaming inference parameters."""

    frame_ms: int = 40
    context_ms: int = 400
    smoothing_window: int = 5
    emit_threshold: float = 0.4
    debounce_ms: int = 150


@dataclass
class SpeechConfig:
    """Controls the procedural phoneme-to-audio synthesizer."""

    sample_rate: int = 16000
    phoneme_duration_ms: int = 80
    release_ms: int = 20
    base_amplitude: float = 0.15
    export_audio_dir: Path = Path("artifacts/neurobridge/audio")

    def __post_init__(self) -> None:
        self.export_audio_dir = Path(self.export_audio_dir)
        self.export_audio_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class NeuroBridgeConfig:
    """Top-level configuration container."""

    dataset: DatasetConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    realtime: RealtimeConfig = field(default_factory=RealtimeConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)

    @classmethod
    def from_dict(cls, data: Dict) -> "NeuroBridgeConfig":
        dataset = DatasetConfig(**data["dataset"])
        model = ModelConfig(**data.get("model", {}))
        training = TrainingConfig(**data.get("training", {}))
        realtime = RealtimeConfig(**data.get("realtime", {}))
        speech = SpeechConfig(**data.get("speech", {}))
        return cls(dataset=dataset, model=model, training=training, realtime=realtime, speech=speech)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "NeuroBridgeConfig":
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return cls.from_dict(data)

    def to_dict(self) -> Dict:
        return {
            "dataset": asdict(self.dataset),
            "model": asdict(self.model),
            "training": asdict(self.training),
            "realtime": asdict(self.realtime),
            "speech": asdict(self.speech),
        }
