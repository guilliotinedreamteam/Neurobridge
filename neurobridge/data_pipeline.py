from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from loguru import logger
from scipy import signal as sp_signal
from scipy.io import loadmat

from .config import DatasetConfig


@dataclass
class PhonemeInventory:
    """Utility helper that keeps phoneme <-> id mappings."""

    symbols: Sequence[str]
    silence_symbol: str = "sil"

    def __post_init__(self) -> None:
        if self.silence_symbol not in self.symbols:
            self.symbols = (self.silence_symbol,) + tuple(self.symbols)
        self.id_to_symbol = list(dict.fromkeys(self.symbols))  # preserve order, drop duplicates
        self.symbol_to_id = {symbol: idx for idx, symbol in enumerate(self.id_to_symbol)}

    @property
    def num_classes(self) -> int:
        return len(self.id_to_symbol)

    def encode(self, label: str) -> int:
        label = (label or "").strip().lower()
        return self.symbol_to_id.get(label, self.symbol_to_id[self.silence_symbol])

    def decode(self, idx: int) -> str:
        return self.id_to_symbol[min(max(idx, 0), self.num_classes - 1)]


def _select_mat_key(content: dict, preferred_key: Optional[str]) -> np.ndarray:
    if preferred_key and preferred_key in content:
        return content[preferred_key]
    candidates = [k for k in content.keys() if not k.startswith("__")]
    if not candidates:
        raise ValueError("No valid data keys found inside .mat file.")
    return content[candidates[0]]


def _bandpass_filter(signal_array: np.ndarray, fs: int, low: float = 0.5, high: float = 200.0) -> np.ndarray:
    nyquist = 0.5 * fs
    low_norm = max(low / nyquist, 1e-4)
    high_norm = min(high / nyquist, 0.999)
    b, a = sp_signal.butter(N=4, Wn=[low_norm, high_norm], btype="band")
    return sp_signal.filtfilt(b, a, signal_array, axis=0)


def _apply_notch_filters(signal_array: np.ndarray, fs: int, freqs: Sequence[float] = (60.0, 120.0, 180.0)) -> np.ndarray:
    filtered = signal_array
    for freq in freqs:
        w0 = freq / (fs / 2)
        b, a = sp_signal.iirnotch(w0=w0, Q=30.0)
        filtered = sp_signal.filtfilt(b, a, filtered, axis=0)
    return filtered


def _downsample(signal_array: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
    if orig_rate == target_rate:
        return signal_array
    gcd = math.gcd(orig_rate, target_rate)
    up = target_rate // gcd
    down = orig_rate // gcd
    return sp_signal.resample_poly(signal_array, up=up, down=down, axis=0)


def _zscore(signal_array: np.ndarray) -> np.ndarray:
    mean = np.mean(signal_array, axis=0, keepdims=True)
    std = np.std(signal_array, axis=0, keepdims=True) + 1e-6
    return (signal_array - mean) / std


def segment_signal(signal_array: np.ndarray, window: int, stride: int) -> np.ndarray:
    windows: List[np.ndarray] = []
    for start in range(0, signal_array.shape[0] - window + 1, stride):
        windows.append(signal_array[start : start + window])
    if not windows:
        return np.empty((0, window, signal_array.shape[1]), dtype=np.float32)
    return np.stack(windows).astype(np.float32)


def segment_labels(label_ids: np.ndarray, num_classes: int, window: int, stride: int) -> np.ndarray:
    label_segments: List[np.ndarray] = []
    eye = np.eye(num_classes, dtype=np.float32)
    for start in range(0, label_ids.shape[0] - window + 1, stride):
        ids = label_ids[start : start + window]
        label_segments.append(eye[ids])
    if not label_segments:
        return np.empty((0, window, num_classes), dtype=np.float32)
    return np.stack(label_segments)


def load_ecog_matrix(path: Path, cfg: DatasetConfig) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        arr = np.load(path)
        keys = [k for k in arr.files if not k.startswith("__")]
        if not keys:
            raise ValueError(f"No arrays found inside {path}")
        return arr[keys[0]]
    if path.suffix == ".mat":
        mat = loadmat(path)
        return _select_mat_key(mat, cfg.mat_key)
    raise ValueError(f"Unsupported file extension for {path}")


def preprocess_ecog(raw: np.ndarray, cfg: DatasetConfig) -> np.ndarray:
    bandpassed = _bandpass_filter(raw, cfg.sampling_rate_hz)
    cleaned = _apply_notch_filters(bandpassed, cfg.sampling_rate_hz)
    downsampled = _downsample(cleaned, cfg.sampling_rate_hz, cfg.target_rate_hz)
    if downsampled.shape[1] > cfg.num_features:
        downsampled = downsampled[:, : cfg.num_features]
    elif downsampled.shape[1] < cfg.num_features:
        padding = np.zeros((downsampled.shape[0], cfg.num_features - downsampled.shape[1]), dtype=downsampled.dtype)
        downsampled = np.concatenate([downsampled, padding], axis=1)
    return _zscore(downsampled).astype(np.float32)


def load_phoneme_intervals(path: Path, cfg: DatasetConfig) -> List[Tuple[float, float, str]]:
    df = pd.read_csv(path)
    start_col = cfg.label_columns["start"]
    end_col = cfg.label_columns["end"]
    label_col = cfg.label_columns["label"]
    intervals: List[Tuple[float, float, str]] = []
    for row in df.itertuples(index=False):
        start = float(getattr(row, start_col))
        end = float(getattr(row, end_col))
        label = str(getattr(row, label_col))
        if end <= start:
            continue
        intervals.append((start, end, label))
    if not intervals:
        logger.warning("Label file %s produced zero intervals.", path)
    return intervals


def rasterize_intervals(
    intervals: List[Tuple[float, float, str]],
    inventory: PhonemeInventory,
    target_rate_hz: int,
    total_samples: int,
) -> np.ndarray:
    labels = np.full((total_samples,), inventory.encode(inventory.silence_symbol), dtype=np.int32)
    for start_sec, end_sec, label in intervals:
        start_idx = max(0, int(start_sec * target_rate_hz))
        end_idx = min(total_samples, int(math.ceil(end_sec * target_rate_hz)))
        if end_idx <= start_idx:
            continue
        labels[start_idx:end_idx] = inventory.encode(label)
    return labels


class ECoGDatasetBuilder:
    """Pairs ECoG recordings with label files and exposes a streaming generator."""

    def __init__(
        self,
        cfg: DatasetConfig,
        inventory: PhonemeInventory,
        split: str = "train",
        val_split: float = 0.1,
        test_split: float = 0.1,
        shuffle_files: bool = True,
    ) -> None:
        self.cfg = cfg
        self.inventory = inventory
        self.split = split
        self.val_split = val_split
        self.test_split = test_split
        self.shuffle_files = shuffle_files
        self.file_pairs = self._scan_pairs()
        self.active_pairs = self._select_split(self.file_pairs)
        logger.info("Initialized %s dataset with %d paired files.", split, len(self.active_pairs))

    def _scan_pairs(self) -> List[Tuple[Path, Path]]:
        ecog_files = sorted(self.cfg.ecog_dir.rglob(f"*{self.cfg.file_extension}"))
        label_lookup = {p.stem: p for p in self.cfg.labels_dir.rglob(f"*{self.cfg.label_extension}")}
        pairs: List[Tuple[Path, Path]] = []
        for ecog_file in ecog_files:
            label_file = label_lookup.get(ecog_file.stem)
            if not label_file:
                logger.warning("Skipping %s because no label file was found.", ecog_file.name)
                continue
            pairs.append((ecog_file, label_file))
            if self.cfg.max_files and len(pairs) >= self.cfg.max_files:
                break
        if not pairs:
            logger.error("No ECoG/label pairs discovered under %s and %s.", self.cfg.ecog_dir, self.cfg.labels_dir)
        if self.shuffle_files:
            random.Random(self.cfg.random_seed).shuffle(pairs)
        return pairs

    def _select_split(self, pairs: List[Tuple[Path, Path]]) -> List[Tuple[Path, Path]]:
        if not pairs:
            return []
        n = len(pairs)
        val_count = int(n * self.val_split)
        test_count = int(n * self.test_split)
        train_count = max(1, n - val_count - test_count)
        train_pairs = pairs[:train_count]
        val_pairs = pairs[train_count : train_count + val_count]
        test_pairs = pairs[train_count + val_count :]
        if self.split == "train":
            return train_pairs
        if self.split == "val":
            return val_pairs if val_pairs else train_pairs[-max(1, int(0.1 * train_count)) :]
        if self.split == "test":
            return test_pairs if test_pairs else train_pairs[-max(1, int(0.1 * train_count)) :]
        raise ValueError(f"Unknown split '{self.split}'")

    def _prepare_sample(self, ecog_path: Path, label_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        # Caching logic
        cache_dir = self.cfg.ecog_dir / ".cache"
        cache_dir.mkdir(exist_ok=True)
        # Simple cache key based on filename and size (could be more robust with hash)
        cache_key = f"{ecog_path.stem}_{ecog_path.stat().st_size}_{self.cfg.target_rate_hz}"
        cache_file = cache_dir / f"{cache_key}.npz"

        if cache_file.exists():
            try:
                data = np.load(cache_file)
                # logger.debug("Loaded %s from cache", ecog_path.name)
                return data["ecog"], data["labels"]
            except Exception as e:
                logger.warning("Failed to load cache for %s: %s", ecog_path.name, e)

        # Original processing logic
        raw = load_ecog_matrix(ecog_path, self.cfg)
        duration = raw.shape[0] / self.cfg.sampling_rate_hz
        if duration < self.cfg.min_signal_duration_s:
            logger.warning("Skipping %s because duration %.2fs is too short.", ecog_path.name, duration)
            empty_ecog = np.empty((0, self.cfg.window_length, self.cfg.num_features), dtype=np.float32)
            empty_labels = np.empty((0, self.cfg.window_length, self.cfg.num_classes), dtype=np.float32)
            return empty_ecog, empty_labels

        ecog = preprocess_ecog(raw, self.cfg)
        intervals = load_phoneme_intervals(label_path, self.cfg)
        label_ids = rasterize_intervals(intervals, self.inventory, self.cfg.target_rate_hz, ecog.shape[0])
        ecog_windows = segment_signal(ecog, self.cfg.window_length, self.cfg.stride_length)
        label_windows = segment_labels(label_ids, self.inventory.num_classes, self.cfg.window_length, self.cfg.stride_length)
        window_count = min(len(ecog_windows), len(label_windows))
        
        if window_count == 0:
            logger.warning("No overlapping windows for %s", ecog_path.name)
            final_ecog = np.empty((0, self.cfg.window_length, self.cfg.num_features), dtype=np.float32)
            final_labels = np.empty((0, self.cfg.window_length, self.cfg.num_classes), dtype=np.float32)
        else:
            final_ecog = ecog_windows[:window_count]
            final_labels = label_windows[:window_count]

        # Save to cache
        try:
            np.savez_compressed(cache_file, ecog=final_ecog, labels=final_labels)
        except Exception as e:
            logger.warning("Failed to write cache for %s: %s", ecog_path.name, e)

        return final_ecog, final_labels

    def generator(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        for ecog_path, label_path in self.active_pairs:
            ecog_windows, label_windows = self._prepare_sample(ecog_path, label_path)
            for window, labels in zip(ecog_windows, label_windows):
                yield window.astype(np.float32), labels.astype(np.float32)

    def build_tf_dataset(self, batch_size: int, cache: bool = True) -> tf.data.Dataset:
        output_signature = (
            tf.TensorSpec(shape=(self.cfg.window_length, self.cfg.num_features), dtype=tf.float32),
            tf.TensorSpec(shape=(self.cfg.window_length, self.cfg.num_classes), dtype=tf.float32),
        )
        ds = tf.data.Dataset.from_generator(self.generator, output_signature=output_signature)
        if self.shuffle_files and self.split == "train":
            ds = ds.shuffle(buffer_size=batch_size * 4, reshuffle_each_iteration=True)
        if cache:
            ds = ds.cache()
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
