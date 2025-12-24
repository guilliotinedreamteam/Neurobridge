
"""
Configuration for the NeuroBridge Speech Neuroprosthesis System.

This module contains constant definitions used across training,
model building, and inference.
"""

from typing import Final

# Based on typical BCI phoneme decoding tasks
NUM_TIMESTEPS: Final[int] = 100    # Sequence length (e.g., 100ms of data)
NUM_FEATURES: Final[int] = 128     # Number of ECoG electrode channels
NUM_PHONEMES: Final[int] = 41      # e.g., 40 phonemes + 1 silence token

# Training parameters
BATCH_SIZE: Final[int] = 32
EPOCHS: Final[int] = 5
TOTAL_MOCK_SAMPLES: Final[int] = 1000

# Conformer Hyperparameters
D_MODEL: Final[int] = 144  # Embedding dimension (must be divisible by num_heads)
NUM_HEADS: Final[int] = 4
KERNEL_SIZE: Final[int] = 15
NUM_LAYERS: Final[int] = 2

# Regularization & Optimization
DROPOUT_RATE: Final[float] = 0.1
WARMUP_STEPS: Final[int] = 4000
CLIP_NORM: Final[float] = 1.0  # Gradient clipping to prevent explosion
PATIENCE: Final[int] = 10      # Early stopping patience
