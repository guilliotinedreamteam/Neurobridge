
"""
Data loading and generation utilities for NeuroBridge.

This module handles the simulation of ECoG data and phoneme labels.
In a production environment, this would interface with actual .mat or .npy files.
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, Generator
from src.config import NUM_TIMESTEPS, NUM_FEATURES, NUM_PHONEMES

def load_mock_ecog_data(
    num_samples: int,
    timesteps: int = NUM_TIMESTEPS,
    features: int = NUM_FEATURES
) -> np.ndarray:
    """
    Simulates loading ECoG (Electrocorticography) data.

    Args:
        num_samples: Number of data samples to generate.
        timesteps: Temporal duration of each sample (frames).
        features: Number of electrode channels.

    Returns:
        A NumPy array of shape (num_samples, timesteps, features) containing
        simulated neural data (float32).
    """
    # In a real scenario, you would load data from ECoG_DATA_PATH here.
    # We use random data to simulate normalized neural activity.
    mock_data = np.random.rand(num_samples, timesteps, features).astype(np.float32)
    return mock_data

def load_mock_phoneme_labels(
    num_samples: int,
    timesteps: int = NUM_TIMESTEPS,
    num_classes: int = NUM_PHONEMES
) -> np.ndarray:
    """
    Simulates loading phoneme labels and converts them to one-hot encoding.

    Args:
        num_samples: Number of label sequences to generate.
        timesteps: Temporal duration of alignment.
        num_classes: Total number of distinct phonemes (including silence).

    Returns:
        A NumPy array of shape (num_samples, timesteps, num_classes) containing
        one-hot encoded labels.
    """
    # In a real scenario, you would load data from PHONEME_TRANSCRIPT_PATH here.

    # Generate sparse phoneme IDs (e.g., 0-indexed phoneme numbers for each timestep)
    mock_phoneme_labels_sparse = np.random.randint(0, num_classes, size=(num_samples, timesteps))

    # Convert sparse phoneme IDs to one-hot encoded format
    mock_phoneme_labels_onehot = tf.keras.utils.to_categorical(
        mock_phoneme_labels_sparse, num_classes=num_classes
    )
    return mock_phoneme_labels_onehot

def data_generator(
    batch_size: int,
    timesteps: int = NUM_TIMESTEPS,
    features: int = NUM_FEATURES,
    num_classes: int = NUM_PHONEMES
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Yields batches of mock ECoG data and one-hot encoded phoneme labels.

    Args:
        batch_size: Size of each batch.
        timesteps: Sequence length.
        features: Number of input channels.
        num_classes: Number of output classes.

    Yields:
        A tuple (ecog_batch, phoneme_batch).
    """
    while True:
        # In a real scenario, this would load data from disk iteratively.
        ecog_batch = load_mock_ecog_data(batch_size, timesteps, features)
        phoneme_batch = load_mock_phoneme_labels(batch_size, timesteps, num_classes)
        yield ecog_batch, phoneme_batch
