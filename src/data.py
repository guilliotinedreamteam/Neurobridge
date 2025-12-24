
"""
Data management and physiological simulation for NeuroBridge.

This module handles:
1. Physiological simulation of ECoG signals (1/f Pink Noise).
2. Structured handling of Patient Health Metadata.
3. Interface for loading real neurophysiological datasets (.mat).
"""

import numpy as np
import tensorflow as tf
from dataclasses import dataclass, field
from typing import Tuple, Generator, Optional, List, Dict
from scipy import signal
import scipy.io
from src.config import NUM_TIMESTEPS, NUM_FEATURES, NUM_PHONEMES

@dataclass
class PatientMetadata:
    """
    Encapsulates patient health and hardware information relevant to BCI decoding.

    Attributes:
        subject_id: Unique identifier (e.g., 'S1', 'ECoG_P4').
        sampling_rate: Data acquisition rate in Hz.
        grid_geometry: Description of electrode array (e.g., '16x8 High-Density').
        hemisphere: 'Left' or 'Right' hemisphere coverage.
        pathology: Clinical indication (e.g., 'Epilepsy', 'ALS').
        bad_channels: List of excluded channel indices due to noise/artifact.
    """
    subject_id: str
    sampling_rate: int
    grid_geometry: str
    hemisphere: str
    pathology: str = "N/A"
    bad_channels: List[int] = field(default_factory=list)

def generate_pink_noise(num_samples: int, timesteps: int, features: int) -> np.ndarray:
    """
    Generates synthetic ECoG data with 1/f spectral characteristics (Pink Noise).

    Neural field potentials typically exhibit a power spectrum that scales as 1/f,
    unlike white noise (random uniform) which has a flat spectrum. This creates
    a scientifically rigorous proxy for real data.

    Args:
        num_samples: Number of trials.
        timesteps: Temporal duration.
        features: Number of channels.

    Returns:
        np.ndarray: Synthetic ECoG data shape (num_samples, timesteps, features).
    """
    # Generate white noise
    white = np.random.randn(num_samples, timesteps, features)

    # Transform to frequency domain
    white_fft = np.fft.fft(white, axis=1)

    # Create 1/f scaling vector (avoid division by zero at DC component)
    freqs = np.fft.fftfreq(timesteps)
    scaling = 1.0 / (np.abs(freqs) + 1e-5)
    scaling = np.sqrt(scaling) # Amplitude scales as sqrt(Power)

    # Reshape scaling to broadcast over samples and features
    scaling = scaling.reshape(1, timesteps, 1)

    # Apply scaling and transform back to time domain
    pink_fft = white_fft * scaling
    pink = np.fft.ifft(pink_fft, axis=1).real

    # Normalize to typical High-Gamma envelope range (roughly 0-1 or z-scored)
    # Here we z-score per channel then scale to simulate processed High-Gamma
    mean = np.mean(pink, axis=1, keepdims=True)
    std = np.std(pink, axis=1, keepdims=True)
    pink_normalized = (pink - mean) / (std + 1e-8)

    return pink_normalized.astype(np.float32)

class NeuralDataLoader:
    """
    Unified interface for loading Real or Synthetic Neural Data.
    """
    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path
        self.metadata = PatientMetadata(
            subject_id="Synthetic_01",
            sampling_rate=1000,
            grid_geometry="8x16 Mock",
            hemisphere="Left"
        )

    def load_data(self,
                  num_samples: int = 100,
                  timesteps: int = NUM_TIMESTEPS,
                  features: int = NUM_FEATURES) -> np.ndarray:
        """
        Loads data from file if path exists, otherwise generates physiological noise.
        """
        if self.data_path and tf.io.gfile.exists(self.data_path):
            try:
                # Stub for loading standard MATLAB files used in BCI (e.g. Naplib format)
                # Structure usually: data['ecog'], data['labels']
                mat = scipy.io.loadmat(self.data_path)
                if 'ecog' in mat:
                    # Assume simple preprocessing/resizing needed here in real scenario
                    # For now, return what's there or handle shape mismatch
                    print(f"Loaded real ECoG data from {self.data_path}")
                    return mat['ecog']
            except Exception as e:
                print(f"Failed to load real data: {e}. Falling back to synthesis.")

        # Fallback to rigorous synthesis
        print(f"Generating {num_samples} samples of physiological 1/f (Pink) noise.")
        return generate_pink_noise(num_samples, timesteps, features)

    def load_labels(self,
                    num_samples: int = 100,
                    timesteps: int = NUM_TIMESTEPS,
                    num_classes: int = NUM_PHONEMES) -> np.ndarray:
        """
        Loads or generates phoneme labels.
        """
        # For prototype, we stick to random labels as we don't have a real aligner
        # In a real scenario, this would load 'phoneme_identity' from the .mat file
        mock_sparse = np.random.randint(0, num_classes, size=(num_samples, timesteps))
        return tf.keras.utils.to_categorical(mock_sparse, num_classes=num_classes)

def data_generator(
    batch_size: int,
    timesteps: int = NUM_TIMESTEPS,
    features: int = NUM_FEATURES,
    num_classes: int = NUM_PHONEMES
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Yields batches of ECoG data and phoneme labels using the NeuralDataLoader.
    """
    loader = NeuralDataLoader()

    while True:
        # Generate batch-sized chunks
        ecog_batch = loader.load_data(batch_size, timesteps, features)
        phoneme_batch = loader.load_labels(batch_size, timesteps, num_classes)
        yield ecog_batch, phoneme_batch
