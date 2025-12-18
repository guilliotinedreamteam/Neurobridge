
import numpy as np
import tensorflow as tf
from src.config import NUM_TIMESTEPS, NUM_FEATURES, NUM_PHONEMES

def load_mock_ecog_data(num_samples, timesteps=NUM_TIMESTEPS, features=NUM_FEATURES):
    """
    Simulates loading ECoG data.
    Returns a NumPy array of shape (num_samples, timesteps, features).
    """
    # In a real scenario, you would load data from ECoG_DATA_PATH here.
    mock_data = np.random.rand(num_samples, timesteps, features).astype(np.float32)
    return mock_data

def load_mock_phoneme_labels(num_samples, timesteps=NUM_TIMESTEPS, num_classes=NUM_PHONEMES):
    """
    Simulates loading phoneme labels and converts them to one-hot encoding.
    Returns a NumPy array of shape (num_samples, timesteps, num_classes).
    """
    # In a real scenario, you would load data from PHONEME_TRANSCRIPT_PATH here.

    # Generate sparse phoneme IDs (e.g., 0-indexed phoneme numbers for each timestep)
    mock_phoneme_labels_sparse = np.random.randint(0, num_classes, size=(num_samples, timesteps))

    # Convert sparse phoneme IDs to one-hot encoded format
    mock_phoneme_labels_onehot = tf.keras.utils.to_categorical(
        mock_phoneme_labels_sparse, num_classes=num_classes
    )
    return mock_phoneme_labels_onehot

def data_generator(batch_size, timesteps=NUM_TIMESTEPS, features=NUM_FEATURES, num_classes=NUM_PHONEMES):
    """
    Yields batches of mock ECoG data and one-hot encoded phoneme labels.
    """
    while True:
        # In a real scenario, this would load data from disk iteratively.
        ecog_batch = load_mock_ecog_data(batch_size, timesteps, features)
        phoneme_batch = load_mock_phoneme_labels(batch_size, timesteps, num_classes)
        yield ecog_batch, phoneme_batch
