
"""
Deep Learning Model Definitions for NeuroBridge.

This module defines the Keras architectures used for decoding neural signals.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, TimeDistributed
from src.config import NUM_TIMESTEPS, NUM_FEATURES, NUM_PHONEMES, D_MODEL, NUM_HEADS, KERNEL_SIZE, NUM_LAYERS
from src.layers import ConformerBlock

def build_neurobridge_decoder(
    timesteps: int = NUM_TIMESTEPS,
    features: int = NUM_FEATURES,
    num_classes: int = NUM_PHONEMES
) -> tf.keras.Model:
    """
    Builds the core Conformer-based RNN for ECoG-to-Phoneme decoding.

    This architecture utilizes a stack of Conformer blocks to capture both
    local (convolutional) and global (attention-based) dependencies.

    Args:
        timesteps: The length of the input sequence.
        features: The number of input features (channels).
        num_classes: The number of output phoneme classes.

    Returns:
        A compiled tf.keras.Model.
    """
    model = Sequential()

    # Input layer specifies the shape of the incoming data
    model.add(Input(shape=(timesteps, features)))

    # Project to d_model dimension
    model.add(TimeDistributed(Dense(D_MODEL)))

    # Conformer Blocks
    for _ in range(NUM_LAYERS):
        model.add(ConformerBlock(
            embed_dim=D_MODEL,
            num_heads=NUM_HEADS,
            kernel_size=KERNEL_SIZE
        ))

    # A TimeDistributed Dense layer applies the same classification
    # logic to each timestep in the sequence.
    model.add(TimeDistributed(Dense(128, activation='relu')))

    # The final output layer maps to the probability of each phoneme
    # at each timestep.
    model.add(TimeDistributed(Dense(num_classes, activation='softmax')))

    return model

# For the real-time decoder in this prototype, we will reuse the main architecture
# because Conformer is generally non-causal due to self-attention.
# True low-latency Conformer would require masking, but for this "never stopping"
# innovation step, we stick to the windowed approach used in inference.py.
# Thus, build_realtime_decoder mirrors build_neurobridge_decoder for compatibility.
def build_realtime_decoder(
    timesteps: int = NUM_TIMESTEPS,
    features: int = NUM_FEATURES,
    num_classes: int = NUM_PHONEMES
) -> tf.keras.Model:
    return build_neurobridge_decoder(timesteps, features, num_classes)
