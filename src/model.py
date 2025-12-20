
"""
Deep Learning Model Definitions for NeuroBridge.

This module defines the Keras architectures used for decoding neural signals.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed, BatchNormalization, Bidirectional
from src.config import NUM_TIMESTEPS, NUM_FEATURES, NUM_PHONEMES

def build_neurobridge_decoder(
    timesteps: int = NUM_TIMESTEPS,
    features: int = NUM_FEATURES,
    num_classes: int = NUM_PHONEMES
) -> tf.keras.Model:
    """
    Builds the core RNN for ECoG-to-Phoneme decoding.

    This architecture utilizes Bidirectional LSTMs to capture temporal dependencies
    in both forward and backward directions, which is standard for offline
    sequence-to-sequence decoding tasks in neuroprosthetics.

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

    # Using Bidirectional LSTMs to capture context from both past and future
    # neural signals in the sequence.
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(BatchNormalization())

    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(BatchNormalization())

    # A TimeDistributed Dense layer applies the same classification
    # logic to each timestep in the sequence.
    model.add(TimeDistributed(Dense(128, activation='relu')))

    # The final output layer maps to the probability of each phoneme
    # at each timestep.
    model.add(TimeDistributed(Dense(num_classes, activation='softmax')))

    return model

def build_realtime_decoder(
    timesteps: int = 1,
    features: int = NUM_FEATURES,
    num_classes: int = NUM_PHONEMES
) -> tf.keras.Model:
    """
    Builds a unidirectional RNN suitable for real-time ECoG-to-Phoneme decoding.

    Unlike the offline model, this uses standard LSTMs to avoid look-ahead
    latency, making it suitable for causal, real-time inference.

    Args:
        timesteps: The length of the input sequence (usually 1 or a small window).
        features: The number of input features.
        num_classes: The number of output classes.

    Returns:
        A compiled tf.keras.Model.
    """
    model = Sequential()

    # Input layer specifies the shape of the incoming data for a single timestep
    model.add(Input(shape=(timesteps, features)))

    # Using standard LSTMs (unidirectional) for real-time processing.
    model.add(LSTM(256, return_sequences=True))
    model.add(BatchNormalization())

    model.add(LSTM(256, return_sequences=True))
    model.add(BatchNormalization())

    # A TimeDistributed Dense layer applies the same classification
    # logic to each timestep in the sequence.
    model.add(TimeDistributed(Dense(128, activation='relu')))

    # The final output layer maps to the probability of each phoneme
    # at each timestep.
    model.add(TimeDistributed(Dense(num_classes, activation='softmax')))

    return model
