
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed, BatchNormalization, Bidirectional
from src.config import NUM_TIMESTEPS, NUM_FEATURES, NUM_PHONEMES

def build_neurobridge_decoder(timesteps=NUM_TIMESTEPS, features=NUM_FEATURES, num_classes=NUM_PHONEMES):
    """
    Builds the core RNN for ECoG-to-Phoneme decoding.
    This architecture is inspired by decoders used in modern speech
    neuroprosthesis research.
    """
    model = Sequential()

    # Input layer specifies the shape of the incoming data
    model.add(Input(shape=(timesteps, features)))

    # Using Bidirectional LSTMs to capture context from both past and future
    # neural signals in the sequence, which is common in offline analysis.
    # For real-time, a standard LSTM or GRU would be used.
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

def build_realtime_decoder(timesteps=1, features=NUM_FEATURES, num_classes=NUM_PHONEMES):
    """
    Builds a unidirectional RNN suitable for real-time ECoG-to-Phoneme decoding.
    Designed to process one timestep at a time (timesteps=1).
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
    # logic to each timestep in the sequence. Since timesteps=1, it's applied once.
    model.add(TimeDistributed(Dense(128, activation='relu')))

    # The final output layer maps to the probability of each phoneme
    # at each timestep (which is just one timestep here).
    model.add(TimeDistributed(Dense(num_classes, activation='softmax')))

    return model
