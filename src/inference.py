
import os
import numpy as np
import tensorflow as tf
from src.model import build_realtime_decoder
from src.config import NUM_FEATURES, NUM_PHONEMES

def predict_realtime_phoneme(ecog_frame, model):
    """
    Simulates real-time prediction of a single phoneme from one ECoG frame.

    Args:
        ecog_frame (np.ndarray): A single ECoG frame of shape (NUM_FEATURES,).
        model (tf.keras.Model): The real-time Keras model.

    Returns:
        np.ndarray: Phoneme probabilities for the current frame, shape (NUM_PHONEMES,).
    """
    # Model expects input of shape (batch_size, timesteps, features)
    # For a single frame, this becomes (1, 1, NUM_FEATURES)
    input_shape_for_model = ecog_frame.reshape(1, 1, -1)

    # Make prediction
    prediction = model.predict(input_shape_for_model, verbose=0)

    # The output will be (1, 1, NUM_PHONEMES). We want (NUM_PHONEMES,)
    return prediction[0, 0, :]

def load_realtime_model(weights_path=None):
    """
    Loads or builds the realtime model.
    If weights_path is provided, it tries to load weights.
    In this conceptual phase, it builds the model and returns it.
    """
    print("\nInstantiating Real-time Decoder Model...")
    realtime_model = build_realtime_decoder(timesteps=1)

    # Explicitly build the realtime_model layers with a dummy input shape
    realtime_model.build(input_shape=(None, 1, NUM_FEATURES))

    realtime_model.compile(optimizer='adam', loss='categorical_crossentropy')

    if weights_path and os.path.exists(weights_path):
         # Loading weights from a bidirectional model to a unidirectional model is non-trivial
         # and requires the 'conceptual transfer' logic from the notebook.
         # For simplicity in this extracted file, we will just return the initialized model
         # or we can implement the transfer logic if needed.
         pass

    return realtime_model

if __name__ == "__main__":
    model = load_realtime_model()

    # Generate a few mock ECoG frames
    mock_realtime_frames = np.random.rand(5, NUM_FEATURES).astype(np.float32)

    print("\nDemonstrating real-time phoneme prediction with mock frames...")
    for i, frame in enumerate(mock_realtime_frames):
        probs = predict_realtime_phoneme(frame, model)
        most_probable_id = np.argmax(probs)
        print(f"Frame {i+1}: Most probable phoneme ID: {most_probable_id} (Prob: {probs[most_probable_id]:.4f})")
