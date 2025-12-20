
import os
import numpy as np
import tensorflow as tf
from collections import deque
from typing import Optional, Deque
from src.model import build_neurobridge_decoder
from src.config import NUM_TIMESTEPS, NUM_FEATURES, NUM_PHONEMES

class RealTimeDecoder:
    """
    A class to handle real-time ECoG-to-Phoneme decoding using a sliding window.
    """
    def __init__(self, model: tf.keras.Model, window_size: int = NUM_TIMESTEPS):
        """
        Initialize the decoder.

        Args:
            model: The trained Keras model.
            window_size: The number of timesteps the model expects.
        """
        self.model = model
        self.window_size = window_size
        # Initialize buffer with zeros
        self.buffer: Deque[np.ndarray] = deque(
            [np.zeros(NUM_FEATURES, dtype=np.float32) for _ in range(window_size)],
            maxlen=window_size
        )

    def process_frame(self, ecog_frame: np.ndarray) -> np.ndarray:
        """
        Process a single new ECoG frame and return the prediction for the current timestep.

        Args:
            ecog_frame: A 1D array of shape (NUM_FEATURES,).

        Returns:
            A 1D array of shape (NUM_PHONEMES,) representing probabilities.
        """
        # Update sliding window
        self.buffer.append(ecog_frame)

        # Prepare input batch of shape (1, window_size, features)
        input_data = np.array(self.buffer)[np.newaxis, ...]

        # Predict
        # Model output is (1, window_size, num_phonemes) if return_sequences=True was used in training
        # We generally want the prediction for the *last* timestep in the window
        prediction = self.model.predict(input_data, verbose=0)

        # Return prediction for the most recent timestep
        return prediction[0, -1, :]

def load_realtime_model(weights_path: Optional[str] = None) -> tf.keras.Model:
    """
    Loads the model architecture consistent with training.

    In a real scenario, we would use the exact same architecture as training
    (Bidirectional LSTM) but since Bidirectional LSTMs look into the 'future',
    true low-latency real-time systems often use Unidirectional LSTMs or
    accept the latency of the window size.

    For this scientific prototype, we will use the `build_neurobridge_decoder`
    (Bidirectional) assuming a latency of `NUM_TIMESTEPS` (100 samples) is acceptable,
    or that we are processing chunk-by-chunk.
    """
    print(f"\nInstantiating Decoder Model (Window Size: {NUM_TIMESTEPS})...")
    # We use the main model architecture to ensure compatibility with trained weights
    model = build_neurobridge_decoder(timesteps=NUM_TIMESTEPS, features=NUM_FEATURES, num_classes=NUM_PHONEMES)

    model.compile(optimizer='adam', loss='categorical_crossentropy')

    if weights_path and os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}...")
        model.load_weights(weights_path)
    else:
        print("No weights found. Using random initialization (Conceptual Mode).")

    return model

if __name__ == "__main__":
    # Simulate a stream of data
    model = load_realtime_model("./neurobridge_decoder_model.h5")
    decoder = RealTimeDecoder(model)

    print(f"\nDemonstrating real-time phoneme prediction (Sliding Window: {NUM_TIMESTEPS})...")

    # Simulate 5 new frames arriving
    for i in range(5):
        # Generate a random frame
        mock_frame = np.random.rand(NUM_FEATURES).astype(np.float32)

        probs = decoder.process_frame(mock_frame)
        most_probable_id = np.argmax(probs)

        print(f"Time {i+1}: Most probable phoneme ID: {most_probable_id} (Prob: {probs[most_probable_id]:.4f})")
