
import unittest
import numpy as np
import tensorflow as tf
from src.model import build_neurobridge_decoder, build_realtime_decoder
from src.config import NUM_TIMESTEPS, NUM_FEATURES, NUM_PHONEMES

class TestNeuroBridgeModel(unittest.TestCase):
    def test_build_neurobridge_decoder(self):
        """Test the offline bidirectional decoder architecture."""
        model = build_neurobridge_decoder(NUM_TIMESTEPS, NUM_FEATURES, NUM_PHONEMES)
        self.assertIsInstance(model, tf.keras.Model)

        # Check input shape (excluding batch size)
        self.assertEqual(model.input_shape[1:], (NUM_TIMESTEPS, NUM_FEATURES))

        # Check output shape (excluding batch size)
        self.assertEqual(model.output_shape[1:], (NUM_TIMESTEPS, NUM_PHONEMES))

    def test_build_realtime_decoder(self):
        """Test the real-time decoder architecture."""
        # For this Conformer prototype, the real-time decoder mirrors the offline architecture
        # and relies on sliding window inference.
        timesteps = NUM_TIMESTEPS
        model = build_realtime_decoder(timesteps, NUM_FEATURES, NUM_PHONEMES)
        self.assertIsInstance(model, tf.keras.Model)

        self.assertEqual(model.input_shape[1:], (timesteps, NUM_FEATURES))
        self.assertEqual(model.output_shape[1:], (timesteps, NUM_PHONEMES))

if __name__ == '__main__':
    unittest.main()
