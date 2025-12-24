
import unittest
import tensorflow as tf
import numpy as np
from src.layers import RelativeMultiHeadSelfAttention, ConvolutionModule, FeedForwardModule, ConformerBlock
from src.config import D_MODEL, NUM_HEADS, KERNEL_SIZE

class TestConformerLayers(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 10
        self.input_shape = (self.batch_size, self.seq_len, D_MODEL)
        self.dummy_input = tf.random.normal(self.input_shape)

    def test_multi_head_self_attention(self):
        """Test the Relative MHSA layer."""
        layer = RelativeMultiHeadSelfAttention(embed_dim=D_MODEL, num_heads=NUM_HEADS)
        output = layer(self.dummy_input)
        self.assertEqual(output.shape, self.input_shape)

    def test_convolution_module(self):
        """Test the Convolution module."""
        layer = ConvolutionModule(embed_dim=D_MODEL, kernel_size=KERNEL_SIZE)
        output = layer(self.dummy_input)
        self.assertEqual(output.shape, self.input_shape)

    def test_feed_forward_module(self):
        """Test the FFN module."""
        layer = FeedForwardModule(embed_dim=D_MODEL)
        output = layer(self.dummy_input)
        self.assertEqual(output.shape, self.input_shape)

    def test_conformer_block(self):
        """Test the full Conformer block composition."""
        layer = ConformerBlock(embed_dim=D_MODEL, num_heads=NUM_HEADS, kernel_size=KERNEL_SIZE)
        output = layer(self.dummy_input)
        self.assertEqual(output.shape, self.input_shape)

if __name__ == '__main__':
    unittest.main()
