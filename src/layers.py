
import tensorflow as tf
from tensorflow.keras import layers
from typing import Optional

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=2, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)
        self.dropout = layers.Dropout(dropout)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        output = self.dropout(output)
        return output

class ConvolutionModule(layers.Layer):
    def __init__(self, embed_dim, kernel_size=31, dropout=0.1):
        super(ConvolutionModule, self).__init__()
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.pointwise_conv1 = layers.Conv1D(
            filters=2 * embed_dim, kernel_size=1, strides=1, padding="valid", activation=None
        )
        self.depthwise_conv = layers.DepthwiseConv1D(
            kernel_size=kernel_size, strides=1, padding="same", activation=None
        )
        self.batch_norm = layers.BatchNormalization()
        self.swish = layers.Activation("swish")
        self.pointwise_conv2 = layers.Conv1D(
            filters=embed_dim, kernel_size=1, strides=1, padding="valid", activation=None
        )
        self.dropout = layers.Dropout(dropout)

    def call(self, inputs):
        x = self.layer_norm(inputs)
        x = self.pointwise_conv1(x)
        # GLU activation: split into two halves, apply sigmoid to one, and multiply
        x, gate = tf.split(x, 2, axis=-1)
        x = x * tf.nn.sigmoid(gate)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        return x

class FeedForwardModule(layers.Layer):
    def __init__(self, embed_dim, expansion_factor=4, dropout=0.1):
        super(FeedForwardModule, self).__init__()
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.dense1 = layers.Dense(embed_dim * expansion_factor, activation="swish")
        self.dropout1 = layers.Dropout(dropout)
        self.dense2 = layers.Dense(embed_dim)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, inputs):
        x = self.layer_norm(inputs)
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        return x

class ConformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads=4, kernel_size=31, dropout=0.1):
        super(ConformerBlock, self).__init__()
        self.ff1 = FeedForwardModule(embed_dim, dropout=dropout)
        self.mhsa = MultiHeadSelfAttention(embed_dim, num_heads=num_heads, dropout=dropout)
        self.conv = ConvolutionModule(embed_dim, kernel_size=kernel_size, dropout=dropout)
        self.ff2 = FeedForwardModule(embed_dim, dropout=dropout)
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        # FFN -> MHSA -> Conv -> FFN
        x = inputs + 0.5 * self.ff1(inputs)
        x = x + self.mhsa(x)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        x = self.layer_norm(x)
        return x
