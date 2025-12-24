
import tensorflow as tf
from tensorflow.keras import layers
from typing import Optional
from src.config import DROPOUT_RATE

class RelativeMultiHeadSelfAttention(layers.Layer):
    """
    Multi-Head Self-Attention with Relative Positional Encodings.
    Based on Transformer-XL style relative attention.
    """
    def __init__(self, embed_dim, num_heads=2, dropout=DROPOUT_RATE):
        super(RelativeMultiHeadSelfAttention, self).__init__()
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
        self.pos_dense = layers.Dense(embed_dim) # Project positional embeddings

        self.combine_heads = layers.Dense(embed_dim)
        self.dropout = layers.Dropout(dropout)

        # Bias u and v for relative attention (learned parameters)
        self.u = self.add_weight(shape=(1, self.num_heads, 1, self.projection_dim),
                                 initializer="glorot_uniform", trainable=True, name="u")
        self.v = self.add_weight(shape=(1, self.num_heads, 1, self.projection_dim),
                                 initializer="glorot_uniform", trainable=True, name="v")

    def _generate_relative_positions_matrix(self, length):
        """Generates relative positions matrix."""
        range_vec = tf.range(length)
        range_mat = tf.expand_dims(range_vec, 0) - tf.expand_dims(range_vec, 1)
        # Shift to be positive indices for embedding lookup if using embedding layer,
        # but here we generate sinusoidal on the fly or project.
        # For simplicity in this advanced prototype, we use fixed sinusoidal encoding generation.
        return range_mat

    def _get_relative_embeddings(self, length):
        """Generates sinusoidal relative embeddings."""
        # Standard sinusoidal encoding
        pos = tf.range(length, dtype=tf.float32)
        dim = tf.cast(self.embed_dim, tf.float32)
        # 10000^(2i/d_model)
        indices = tf.range(self.embed_dim // 2, dtype=tf.float32)
        exponent = 2 * indices / dim
        inv_freq = 1.0 / (10000 ** exponent)

        sinusoid_inp = tf.einsum('i,j->ij', pos, inv_freq)
        pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], axis=-1)

        return pos_emb # (length, embed_dim)

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def _relative_shift(self, x):
        """Performs relative shift to align attention scores."""
        # Input shape: (batch, heads, seq_len, seq_len)
        # Implementation of Transformer-XL relative shift
        shape = tf.shape(x)
        batch, heads, seq_len, _ = shape[0], shape[1], shape[2], shape[3]

        # Pad columns
        x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [1, 0]])
        x = tf.reshape(x, [batch, heads, seq_len + 1, seq_len])
        x = tf.slice(x, [0, 0, 1, 0], [-1, -1, -1, -1])
        x = tf.reshape(x, [batch, heads, seq_len, seq_len])
        return x

    def call(self, inputs):
        # inputs shape: (batch, seq_len, embed_dim)
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        query = self.query_dense(inputs) # (B, L, D)
        key = self.key_dense(inputs)     # (B, L, D)
        value = self.value_dense(inputs) # (B, L, D)

        # Relative Position Generation
        # Generate embeddings for length: 0 to seq_len-1
        # In full RelAttn, we often need 2*seq_len-1 context, but for Conformer (SOTA),
        # standard relative positional encoding often considers simple distance.
        # We will use the standard Transformer-XL definition: R matrix.
        # We need relative embeddings for distances up to seq_len.
        # In a self-attention block, positions range from -(L-1) to (L-1).
        # We simplify to sinusoidal embeddings of the sequence itself.
        pos_emb = self._get_relative_embeddings(seq_len) # (L, D)
        pos_emb = self.pos_dense(pos_emb) # Project R to R_hat (L, D)

        # Separate Heads
        query = self.separate_heads(query, batch_size) # (B, H, L, d_k)
        key = self.separate_heads(key, batch_size)     # (B, H, L, d_k)
        value = self.separate_heads(value, batch_size) # (B, H, L, d_k)

        # Reshape Position Embeddings for heads
        # pos_emb is (L, D) -> (L, H, d_k) -> (1, H, L, d_k) - effectively broadcast later
        pos_emb = tf.reshape(pos_emb, (seq_len, self.num_heads, self.projection_dim))
        pos_emb = tf.transpose(pos_emb, perm=[1, 0, 2]) # (H, L, d_k)
        pos_emb = tf.expand_dims(pos_emb, 0) # (1, H, L, d_k)

        # Content-Content Score: Q * K^T
        # (Q + u) * K^T
        query_with_u = query + self.u
        content_score = tf.matmul(query_with_u, key, transpose_b=True) # (B, H, L, L)

        # Content-Position Score: Q * R^T
        # (Q + v) * R^T
        query_with_v = query + self.v
        # Matmul (B, H, L, d_k) with (1, H, L, d_k)^T -> (B, H, L, L)
        pos_score = tf.matmul(query_with_v, pos_emb, transpose_b=True)
        # Apply relative shift
        pos_score = self._relative_shift(pos_score)

        # Total Attention Score
        score = content_score + pos_score

        dim_key = tf.cast(self.projection_dim, tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)

        weights = tf.nn.softmax(scaled_score, axis=-1)
        weights = self.dropout(weights)

        output = tf.matmul(weights, value) # (B, H, L, d_k)

        # Merge Heads
        output = tf.transpose(output, perm=[0, 2, 1, 3]) # (B, L, H, d_k)
        concat_output = tf.reshape(output, (batch_size, -1, self.embed_dim))

        output = self.combine_heads(concat_output)
        output = self.dropout(output)
        return output

class ConvolutionModule(layers.Layer):
    def __init__(self, embed_dim, kernel_size=31, dropout=DROPOUT_RATE):
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
    def __init__(self, embed_dim, expansion_factor=4, dropout=DROPOUT_RATE):
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
    def __init__(self, embed_dim, num_heads=4, kernel_size=31, dropout=DROPOUT_RATE):
        super(ConformerBlock, self).__init__()
        self.ff1 = FeedForwardModule(embed_dim, dropout=dropout)
        # Replaced standard MHSA with Relative MHSA
        self.mhsa = RelativeMultiHeadSelfAttention(embed_dim, num_heads=num_heads, dropout=dropout)
        self.conv = ConvolutionModule(embed_dim, kernel_size=kernel_size, dropout=dropout)
        self.ff2 = FeedForwardModule(embed_dim, dropout=dropout)
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        # Macaron Architecture: FFN -> MHSA -> Conv -> FFN
        x = inputs + 0.5 * self.ff1(inputs)
        x = x + self.mhsa(x)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        x = self.layer_norm(x)
        return x
