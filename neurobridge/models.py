from __future__ import annotations

from typing import List

import tensorflow as tf
from loguru import logger

from .config import DatasetConfig, ModelConfig


def _conv_frontend(inputs: tf.Tensor, cfg: ModelConfig) -> tf.Tensor:
    x = inputs
    for filters in cfg.conv_filters:
        x = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=cfg.conv_kernel_size,
            padding="same",
            activation="relu",
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(cfg.dropout_rate)(x)
    return x


def _rnn_stack(inputs: tf.Tensor, cfg: ModelConfig, bidirectional: bool = True) -> tf.Tensor:
    x = inputs
    stack = cfg.rnn_units if bidirectional else cfg.realtime_units
    for units in stack:
        rnn_layer = tf.keras.layers.LSTM(
            units,
            return_sequences=True,
            activation="tanh",
            recurrent_activation="sigmoid",
        )
        if bidirectional:
            rnn_layer = tf.keras.layers.Bidirectional(rnn_layer)
        x = rnn_layer(x)
        if cfg.layer_norm:
            x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dropout(cfg.dropout_rate)(x)
    return x


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(ff_dim, activation="relu"),
                tf.keras.layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def _transformer_stack(inputs: tf.Tensor, cfg: ModelConfig) -> tf.Tensor:
    x = inputs
    # Project to d_model if necessary, or just rely on conv output
    # Assuming conv output filters match desired d_model or close enough
    # If explicit projection needed:
    # x = tf.keras.layers.Dense(cfg.d_model)(x)
    
    for _ in range(cfg.transformer_layers):
        # We use the last conv filter size as the embedding dimension
        embed_dim = inputs.shape[-1]
        x = TransformerBlock(embed_dim, cfg.num_heads, cfg.ff_dim, cfg.dropout_rate)(x)
    return x


def build_offline_decoder(data_cfg: DatasetConfig, model_cfg: ModelConfig) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(data_cfg.window_length, data_cfg.num_features), name="ecog_window")
    x = _conv_frontend(inputs, model_cfg)
    
    if model_cfg.architecture == "transformer":
        x = _transformer_stack(x, model_cfg)
    else:
        x = _rnn_stack(x, model_cfg, bidirectional=True)
        
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(model_cfg.dense_units, activation="relu"),
        name="time_dense",
    )(x)
    outputs = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(data_cfg.num_classes, activation="softmax"), name="phoneme"
    )(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="neurobridge_offline_decoder")
    logger.info(
        "Offline decoder (%s) instantiated with %d trainable parameters.", 
        model_cfg.architecture, model.count_params()
    )
    return model


def build_realtime_decoder(data_cfg: DatasetConfig, model_cfg: ModelConfig) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(data_cfg.window_length, data_cfg.num_features), name="realtime_window")
    x = _conv_frontend(inputs, model_cfg)
    x = _rnn_stack(x, model_cfg, bidirectional=False)
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(model_cfg.dense_units, activation="relu"),
        name="realtime_time_dense",
    )(x)
    outputs = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(data_cfg.num_classes, activation="softmax"), name="realtime_phoneme"
    )(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="neurobridge_realtime_decoder")
    logger.info(
        "Real-time decoder instantiated with %d trainable parameters.", model.count_params()
    )
    return model


def initialize_realtime_from_offline(offline: tf.keras.Model, realtime: tf.keras.Model) -> None:
    """Copy overlapping weights from the offline model into the realtime variant."""

    def _filtered_layers(model: tf.keras.Model) -> List[tf.keras.layers.Layer]:
        return [layer for layer in model.layers if layer.weights]

    offline_layers = _filtered_layers(offline)
    realtime_layers = _filtered_layers(realtime)
    copied = 0
    for src_layer, dst_layer in zip(offline_layers, realtime_layers):
        src_weights = src_layer.get_weights()
        dst_weights = dst_layer.get_weights()
        if len(src_weights) != len(dst_weights):
            continue
        try:
            dst_layer.set_weights(src_weights)
            copied += 1
        except ValueError:
            continue
    logger.info("Transferred %d layer weights from offline -> realtime decoder.", copied)
