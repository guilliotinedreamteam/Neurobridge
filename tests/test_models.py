import pytest
import tensorflow as tf
from neurobridge.models import build_offline_decoder, build_realtime_decoder
from neurobridge.config import DatasetConfig, ModelConfig

@pytest.fixture
def mock_configs():
    data_cfg = DatasetConfig(
        window_duration_ms=50,
        target_rate_hz=200,
        num_features=64,
        phonemes=["a", "b", "c"],
        ecog_dir=".",
        labels_dir="."
    )
    # window_length will be (200 * 0.05) = 10
    model_cfg = ModelConfig(
        architecture="rnn",
        conv_filters=[16],
        rnn_units=[32],
        dense_units=16,
        dropout_rate=0.1
    )
    return data_cfg, model_cfg

def test_build_offline_decoder(mock_configs):
    data_cfg, model_cfg = mock_configs
    model = build_offline_decoder(data_cfg, model_cfg)
    assert isinstance(model, tf.keras.Model)
    assert model.input_shape == (None, 10, 64)
    # Output should be (None, window_length, num_classes)
    # num_classes = 3 symbols + 'sil' = 4
    assert model.output_shape == (None, 10, 4)

def test_build_realtime_decoder(mock_configs):
    data_cfg, model_cfg = mock_configs
    model = build_realtime_decoder(data_cfg, model_cfg)
    assert isinstance(model, tf.keras.Model)
    assert model.input_shape == (None, 10, 64)
    assert model.output_shape == (None, 10, 4)

def test_transformer_architecture(mock_configs):
    data_cfg, model_cfg = mock_configs
    model_cfg.architecture = "transformer"
    model_cfg.transformer_layers = 1
    model_cfg.num_heads = 2
    model_cfg.ff_dim = 32
    
    model = build_offline_decoder(data_cfg, model_cfg)
    assert isinstance(model, tf.keras.Model)
    assert model.output_shape == (None, 10, 4)
