
"""
Training Pipeline for NeuroBridge.

This module orchestrates the model creation, data generation, and training loop.
It saves the trained model artifact for later inference.
"""

import os
import tensorflow as tf
from src.model import build_neurobridge_decoder
from src.data import data_generator
from src.config import NUM_TIMESTEPS, NUM_FEATURES, NUM_PHONEMES, BATCH_SIZE, EPOCHS, TOTAL_MOCK_SAMPLES, D_MODEL

class WarmUpSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Applies a linear warmup followed by an inverse square root decay.
    This is critical for stabilizing Transformer/Conformer training.
    """
    def __init__(self, d_model: int, warmup_steps: int = 4000):
        super(WarmUpSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {
            "d_model": self.d_model,
            "warmup_steps": self.warmup_steps,
        }

def train_model() -> tf.keras.callbacks.History:
    """
    Executes the training process.

    Returns:
        The Keras History object containing training metrics.
    """
    print("\nInstantiating NeuroBridge Decoder Model...")
    model = build_neurobridge_decoder(NUM_TIMESTEPS, NUM_FEATURES, NUM_PHONEMES)

    # Initialize Custom Learning Rate Scheduler
    learning_rate = WarmUpSchedule(D_MODEL)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    print("Compiling Model with Warmup Scheduler...")
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    steps_per_epoch = TOTAL_MOCK_SAMPLES // BATCH_SIZE

    print(f"\nTraining Parameters: Epochs={EPOCHS}, Batch Size={BATCH_SIZE}, Steps per Epoch={steps_per_epoch}")

    print("\nStarting model training with mock data...")
    history = model.fit(
        data_generator(BATCH_SIZE, NUM_TIMESTEPS, NUM_FEATURES, NUM_PHONEMES),
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        verbose=1
    )
    print("Model training complete.")

    model_save_path = "./neurobridge_decoder_model.h5"
    print(f"Saving trained model to: {model_save_path}")
    model.save(model_save_path)
    print("Model saved successfully.")

    return history

if __name__ == "__main__":
    train_model()
