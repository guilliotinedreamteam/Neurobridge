
"""
Training Pipeline for NeuroBridge.

This module orchestrates the model creation, data generation, and training loop.
It saves the trained model artifact for later inference.
"""

import os
import tensorflow as tf
from src.model import build_neurobridge_decoder
from src.data import data_generator
from src.config import NUM_TIMESTEPS, NUM_FEATURES, NUM_PHONEMES, BATCH_SIZE, EPOCHS, TOTAL_MOCK_SAMPLES

def train_model() -> tf.keras.callbacks.History:
    """
    Executes the training process.

    Returns:
        The Keras History object containing training metrics.
    """
    print("\nInstantiating NeuroBridge Decoder Model...")
    model = build_neurobridge_decoder(NUM_TIMESTEPS, NUM_FEATURES, NUM_PHONEMES)

    print("Compiling Model...")
    model.compile(
        optimizer='adam',
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
