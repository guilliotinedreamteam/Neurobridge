from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix

from .config import NeuroBridgeConfig
from .data_pipeline import ECoGDatasetBuilder, PhonemeInventory
from .models import (
    build_offline_decoder,
    build_realtime_decoder,
    initialize_realtime_from_offline,
)


def _build_datasets(config: NeuroBridgeConfig, inventory: PhonemeInventory):
    builders = {
        "train": ECoGDatasetBuilder(
            config.dataset,
            inventory,
            split="train",
            val_split=config.training.val_split,
            test_split=config.training.test_split,
        ),
        "val": ECoGDatasetBuilder(
            config.dataset,
            inventory,
            split="val",
            val_split=config.training.val_split,
            test_split=config.training.test_split,
            shuffle_files=False,
        ),
        "test": ECoGDatasetBuilder(
            config.dataset,
            inventory,
            split="test",
            val_split=config.training.val_split,
            test_split=config.training.test_split,
            shuffle_files=False,
        ),
    }
    datasets = {
        name: builder.build_tf_dataset(config.training.batch_size, cache=config.training.cache_dataset)
        for name, builder in builders.items()
    }
    return datasets


def _collect_predictions(model: tf.keras.Model, dataset: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    y_true_segments = []
    y_pred_segments = []
    for batch_ecog, batch_labels in dataset:
        preds = model.predict(batch_ecog, verbose=0)
        y_pred_segments.append(np.argmax(preds, axis=-1).reshape(-1))
        y_true_segments.append(np.argmax(batch_labels.numpy(), axis=-1).reshape(-1))
    if not y_true_segments:
        return np.array([]), np.array([])
    return np.concatenate(y_true_segments), np.concatenate(y_pred_segments)


def _save_metrics(path: Path, metrics: Dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    logger.info("Wrote training metrics to %s", path)


def train_and_evaluate(config: NeuroBridgeConfig) -> Dict:
    inventory = PhonemeInventory(config.dataset.phonemes)
    datasets = _build_datasets(config, inventory)

    model = build_offline_decoder(config.dataset, config.model)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.training.learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="frame_accuracy"),
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_accuracy"),
        ],
    )

    checkpoint_path = config.training.checkpoint_dir / "offline_decoder.keras"
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, verbose=1),
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir=config.training.log_dir),
    ]

    logger.info("Starting offline decoder training.")
    history = model.fit(
        datasets["train"],
        validation_data=datasets["val"],
        epochs=config.training.max_epochs,
        steps_per_epoch=config.training.steps_per_epoch,
        callbacks=callbacks,
    )

    logger.info("Evaluating on held-out test split.")
    test_results = model.evaluate(datasets["test"], return_dict=True)
    y_true, y_pred = _collect_predictions(model, datasets["test"])
    if y_true.size == 0:
        report = {}
        confusion = []
    else:
        report = classification_report(
            y_true,
            y_pred,
            target_names=inventory.id_to_symbol,
            zero_division=0,
            output_dict=True,
        )
        confusion = confusion_matrix(y_true, y_pred).tolist()

    metrics = {
        "training_history": history.history,
        "test_results": test_results,
        "classification_report": report,
        "confusion_matrix": confusion,
        "checkpoint_path": str(checkpoint_path),
    }
    _save_metrics(config.training.metrics_path, metrics)

    realtime_model = build_realtime_decoder(config.dataset, config.model)
    initialize_realtime_from_offline(model, realtime_model)
    realtime_checkpoint = config.training.checkpoint_dir / "realtime_decoder.keras"
    realtime_model.save(realtime_checkpoint)
    logger.info("Saved realtime decoder to %s", realtime_checkpoint)

    metrics["realtime_checkpoint_path"] = str(realtime_checkpoint)
    return metrics
