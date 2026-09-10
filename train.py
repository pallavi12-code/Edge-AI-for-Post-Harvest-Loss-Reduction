"""Train a reproducible fruit-quality classifier with MobileNetV2.

Dataset layout:
    data/
      class_a/
      class_b/

Example:
    python train.py --data-dir data --output-dir artifacts
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from fruit_quality_utils import (
    validate_dataset_dir,
    validate_fraction,
    validate_positive_int,
    validate_seed,
)


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        # TensorFlow may already be initialized by a caller.
        pass


def build_datasets(
    data_dir: Path,
    image_size: tuple[int, int],
    batch_size: int,
    validation_split: float,
    seed: int,
):
    data_dir = validate_dataset_dir(data_dir)
    train = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True,
    )
    validation = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    )
    return (
        train.prefetch(tf.data.AUTOTUNE),
        validation.prefetch(tf.data.AUTOTUNE),
        train.class_names,
    )


def build_model(num_classes: int, image_size: tuple[int, int]) -> Model:
    inputs = tf.keras.Input(shape=(*image_size, 3))
    x = layers.RandomRotation(0.05)(inputs)
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)
    backbone = MobileNetV2(include_top=False, weights="imagenet", input_shape=(*image_size, 3))
    backbone.trainable = False
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inputs, outputs, name="fruit_quality_mobilenetv2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_positive_int(args.epochs, "epochs")
    validate_positive_int(args.batch_size, "batch-size")
    validate_seed(args.seed)
    validate_fraction(args.validation_split, "validation-split")
    set_seed(args.seed)
    image_size = (224, 224)
    train_ds, val_ds, class_names = build_datasets(
        args.data_dir,
        image_size,
        args.batch_size,
        args.validation_split,
        args.seed,
    )

    model = build_model(len(class_names), image_size)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(args.output_dir / "best_model.keras", monitor="val_accuracy", save_best_only=True),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    metrics = model.evaluate(val_ds, return_dict=True, verbose=0)

    model.save(args.output_dir / "final_model.keras")
    with open(args.output_dir / "class_names.json", "w", encoding="utf-8") as handle:
        json.dump(class_names, handle, indent=2)
    with open(args.output_dir / "training_summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "classes": class_names,
                "metrics": {key: float(value) for key, value in metrics.items()},
                "history": history.history,
                "config": {
                    "image_size": image_size,
                    "batch_size": args.batch_size,
                    "epochs": args.epochs,
                    "seed": args.seed,
                    "validation_split": args.validation_split,
                },
            },
            handle,
            indent=2,
        )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
