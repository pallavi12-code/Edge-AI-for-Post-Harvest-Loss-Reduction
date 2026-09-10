"""Evaluate a saved classifier on the deterministic validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf

from fruit_quality_utils import validate_dataset_dir, validate_fraction, validate_positive_int, validate_seed
from train import build_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_dataset_dir(args.data_dir)
    validate_positive_int(args.batch_size, "batch-size")
    validate_seed(args.seed)
    validate_fraction(args.validation_split, "validation-split")
    _, validation, class_names = build_datasets(
        args.data_dir, (224, 224), args.batch_size, args.validation_split, args.seed
    )
    model = tf.keras.models.load_model(args.model)
    metrics = {key: float(value) for key, value in model.evaluate(validation, return_dict=True).items()}
    result = {"classes": class_names, "metrics": metrics}
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
