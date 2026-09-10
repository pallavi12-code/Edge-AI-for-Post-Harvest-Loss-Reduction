"""Predict the class and confidence for one image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from fruit_quality_utils import IMAGE_EXTENSIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--class-names", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.is_file() or args.image.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported or missing image: {args.image}")
    class_names = json.loads(args.class_names.read_text(encoding="utf-8"))
    image = tf.keras.utils.load_img(args.image, target_size=(224, 224))
    scores = tf.keras.models.load_model(args.model).predict(
        np.expand_dims(tf.keras.utils.img_to_array(image), axis=0), verbose=0
    )[0]
    index = int(np.argmax(scores))
    print(json.dumps({"class": class_names[index], "confidence": float(scores[index])}, indent=2))


if __name__ == "__main__":
    main()
