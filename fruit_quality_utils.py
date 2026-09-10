"""Pure-Python helpers shared by the command-line entry points."""

from __future__ import annotations

from pathlib import Path

IMAGE_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".webp"})


def validate_dataset_dir(data_dir: Path) -> Path:
    """Validate a directory-based image dataset without changing it."""
    data_dir = data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise ValueError(f"Dataset directory does not exist: {data_dir}")

    class_dirs = sorted(path for path in data_dir.iterdir() if path.is_dir())
    if len(class_dirs) < 2:
        raise ValueError("Dataset must contain at least two class directories")
    for class_dir in class_dirs:
        if not any(
            path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            for path in class_dir.rglob("*")
        ):
            raise ValueError(f"Class directory contains no supported images: {class_dir.name}")
    return data_dir


def validate_positive_int(value: int, name: str) -> int:
    if value < 1:
        raise ValueError(f"{name} must be greater than zero")
    return value


def validate_seed(value: int) -> int:
    if value < 0:
        raise ValueError("seed must be zero or greater")
    return value


def validate_fraction(value: float, name: str) -> float:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return value
