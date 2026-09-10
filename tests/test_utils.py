from pathlib import Path

import pytest

from fruit_quality_utils import (
    validate_dataset_dir,
    validate_fraction,
    validate_positive_int,
    validate_seed,
)


def test_validate_dataset_dir_accepts_two_nonempty_classes(tmp_path: Path) -> None:
    for class_name in ("fresh", "rotten"):
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        (class_dir / "sample.jpg").write_bytes(b"not decoded by validation")

    assert validate_dataset_dir(tmp_path) == tmp_path.resolve()


def test_validate_dataset_dir_rejects_missing_class_images(tmp_path: Path) -> None:
    (tmp_path / "fresh").mkdir()
    (tmp_path / "rotten").mkdir()
    (tmp_path / "fresh" / "sample.txt").write_text("x")

    with pytest.raises(ValueError, match="no supported images"):
        validate_dataset_dir(tmp_path)


@pytest.mark.parametrize("value", [0, -1])
def test_validate_positive_int_rejects_non_positive_values(value: int) -> None:
    with pytest.raises(ValueError):
        validate_positive_int(value, "epochs")


def test_validate_seed_accepts_zero() -> None:
    assert validate_seed(0) == 0


def test_validate_seed_rejects_negative_values() -> None:
    with pytest.raises(ValueError):
        validate_seed(-1)


@pytest.mark.parametrize("value", [0.0, 1.0, -0.1, 1.1])
def test_validate_fraction_rejects_out_of_range_values(value: float) -> None:
    with pytest.raises(ValueError):
        validate_fraction(value, "validation-split")
