# Edge AI for Fruit Quality Classification

This project trains a MobileNetV2 transfer-learning classifier for fruit-quality images. It is an edge-oriented research/portfolio project: the model architecture is compact, but no Raspberry Pi, mobile, or other device deployment is included.

## Architecture

```text
image (224x224x3)
  -> training-only augmentation (rotation, horizontal flip, zoom)
  -> MobileNetV2 ImageNet backbone (frozen)
  -> global average pooling
  -> dropout
  -> softmax class head
```

The model contains the input rescaling layer, so the same saved `.keras` model is used for evaluation and prediction. Training uses an Adam optimizer, categorical cross-entropy, early stopping, learning-rate reduction, and best-checkpoint saving.

## Dataset structure

Provide a directory with one subdirectory per class. The training code only reads image files; it never downloads, extracts, deletes, renames, or otherwise modifies the dataset.

```text
data/
├── fresh/
│   ├── image-001.jpg
│   └── image-002.jpg
└── rotten/
    ├── image-003.jpg
    └── image-004.jpg
```

The [Fruit Quality Classification dataset](https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification) is one compatible source. Download it separately and pass its extracted class-directory root to the scripts.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Training

```bash
python train.py \
  --data-dir /path/to/data \
  --output-dir artifacts/fruit-quality \
  --epochs 10 \
  --batch-size 32 \
  --validation-split 0.2 \
  --seed 42
```

The output directory contains `best_model.keras`, `final_model.keras`, `class_names.json`, and `training_summary.json`. The dataset path is intentionally external to the repository and is ignored by Git.

## Evaluation and prediction

Evaluation recreates the same deterministic validation split:

```bash
python evaluate.py \
  --model artifacts/fruit-quality/best_model.keras \
  --data-dir /path/to/data \
  --output artifacts/fruit-quality/evaluation.json
```

Predict one image using the saved class ordering:

```bash
python predict.py \
  --model artifacts/fruit-quality/best_model.keras \
  --class-names artifacts/fruit-quality/class_names.json \
  --image /path/to/image.jpg
```

Both commands print JSON. This repository does not publish accuracy, latency, or model-size claims because those depend on the dataset, trained checkpoint, and hardware used.

## Project structure

```text
├── train.py                 # reproducible training entry point
├── evaluate.py              # validation evaluation entry point
├── predict.py               # single-image inference entry point
├── fruit_quality_utils.py   # dataset and argument validation
├── tests/test_utils.py      # dataset-independent tests
├── requirements.txt
└── .github/workflows/ci.yml
```

## Reproducibility and CI

Training accepts explicit dataset, output, epoch, batch-size, validation-split, and seed arguments. Python, NumPy, and TensorFlow seeds are set, and TensorFlow deterministic operations are enabled where supported. Results are still dependent on TensorFlow version, hardware, and pretrained-weight availability.

GitHub Actions installs `requirements.txt` on Python 3.11 and runs the non-GPU, non-dataset-dependent test suite on every pull request and push to `main`.

## Edge deployment roadmap

The saved Keras model is a portable artifact for future conversion and benchmarking. Actual TensorFlow Lite conversion, quantization, device packaging, latency measurement, and memory profiling are not implemented here and should be added only with measurements from the target hardware.
