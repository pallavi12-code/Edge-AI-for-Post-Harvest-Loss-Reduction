# 🍎 Edge AI for Fruit Quality Classification

A lightweight computer-vision project using **MobileNetV2 transfer learning** to classify fruit images as fresh or rotten, with the goal of supporting faster post-harvest quality inspection.

## Problem

Manual fruit-quality inspection can be slow and inconsistent. This project explores a compact deep-learning model suitable for resource-constrained deployment scenarios such as farm or warehouse inspection.

## Model

| Component | Configuration |
|---|---|
| Backbone | MobileNetV2 pretrained on ImageNet |
| Input | 224 × 224 × 3 |
| Head | GlobalAveragePooling2D → Dense/Softmax |
| Optimizer | Adam |
| Loss | Categorical Crossentropy |
| Epochs | 10 |
| Batch size | 32 |

MobileNetV2 was selected because its lightweight architecture makes it a practical candidate for edge-oriented computer-vision applications.

## Results

| Metric | Result |
|---|---:|
| Validation accuracy | ~94% |

The reported result is from this project's validation setup and should not be treated as a guarantee of real-world performance.

## Dataset

Fruit Quality Classification dataset from Kaggle: https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification

The project uses an 80/20 train-validation split with image augmentation.

## Features

- Transfer learning with MobileNetV2
- Image augmentation
- Single-image inference with confidence score
- Confusion matrix and classification report
- Training/validation curves
- Saved Keras model for reuse

## Project structure

```text
Edge-AI-for-Post-Harvest-Loss-Reduction/
├── aicte.py
├── requirements.txt
└── README.md
```

## Run locally

```bash
git clone https://github.com/pallavi12-code/Edge-AI-for-Post-Harvest-Loss-Reduction.git
cd Edge-AI-for-Post-Harvest-Loss-Reduction
pip install -r requirements.txt
python aicte.py
```

## Tech stack

**Python · TensorFlow · Keras · MobileNetV2 · NumPy · Pandas · Scikit-learn · Matplotlib · Seaborn**

## Future improvements

- Benchmark inference latency and model size on real edge hardware
- Add a held-out test set and cross-dataset evaluation
- Quantize the model for constrained devices
- Add a lightweight web/mobile inference interface

## Author

**Pallavi Reddy**  
B.E. Artificial Intelligence & Machine Learning, CBIT Hyderabad
