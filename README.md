# Edge-AI-for-Post-Harvest-Loss-Reduction
# 🍎 Edge AI for Fruit Quality Classification

> Transfer learning with MobileNetV2 to detect fruit quality (fresh vs. rotten) for post-harvest loss reduction in agriculture.

---

## 📌 Problem Statement

Post-harvest food losses cost billions annually in developing economies. Manual quality inspection is slow, inconsistent, and unscalable. This project builds a lightweight Edge AI model capable of running on low-resource devices (Raspberry Pi, mobile) to classify fruit quality in real time — enabling faster, automated quality control at the farm or warehouse level.

---

## 🧠 Model Architecture

| Component | Detail |
|---|---|
| Base Model | MobileNetV2 (pretrained on ImageNet) |
| Custom Head | GlobalAveragePooling2D → Dense (Softmax) |
| Input Size | 224 × 224 × 3 |
| Optimizer | Adam |
| Loss | Categorical Crossentropy |
| Epochs | 10 |
| Batch Size | 32 |

**Why MobileNetV2?** It is specifically designed for edge deployment — low latency, small footprint, and high accuracy on vision tasks, making it ideal for Raspberry Pi or similar constrained hardware.

---

## 📊 Results

| Metric | Value |
|---|---|
| Validation Accuracy | ~94% |
| Framework | TensorFlow / Keras |
| Model Format | `.keras` |

*Training/validation accuracy and loss curves generated during training (see Outputs section).*

---

## 🗂️ Dataset

- **Source:** [Fruit Quality Classification – Kaggle](https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification)
- Multi-class image dataset of fresh and rotten fruits
- 80/20 train-validation split with augmentation

---

## ⚙️ Tech Stack

`Python` `TensorFlow` `Keras` `MobileNetV2` `NumPy` `Matplotlib` `Seaborn` `Scikit-learn` `Google Colab`

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/edge-ai-fruit-quality.git
cd edge-ai-fruit-quality
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download dataset
Set up your Kaggle API key (`kaggle.json`) and run:
```bash
kaggle datasets download ryandpark/fruit-quality-classification -p ./data
```

### 4. Run training
```bash
python aicte.py
```

### 5. Predict on a new image
The script prompts you to upload an image and outputs the predicted class with confidence score.

---

## 📁 Project Structure

```
edge-ai-fruit-quality/
├── aicte.py               # Full training + evaluation + inference pipeline
├── requirements.txt       # Python dependencies
└── README.md
```

---

## 🔍 Key Features

- ✅ Transfer learning with frozen MobileNetV2 backbone
- ✅ Real-time single-image inference with confidence score
- ✅ Confusion matrix + classification report for evaluation
- ✅ Accuracy and loss curves plotted over epochs
- ✅ Model saved in `.keras` format for reuse/deployment
- ✅ Lightweight architecture suitable for edge hardware (Raspberry Pi)

---

## 👩‍💻 Author

** (Marikanti Pallavi Reddy)**  
B.E. AI & ML — Chaitanya Bharathi Institute of Technology, Hyderabad  
[LinkedIn](www.linkedin.com/in/pallavi-reddy-4865703a9) · [GitHub](https://github.com/pallavi12-code)
