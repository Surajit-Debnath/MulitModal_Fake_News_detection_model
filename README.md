# 📰 Multimodal Fake News Detection using Deep Learning

A deep learning-based **multimodal fake news detection system** that combines **textual** and **visual** information to classify news articles as **Real** or **Fake**. The project leverages **SentenceTransformer** for text representation and a pretrained **ResNet152** for image feature extraction, followed by a feature fusion network for binary classification.

---

## 📖 Overview

Fake news often contains both misleading text and manipulated or unrelated images. Traditional text-only models ignore visual information, while image-only models miss textual context.

This project addresses the problem by combining both modalities into a single deep learning model for improved fake news detection.

---

## ✨ Features

- 📄 Text embedding using **SentenceTransformer (all-mpnet-base-v2)**
- 🖼️ Image feature extraction using pretrained **ResNet152**
- 🔗 Multimodal feature fusion
- ⚡ PyTorch Lightning training pipeline
- 📊 Comprehensive evaluation using multiple performance metrics
- 🧹 Dataset cleaning and preprocessing
- 💾 Offline embedding generation for faster training

---

# 🏗️ Model Architecture

```
                 News Text
                     │
                     ▼
      SentenceTransformer Encoder
                     │
              768-D Embedding
                     │
              Fully Connected
                     │
               300 Features
                     │
                     ├──────────────┐
                     │              │
                     ▼              ▼
               Feature Concatenation
                     │
                Fusion Network
                     │
             Fully Connected Layers
                     │
             Binary Classification
                     ▲
                     │
               300 Features
                     │
             Fully Connected
                     │
          Pretrained ResNet152
                     │
                  News Image
```

---

# 📂 Project Structure

```text
multimodal-fake-news-detection/
│
├── data/
│   ├── processed/              # Processed CSV files
│   ├── embeddings/             # Precomputed text embeddings
│   └── ...
│
├── images/                     # News images
│
├── lightning_logs/             # Training logs and model checkpoints
│
├── notebooks/
│   ├── Data_Preparation.ipynb
│   ├── Data_cleaning.ipynb
│   ├── Data_recleaning.ipynb
│   ├── Generate_Embeddings.ipynb
│   ├── Multimodal_model.ipynb
│   └── newmodel.ipynb
│
├── src/
│   ├── dataloader.py
│   ├── dataset.py
│   ├── model.py
│   ├── transforms.py
│   └── distilled_requirements.txt
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

# 🛠️ Technologies Used

- Python
- PyTorch
- PyTorch Lightning
- Torchvision
- SentenceTransformers
- ResNet152
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Pillow

---

# 📊 Dataset

The dataset contains:

- News text/headlines
- Associated news images
- Binary labels (Real / Fake)

During preprocessing:

- Invalid and corrupted images were removed.
- Text data was cleaned.
- Sentence embeddings were generated offline.
- Dataset was split into training, validation, and test sets.

---

# ⚙️ Model Pipeline

### Text Branch

- SentenceTransformer (`all-mpnet-base-v2`)
- 768-dimensional sentence embeddings
- Fully connected projection layer

### Image Branch

- Pretrained ResNet152
- Frozen convolutional backbone
- Trainable projection layer

### Fusion

- Concatenation of text and image features
- Fully connected fusion network
- Binary classifier

---

# 🚀 Training Configuration

| Parameter        |             Value |
| ---------------- | ----------------: |
| Batch Size       |                32 |
| Learning Rate    |              1e-4 |
| Optimizer        |              Adam |
| Loss Function    |  CrossEntropyLoss |
| Framework        | PyTorch Lightning |
| Early Stopping   |                ✅ |
| Model Checkpoint |                ✅ |

---

# 📈 Performance

| Metric                     |      Score |
| -------------------------- | ---------: |
| **Test Accuracy**          | **81.30%** |
| **Precision**              |   **0.81** |
| **Recall**                 |   **0.81** |
| **F1 Score**               |   **0.81** |
| **ROC AUC**                |  **0.890** |
| **Average Precision (AP)** |  **0.899** |

---

# 📊 Evaluation Metrics

The model was evaluated using:

- ✅ Accuracy
- ✅ Precision
- ✅ Recall
- ✅ F1 Score
- ✅ Confusion Matrix
- ✅ ROC Curve
- ✅ Precision–Recall Curve
- ✅ Area Under ROC Curve (AUC)
- ✅ Average Precision (AP)

---

# 📷 Results

Include screenshots of:

- Confusion Matrix
- ROC Curve
- Precision–Recall Curve

Example:

```
results/
├── confusion_matrix.png
├── roc_curve.png
└── precision_recall_curve.png
```

---

# ⚡ Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/multimodal-fake-news-detection.git
```

Move into the project directory

```bash
cd multimodal-fake-news-detection
```

Install the required packages

```bash
pip install -r requirements.txt
```

---

# ▶️ Usage

1. Prepare the dataset.
2. Generate text embeddings.
3. Train the model.
4. Evaluate the model using the provided notebooks.

---

# 🔮 Future Improvements

- Fine-tune the SentenceTransformer encoder
- Train the ResNet backbone
- Experiment with Vision Transformers (ViT)
- Attention-based multimodal fusion
- Explainable AI using Grad-CAM and SHAP
- Deploy as a Streamlit web application

---

# 👨‍💻 Author

**Surajit Debnath**

B.Tech in Computer Science & Engineering  
Jalpaiguri Government Engineering College

GitHub: https://github.com/Surajit-Debnath

LinkedIn: https://linkedin.com/in/YOUR_PROFILE

---

# 📄 License

This project is licensed under the **MIT License**.

---

## ⭐ Support

If you found this project helpful, consider giving it a **⭐ Star** on GitHub.
