
# 🧠 CogniScan – Real-Time Attention State Classifier

> A machine learning-based system for detecting **focused vs distracted** mental states using facial expression data and raw pixel features.

---

![CogniScan Demo](assets/demo.gif)

## 📌 Project Overview

**CogniScan** is a multi-model machine learning system that classifies facial expressions into *focused* or *distracted* states using raw image data. It evaluates the performance of:
- 🎯 **SVM (Support Vector Machine)**
- ⚡ **XGBoost Classifier**
- 🧠 **Convolutional Neural Network (CNN)**

The goal is to compare classical machine learning with deep learning to detect real-time attention patterns—laying the foundation for an intelligent productivity assistant.

---

## 🚀 Features

- Preprocessing pipeline for image normalization and resizing
- Training & evaluation of 3 separate models
- Clean model comparison & metrics output
- Ready-to-use notebook for inference demos
- A modular codebase for future AU/gaze integration (e.g., OpenFace)

---

## 🛠️ Tech Stack

- **Languages**: Python
- **Libraries**:
  - `scikit-learn`
  - `xgboost`
  - `tensorflow/keras`
  - `opencv-python`
  - `matplotlib`, `seaborn`, `numpy`, `pandas`
- **Tools**: Jupyter, GitHub, VS Code

---

## 📁 Project Structure

```
CogniScan/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/                # Raw and preprocessed images (not uploaded)
├── models/              # Saved SVM, XGBoost, and CNN models
├── notebooks/           # Jupyter notebooks for training & demo
├── scripts/             # Python scripts for training & preprocessing
├── utils/               # Helper functions
└── assets/              # Demo GIFs, model performance plots, etc.
```

---

## 📊 Results

| Model      | Accuracy | F1-Score | Notes                          |
|------------|----------|----------|--------------------------------|
| SVM        | -        | -        | Fast to train, small size      |
| XGBoost    | -        | -        | Strong generalization          |
| CNN        | -        | -        | Best performance, deeper model |

> *Note: Results may vary slightly based on dataset splits and hyperparameters.*

---

## 🧪 Getting Started

### 1. Clone the repo:
```bash
git clone https://github.com/your-username/CogniScan.git
cd CogniScan
```

### 2. Create virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install dependencies:
```bash
pip install -r requirements.txt
```

### 4. Run a notebook or training script:
```bash
jupyter notebook notebooks/compare_models.ipynb
```

---

## 🖼️ Demo

![Model Comparison Chart](assets/model_accuracy_plot.png)

> Example predictions made by each model using unseen facial images.

---

## 💡 Future Additions

- [ ] Integrate **OpenFace** AU features
- [ ] Add **real-time webcam inference**
- [ ] Build a **Streamlit dashboard**
- [ ] Add **distraction heatmap overlays**

---

## 👤 Author

**Warren Dmello**  
💼 [LinkedIn](https://www.linkedin.com/in/warrenlukedmello)  
📧 warrenlukedmello@gmail.com  
🧠 Focused on AI for productivity and behavioral analysis.

---
