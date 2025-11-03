# 🔬 AgriVision Plant Disease Analyzer

This project is an end-to-end deep learning pipeline built to identify **10 different classes of tomato plant disease** from leaf images. It includes a rigorous comparative analysis of four advanced deep learning models — **ResNet50**, **AlexNet**, **EfficientNet-B4**, and **ViT** — and culminates in a 4-model interactive web application that provides real-time predictions and model interpretability via **Grad-CAM** visualizations.

---

## 🚀 Final Application

The final product is an **interactive Streamlit web application**. Upon uploading a leaf image, the app automatically provides a prediction and a heatmap visualization from our **top-performing model (ResNet50)**. The user can then compare this result against the other three trained models in real time.

🎬[AgriVision App Demo](https://drive.google.com/drive/folders/1G4s7_7IATuH9qAsQK7oR0Wt5s-y8qtOw?usp=sharing)


### 🌿 Features

* ⚙️ **Automatic Analysis:** Instant prediction from the top-performing model (ResNet50) on image upload.
* 🧩 **4-Model Comparison:** Instantly compare the primary model's results against AlexNet, EfficientNet-B4, and ViT.
* 🔬 **Visual Interpretability:** A Grad-CAM (Gradient-weighted Class Activation Mapping) visualization is generated for *every* model, showing *why* the model made its prediction by highlighting focus regions.
* 🎨 **Custom Theming:** A custom **Material Dark theme** (`.streamlit/config.toml`) for a polished, professional UI.

---

## 🏆 Final Model Analysis

A core goal of this project was to identify the best model architecture for tomato leaf disease classification. All four models were trained and evaluated on the same standardized dataset.

The results were definitive: **ResNet50 was identified as the optimal architecture.**

### 📊 Final Results & Visuals

All high-resolution output artifacts (learning curves, confusion matrices, classification heatmaps, etc.) for all experiments — including undeployed ones — are archived and available here:

➡️ **[View All Project Results Here](https://drive.google.com/drive/folders/1NxC2321my0TYRSzgegLzIKzo3vjpYKa_?usp=sharing)**

### Final Test Set Accuracy (Deployed Models)

| Rank  | Model Architecture           | Model File                 | Final Test Accuracy |
| ----- | ---------------------------- | -------------------------- | ------------------- |
| 🥇 #1 | **ResNet50 (Primary Model)** | `resnet50_best_p15.pth`    | **91.25%**          |
| 🥈 #2 | AlexNet (Secondary Model)    | `alexnet_best_p30.pth`     | 90.85%              |
| 🥉 #3 | EfficientNet-B4              | `efficient_b4_best.pt`     | 87.90%              |
| 🎯 #4 | Vision Transformer (ViT)     | `vit_base_patch16_best.pt` | 78.80%              |

---

## 🏗️ Project Architecture

A modular and clean structure is maintained for both deep learning and classical ML pipelines.

```
agrivision-project/
├── .streamlit/
│   └── config.toml                  <-- Custom "Material Dark" theme
├── _archive/                        <-- Archived experimental/test scripts
│   ├── main_app_ex.py               <-- Workflow test app
│   └── ...                          <-- Other archived scripts
├── data/
│   ├── train/
│   ├── val/
│   └── test/
├── dl_pipeline/
│   ├── train_alexnet.py
│   ├── test_alexnet.py
│   ├── train_resnet.py
│   ├── test_resnet.py
│   └── ... (other training/testing scripts)
├── ml_pipeline/
│   ├── train_random_forest.py
│   └── train_catboost.py
├── models/
│   ├── resnet50_best_p15.pth        <-- Primary deployed model
│   ├── alexnet_best_p30.pth         <-- Secondary deployed model
│   ├── efficient_b4_best.pt         <-- Teammate’s model
│   ├── vit_base_patch16_best.pt     <-- Teammate’s model
│   └── ... (other models)
├── output/
│   ├── resnet_p15/
│   │   ├── confusion_matrix.png
│   │   ├── classification_report_heatmap.png
│   │   └── ... (final artifacts)
│   └── ... (other outputs)
├── .gitignore
├── interpret_cam.py                 <-- 4-model Grad-CAM engine
├── main_app.py                      <-- Streamlit web app
└── requirements.txt
```

---

## ⚙️ Setup & Installation

### Prerequisites

* Python 3.10+
* Git
* (Optional) NVIDIA GPU with CUDA for DL training

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/jbzmcs/agrivision-project.git
cd agrivision-project
```

### 2️⃣ Create and Activate a Virtual Environment

**Windows (PowerShell or Git Bash):**

```bash
python -m venv venv
.\venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run the Final Web Application

This is the main entry point of the project.

```bash
streamlit run main_app.py
```

Your browser should open automatically at [http://localhost:8501](http://localhost:8501)

### Re-run Deep Learning Experiments

You can retrain or test any of the models using the provided scripts.

```bash
# Train ResNet50
python dl_pipeline/train_resnet.py

# Test ResNet50
python dl_pipeline/test_resnet.py
```

---

## 🧠 Technology Stack

| Category                | Technologies                                                        |
| ----------------------- | ------------------------------------------------------------------- |
| **Core Deep Learning**  | PyTorch                                                             |
| **Web Application**     | Streamlit                                                           |
| **Model Architectures** | torchvision.models (AlexNet, ResNet50), timm (ViT, EfficientNet-B4) |
| **Interpretability**    | pytorch-grad-cam                                                    |
| **ML & Evaluation**     | scikit-learn, pandas, seaborn, matplotlib                           |
| **Image Processing**    | Pillow (PIL), OpenCV (cv2)                                          |

---

## 💬 Acknowledgments

Special thanks to the open-source community and the developers of **PyTorch**, **Streamlit**, and **TIMM** for their exceptional frameworks that made this project possible.
