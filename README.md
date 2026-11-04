# 🧠 An Intelligent Deep Learning Framework for Handling Imbalanced and Noisy Real-World Data

## 📌 Overview
Deep learning models often fail in high-stakes domains (healthcare, finance, autonomous systems) due to **class imbalance** and **noisy labels**. Traditional methods treat these issues in isolation. 

This project proposes a unified, end-to-end deep learning framework that simultaneously addresses both challenges using an adaptive, feedback-driven pipeline.

## ✨ Key Features
* **Automated Noise Detection:** Uses DBSCAN and confidence-thresholding to identify and correct mislabeled data dynamically.
* **Adaptive Class Balancing:** Reinforcement Learning (RL) agent dynamically adjusts SMOTE/ADASYN resampling ratios based on epoch-level performance.
* **Robust Loss Functions:** Combines Generalized Cross Entropy (GCE) and Focal Loss to handle noise while focusing on hard minority samples.
* **Attention-Based Architecture:** Custom ResNet with Squeeze-and-Excitation (SE) blocks to suppress noise-contaminated feature channels.
* **Integrated Explainability:** Generates attention maps and SHAP values for transparent decision-making.

## 📊 Key Results
* **> 20% improvement** in F1-score and detection rates for minority classes compared to baseline models.
* Maintains **> 75% F1-score** even with **20% label noise** (compared to baseline dropping below 60%).
* The RL feedback loop alone contributes a **~10% gain** in minority F1-score.

## 🛠️ Technology Stack
* **Deep Learning:** PyTorch, Torchvision
* **Data Processing:** NumPy, Pandas, Scikit-learn, Imbalanced-learn
* **Deployment:** Flask, Docker, TorchScript, Gunicorn
* **Experiment Tracking:** Weights & Biases (wandb)

## 📂 Repository Structure
```text
├── /data          # Dataset download and preprocessing scripts
├── /models        # Model architecture definitions (ResNet + SE Blocks)
├── /training      # Training pipeline and RL feedback controller
├── /evaluation    # Metric computation and visualization
├── /deployment    # Docker, Flask API, and TorchScript export
└── /notebooks     # Jupyter notebooks for exploratory analysis
