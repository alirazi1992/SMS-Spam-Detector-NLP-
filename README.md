# 📩 Day 11 — SMS Spam Detector (NLP)

This project builds a machine learning model that detects **spam** vs **ham** (normal) SMS messages using **TF-IDF vectorization** and **linear classifiers**.  
The goal is to demonstrate a complete **NLP classification pipeline** from raw text → preprocessing → vectorization → model selection → evaluation → inference.

---

## 🧠 Project Overview

Text messages often contain patterns that distinguish **spam** (promotional, scam, automated) from **ham** (normal human conversation).  
This project extracts those patterns using:

- **Text cleaning & normalization**
- **TF-IDF (Term Frequency–Inverse Document Frequency)**
- **Linear SVM**, **Logistic Regression**, and **Naive Bayes**
- **Performance comparison using Precision/Recall/F1**
- **Error analysis to inspect misclassified messages**
- **Model export (`.joblib`) for real-world use**

---

## 🗂️ Dataset

Dataset used: **UCI SMS Spam Collection**  
Kaggle mirror: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

| Label | Meaning |
|-------|---------|
| `ham` | Normal user message |
| `spam` | Advertising, phishing, scams |

---

## 🧼 Preprocessing Steps

1. Lowercasing  
2. Removing URLs, numbers, and noise  
3. Normalizing whitespaces  
4. Converting text → TF-IDF vectors (word + bi-gram features)

---

## 🔥 Models Compared

| Model                 | Why we use it |
|----------------------|---------------|
| **Linear SVM (Calibrated)** | Strong baseline for sparse NLP data |
| **Logistic Regression**     | Stable, interpretable linear classifier |
| **Multinomial Naive Bayes** | Extremely fast + very effective for text |

---

## 📊 Example Results

| Model                   | Accuracy | Precision | Recall |   F1   | ROC-AUC |
|------------------------|:--------:|:---------:|:------:|:------:|:-------:|
| **Linear SVM (Cal.)**  | **0.98** | **0.98**  | **0.98** | **0.98** | **0.99** |
| Logistic Regression     | 0.98     | 0.98      | 0.97   | 0.98   | 0.99    |
| Multinomial Naive Bayes | 0.97     | 0.96      | 0.97   | 0.96   | 0.98    |

> Results may vary slightly depending on dataset version and train/test split.

---

## 📈 Visualizations Included

- Label distribution plot
- Text length histogram by class
- Confusion matrices for all models
- ROC curves for probabilistic models
- Top spam-triggering keywords (feature inspection)

---

## 💾 Model Export

After training, the best model is saved as:

sms_spam_detector.joblib

### Example Usage:
```python
from joblib import load
model = load("sms_spam_detector.joblib")
model.predict(["hello how are you"])
model.predict(["WIN a FREE vacation now!!! CLICK here!!!"])

```

### 📂 Repository Structure

│── Day11_SMS_Spam_Detector.ipynb     # Full notebook
│── sms_spam_detector.joblib          # Saved trained model
│── README.md                         # Project documentation
