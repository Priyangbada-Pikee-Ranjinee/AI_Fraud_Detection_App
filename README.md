# AI Fraud Detection System

This is a machine learning-based web application that detects fraudulent credit card transactions using multiple classification models. The system is deployed using Streamlit for an interactive user interface.

---

## Features

- Real-time fraud detection
- Multiple machine learning models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - K-Nearest Neighbors
- Probability-based risk analysis
- Interactive Streamlit dashboard

---

## How It Works

1. User enters transaction details (Amount, Time, V1–V28 features)
2. Data is preprocessed using scaling and feature engineering
3. Selected machine learning model predicts fraud probability
4. Output is classified as:
   - Low Risk
   - Medium Risk
   - High Risk

---

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib

---

## Model Information

The models were trained on a credit card fraud dataset with highly imbalanced classes. Logistic Regression performed best due to the dataset being linearly separable after PCA transformation.

---

## How to Run Locally

```bash
pip install -r requirements.txt
python -m streamlit run app.py