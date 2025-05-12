# Real-Time Fraud Detection – Catch It Before It Hits

This is a Streamlit web application that predicts whether a credit card transaction is fraudulent using a machine learning model trained on anonymized transaction data.

**Live App**: https://sharmitaranii-real-time-fraud-detection-ml-streamlit-app-oupvde.streamlit.app/  
**Colab Notebook**: https://colab.research.google.com/drive/1HzauY24UJQ2RoQvspnhl4qJZ7v6R59Bb?usp=sharing

---

## 🧠 Project Overview

Fraud detection in financial transactions is a critical real-world challenge. This application aims to detect fraudulent transactions in real-time using historical credit card data and machine learning. The app predicts whether a transaction is likely to be fraudulent based on key numerical features derived from PCA and engineered time-amount features.

---

## 🚀 Features

- Interactive Streamlit web application
- Real-time fraud prediction using a trained XGBoost model
- Upload CSV functionality for batch predictions
- Clean user interface with alert summary and results table
- End-to-end pipeline: EDA → Model → Deployment

---

## 📊 Dataset

The dataset used is the [Credit Card Fraud Detection dataset]([https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud]) from Kaggle. It contains anonymized credit card transactions made by European cardholders in 2013. The dataset includes:

- 284,807 transactions
- 492 frauds (0.17% of data)
- 28 PCA-transformed numerical features + `Time`, `Amount`, and `Class`

---

## 🔍 Exploratory Data Analysis

A comprehensive EDA was performed to understand data patterns and build intuition for feature selection and modeling. Key EDA steps:

- Class imbalance visualization
- Feature distributions via KDE plots
- Boxplots to compare legit vs. fraud transactions
- Time-based fraud trends
- Correlation matrix heatmap
- Feature importance from XGBoost
- ROC curve + AUC
- Error analysis of false positives/negatives

Refer to the Colab notebook for detailed visualizations and insights.

---

## 🧠 Modeling

Several models were evaluated, including:

- Random Forest
- XGBoost (final model)

XGBoost was selected for its superior recall and AUC performance. Class imbalance was handled using SMOTE, and model performance was evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

---

##  Deployment

The final model was deployed using Streamlit to create an intuitive UI where users can upload new transaction data and receive fraud predictions in real time.

---

## Installation

To run the app locally:

# Clone the repository
git clone https://github.com/sharmitaranii/real-time-fraud-detection-ml.git

# Navigate into the folder
cd real-time-fraud-detection-ml

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_app.py

## Usage
   After running the app, a browser window will open.

   Upload a CSV file containing transaction data (matching the trained feature format).

   The app will display:

   Preview of your data

   Count of predicted fraudulent transactions

   Full table with prediction column
   
## Results
   The XGBoost model achieved the following performance:

    Accuracy: ~99.9%

    Precision (fraud): 73%

    Recall (fraud): 89%

    AUC Score: 0.979

    The model performs strongly, especially in detecting fraudulent transactions while minimizing false negatives.

## Future Work
   Add live transaction stream simulation

   Incorporate SHAP for model explainability

   Deploy on Streamlit Cloud with database integration

   Alert notification system (email or SMS)

 ## Author
   Sharmita Rani
   LinkedIn Profile: https://www.linkedin.com/in/sharmitarani/
