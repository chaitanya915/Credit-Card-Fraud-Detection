# Next-Generation Adaptive Credit Card Fraud Detection Using Hybrid AI and Explainable AI
## Project Overview

This project detects fraudulent credit card transactions using Machine Learning, Deep Learning, and Explainable AI.

The system compares multiple algorithms, predicts fraudulent transactions, generates a fraud risk score, and explains every prediction using SHAP and LIME.

## Features

- Data preprocessing
- Exploratory Data Analysis
- Feature Engineering
- SMOTE for class balancing
- Multiple ML models
- Autoencoder anomaly detection
- Explainable AI (SHAP & LIME)
- Fraud Risk Score
- Streamlit Dashboard
- FastAPI Integration

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- TensorFlow
- SHAP
- LIME
- Streamlit
- FastAPI
- Git
- GitHub

## Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── data/
├── notebooks/
├── src/
├── models/
├── reports/
├── images/
├── tests/
├── app.py
├── requirements.txt
└── README.md
```
## Installation

1. Clone the repository

```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
```

2. Open the project folder

3. Create a virtual environment

```bash
python -m venv venv
```

4. Activate it

```bash
venv\Scripts\activate
```

5. Install dependencies

```bash
pip install -r requirements.txt
```
## Run the Dashboard

```bash
streamlit run app.py
```

## Run the API

```bash
uvicorn src.api.main:app --reload
```

## Machine Learning Models

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Isolation Forest
- Autoencoder

## Explainable AI

The project uses:

- SHAP
- LIME

These techniques explain why a transaction was predicted as fraudulent and highlight the most influential features.

## Future Enhancements

- Real-time streaming with Apache Kafka
- Federated Learning
- Graph Neural Networks
- Online Learning
- Blockchain integration

## Author

Name: Chaitanya Gurav

B.E. Computer Engineering

SSPM College of Engineering, Kankavli

README.md

│
├── Project Title
├── Overview
├── Features
├── Technologies
├── Installation
├── Project Structure
├── Machine Learning Models
├── Explainable AI
├── Future Enhancements
└── Author

