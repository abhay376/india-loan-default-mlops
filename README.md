[README.md](https://github.com/user-attachments/files/25622425/README.md)[Uploadin# 🇮🇳 India Loan Default Prediction — MLOps Pipeline

> An end-to-end MLOps project predicting loan default risk for Indian applicants — built as a production-ready fintech API.

![API Status](https://img.shields.io/badge/API-Live-brightgreen) ![Python](https://img.shields.io/badge/Python-3.13-blue) ![Accuracy](https://img.shields.io/badge/Accuracy-93%25-success) ![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🚀 Live Demo

```
{"message": "India Loan Default Prediction API", "status": "running"}
```

**API Endpoints:**
- `GET /` — Health check
- `POST /predict` — Predict loan default risk for an applicant

---

## 🎯 What This Project Does

This project replicates what fintech startups like **KreditBee**, **CreditSea**, and **Slice** do at their core — predict whether a loan applicant is likely to default, using machine learning.

Given an applicant's profile (income, age, occupation, etc.), the API returns:
- ✅ **Default Risk** or **No Default Risk**
- 📊 **Confidence score** (e.g. 0.93)

---

## 🏗️ Architecture

```
Raw Data (100k Indian loan records)
        ↓
Data Preprocessing (src/data_preprocessing.py)
        ↓
Model Training — 3 Models Compared in MLflow
   ├── GradientBoosting  → F1: 0.8154
   ├── RandomForest      → F1: 0.9331 ✅ WINNER
   └── XGBoost           → F1: 0.9084
        ↓
Best Model Saved (models/model.joblib)
        ↓
FastAPI Prediction Server (src/api.py)
        ↓
Docker Container (Dockerfile)
```

---

## 📊 Model Performance

| Model | F1 Score | Accuracy |
|-------|----------|----------|
| GradientBoosting | 0.8154 | 87.31% |
| **RandomForest** | **0.9331** | **93.04%** ⭐ |
| XGBoost | 0.9084 | 91.43% |

> Best model: **RandomForest** with **93% accuracy** on 100,000 real Indian loan records

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| ML Models | Scikit-Learn, XGBoost |
| Experiment Tracking | MLflow |
| Pipeline Automation | Prefect |
| API Server | FastAPI + Uvicorn |
| Containerization | Docker |
| Language | Python 3.13 |

---

## 📁 Project Structure

```
india-loan-default-mlops/
├── data/
│   └── Applicant_Details_For_Loan_Approve.csv
├── models/
│   ├── model.joblib          # Best trained model
│   ├── scaler.joblib         # Feature scaler
│   └── feature_names.joblib  # Feature names
├── src/
│   ├── api.py                # FastAPI prediction server
│   ├── data_preprocessing.py # Data cleaning & feature engineering
│   ├── train_model.py        # Model training + MLflow logging
│   └── prefect_flow.py       # Automated retraining pipeline
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/abhay376/india-loan-default-mlops.git
cd india-loan-default-mlops
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Preprocess data
```bash
python src/data_preprocessing.py
```

### 4. Train models
```bash
python src/train_model.py

Stratified train-test split

Class imbalance handled using class weights

Evaluated using F1 score
```

### 5. Start the API
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### 6. Test a prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Annual_Income": 500000,
    "Applicant_Age": 35,
    "Work_Experience": 8,
    "Marital_Status": "Married",
    "House_Ownership": "Owned",
    "Vehicle_Ownership": "Yes",
    "Occupation": "Software Engineer",
    "Residence_City": "Mumbai",
    "Residence_State": "Maharashtra",
    "Years_in_Current_Employment": 5,
    "Years_in_Current_Residence": 3
  }'
```

**Response:**
```json
{
  "prediction": "No Default Risk",
  "confidence": 0.9304
}
```

---

## 🐳 Run with Docker

```bash
docker build -t loan-default-api .
docker run -p 8000:8000 loan-default-api
```

---

## 📈 View MLflow Experiments

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Then open `http://localhost:5000` to compare all model runs.

---

## 🔄 Automated Retraining (Prefect)

```bash
python src/prefect_flow.py
```

This schedules automatic model retraining when new data arrives.

---

## 📌 Dataset

- **Source:** Kaggle — Applicant Details For Loan Approve (India)
- **Size:** 100,000 records
- **Features:** Income, Age, Work Experience, Marital Status, House Ownership, Vehicle Ownership, Occupation, City, State, Employment Years, Residence Years
- **Target:** Loan Default Risk (Binary)

---

## 💡 Business Use Case

India has **190M+ underbanked citizens** who lack traditional credit scores. This model uses alternative data to assess loan risk — the same approach used by India's fastest-growing fintech startups.

**Potential integrations:**
- NBFCs (Non-Banking Financial Companies)
- Digital lending platforms
- Credit risk assessment tools

---

## 🔐 Production Considerations

- Model versioning with MLflow
- API containerized using Docker
- Reproducible pipeline via Prefect
- Scalable deployment ready

## 👨‍💻 Author

**Abhay** — [@abhay376](https://github.com/abhay376)

---

## 📄 License

MIT License — feel free to use this for your own projects!
g README.md…]()
