# 🇮🇳 India Loan Default Prediction — MLOps Pipeline

End-to-end MLOps project for predicting loan defaults for Indian applicants.

**Stack:** Python · Scikit-Learn · XGBoost · MLflow · Prefect · FastAPI · Docker

---

## 📁 Project Structure

```
├── data/
│   └── Applicant_Details_For_Loan_Approve.csv   # raw dataset
├── models/                                       # saved model artifacts
├── src/
│   ├── data_preprocessing.py                     # data cleaning & feature engineering
│   ├── train_model.py                            # model training + MLflow tracking
│   ├── prefect_flow.py                           # Prefect retraining orchestration
│   └── api.py                                    # FastAPI prediction service
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Preprocessing

```bash
python src/data_preprocessing.py
```

Creates `data/processed_data.csv` and `models/preprocessor.joblib`.

### 3. Train Models

```bash
python src/train_model.py
```

Trains **GradientBoosting**, **RandomForest**, and **XGBoost**. All runs are logged to MLflow. The best model (by F1 score) is saved to `models/model.joblib`.

### 4. View MLflow Dashboard

```bash
mlflow ui
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) to compare runs.

### 5. Start the API

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for interactive Swagger docs.

### 6. Run Prefect Retraining Flow

```bash
python src/prefect_flow.py
```

Or start the Prefect server and register a deployment:

```bash
prefect server start
```

---

## 🐳 Docker

```bash
# Build the image (includes preprocessing + training)
docker build -t loan-prediction-api .

# Run the container
docker run -p 8000:8000 loan-prediction-api
```

---

## 📡 API Usage

### Health Check

```bash
curl http://127.0.0.1:8000/health
```

### Make a Prediction

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "1",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 1500,
    "LoanAmount": 128,
    "Loan_Amount_Term": 360,
    "Credit_History": 1,
    "Property_Area": "Urban"
  }'
```

**Response:**

```json
{
  "prediction": "Y",
  "probability": 0.8732,
  "model_used": "model.joblib"
}
```

---

## 📊 Models Compared

| Model              | Logged to MLflow | Notes                        |
|---------------------|-----------------|------------------------------|
| GradientBoosting    | ✅              | Strong default for tabular   |
| RandomForest        | ✅              | Robust, interpretable        |
| XGBoost             | ✅              | State-of-the-art boosting    |

Best model (by F1 score) is auto-selected and saved for serving.
