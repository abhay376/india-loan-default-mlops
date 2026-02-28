from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI(title="India Loan Default Prediction API")

model = joblib.load('models/model.joblib')
scaler = joblib.load('models/scaler.joblib')
feature_names = joblib.load('models/feature_names.joblib')

class ApplicantData(BaseModel):
    Annual_Income: float
    Applicant_Age: int
    Work_Experience: int
    Marital_Status: str
    House_Ownership: str
    Vehicle_Ownership: str
    Occupation: str
    Residence_City: str
    Residence_State: str
    Years_in_Current_Employment: int
    Years_in_Current_Residence: int

@app.get("/")
def home():
    return {"message": "India Loan Default Prediction API", "status": "running"}

@app.post("/predict")
def predict(data: ApplicantData):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    input_dict = data.dict()
    cat_cols = ['Marital_Status', 'House_Ownership', 'Vehicle_Ownership', 'Occupation', 'Residence_City', 'Residence_State']
    for col in cat_cols:
        input_dict[col] = hash(input_dict[col]) % 100
    features = np.array([[input_dict[f] for f in feature_names]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0].max()
    return {
        "prediction": "Default Risk" if prediction == 1 else "No Default Risk",
        "confidence": round(float(probability), 4)
    }
