import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

def load_data(path):
    df = pd.read_csv(path)
    print(f"[INFO] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def preprocess(data_path='data/Applicant-details.csv'):
    df = load_data(data_path)
    df = df.drop(columns=['Applicant_ID'], errors='ignore')
    print("[INFO] Missing values after cleaning:")
    print(df.isnull().sum().sum(), "total")
    df = df.fillna(df.mode().iloc[0])
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    if 'Loan_Default_Risk' in cat_cols:
        cat_cols.remove('Loan_Default_Risk')
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    y = le.fit_transform(df['Loan_Default_Risk'].astype(str))
    X = df.drop(columns=['Loan_Default_Risk'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(list(X.columns), 'models/feature_names.joblib')
    np.save('models/X_train.npy', X_train)
    np.save('models/X_test.npy', X_test)
    np.save('models/y_train.npy', y_train)
    np.save('models/y_test.npy', y_test)
    print("[INFO] Preprocessing complete. Files saved to models/")

if __name__ == '__main__':
    preprocess()
