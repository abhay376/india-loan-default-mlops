import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from xgboost import XGBClassifier
import os

def train():
    X_train = np.load('models/X_train.npy')
    X_test = np.load('models/X_test.npy')
    y_train = np.load('models/y_train.npy')
    y_test = np.load('models/y_test.npy')

    models = {
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    }

    best_model = None
    best_score = 0
    best_name = ''

    mlflow.set_experiment('india_loan_default')

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            f1 = f1_score(y_test, preds, average='weighted')
            acc = accuracy_score(y_test, preds)
            mlflow.log_param('model', name)
            mlflow.log_metric('f1_score', f1)
            mlflow.log_metric('accuracy', acc)
            mlflow.sklearn.log_model(model, name)
            print(f"[INFO] {name} - F1: {f1:.4f}, Accuracy: {acc:.4f}")
            if f1 > best_score:
                best_score = f1
                best_model = model
                best_name = name

    joblib.dump(best_model, 'models/model.joblib')
    print(f"[INFO] Best model: {best_name} with F1: {best_score:.4f}")
    print("[INFO] Model saved to models/model.joblib")

if __name__ == '__main__':
    train()
