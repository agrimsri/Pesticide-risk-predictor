import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score
import xgboost as xgb
from ..config import MODEL_PATH, SAFE_THRESHOLD

def generate_synthetic_training_data(n_samples=500):
    np.random.seed(42)
    
    hours_to_rain = np.random.uniform(0, 48, n_samples)
    rainfall_intensity = np.random.exponential(2, n_samples)
    rainfall_prob = np.random.uniform(0, 100, n_samples)
    temp_app = np.random.uniform(15, 35, n_samples)
    avg_temp = np.random.uniform(15, 30, n_samples)
    wind_speed = np.random.exponential(3, n_samples)
    wind_var = np.random.exponential(1, n_samples)
    humidity = np.random.uniform(20, 90, n_samples)
    time_above_20 = np.random.poisson(12, n_samples)
    time_above_25 = np.random.poisson(6, n_samples)
    cum_rainfall = rainfall_intensity * (hours_to_rain / 10 + 1)
    heavy_rain = (rainfall_intensity > 5).astype(int) * np.random.poisson(2, n_samples)
    efficacy = np.exp(-0.05 * hours_to_rain - 0.1 * rainfall_intensity)
    
    X = np.column_stack([
        hours_to_rain, rainfall_intensity, rainfall_prob,
        temp_app, avg_temp, wind_speed, wind_var, humidity,
        time_above_20, time_above_25, cum_rainfall, heavy_rain, efficacy
    ])
    
    y = ((hours_to_rain > 8) & (efficacy > SAFE_THRESHOLD) | 
         ((rainfall_prob < 50) & (rainfall_intensity < 5))).astype(int)
    
    return X, y

def train_models():
    X, y = generate_synthetic_training_data(500)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    ensemble_pred = ((lr_proba + xgb_proba) / 2 > 0.5).astype(int)
    
    results = {
        'logistic_regression': {
            'accuracy': accuracy_score(y_test, lr_pred),
            'precision': precision_score(y_test, lr_pred),
            'recall': recall_score(y_test, lr_pred),
            'auc': roc_auc_score(y_test, lr_proba)
        },
        'xgboost': {
            'accuracy': accuracy_score(y_test, xgb_pred),
            'precision': precision_score(y_test, xgb_pred),
            'recall': recall_score(y_test, xgb_pred),
            'auc': roc_auc_score(y_test, xgb_proba)
        },
        'ensemble': {
            'accuracy': accuracy_score(y_test, ensemble_pred),
            'precision': precision_score(y_test, ensemble_pred),
            'recall': recall_score(y_test, ensemble_pred),
            'auc': roc_auc_score(y_test, (lr_proba + xgb_proba) / 2)
        }
    }
    
    os.makedirs(MODEL_PATH, exist_ok=True)
    pickle.dump(lr_model, open(f'{MODEL_PATH}/lr_model.pkl', 'wb'))
    pickle.dump(xgb_model, open(f'{MODEL_PATH}/xgb_model.pkl', 'wb'))
    pickle.dump(scaler, open(f'{MODEL_PATH}/scaler.pkl', 'wb'))
    
    return lr_model, xgb_model, scaler, results

def load_models():
    lr_model = pickle.load(open(f'{MODEL_PATH}/lr_model.pkl', 'rb'))
    xgb_model = pickle.load(open(f'{MODEL_PATH}/xgb_model.pkl', 'rb'))
    scaler = pickle.load(open(f'{MODEL_PATH}/scaler.pkl', 'rb'))
    return lr_model, xgb_model, scaler

def predict(X, models=None, scaler=None):
    if models is None:
        lr_model, xgb_model, scaler = load_models()
    else:
        lr_model, xgb_model = models
    
    X_scaled = scaler.transform(X)
    lr_proba = lr_model.predict_proba(X_scaled)[:, 1][0]
    xgb_proba = xgb_model.predict_proba(X)[:, 1][0]
    
    ensemble_proba = (lr_proba + xgb_proba) / 2
    prediction = int(ensemble_proba > 0.5)
    
    return prediction, ensemble_proba
