#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pesticide_risk.database import init_db
from pesticide_risk.core.ml import train_models

def setup():
    print("Pesticide Risk Predictor - Setup")
    print("-" * 50)
    
    print("\n1. Initializing database...")
    init_db()
    print("Database initialized")
    
    print("\n2. Training models...")
    lr_model, xgb_model, scaler, results = train_models()
    print("Models trained")
    
    print("\n3. Model Performance:")
    for model_name, metrics in results.items():
        print(f"\n  {model_name.upper()}:")
        print(f"   - Accuracy:  {metrics['accuracy']:.3f}")
        print(f"   - Precision: {metrics['precision']:.3f}")
        print(f"   - Recall:    {metrics['recall']:.3f}")
        print(f"   - AUC-ROC:   {metrics['auc']:.3f}")
    
    print("\n4. Ready to run!")
    print("   Run: python app.py")
    print("   Then open: http://localhost:5000")

if __name__ == '__main__':
    setup()
