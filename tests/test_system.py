#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pesticide_risk.database import init_db
from pesticide_risk.core.ml import generate_synthetic_training_data, train_models, load_models, predict
from pesticide_risk.core.features import create_training_dataset
from pesticide_risk.core.decay import calculate_washoff_risk
from pesticide_risk.config import PESTICIDE_DATA

def test_data_generation():
    print("Testing synthetic data generation...")
    X, y = generate_synthetic_training_data(100)
    assert X.shape == (100, 13), "Feature matrix shape mismatch"
    assert len(y) == 100, "Label count mismatch"
    print("✓ Data generation: PASSED\n")

def test_database():
    print("Testing database...")
    init_db()
    print("✓ Database initialization: PASSED\n")

def test_decay_model():
    print("Testing decay model...")
    for pesticide in ['glyphosate', 'imidacloprid', 'atrazine']:
        efficacy = calculate_washoff_risk(
            pesticide,
            temperature=25,
            wind_speed=5,
            rainfall_prob=60,
            hours_to_rain=12,
            rain_intensity=5
        )
        assert 0 <= efficacy <= 1, f"Invalid efficacy: {efficacy}"
    print("✓ Decay model: PASSED\n")

def test_model_training():
    print("Testing model training...")
    lr_model, xgb_model, scaler, results = train_models()
    
    for model_name in ['logistic_regression', 'xgboost', 'ensemble']:
        metrics = results[model_name]
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['auc'] <= 1
        print(f"  {model_name}:")
        print(f"    Accuracy: {metrics['accuracy']:.3f}")
        print(f"    AUC-ROC: {metrics['auc']:.3f}")
    print("✓ Model training: PASSED\n")

def test_predictions():
    print("Testing predictions...")
    lr_model, xgb_model, scaler = load_models()
    
    X = np.random.randn(1, 13)
    prediction, confidence = predict(X, (lr_model, xgb_model), scaler)
    
    assert prediction in [0, 1], "Invalid prediction"
    assert 0 <= confidence <= 1, "Invalid confidence"
    print(f"  Sample prediction: {prediction}")
    print(f"  Confidence: {confidence:.3f}")
    print("✓ Predictions: PASSED\n")

def test_pesticide_data():
    print("Testing pesticide database...")
    for pesticide, data in PESTICIDE_DATA.items():
        assert 'half_life_days' in data
        assert 'rain_fastness_hours' in data
        assert data['half_life_days'] > 0
        assert data['rain_fastness_hours'] > 0
    print(f"✓ {len(PESTICIDE_DATA)} pesticides configured: PASSED\n")

def main():
    print("=" * 60)
    print("PESTICIDE RISK PREDICTOR - TEST SUITE")
    print("=" * 60 + "\n")
    
    try:
        test_pesticide_data()
        test_database()
        test_data_generation()
        test_decay_model()
        test_model_training()
        test_predictions()
        
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        print("\nYou can now run:")
        print("  python app.py          # Start web server")
        print("  python cli.py 28.7041 77.1025 glyphosate  # CLI test")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
