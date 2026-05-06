#!/usr/bin/env python3
import sys
from .core.weather import get_forecast_window
from .core.features import engineer_features, prepare_feature_matrix
from .core.decay import predict_safe_window
from .core.ml import predict, load_models
from .config import PESTICIDE_DATA

def main():
    if len(sys.argv) < 4:
        print("Usage: python cli.py <latitude> <longitude> <pesticide>")
        print("\nAvailable pesticides:")
        for p in PESTICIDE_DATA.keys():
            print(f"  - {p}")
        sys.exit(1)
    
    latitude = float(sys.argv[1])
    longitude = float(sys.argv[2])
    pesticide = sys.argv[3]
    
    if pesticide not in PESTICIDE_DATA:
        print(f"Error: Unknown pesticide '{pesticide}'")
        sys.exit(1)
    
    print(f"\n🌾 Analyzing: {pesticide.upper()} @ ({latitude}, {longitude})")
    print("-" * 60)
    
    print("\n[1/4] Fetching weather forecast...")
    forecast = get_forecast_window(latitude, longitude, 24)
    if forecast is None or forecast.empty:
        print("Error: Could not fetch weather data")
        sys.exit(1)
    print(f"✓ Retrieved {len(forecast)} hourly forecasts")
    
    print("\n[2/4] Engineering features...")
    temp_at_app = forecast['temperature'].iloc[0]
    features = engineer_features(forecast, pesticide, temp_at_app)
    print(f"✓ Generated {len(features)} features")
    
    print("\n[3/4] Physical model analysis...")
    safe_window = predict_safe_window(forecast, pesticide)
    print(f"✓ Hours until rain risk: {safe_window['hours_until_risk']:.1f}")
    print(f"✓ Estimated efficacy: {safe_window['efficacy']*100:.1f}%")
    
    print("\n[4/4] ML ensemble prediction...")
    lr_model, xgb_model, scaler = load_models()
    X, feature_cols = prepare_feature_matrix(features)
    prediction, confidence = predict(X, (lr_model, xgb_model), scaler)
    print(f"✓ Model ensemble confidence: {confidence*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("RESULT:")
    print("=" * 60)
    status = "✓ SAFE TO APPLY" if prediction else "✗ AT-RISK, WAIT"
    print(f"Status: {status}")
    print(f"Confidence: {confidence*100:.1f}%")
    
    print("\nWeather Summary (Next 24h):")
    print(f"  Current Temperature: {temp_at_app:.1f}°C")
    print(f"  Avg Temperature: {features['average_temperature_24h']:.1f}°C")
    print(f"  Avg Wind Speed: {features['average_wind_speed']:.1f} m/s")
    print(f"  Max Rainfall Probability: {features['rainfall_probability_max']:.0f}%")
    print(f"  Cumulative Rainfall: {features['cumulative_rainfall']:.1f} mm")
    print(f"  Hours to Rain: {features['hours_to_rain']:.1f}")
    
    print("\nPhysical Model:")
    print(f"  Half-life: {PESTICIDE_DATA[pesticide]['half_life_days']} days")
    print(f"  Rain-fastness: {PESTICIDE_DATA[pesticide]['rain_fastness_hours']} hours")
    print(f"  Efficacy at application: 100%")
    print(f"  Efficacy in {features['hours_to_rain']:.0f}h: {features['estimated_efficacy']*100:.1f}%")
    
    print("\nRecommendation:")
    if prediction:
        print(f"  ✓ Apply pesticide now. Conditions are favorable.")
        print(f"    Safe application window: next {safe_window['hours_until_risk']:.0f} hours")
    else:
        print(f"  ✗ Wait {safe_window['hours_until_risk']:.0f} hours before applying.")
        print(f"    Better conditions expected after rain passes.")
    
    print("\n" + "=" * 60)

if __name__ == '__main__':
    main()
