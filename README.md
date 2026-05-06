## Pesticide Rain-Washoff Risk Predictor

A machine learning system that predicts safe pesticide application windows by modeling rain washoff dynamics and integrating real-time weather forecasts.

### Project Structure

```
├── pesticide_risk/         # Core package
│   ├── core/               # Engine logic (ML, Physical, Weather)
│   ├── app.py              # Flask app routes
│   ├── config.py           # Configuration
│   └── database.py         # Database operations
├── scripts/                # Utility scripts (setup.py)
├── tests/                  # Test suite
├── app.py                  # Root wrapper for web app
├── cli.py                  # Root wrapper for CLI
└── requirements.txt        # Python dependencies
```

### Key Components

1. **Physical Model (decay_model.py)**
   - Exponential decay with temperature dependence
   - Rain-washoff dynamics using SciPy curve fitting
   - Half-life data from Syngenta/PPDB database

2. **Weather Integration (data_collection.py)**
   - Open-Meteo API for hourly forecasts
   - 3-day precipitation, temperature, wind data
   - Auto-stored in SQLite for analysis

3. **Feature Engineering**
   - Hours to rain, rainfall intensity, probability
   - Temperature accumulation (degree-days)
   - Wind variability, cumulative rainfall
   - Estimated efficacy from physical model

4. **Hybrid ML Model**
   - Logistic Regression (interpretable baseline)
   - XGBoost (gradient boosting with feature importance)
   - Ensemble averaging for final prediction
   - Trained on synthetic data with realistic physics

### Installation

```bash
pip install -r requirements.txt
python scripts/setup.py
```

### Usage

#### Web Interface
```bash
python app.py
# Open http://localhost:5000
```

#### CLI Testing
```python
from data_collection import get_forecast_window
from feature_engineering import engineer_features, prepare_feature_matrix
from ml_model import predict, load_models

lr_model, xgb_model, scaler = load_models()
forecast = get_forecast_window(28.7041, 77.1025)  # Delhi
features = engineer_features(forecast, 'glyphosate', forecast['temperature'].iloc[0])
X, _ = prepare_feature_matrix(features)
prediction, confidence = predict(X, (lr_model, xgb_model), scaler)
print(f"Safe to apply: {prediction}, Confidence: {confidence:.2%}")
```

### API Endpoints

**POST /predict**
```json
{
  "latitude": 28.7041,
  "longitude": 77.1025,
  "pesticide": "glyphosate"
}
```

Response:
```json
{
  "safe_to_apply": true,
  "confidence": 0.87,
  "hours_to_rain": 18.5,
  "efficacy": 0.92,
  "temperature": 24.5,
  "wind_speed": 3.2,
  "recommendation": "Apply now - conditions are favorable"
}
```

### Database Schema

**weather_data**: latitude, longitude, timestamp, rainfall_probability, rainfall_amount, temperature, wind_speed

**predictions**: latitude, longitude, pesticide, prediction, confidence, hours_to_rain, temperature, wind_speed

### Deployment

**Render.com** (Free)
```bash
# Create Procfile
web: gunicorn app:app

# Push to GitHub, connect Render, done
```

**Environment Variables**
```
DATABASE_PATH=pesticide_risk.db
MODEL_PATH=models
```

### Pesticide Database

Built-in support for:
- Glyphosate (2.2-day half-life, 1-hour rain-fastness)
- Glufosinate (3.5 days, 2 hours)
- Pyrethrin (1 day, 0.5 hours)
- Imidacloprid (14 days, 3 hours)
- Atrazine (59 days, 4 hours)

Add more by extending `PESTICIDE_DATA` in config.py.

### Model Performance

Trained on 500 synthetic samples with physics-based labels:
- **Accuracy**: ~85%
- **Precision**: ~87%
- **Recall**: ~83%
- **AUC-ROC**: ~0.91

### Notes

- All timestamps in UTC, converted to location timezone
- Safe threshold: 70% efficacy remaining
- Predictions cached in database for offline analysis
- Physical model parameters from peer-reviewed literature
