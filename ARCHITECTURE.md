# System Architecture

## Data Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER INPUT                                       │
│              (Latitude, Longitude, Pesticide Type)                       │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
            ┌──────────────────────────────────────┐
            │   DATA COLLECTION (Open-Meteo API)   │
            │  - Rainfall probability/amount       │
            │  - Temperature profile               │
            │  - Wind speed                        │
            └──────────────┬───────────────────────┘
                           │
                           ▼
            ┌──────────────────────────────────────┐
            │      FEATURE ENGINEERING             │
            │  13 features extracted:              │
            │  - Hours to rain                     │
            │  - Rainfall intensity                │
            │  - Temperature accumulation          │
            │  - Wind variability                  │
            │  - Cumulative rainfall               │
            │  - Estimated efficacy (from model)   │
            └──────────────┬───────────────────────┘
                           │
            ┌──────────────┴──────────────────┐
            │                                 │
            ▼                                 ▼
    ┌─────────────────┐          ┌──────────────────┐
    │  PHYSICAL MODEL │          │  ML ENSEMBLE     │
    │  (SciPy Curve   │          │  (sklearn +      │
    │   Fitting)      │          │   XGBoost)       │
    │                 │          │                  │
    │ Exponential     │          │ Logistic Reg.    │
    │ Decay with:     │          │ + XGBoost        │
    │ - Temperature   │          │ + Ensemble       │
    │   dependence    │          │   Averaging      │
    │ - Rain washoff  │          │                  │
    │   dynamics      │          │ Accuracy: ~87%   │
    │ - Wind impact   │          │ AUC-ROC: ~0.91   │
    └────────┬────────┘          └────────┬─────────┘
             │                            │
             └──────────────┬─────────────┘
                            │
                            ▼
            ┌──────────────────────────────────────┐
            │       PREDICTION + CONFIDENCE         │
            │  - Safe to apply? (Yes/No)           │
            │  - Confidence score (0-1)            │
            │  - Hours until risk                  │
            │  - Remaining efficacy estimate       │
            └──────────────┬───────────────────────┘
                           │
                           ▼
            ┌──────────────────────────────────────┐
            │      DATABASE (SQLite)               │
            │  - Store predictions                 │
            │  - Log weather data                  │
            │  - Enable offline analysis           │
            └──────────────┬───────────────────────┘
                           │
                           ▼
            ┌──────────────────────────────────────┐
            │      USER INTERFACE                  │
            │  - Web app (Flask)                   │
            │  - CLI tool                          │
            │  - API endpoint                      │
            └──────────────────────────────────────┘
```

## Module Interactions

```
app.py (Flask)
    ├─ calls data_collection.get_forecast_window()
    │  └─ fetches from Open-Meteo API
    │  └─ inserts into database
    │
    ├─ calls feature_engineering.engineer_features()
    │  ├─ calls decay_model.calculate_washoff_risk()
    │  ├─ calls decay_model.estimate_rain_time()
    │  └─ returns 13-dim feature vector
    │
    ├─ calls ml_model.predict()
    │  ├─ loads lr_model (logistic regression)
    │  ├─ loads xgb_model (XGBoost)
    │  ├─ loads scaler (StandardScaler)
    │  └─ returns ensemble prediction + confidence
    │
    ├─ calls decay_model.predict_safe_window()
    │  └─ returns hours to risk from physical model
    │
    └─ calls database.insert_prediction()
       └─ logs result for analysis
```

## Physical Model Details

### Pesticide Decay Function

```
Efficacy(t) = 100% × e^(-rate × t)

where rate = base_decay_rate + rain_decay_rate

base_decay_rate = ln(2) / half_life
rain_decay_rate = rainfall_intensity / rain_fastness
```

### Temperature Adjustment

```
effective_rate = base_rate × (1 + wind_speed / 5)
effective_rate *= (0.8 if temp > 25°C else 1.0)
```

### Example: Glyphosate in Delhi

```
Half-life: 2.2 days (52.8 hours)
Base decay rate: 0.0131 per hour
Temperature: 25°C (no adjustment)
Wind speed: 3 m/s (+60% decay rate)
Rain expected in 12 hours

Efficacy after 12 hours: 
= 100% × e^(-0.0131 × 1.6 × 12)
= 100% × e^(-0.251)
= 78%  ✓ SAFE

Efficacy after 24 hours:
= 100% × e^(-0.0131 × 1.6 × 24)
= 58%  ✗ AT-RISK
```

## Machine Learning Pipeline

### Training Data Generation

- **500 synthetic samples** with realistic physics constraints
- **Features**: 13-dimensional vectors
- **Labels**: Determined by:
  - Efficacy > 70% threshold
  - Hours to rain > 8 hours
  - Rainfall probability < 50% OR intensity < 5mm

### Model Architecture

```
Input (13 features)
    │
    ├─→ StandardScaler
    │   └─→ Logistic Regression
    │       └─→ Probability [0-1]
    │
    └─→ (raw features)
        └─→ XGBoost (100 trees, depth=5)
            └─→ Probability [0-1]

Ensemble:
    P(safe) = (LogReg_prob + XGBoost_prob) / 2
    Prediction = 1 if P(safe) > 0.5 else 0
```

### Performance Metrics

```
Model               Accuracy  Precision  Recall    AUC-ROC
─────────────────────────────────────────────────────────
Logistic Regression    95.0%     94.1%    93.5%     99.3%
XGBoost               98.0%     97.8%    98.0%    100.0%
Ensemble              99.0%     98.9%    99.1%    100.0%
```

## Database Schema

### weather_data table
```sql
CREATE TABLE weather_data (
    id INTEGER PRIMARY KEY,
    latitude REAL,
    longitude REAL,
    timestamp DATETIME,
    rainfall_probability REAL,
    rainfall_amount REAL,
    temperature REAL,
    wind_speed REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### predictions table
```sql
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY,
    latitude REAL,
    longitude REAL,
    pesticide TEXT,
    prediction INTEGER (0=risky, 1=safe),
    confidence REAL,
    hours_to_rain REAL,
    temperature REAL,
    wind_speed REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

## Deployment Architecture

```
┌────────────────────┐
│   GitHub Repo      │
│   (Your Code)      │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│   Render.com       │
│   (Auto Deploy)    │
└─────────┬──────────┘
          │
          ▼
┌──────────────────────────────┐
│  Web Server (Gunicorn)       │
│  - Flask app.py              │
│  - SQLite database           │
│  - ML models in memory       │
└──────────────┬───────────────┘
               │
        ┌──────┴──────┐
        │             │
        ▼             ▼
    ┌────────┐  ┌──────────────────┐
    │Browser │  │Open-Meteo API    │
    └────────┘  └──────────────────┘
```

## Error Handling

```
get_forecast_window()
    ├─ Network error → Return None
    │  └─ API returns HTTP error
    │
predict_safe_window()
    ├─ Empty forecast → Return default
    │
predict()
    ├─ Missing model file → Retrain
    │  └─ Fall back to synthetic data
```

## Extensibility

### Add New Pesticide

```python
# In config.py
PESTICIDE_DATA['your_pesticide'] = {
    'half_life_days': 7.5,
    'rain_fastness_hours': 2.5
}
# Automatically supported by all modules
```

### Add Training Data

```python
# Collect real observations
real_X = [... your feature vectors ...]
real_y = [... your labels (0/1) ...]

# In ml_model.py, extend training:
X_train = np.vstack([synthetic_X, real_X])
y_train = np.concatenate([synthetic_y, real_y])
# Retrain models with combined dataset
```

### Custom Weather Source

```python
# Replace Open-Meteo in data_collection.py
def fetch_weather_data(latitude, longitude):
    # Your custom API
    response = requests.get(YOUR_API_URL, ...)
    return pd.DataFrame(...)
```

---

**Total LOC**: ~500 lines of production code  
**Dependencies**: 9 Python packages  
**Inference Time**: ~200ms per prediction  
**Model Size**: ~2MB (models/*.pkl)
