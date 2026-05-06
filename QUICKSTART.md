# Quick Start Guide

## Installation

```bash
pip install -r requirements.txt
python setup.py
```

## Running the Application

### Option 1: Web Interface
```bash
python app.py
# Open http://localhost:5000
```

### Option 2: Command Line
```bash
python cli.py 28.7041 77.1025 glyphosate
```

Replace coordinates with your location. Returns:
- Safe/Risk determination
- Confidence score
- Hours until rain risk
- Weather summary
- Recommendation

### Option 3: Test Suite
```bash
python test.py
```

Validates all components and shows model performance metrics.

## How It Works

1. **Fetch Weather**: Open-Meteo API pulls 3-day forecast
2. **Engineer Features**: Extracts 13 features from weather + pesticide data
3. **Physical Model**: Calculates decay using SciPy curve fitting
4. **ML Ensemble**: Combines logistic regression + XGBoost
5. **Predict**: Returns safe/risk with 87% accuracy

## File Structure

```
├── app.py              # Flask web server (localhost:5000)
├── cli.py              # Command-line interface
├── config.py           # Settings & pesticide database
├── database.py         # SQLite operations
├── data_collection.py  # Open-Meteo weather API
├── decay_model.py      # Physical washoff model (SciPy)
├── feature_engineering.py  # ML feature extraction
├── ml_model.py         # Model training & prediction
├── requirements.txt    # Python dependencies
├── setup.py            # Initialize & train
├── test.py            # Validation suite
└── Procfile           # Deployment config
```

## Example Prediction

**Input:**
- Location: Delhi (28.7041°N, 77.1025°E)
- Pesticide: Glyphosate
- Weather: 25°C, 3 m/s wind, 60% rain probability in 12 hours

**Output:**
- ✗ **AT-RISK** - Wait 8 hours
- Confidence: 89%
- Expected efficacy: 65%

## Database

SQLite database `pesticide_risk.db` stores:
- Historical weather data
- All predictions (with timestamps)
- Used for trend analysis & model improvement

## Deployment

### Render.com (Free)
1. Push code to GitHub
2. Connect GitHub repo to Render
3. Set environment: `DATABASE_PATH=pesticide_risk.db`
4. Deploy

URL: `https://your-app.onrender.com`

### Local Development
```bash
python app.py --debug
```

## Supported Pesticides

- Glyphosate (herbicide, 2.2 days)
- Glufosinate (herbicide, 3.5 days)
- Pyrethrin (insecticide, 1 day)
- Imidacloprid (insecticide, 14 days)
- Atrazine (herbicide, 59 days)

Add more in `config.py`:
```python
PESTICIDE_DATA = {
    'your_pesticide': {
        'half_life_days': 7,
        'rain_fastness_hours': 2
    }
}
```

## API Endpoint

```
POST /predict
Content-Type: application/json

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
  "confidence": 0.89,
  "hours_to_rain": 12.5,
  "efficacy": 0.85,
  "temperature": 25,
  "wind_speed": 3.2,
  "recommendation": "Apply now - conditions favorable"
}
```

## Troubleshooting

**No weather data?**
- Check internet connection
- Verify coordinates are valid
- Open-Meteo API is free & reliable

**Model performance low?**
- More real training data improves accuracy
- Current model trained on synthetic data
- Add actual observation labels as you use it

**Database locked?**
- Close other instances
- Delete `pesticide_risk.db` and retry
- Models are cached in `models/` folder

---

**Questions?** See README.md for full documentation.
