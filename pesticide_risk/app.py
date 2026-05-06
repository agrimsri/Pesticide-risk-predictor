from flask import Flask, render_template_string, request, jsonify
import json
from .core.weather import get_forecast_window
from .core.features import engineer_features, prepare_feature_matrix
from .core.ml import predict, load_models, train_models
from .core.decay import predict_safe_window
from .database import init_db, insert_prediction
from .config import PESTICIDE_DATA

app = Flask(__name__)

init_db()

try:
    lr_model, xgb_model, scaler = load_models()
except:
    lr_model, xgb_model, scaler, _ = train_models()

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Pesticide Rain-Washoff Risk Predictor</title>
    <style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    body {
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: radial-gradient(circle at top, #1f1f1f 0%, #0b0b0b 55%);
        min-height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 24px;
        color: #f5f5f5;
    }

    .container {
        width: 100%;
        max-width: 620px;
        background: rgba(18, 18, 18, 0.82);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 42px;
        backdrop-filter: blur(18px);
        box-shadow:
            0 10px 40px rgba(0, 0, 0, 0.6),
            inset 0 1px 0 rgba(255,255,255,0.03);
    }

    h1 {
        font-size: 30px;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 36px;
        color: #ffffff;
        text-align: center;
    }

    .form-group {
        margin-bottom: 22px;
    }

    label {
        display: block;
        margin-bottom: 10px;
        color: #cfcfcf;
        font-size: 14px;
        font-weight: 500;
        letter-spacing: 0.2px;
    }

    input,
    select {
        width: 100%;
        padding: 15px 16px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        color: #ffffff;
        font-size: 14px;
        transition: all 0.25s ease;
    }

    input::placeholder {
        color: #777;
    }

    input:focus,
    select:focus {
        outline: none;
        border-color: rgba(255,255,255,0.18);
        background: rgba(255,255,255,0.06);
        box-shadow: 0 0 0 4px rgba(255,255,255,0.03);
    }

    select option {
        background: #111;
        color: white;
    }

    button {
        width: 100%;
        padding: 16px;
        margin-top: 10px;
        border: none;
        border-radius: 14px;
        background: linear-gradient(
            135deg,
            #ffffff 0%,
            #d6d6d6 100%
        );
        color: #0b0b0b;
        font-size: 15px;
        font-weight: 700;
        cursor: pointer;
        transition: all 0.25s ease;
    }

    button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(255,255,255,0.12);
    }

    button:active {
        transform: scale(0.99);
    }

    .loading {
        display: none;
        text-align: center;
        margin-top: 30px;
        color: #d0d0d0;
    }

    .spinner {
        width: 42px;
        height: 42px;
        border-radius: 50%;
        border: 3px solid rgba(255,255,255,0.08);
        border-top: 3px solid #ffffff;
        animation: spin 0.8s linear infinite;
        margin: 0 auto 14px;
    }

    @keyframes spin {
        100% {
            transform: rotate(360deg);
        }
    }

    .result {
        margin-top: 28px;
        padding: 24px;
        border-radius: 18px;
        display: none;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.03);
        animation: fadeIn 0.3s ease;
    }

    .result.safe {
        border-left: 4px solid #22c55e;
    }

    .result.risk {
        border-left: 4px solid #ef4444;
    }

    .result h3 {
        margin-bottom: 14px;
        font-size: 20px;
        color: #fff;
    }

    .details {
        color: #c9c9c9;
        line-height: 1.9;
        font-size: 14px;
    }

    .details strong {
        color: #ffffff;
        font-weight: 600;
    }

    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @media (max-width: 640px) {
        .container {
            padding: 28px;
        }

        h1 {
            font-size: 24px;
        }
    }
</style>
</head>
<body>
    <div class="container">
        <h1>🌾 Pesticide Rain-Washoff Risk Predictor</h1>
        <form id="predictionForm">
            <div class="form-group">
                <label>Latitude</label>
                <input type="number" id="latitude" placeholder="e.g., 28.7041" step="0.0001" required>
            </div>
            <div class="form-group">
                <label>Longitude</label>
                <input type="number" id="longitude" placeholder="e.g., 77.1025" step="0.0001" required>
            </div>
            <div class="form-group">
                <label>Pesticide Type</label>
                <select id="pesticide" required>
                    <option value="">Select pesticide...</option>
                    <option value="glyphosate">Glyphosate</option>
                    <option value="glufosinate">Glufosinate</option>
                    <option value="pyrethrin">Pyrethrin</option>
                    <option value="imidacloprid">Imidacloprid</option>
                    <option value="atrazine">Atrazine</option>
                </select>
            </div>
            <button type="submit">Analyze Weather & Predict Risk</button>
        </form>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Fetching weather data and analyzing...</p>
        </div>
        
        <div class="result" id="result">
            <h3 id="resultTitle"></h3>
            <div class="details" id="resultDetails"></div>
        </div>
    </div>
    
    <script>
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            loading.style.display = 'block';
            result.style.display = 'none';
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        latitude: parseFloat(document.getElementById('latitude').value),
                        longitude: parseFloat(document.getElementById('longitude').value),
                        pesticide: document.getElementById('pesticide').value
                    })
                });
                const data = await response.json();
                loading.style.display = 'none';
                
                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }
                
                const resultDiv = document.getElementById('result');
                resultDiv.className = 'result ' + (data.safe_to_apply ? 'safe' : 'risk');
                document.getElementById('resultTitle').textContent = data.safe_to_apply ? '✓ SAFE TO APPLY' : '✗ RISK - WAIT';
                document.getElementById('resultDetails').innerHTML = `
                    <strong>Confidence:</strong> ${(data.confidence * 100).toFixed(1)}%<br>
                    <strong>Hours until rain risk:</strong> ${data.hours_to_rain.toFixed(1)}<br>
                    <strong>Estimated efficacy:</strong> ${(data.efficacy * 100).toFixed(1)}%<br>
                    <strong>Temperature:</strong> ${data.temperature.toFixed(1)}°C<br>
                    <strong>Wind speed:</strong> ${data.wind_speed.toFixed(1)} m/s<br>
                    <strong>Recommendation:</strong> ${data.recommendation}
                `;
                resultDiv.style.display = 'block';
            } catch (error) {
                loading.style.display = 'none';
                alert('Error: ' + error.message);
            }
        });
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def make_prediction():
    data = request.json
    latitude = data.get('latitude')
    longitude = data.get('longitude')
    pesticide = data.get('pesticide')
    
    if not all([latitude, longitude, pesticide]):
        return jsonify({'error': 'Missing parameters'}), 400
    
    if pesticide not in PESTICIDE_DATA:
        return jsonify({'error': 'Invalid pesticide'}), 400
    
    forecast = get_forecast_window(latitude, longitude, hours_ahead=24)
    if forecast is None or forecast.empty:
        return jsonify({'error': 'Could not fetch weather data'}), 500
    
    temp_at_app = forecast['temperature'].iloc[0]
    features = engineer_features(forecast, pesticide, temp_at_app)
    X, feature_cols = prepare_feature_matrix(features)
    
    prediction, confidence = predict(X, (lr_model, xgb_model), scaler)
    
    safe_window = predict_safe_window(forecast, pesticide)
    
    insert_prediction(
        latitude, longitude, pesticide,
        prediction, confidence, 
        features['hours_to_rain'],
        temp_at_app,
        features['average_wind_speed']
    )
    
    recommendation = "Apply now - conditions are favorable" if prediction else f"Wait {int(safe_window['hours_until_risk'])} hours for better conditions"
    
    return jsonify({
        'safe_to_apply': bool(prediction),
        'confidence': confidence,
        'hours_to_rain': features['hours_to_rain'],
        'efficacy': features['estimated_efficacy'],
        'temperature': temp_at_app,
        'wind_speed': features['average_wind_speed'],
        'recommendation': recommendation
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
