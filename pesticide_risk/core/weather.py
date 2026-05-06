import requests
import pandas as pd
from datetime import datetime
from ..config import API_TIMEOUT
from ..database import insert_weather

def fetch_weather_data(latitude, longitude):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "precipitation_probability,precipitation,temperature_2m,wind_speed_10m",
        "timezone": "auto",
        "forecast_days": 3
    }
    
    try:
        response = requests.get(url, params=params, timeout=API_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame({
            'time': data['hourly']['time'],
            'rainfall_probability': data['hourly']['precipitation_probability'],
            'rainfall_amount': data['hourly']['precipitation'],
            'temperature': data['hourly']['temperature_2m'],
            'wind_speed': data['hourly']['wind_speed_10m']
        })
        
        for idx, row in df.iterrows():
            insert_weather(
                latitude, longitude,
                row['rainfall_probability'],
                row['rainfall_amount'],
                row['temperature'],
                row['wind_speed']
            )
        
        return df
    except requests.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def get_forecast_window(latitude, longitude, hours_ahead=24):
    df = fetch_weather_data(latitude, longitude)
    if df is not None:
        return df.head(hours_ahead)
    return None
