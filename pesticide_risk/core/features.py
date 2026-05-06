import numpy as np
import pandas as pd
from .decay import estimate_rain_time, calculate_washoff_risk
from ..config import PESTICIDE_DATA

def engineer_features(forecast_df, pesticide, temperature_at_app):
    features = {}
    
    features['hours_to_rain'] = estimate_rain_time(forecast_df)
    features['rainfall_intensity'] = forecast_df['rainfall_amount'].max()
    features['rainfall_probability_max'] = forecast_df['rainfall_probability'].max()
    features['temperature_at_application'] = temperature_at_app
    features['average_temperature_24h'] = forecast_df['temperature'].mean()
    features['average_wind_speed'] = forecast_df['wind_speed'].mean()
    features['wind_variability'] = forecast_df['wind_speed'].std()
    
    features['humidity_impact'] = forecast_df['rainfall_probability'].mean()
    features['time_above_20c'] = len(forecast_df[forecast_df['temperature'] > 20])
    features['time_above_25c'] = len(forecast_df[forecast_df['temperature'] > 25])
    
    features['cumulative_rainfall'] = forecast_df['rainfall_amount'].sum()
    features['heavy_rain_hours'] = len(forecast_df[forecast_df['rainfall_amount'] > 5])
    
    washoff_risk = calculate_washoff_risk(
        pesticide,
        temperature_at_app,
        forecast_df['wind_speed'].mean(),
        forecast_df['rainfall_probability'].max(),
        features['hours_to_rain'],
        features['rainfall_intensity']
    )
    features['estimated_efficacy'] = washoff_risk
    
    return features

def create_training_dataset(weather_samples, labels, pesticide):
    data = []
    
    for idx, sample in enumerate(weather_samples):
        df = pd.DataFrame(sample)
        features = engineer_features(df, pesticide, df['temperature'].iloc[0])
        features['label'] = labels[idx]
        data.append(features)
    
    return pd.DataFrame(data)

def prepare_feature_matrix(features_dict):
    feature_cols = [
        'hours_to_rain', 'rainfall_intensity', 'rainfall_probability_max',
        'temperature_at_application', 'average_temperature_24h', 'average_wind_speed',
        'wind_variability', 'humidity_impact', 'time_above_20c', 'time_above_25c',
        'cumulative_rainfall', 'heavy_rain_hours', 'estimated_efficacy'
    ]
    
    X = np.array([features_dict[col] for col in feature_cols]).reshape(1, -1)
    return X, feature_cols
