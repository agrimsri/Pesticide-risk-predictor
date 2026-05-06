import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import odeint
from ..config import PESTICIDE_DATA

def exponential_decay(t, initial, rate):
    return initial * np.exp(-rate * t)

def washoff_decay(t, initial, rain_intensity, rain_fastness):
    base_decay_rate = np.log(2) / 24
    rain_decay_rate = rain_intensity / max(rain_fastness, 0.1)
    total_rate = base_decay_rate + rain_decay_rate
    return initial * np.exp(-total_rate * t)

def estimate_rain_time(forecast_df, rain_threshold=10):
    for idx, row in forecast_df.iterrows():
        if row['rainfall_amount'] > rain_threshold / 100:
            return idx
    return len(forecast_df)

def calculate_washoff_risk(pesticide, temperature, wind_speed, rainfall_prob, hours_to_rain, rain_intensity=0):
    pesti_info = PESTICIDE_DATA.get(pesticide, PESTICIDE_DATA['glyphosate'])
    half_life = pesti_info['half_life_days'] * 24
    rain_fastness = pesti_info['rain_fastness_hours']
    
    decay_rate = np.log(2) / half_life
    effective_rate = decay_rate * (1 + wind_speed / 5)
    effective_rate *= (0.8 if temperature > 25 else 1.0)
    
    if rainfall_prob > 50 and hours_to_rain > 0:
        rain_decay = (100 - rainfall_prob) / (rain_fastness + 0.1)
        effective_rate += rain_decay / 100
    
    efficacy_remaining = np.exp(-effective_rate * hours_to_rain)
    
    return efficacy_remaining

def fit_decay_curve(time_data, efficacy_data, pesticide):
    pesti_info = PESTICIDE_DATA[pesticide]
    half_life = pesti_info['half_life_days'] * 24
    
    try:
        popt, _ = curve_fit(exponential_decay, time_data, efficacy_data, 
                           p0=[1.0, np.log(2) / half_life],
                           maxfev=5000)
        return popt
    except:
        decay_rate = np.log(2) / half_life
        return [1.0, decay_rate]

def predict_safe_window(forecast_df, pesticide, min_efficacy=0.7):
    rain_fastness = PESTICIDE_DATA.get(pesticide, PESTICIDE_DATA['glyphosate'])['rain_fastness_hours']
    
    for idx, row in forecast_df.iterrows():
        hours_to_rain = idx
        efficacy = calculate_washoff_risk(
            pesticide,
            row['temperature'],
            row['wind_speed'],
            row['rainfall_probability'],
            hours_to_rain,
            row['rainfall_amount']
        )
        
        if efficacy < min_efficacy:
            return {'safe_apply': False, 'hours_until_risk': hours_to_rain, 'efficacy': efficacy}
    
    return {'safe_apply': True, 'hours_until_risk': len(forecast_df), 'efficacy': 1.0}
