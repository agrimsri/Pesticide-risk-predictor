import os

DATABASE_PATH = os.getenv('DATABASE_PATH', 'pesticide_risk.db')
MODEL_PATH = os.getenv('MODEL_PATH', 'models')
API_TIMEOUT = 10

PESTICIDE_DATA = {
    'glyphosate': {'half_life_days': 2.2, 'rain_fastness_hours': 1},
    'glufosinate': {'half_life_days': 3.5, 'rain_fastness_hours': 2},
    'pyrethrin': {'half_life_days': 1.0, 'rain_fastness_hours': 0.5},
    'imidacloprid': {'half_life_days': 14, 'rain_fastness_hours': 3},
    'atrazine': {'half_life_days': 59, 'rain_fastness_hours': 4},
}

SAFE_THRESHOLD = 0.7
