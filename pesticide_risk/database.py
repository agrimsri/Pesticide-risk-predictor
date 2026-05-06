import sqlite3
from datetime import datetime
from .config import DATABASE_PATH

def init_db():
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS weather_data (
        id INTEGER PRIMARY KEY,
        latitude REAL,
        longitude REAL,
        timestamp DATETIME,
        rainfall_probability REAL,
        rainfall_amount REAL,
        temperature REAL,
        wind_speed REAL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY,
        latitude REAL,
        longitude REAL,
        pesticide TEXT,
        prediction INTEGER,
        confidence REAL,
        hours_to_rain REAL,
        temperature REAL,
        wind_speed REAL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    
    conn.commit()
    conn.close()

def insert_weather(latitude, longitude, rainfall_prob, rainfall_amt, temp, wind):
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO weather_data 
        (latitude, longitude, timestamp, rainfall_probability, rainfall_amount, temperature, wind_speed)
        VALUES (?, ?, ?, ?, ?, ?, ?)''',
        (latitude, longitude, datetime.now(), rainfall_prob, rainfall_amt, temp, wind))
    conn.commit()
    conn.close()

def insert_prediction(latitude, longitude, pesticide, prediction, confidence, hours_to_rain, temp, wind):
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO predictions 
        (latitude, longitude, pesticide, prediction, confidence, hours_to_rain, temperature, wind_speed)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
        (latitude, longitude, pesticide, prediction, confidence, hours_to_rain, temp, wind))
    conn.commit()
    conn.close()

def get_recent_weather(hours=24):
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute('''SELECT * FROM weather_data WHERE 
        created_at > datetime('now', '-' || ? || ' hours')''', (hours,))
    data = c.fetchall()
    conn.close()
    return data
