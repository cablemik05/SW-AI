"""
Sample data generator for ddareungi.csv and weather.csv
Generates 2 years (2022-2023) of realistic synthetic data.
"""
import pandas as pd
import numpy as np
import os

np.random.seed(42)

# ── Weather data ─────────────────────────────────────────────
dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')

def temp_for_day(d):
    """Seoul-like average temperature with seasonal curve."""
    yday = d.day_of_year
    base = 12.5 + 14.5 * np.sin(2 * np.pi * (yday - 80) / 365)
    return round(base + np.random.normal(0, 3), 1)

def pm10_for_day(d):
    """PM10 with higher values in spring/winter."""
    month = d.month
    if month in [3, 4, 5]:        # spring: yellow dust
        base = 65
    elif month in [12, 1, 2]:     # winter: heating pollution
        base = 55
    elif month in [6, 7, 8]:      # summer: rain clears air
        base = 28
    else:
        base = 40
    val = base + np.random.exponential(15) + np.random.normal(0, 8)
    return max(5, round(val, 1))

weather_rows = []
for d in dates:
    weather_rows.append({
        '날짜': d.strftime('%Y-%m-%d'),
        '평균기온': temp_for_day(d),
        'PM10': pm10_for_day(d),
    })

df_weather = pd.DataFrame(weather_rows)
df_weather.to_csv('weather.csv', index=False, encoding='utf-8-sig')
print(f"weather.csv 생성 완료 — {len(df_weather)}행")
print(df_weather.head())

# ── Bike rental data ─────────────────────────────────────────
# Each row = one rental record (대여일시 = rental datetime)
# Generate ~800-3500 rentals per day depending on season/weekday/weather
records = []
for _, row in df_weather.iterrows():
    date = pd.to_datetime(row['날짜'])
    temp = row['평균기온']
    pm10 = row['PM10']

    # Base rentals based on temperature (bell-curve peak ~22°C)
    temp_factor = max(0.1, 1 - ((temp - 22) / 22) ** 2)
    # PM10 penalty
    pm_factor = max(0.6, 1 - max(0, pm10 - 30) / 250)
    # Weekend boost
    weekend_factor = 1.25 if date.weekday() >= 5 else 1.0
    # Monthly seasonal multiplier (summer spike)
    month_mult = {1: 0.45, 2: 0.50, 3: 0.75, 4: 1.0, 5: 1.15,
                  6: 1.20, 7: 1.10, 8: 1.05, 9: 1.15, 10: 1.10,
                  11: 0.80, 12: 0.55}[date.month]

    n_rentals = int(
        1800 * temp_factor * pm_factor * weekend_factor * month_mult
        + np.random.normal(0, 150)
    )
    n_rentals = max(50, n_rentals)

    # Spread rentals across hours (rush hour peaks at 8 & 18)
    hour_weights = np.array([
        0.5, 0.3, 0.2, 0.2, 0.3, 0.8,
        2.0, 4.5, 3.5, 2.5, 2.0, 2.2,
        2.5, 2.3, 2.0, 2.2, 2.8, 4.8,
        4.0, 3.0, 2.5, 2.0, 1.5, 0.8,
    ])
    hour_weights /= hour_weights.sum()
    hours = np.random.choice(24, size=n_rentals, p=hour_weights)
    minutes = np.random.randint(0, 60, size=n_rentals)

    for h, m in zip(hours, minutes):
        dt = date.replace(hour=int(h), minute=int(m))
        records.append({'대여일시': dt.strftime('%Y-%m-%d %H:%M:%S')})

df_bike = pd.DataFrame(records)
df_bike.to_csv('ddareungi.csv', index=False, encoding='utf-8-sig')
print(f"\nddareungi.csv 생성 완료 — {len(df_bike):,}행")
print(df_bike.head())
