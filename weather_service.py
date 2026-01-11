"""
Weather Service for Crop Yield Prediction
Fetches weather data for a specific district and calculates required statistics
"""

import requests
import pandas as pd
from datetime import datetime
from typing import Dict, Any

# Gujarat Districts with their coordinates (latitude, longitude)
GUJARAT_DISTRICTS = {
    "Ahmedabad": (23.0225, 72.5714),
    "Amreli": (21.6032, 71.2225),
    "Anand": (22.5645, 72.9289),
    "Aravalli": (23.0333, 73.1500),
    "Banaskantha": (24.1667, 72.4333),
    "Bharuch": (21.7051, 72.9959),
    "Bhavnagar": (21.7645, 72.1519),
    "Botad": (22.1694, 71.6686),
    "Chhota Udaipur": (22.3039, 74.0180),
    "Dahod": (22.8333, 74.2500),
    "Dang": (20.7500, 73.7500),
    "Devbhoomi Dwarka": (22.2029, 69.6575),
    "Gandhinagar": (23.2156, 72.6369),
    "Gir Somnath": (20.8880, 70.4017),
    "Jamnagar": (22.4707, 70.0577),
    "Junagadh": (21.5222, 70.4579),
    "Kheda": (22.7500, 72.6833),
    "Kutch": (23.7337, 69.8597),
    "Mahisagar": (23.1167, 73.6333),
    "Mehsana": (23.5880, 72.3693),
    "Morbi": (22.8173, 70.8370),
    "Narmada": (21.8787, 73.4996),
    "Navsari": (20.9467, 72.9520),
    "Panchmahal": (22.7500, 73.6000),
    "Patan": (23.8493, 72.1266),
    "Porbandar": (21.6417, 69.6293),
    "Rajkot": (22.3039, 70.8022),
    "Sabarkantha": (23.6000, 73.0500),
    "Surat": (21.1702, 72.8311),
    "Surendranagar": (22.7000, 71.6500),
    "Tapi": (21.1167, 73.4167),
    "Vadodara": (22.3072, 73.1812),
    "Valsad": (20.5992, 72.9342),
}

# Open-Meteo Historical Weather API endpoint
BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Date range for LAST YEAR monsoon season (June to November of previous year)
CURRENT_YEAR = datetime.now().year
LAST_YEAR = CURRENT_YEAR - 1  # Use last year's data for prediction
START_DATE = f"{LAST_YEAR}-06-01"
END_DATE = f"{LAST_YEAR}-11-30"


async def fetch_weather_data(district_name: str, lat: float, lon: float, 
                              start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical weather data for a specific district from Open-Meteo API.
    Returns DataFrame with: district, date, min_temp, max_temp, rain, humidity
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "relative_humidity_2m_mean"
        ],
        "timezone": "Asia/Kolkata"
    }
    
    response = requests.get(BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    
    if "daily" not in data:
        raise ValueError(f"No daily data found for {district_name}")
    
    daily_data = data["daily"]
    df = pd.DataFrame({
        "district": district_name,
        "date": daily_data["time"],
        "min_temp": daily_data.get("temperature_2m_min"),
        "max_temp": daily_data.get("temperature_2m_max"),
        "rain": daily_data.get("precipitation_sum"),
        "humidity": daily_data.get("relative_humidity_2m_mean")
    })
    return df


def calculate_district_statistics(daily_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate statistics for a single district.
    Returns dict with all required model features.
    """
    daily_df = daily_df.copy()
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_df = daily_df.sort_values("date").reset_index(drop=True)
    
    # Identify rainy days (rain > 0)
    rainy_days_df = daily_df[daily_df["rain"] > 0]
    rainy_days_count = len(rainy_days_df)
    
    # First and last rain date
    if rainy_days_count > 0:
        first_rain_date = rainy_days_df["date"].min()
        last_rain_date = rainy_days_df["date"].max()
        first_rain_date_str = first_rain_date.strftime("%Y-%m-%d")
        last_rain_date_str = last_rain_date.strftime("%Y-%m-%d")
    else:
        first_rain_date = None
        last_rain_date = None
        first_rain_date_str = None
        last_rain_date_str = None
    
    # Total rainfall
    total_rainfall = daily_df["rain"].sum()
    
    # Calculate dry spell lengths (consecutive days without rain)
    daily_df["is_dry"] = daily_df["rain"] <= 0
    dry_spell_lengths = []
    current_dry_spell = 0
    
    for is_dry in daily_df["is_dry"]:
        if is_dry:
            current_dry_spell += 1
        else:
            if current_dry_spell > 0:
                dry_spell_lengths.append(current_dry_spell)
            current_dry_spell = 0
    
    if current_dry_spell > 0:
        dry_spell_lengths.append(current_dry_spell)
    
    # Mean dry spell length
    mean_dry_spell_length = sum(dry_spell_lengths) / len(dry_spell_lengths) if dry_spell_lengths else 0
    
    # Average temperatures and humidity
    average_tmax = daily_df["max_temp"].mean()
    average_tmin = daily_df["min_temp"].mean()
    average_humidity = daily_df["humidity"].mean()
    
    # Rain categories
    first_rain_category = categorize_first_rain(first_rain_date)
    last_rain_category = categorize_last_rain(last_rain_date)
    
    return {
        "season": LAST_YEAR,
        "first_rain_date": first_rain_date_str,
        "last_rain_date": last_rain_date_str,
        "first_rain_category": first_rain_category,
        "last_rain_category": last_rain_category,
        "mean_dry_spell_length": round(mean_dry_spell_length, 2),
        "total_rainfall": round(total_rainfall, 2),
        "rainy_days": rainy_days_count,
        "average_tmax": round(average_tmax, 2) if pd.notna(average_tmax) else None,
        "average_tmin": round(average_tmin, 2) if pd.notna(average_tmin) else None,
        "average_humidity": round(average_humidity, 2) if pd.notna(average_humidity) else None,
    }


def categorize_first_rain(date_obj) -> str:
    """Categorize first rain date as early, on_time, or late."""
    if date_obj is None or pd.isna(date_obj):
        return 'No Rain'
    
    month_day = (date_obj.month, date_obj.day)
    if (6, 1) <= month_day <= (6, 8):
        return 'early'
    elif (6, 9) <= month_day <= (6, 22):
        return 'on_time'
    elif (6, 23) <= month_day <= (12, 31):
        return 'late'
    return 'N/A'


def categorize_last_rain(date_obj) -> str:
    """Categorize last rain date as early, on_time, or late."""
    if date_obj is None or pd.isna(date_obj):
        return 'No Rain'
    
    month_day = (date_obj.month, date_obj.day)
    if (1, 1) <= month_day <= (10, 8):
        return 'early'
    elif (10, 9) <= month_day <= (10, 25):
        return 'on_time'
    elif (10, 26) <= month_day <= (11, 30):
        return 'late'
    return 'N/A'


async def fetch_and_process_agricultural_data(district: str, 
                                               start_date: str = None, 
                                               end_date: str = None) -> Dict[str, Any]:
    """
    Main function to fetch and process weather data for a specific district.
    This is the function called by app.py for prediction.
    
    Args:
        district: Name of the Gujarat district (e.g., "Surat", "Ahmedabad")
        start_date: Start date for weather data (default: current year monsoon start)
        end_date: End date for weather data (default: current year monsoon end)
    
    Returns:
        Dict containing all calculated statistics needed for model prediction
    """
    # Normalize district name (title case)
    district_normalized = district.strip().title()
    
    # Check if district exists
    if district_normalized not in GUJARAT_DISTRICTS:
        available_districts = ", ".join(sorted(GUJARAT_DISTRICTS.keys()))
        raise ValueError(
            f"District '{district}' not found. Available districts: {available_districts}"
        )
    
    # Get coordinates
    lat, lon = GUJARAT_DISTRICTS[district_normalized]
    
    # Use default dates if not provided
    if start_date is None:
        start_date = START_DATE
    if end_date is None:
        end_date = END_DATE
    
    # Fetch weather data
    daily_df = await fetch_weather_data(district_normalized, lat, lon, start_date, end_date)
    
    # Calculate statistics
    stats = calculate_district_statistics(daily_df)
    
    return stats


def get_available_districts() -> list:
    """Return list of available districts."""
    return sorted(GUJARAT_DISTRICTS.keys())
