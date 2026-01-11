from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
import joblib
import pandas as pd
import numpy as np
from weather_service import fetch_and_process_agricultural_data, get_available_districts

app = FastAPI(
    title="Crop Yield Predictor API",
    description="Predict crop yield based on district, crop type, area and weather data",
    version="1.0.0"
)

# Load the model package
try:
    package = joblib.load("crop_yield.joblib")
except FileNotFoundError:
    package = None
    print("Warning: crop_yield.joblib not found. Model predictions will not work.")


# T(base) values for different crop types
CROP_TBASE = {
    "TOTAL COTTON (LINT)": 15.5,
    "TOTAL GROUNDNUT": 10,
    "TOTAL BAJRA": 10,
    "CASTOR": 15,
    "TOTAL RICE": 10,
    "TOTAL TOBACCO": 18.5,
}


# Enum for valid crop types - user can ONLY select from these 6 crops
class CropType(str, Enum):
    COTTON = "TOTAL COTTON (LINT)"
    GROUNDNUT = "TOTAL GROUNDNUT"
    BAJRA = "TOTAL BAJRA"
    CASTOR = "CASTOR"
    RICE = "TOTAL RICE"
    TOBACCO = "TOTAL TOBACCO"


# District-Crop mapping: Which crops are grown in which districts
# Add districts where each crop is cultivated
DISTRICT_CROPS = {
    "Ahmedabad": ["TOTAL COTTON (LINT)", "TOTAL GROUNDNUT", "TOTAL BAJRA", "CASTOR", "TOTAL RICE"],
    "Amreli": ["TOTAL COTTON (LINT)", "TOTAL GROUNDNUT", "TOTAL BAJRA", "CASTOR"],
    "Anand": ["TOTAL RICE", "TOTAL TOBACCO", "TOTAL COTTON (LINT)"],
    "Aravalli": ["TOTAL BAJRA", "CASTOR", "TOTAL RICE"],
    "Banaskantha": ["TOTAL BAJRA", "CASTOR", "TOTAL GROUNDNUT", "TOTAL COTTON (LINT)"],
    "Bharuch": ["TOTAL COTTON (LINT)", "TOTAL RICE", "TOTAL GROUNDNUT"],
    "Bhavnagar": ["TOTAL COTTON (LINT)", "TOTAL GROUNDNUT", "TOTAL BAJRA"],
    "Botad": ["TOTAL COTTON (LINT)", "TOTAL GROUNDNUT", "TOTAL BAJRA"],
    "Chhota Udaipur": ["TOTAL BAJRA", "TOTAL RICE", "CASTOR"],
    "Dahod": ["TOTAL BAJRA", "TOTAL RICE"],
    "Dang": ["TOTAL RICE"],
    "Devbhoomi Dwarka": ["TOTAL GROUNDNUT", "TOTAL BAJRA", "TOTAL COTTON (LINT)"],
    "Gandhinagar": ["TOTAL BAJRA", "CASTOR", "TOTAL RICE"],
    "Gir Somnath": ["TOTAL GROUNDNUT", "TOTAL BAJRA", "TOTAL COTTON (LINT)"],
    "Jamnagar": ["TOTAL GROUNDNUT", "TOTAL BAJRA", "TOTAL COTTON (LINT)"],
    "Junagadh": ["TOTAL GROUNDNUT", "TOTAL BAJRA", "TOTAL COTTON (LINT)"],
    "Kheda": ["TOTAL RICE", "TOTAL TOBACCO", "TOTAL COTTON (LINT)"],
    "Kutch": ["TOTAL BAJRA", "CASTOR", "TOTAL GROUNDNUT"],
    "Mahisagar": ["TOTAL BAJRA", "TOTAL RICE", "CASTOR"],
    "Mehsana": ["TOTAL BAJRA", "CASTOR", "TOTAL GROUNDNUT"],
    "Morbi": ["TOTAL COTTON (LINT)", "TOTAL GROUNDNUT", "TOTAL BAJRA"],  # No TOBACCO here
    "Narmada": ["TOTAL RICE", "TOTAL COTTON (LINT)"],
    "Navsari": ["TOTAL RICE", "TOTAL GROUNDNUT"],
    "Panchmahal": ["TOTAL BAJRA", "TOTAL RICE"],
    "Patan": ["TOTAL BAJRA", "CASTOR", "TOTAL GROUNDNUT"],
    "Porbandar": ["TOTAL GROUNDNUT", "TOTAL BAJRA"],
    "Rajkot": ["TOTAL COTTON (LINT)", "TOTAL GROUNDNUT", "TOTAL BAJRA"],
    "Sabarkantha": ["TOTAL BAJRA", "CASTOR", "TOTAL RICE"],
    "Surat": ["TOTAL RICE", "TOTAL COTTON (LINT)", "TOTAL GROUNDNUT"],
    "Surendranagar": ["TOTAL COTTON (LINT)", "TOTAL GROUNDNUT", "TOTAL BAJRA"],
    "Tapi": ["TOTAL RICE", "TOTAL COTTON (LINT)"],
    "Vadodara": ["TOTAL RICE", "TOTAL TOBACCO", "TOTAL COTTON (LINT)"],
    "Valsad": ["TOTAL RICE", "TOTAL GROUNDNUT"],
}


def get_tbase(crop_type: str) -> Optional[float]:
    """Get base temperature for a crop type."""
    crop_upper = crop_type.upper().strip()
    return CROP_TBASE.get(crop_upper, None)


def validate_district_crop(district: str, crop: str) -> tuple[bool, str]:
    """
    Validate if a crop is grown in a specific district.
    Returns (is_valid, error_message)
    """
    district_normalized = district.strip().title()
    
    if district_normalized not in DISTRICT_CROPS:
        return False, f"District '{district}' not found in our database."
    
    available_crops = DISTRICT_CROPS[district_normalized]
    
    if crop not in available_crops:
        crop_names = [c.replace("TOTAL ", "") for c in available_crops]
        return False, (
            f"'{crop.replace('TOTAL ', '')}' is not cultivated in {district_normalized} district. "
            f"Available crops in {district_normalized}: {', '.join(crop_names)}"
        )
    
    return True, ""


def calculate_gdd(avg_tmax: float, avg_tmin: float, tbase: float) -> float:
    """Calculate Growing Degree Days (GDD) = (Tmax + Tmin)/2 - Tbase"""
    if avg_tmax is None or avg_tmin is None or tbase is None:
        return None
    return ((avg_tmax + avg_tmin) / 2) - tbase


class YieldInput(BaseModel):
    district: str = Field(..., example="Surat", description="Gujarat district name")
    crop: CropType = Field(..., example=CropType.CASTOR, description="Crop type - select from available options")
    area: float = Field(..., gt=0, example=100.0, description="Area in hectares")

    class Config:
        json_schema_extra = {
            "example": {
                "district": "Surat",
                "crop": "CASTOR",
                "area": 100.0
            }
        }


class PredictionResponse(BaseModel):
    district: str
    crop: str
    area: float
    tbase: float
    gdd: float
    weather_data: dict
    prediction: float
    unit: str = "kg/hectare"


class DistrictsResponse(BaseModel):
    districts: List[str]
    total: int


@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Crop Yield Predictor API is running"}


@app.get("/districts", response_model=DistrictsResponse, tags=["Districts"])
async def list_districts():
    """Get list of all available Gujarat districts."""
    districts = get_available_districts()
    return {"districts": districts, "total": len(districts)}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def get_prediction(data: YieldInput):
    """
    Predict crop yield for a specific district.
    
    The API will automatically:
    1. Validate if the crop is cultivated in the selected district
    2. Fetch weather data for the selected district (last year's monsoon season)
    3. Calculate required statistics (rainfall, temperature, humidity, etc.)
    4. Use the trained model to predict yield
    """
    if package is None:
        raise HTTPException(
            status_code=500, 
            detail="Model not loaded. Please ensure crop_yield.joblib exists."
        )
    
    # Get crop type value
    crop_type = data.crop.value  # Get the enum value (e.g., "CASTOR", "TOTAL RICE")
    district_normalized = data.district.strip().title()
    
    # 1. Validate district-crop combination
    is_valid, error_msg = validate_district_crop(district_normalized, crop_type)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    # 2. Fetch weather data for the selected district (last year's monsoon: June-November)
    try:
        enriched_data = await fetch_and_process_agricultural_data(data.district)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=503, 
            detail=f"Failed to fetch weather data: {str(e)}"
        )

    # 3. Calculate T(base) and GDD based on crop type
    tbase = get_tbase(crop_type)
    
    avg_tmax = enriched_data.get("average_tmax")
    avg_tmin = enriched_data.get("average_tmin")
    gdd = calculate_gdd(avg_tmax, avg_tmin, tbase)

    # 4. Create ONE-HOT ENCODED features matching the training data exactly
    # Numeric features
    model_input = {
        "Area": data.area,
        "mean_dry_spell_length": enriched_data.get("mean_dry_spell_length"),
        "total_rainfall": enriched_data.get("total_rainfall"),
        "rainy_days": enriched_data.get("rainy_days"),
        "average_tmax": avg_tmax,
        "average_tmin": avg_tmin,
        "average_humidity": enriched_data.get("average_humidity"),
        "T(base)": tbase,
        "GDD": round(gdd, 2) if gdd is not None else None,
    }
    
    # One-hot encode DISTRICTS (all districts from training)
    all_districts = [
        "Amreli", "Anand", "Aravalli", "Banaskantha", "Bharuch", "Bhavnagar",
        "Botad", "Chhota Udaipur", "Dahod", "Dang", "Devbhumi Dwarka", 
        "Gandhinagar", "Gir Somnath", "Jamnagar", "Junagadh", "Kheda", "Kutch",
        "Mahisagar", "Mehsana", "Morbi", "Narmada", "Navsari", "Panchmahal",
        "Patan", "Porbandar", "Rajkot", "Sabarkantha", "Surat", "Surendranagar",
        "Tapi", "Vadodara", "Valsad"
    ]
    for d in all_districts:
        model_input[f"district_{d}"] = 1.0 if district_normalized == d else 0.0
    
    # One-hot encode CROP TYPES (5 crops from training - CASTOR is baseline/dropped)
    all_crops = ["TOTAL BAJRA", "TOTAL COTTON (LINT)", "TOTAL GROUNDNUT", "TOTAL RICE", "TOTAL TOBACCO"]
    for c in all_crops:
        model_input[f"Crop Type_{c}"] = 1.0 if crop_type == c else 0.0
    
    # One-hot encode FIRST RAIN CATEGORY (on_time only, early is baseline/dropped)
    first_rain = enriched_data.get("first_rain_category", "late")
    model_input["first_rain_category_on_time"] = 1.0 if first_rain == "on_time" else 0.0
    
    # One-hot encode LAST RAIN CATEGORY (late and on_time, early is baseline/dropped)
    last_rain = enriched_data.get("last_rain_category", "early")
    model_input["last_rain_category_late"] = 1.0 if last_rain == "late" else 0.0
    model_input["last_rain_category_on_time"] = 1.0 if last_rain == "on_time" else 0.0
    
    # One-hot encode SEASON (2017-2023, some baseline dropped)
    season = enriched_data.get("season", 2025)
    for year in [2017, 2018, 2019, 2020, 2021, 2022, 2023]:
        model_input[f"season_{year}"] = 1.0 if season == year else 0.0
    
    # Define EXACT column order as used in model training
    FEATURE_ORDER = [
        "Area", "mean_dry_spell_length", "total_rainfall", "rainy_days",
        "average_tmax", "average_tmin", "average_humidity", "T(base)", "GDD",
        # Districts (one-hot)
        "district_Amreli", "district_Anand", "district_Aravalli", "district_Banaskantha",
        "district_Bharuch", "district_Bhavnagar", "district_Botad", "district_Chhota Udaipur",
        "district_Dahod", "district_Dang", "district_Devbhumi Dwarka", "district_Gandhinagar",
        "district_Gir Somnath", "district_Jamnagar", "district_Junagadh", "district_Kheda",
        "district_Kutch", "district_Mahisagar", "district_Mehsana", "district_Morbi",
        "district_Narmada", "district_Navsari", "district_Panchmahal", "district_Patan",
        "district_Porbandar", "district_Rajkot", "district_Sabarkantha", "district_Surat",
        "district_Surendranagar", "district_Tapi", "district_Vadodara", "district_Valsad",
        # Crop Types (one-hot)
        "Crop Type_TOTAL BAJRA", "Crop Type_TOTAL COTTON (LINT)", "Crop Type_TOTAL GROUNDNUT",
        "Crop Type_TOTAL RICE", "Crop Type_TOTAL TOBACCO",
        # Rain categories (one-hot)
        "first_rain_category_on_time", "last_rain_category_late", "last_rain_category_on_time",
        # Seasons (one-hot)
        "season_2017", "season_2018", "season_2019", "season_2020", "season_2021", 
        "season_2022", "season_2023"
    ]

    # 5. Transform & Predict
    try:
        input_df = pd.DataFrame([model_input])
        
        # Reorder columns to match exact training order
        input_df = input_df[FEATURE_ORDER]
        
        # Make prediction
        prediction = package["model"].predict(input_df)[0]
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}"
        )

    return {
        "district": data.district.strip().title(),
        "crop": crop_type,
        "area": data.area,
        "tbase": tbase,
        "gdd": round(gdd, 2) if gdd is not None else None,
        "weather_data": enriched_data,
        "prediction": round(float(prediction), 2),
        "unit": "kg/hectare"
    }


@app.get("/crops/{district}", tags=["Crops"])
async def get_district_crops(district: str):
    """
    Get list of crops cultivated in a specific district.
    Use this to check which crops are available before making a prediction.
    """
    district_normalized = district.strip().title()
    
    if district_normalized not in DISTRICT_CROPS:
        available_districts = ", ".join(sorted(DISTRICT_CROPS.keys()))
        raise HTTPException(
            status_code=400,
            detail=f"District '{district}' not found. Available districts: {available_districts}"
        )
    
    crops = DISTRICT_CROPS[district_normalized]
    crop_names = [c.replace("TOTAL ", "") for c in crops]
    
    return {
        "district": district_normalized,
        "available_crops": crop_names,
        "crop_values": crops,  # Actual values to use in API
        "total": len(crops)
    }


@app.get("/weather/{district}", tags=["Weather"])
async def get_district_weather(district: str):
    """
    Get last year's monsoon weather statistics for a specific district.
    Data is fetched from June to November of the previous year.
    """
    try:
        weather_data = await fetch_and_process_agricultural_data(district)
        return {
            "district": district.strip().title(),
            "data_period": f"{weather_data.get('season')}-06-01 to {weather_data.get('season')}-11-30",
            "weather_statistics": weather_data
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=503, 
            detail=f"Failed to fetch weather data: {str(e)}"
        )