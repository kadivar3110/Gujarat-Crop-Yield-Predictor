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

try:
    package = joblib.load("crop_yield.joblib")
except FileNotFoundError:
    package = None
    print("Warning: crop_yield.joblib not found. Model predictions will not work.")

CROP_TBASE = {
    "TOTAL COTTON (LINT)": 15.5,
    "TOTAL GROUNDNUT": 10,
    "TOTAL BAJRA": 10,
    "CASTOR": 15,
    "TOTAL RICE": 10,
    "TOTAL TOBACCO": 18.5,
}


class CropType(str, Enum):
    COTTON = "TOTAL COTTON (LINT)"
    GROUNDNUT = "TOTAL GROUNDNUT"
    BAJRA = "TOTAL BAJRA"
    CASTOR = "CASTOR"
    RICE = "TOTAL RICE"
    TOBACCO = "TOTAL TOBACCO"



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

    crop_upper = crop_type.upper().strip()
    return CROP_TBASE.get(crop_upper, None)



def validate_district_crop(district: str, crop: str) -> tuple[bool, str]:

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

    if avg_tmax is None or avg_tmin is None or tbase is None:
        return None
    return ((avg_tmax + avg_tmin) / 2) - tbase


class YieldInput(BaseModel):
    district: str = Field(..., example="Morbi", description="Gujarat district name")
    crop: CropType = Field(..., example=CropType.COTTON, description="Crop type - select from available options")
    area: float = Field(..., gt=0, example=100.0, description="Area in hectares")

    class Config:
        json_schema_extra = {
            "example": {
                "district": "Morbi",
                "crop": "COTTON",
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
    return {"status": "healthy", "message": "Crop Yield Predictor API is running"}


@app.get("/districts", response_model=DistrictsResponse, tags=["Districts"])
async def list_districts():
    districts = get_available_districts()
    return {"districts": districts, "total": len(districts)}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def get_prediction(data: YieldInput):

    if package is None:
        raise HTTPException(
            status_code=500, 
            detail="Model not loaded. Please ensure crop_yield.joblib exists."
        )
    

    crop_type = data.crop.value 
    district_normalized = data.district.strip().title()
    

    is_valid, error_msg = validate_district_crop(district_normalized, crop_type)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    
    try:
        enriched_data = await fetch_and_process_agricultural_data(data.district)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        raise HTTPException(
            status_code=503, 
            detail=f"Failed to fetch weather data: {str(e)}"
        )


    tbase = get_tbase(crop_type)
    
    avg_tmax = enriched_data.get("average_tmax")
    avg_tmin = enriched_data.get("average_tmin")
    gdd = calculate_gdd(avg_tmax, avg_tmin, tbase)

   

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
    
    
    all_crops = ["TOTAL BAJRA", "TOTAL COTTON (LINT)", "TOTAL GROUNDNUT", "TOTAL RICE", "TOTAL TOBACCO"]
    for c in all_crops:
        model_input[f"Crop Type_{c}"] = 1.0 if crop_type == c else 0.0
    
    
    first_rain = enriched_data.get("first_rain_category", "late")
    model_input["first_rain_category_on_time"] = 1.0 if first_rain == "on_time" else 0.0
    
    
    last_rain = enriched_data.get("last_rain_category", "early")
    model_input["last_rain_category_late"] = 1.0 if last_rain == "late" else 0.0
    model_input["last_rain_category_on_time"] = 1.0 if last_rain == "on_time" else 0.0
    
    

    season = enriched_data.get("season", 2025)
    for year in [2017, 2018, 2019, 2020, 2021, 2022, 2023]:
        model_input[f"season_{year}"] = 1.0 if season == year else 0.0
    

    FEATURE_ORDER = [
        "Area", "mean_dry_spell_length", "total_rainfall", "rainy_days",
        "average_tmax", "average_tmin", "average_humidity", "T(base)", "GDD",
        
        "district_Amreli", "district_Anand", "district_Aravalli", "district_Banaskantha",
        "district_Bharuch", "district_Bhavnagar", "district_Botad", "district_Chhota Udaipur",
        "district_Dahod", "district_Dang", "district_Devbhumi Dwarka", "district_Gandhinagar",
        "district_Gir Somnath", "district_Jamnagar", "district_Junagadh", "district_Kheda",
        "district_Kutch", "district_Mahisagar", "district_Mehsana", "district_Morbi",
        "district_Narmada", "district_Navsari", "district_Panchmahal", "district_Patan",
        "district_Porbandar", "district_Rajkot", "district_Sabarkantha", "district_Surat",
        "district_Surendranagar", "district_Tapi", "district_Vadodara", "district_Valsad",
        
        "Crop Type_TOTAL BAJRA", "Crop Type_TOTAL COTTON (LINT)", "Crop Type_TOTAL GROUNDNUT",
        "Crop Type_TOTAL RICE", "Crop Type_TOTAL TOBACCO",
        
        "first_rain_category_on_time", "last_rain_category_late", "last_rain_category_on_time",
   
        "season_2017", "season_2018", "season_2019", "season_2020", "season_2021", 
        "season_2022", "season_2023"
    ]


    try:
        input_df = pd.DataFrame([model_input])
        
        input_df = input_df[FEATURE_ORDER]
        
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
        "crop_values": crops,  
        "total": len(crops)
    }


@app.get("/weather/{district}", tags=["Weather"])
async def get_district_weather(district: str):
 
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