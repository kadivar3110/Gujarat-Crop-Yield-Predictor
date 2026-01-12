import streamlit as st
import requests

st.set_page_config(page_title="Crop Yield Predictor", layout="wide")
st.title("Crop Yield Predictor")

st.write("""
This is a simple web app to predict crop yield for Gujarat districts using weather data and a machine learning model. Please fill in the details below and click Predict.
""")

API_URL = "http://13.60.224.58:8000"

@st.cache_data
def get_districts():
    try:
        resp = requests.get(f"{API_URL}/districts")
        if resp.status_code == 200:
            return resp.json()["districts"]
        else:
            return []
    except Exception:
        return []

districts = get_districts()

selected_district = st.selectbox("Select District", districts)

@st.cache_data
def get_crops(district):
    try:
        resp = requests.get(f"{API_URL}/crops/{district}")

        if resp.status_code == 200:
            return resp.json()["crop_values"]
        else:
            return []
    except Exception:
        return []

crops = get_crops(selected_district) if selected_district else []
selected_crop = st.selectbox("Select Crop", crops)


area = st.number_input("Area (hectares)", min_value=0.1, value=1.0, step=0.1)

if 'prediction_result' not in st.session_state:
    st.session_state['prediction_result'] = None

if 'prediction_error' not in st.session_state:
    st.session_state['prediction_error'] = None

if st.button("Predict"):

    st.session_state['district_yields'] = None
    st.session_state['map_crop'] = None
    st.session_state['show_map'] = False

    if not selected_district or not selected_crop or area <= 0:
        st.session_state['prediction_result'] = None
        st.session_state['prediction_error'] = "Please fill all fields correctly."
    else:
        with st.spinner("Predicting..."):
            payload = {
                "district": selected_district,
                "crop": selected_crop,
                "area": area
            }

            try:
                resp = requests.post(f"{API_URL}/predict", json=payload)
                if resp.status_code == 200:
                    result = resp.json()
                    st.session_state['prediction_result'] = result
                    st.session_state['prediction_error'] = None
                else:
                    st.session_state['prediction_result'] = None
                    st.session_state['prediction_error'] = f"Error: {resp.json().get('detail', 'Unknown error')}"
            except Exception as e:
                st.session_state['prediction_result'] = None
                st.session_state['prediction_error'] = f"Request failed: {e}"



if st.session_state.get('prediction_result'):
    result = st.session_state['prediction_result']

    st.success(f"Predicted Yield: {result['prediction']} {result['unit']}")
    st.write("Weather Data:")
    st.json(result["weather_data"])

elif st.session_state.get('prediction_error'):
    st.error(st.session_state['prediction_error'])


## AI
# --- Map Visualization Imports ---
import json
import folium
from streamlit_folium import st_folium
import time
import pandas as pd

st.write("\n---\n")

# --- Gujarat Map Crop Yield Visualization (Beginner Style) ---
st.header("Gujarat District Map: Crop Yield Prediction")

# Load GeoJSON (no caching, just open file)
try:
    with open("gujarat.geojson", "r", encoding="utf-8") as f:
        geojson_data = json.load(f)
except Exception as e:
    st.error(f"Could not load Gujarat GeoJSON: {e}")
    geojson_data = None


# --- Session state for map persistence ---
if 'district_yields' not in st.session_state:
    st.session_state['district_yields'] = None
if 'map_crop' not in st.session_state:
    st.session_state['map_crop'] = None
if 'show_map' not in st.session_state:
    st.session_state['show_map'] = False

# Use the crop selected above for both prediction and map
selected_map_crop = selected_crop
default_area = area


if st.button("Show Gujarat Map for Selected Crop"):
    if not geojson_data:
        st.error("GeoJSON not loaded!")
    elif not selected_map_crop:
        st.warning("Please select a crop.")
    else:
        st.write("Predicting yield for all districts... (this may take a while)")
        district_yields = {}
        for feature in geojson_data["features"]:
            district = feature["properties"]["district"].strip().title()
            payload = {"district": district, "crop": selected_map_crop, "area": default_area}
            try:
                resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                if resp.status_code == 200:
                    pred = resp.json()["prediction"]
                else:
                    pred = None
            except Exception:
                pred = None
            district_yields[district] = pred
            time.sleep(0.1)
        st.session_state['district_yields'] = district_yields
        st.session_state['map_crop'] = selected_map_crop
        st.session_state['show_map'] = True

if st.session_state.get('district_yields') and st.session_state.get('show_map'):
    st.write(f"### Predicted Yield Map for Crop: {st.session_state['map_crop']}")
    # --- Map rendering block ---
    district_yields = st.session_state['district_yields']
    selected_map_crop = st.session_state.get('map_crop', selected_map_crop)
    # Add yield to geojson for coloring
    for feature in geojson_data["features"]:
        district = feature["properties"]["district"].strip().title()
        feature["properties"]["pred_yield"] = district_yields.get(district)
    m = folium.Map(location=[22.5, 72.5], zoom_start=7)
    choropleth_data = []
    for d, y in district_yields.items():
        try:
            val = float(y)
            if val != val:
                continue
        except Exception:
            continue
        choropleth_data.append({"district": d, "yield": val})
    if len(choropleth_data) == 0:
        st.error("No valid yield data to display on map.")
    else:
        df = pd.DataFrame(choropleth_data)
        df["yield"] = pd.to_numeric(df["yield"], errors="coerce")
        df = df.dropna(subset=["yield"])
        df["yield"] = df["yield"].astype(float)
        try:
            import branca.colormap as bcm
        except Exception:
            bcm = None
        values = df["yield"].tolist()
        vmin = min(values)
        vmax = max(values)
        if bcm is not None and vmin < vmax:
            colormap = bcm.LinearColormap(["green", "yellow", "red"], vmin=vmin, vmax=vmax)
        else:
            colormap = None
        yield_lookup = {row["district"].strip().title(): float(row["yield"]) for _, row in df.iterrows()}
        def style_function(feature):
            d = feature["properties"].get("district", "").strip().title()
            val = yield_lookup.get(d)
            if val is None or colormap is None:
                color = "gray"
            else:
                try:
                    color = colormap(val)
                except Exception:
                    color = "gray"
            return {
                "fillColor": color,
                "color": "black",
                "weight": 0.5,
                "fillOpacity": 0.7,
            }
        folium.GeoJson(geojson_data, style_function=style_function, name="Yield").add_to(m)
        if colormap is not None:
            colormap.caption = "Predicted Yield (kg/hectare)"
            colormap.add_to(m)
    # Add markers
    def extract_points(coords):
        pts = []
        if coords is None:
            return pts
        def traverse(c):
            if isinstance(c, (list, tuple)) and len(c) > 0:
                if isinstance(c[0], (list, tuple)):
                    for item in c:
                        traverse(item)
                else:
                    try:
                        lon = float(c[0])
                        lat = float(c[1])
                        pts.append((lon, lat))
                    except Exception:
                        return
        traverse(coords)
        return pts
    for feature in geojson_data["features"]:
        district = feature["properties"]["district"]
        coords_raw = feature["geometry"].get("coordinates")
        pts = extract_points(coords_raw)
        if not pts:
            continue
        avg_lon = sum(p[0] for p in pts) / len(pts)
        avg_lat = sum(p[1] for p in pts) / len(pts)
        try:
            folium.Marker(
                location=[float(avg_lat), float(avg_lon)],
                popup=f"{district}: {feature['properties'].get('pred_yield', 'N/A')}",
                icon=folium.Icon(icon="info-sign", color="blue", icon_color="white")
            ).add_to(m)
        except Exception:
            continue
    st_folium(m, width=800, height=600, key="gujarat_map")