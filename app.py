import pandas as pd
import numpy as np
import datetime
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

# Enable CORS for Expo/Mobile access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 1. GLOBAL ENGINE (Trains on Startup)
# ==========================================
print("Loading data and training model... please wait.")
df = pd.read_csv("daily-shelter-overnight-service-occupancy-capacity-2024.csv")
df["OCCUPANCY_DATE"] = pd.to_datetime(df["OCCUPANCY_DATE"])
df["available_beds"] = df["UNOCCUPIED_BEDS"].fillna(0)

# Feature Prep
df["month"] = df["OCCUPANCY_DATE"].dt.month
df["day_of_week"] = df["OCCUPANCY_DATE"].dt.dayofweek
df["capacity"] = df["CAPACITY_ACTUAL_BED"].fillna(0)

# Averages and Encoders
shelter_means = df.groupby("SHELTER_ID")["available_beds"].mean()
df["shelter_avg_avail"] = df["SHELTER_ID"].map(shelter_means)

le_sector = LabelEncoder()
le_model = LabelEncoder()
df["SECTOR_ENC"] = le_sector.fit_transform(df["SECTOR"].astype(str))
df["MODEL_ENC"] = le_model.fit_transform(df["PROGRAM_MODEL"].astype(str))

# Train the model
features = ["month", "day_of_week", "capacity", "SECTOR_ENC", "MODEL_ENC", "shelter_avg_avail"]
final_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
final_model.fit(df[features], df["available_beds"])

# Create a "Registry" of shelters to reference names/addresses easily
shelter_registry = df.sort_values('OCCUPANCY_DATE').groupby('SHELTER_ID').tail(1).copy()
print("Model trained and ready!")

# ==========================================
# 2. API ENDPOINTS
# ==========================================

@app.get("/forecast")
def get_shelter_forecast(date: str = None, sector: str = None):
    # 1. Handle Date (Default to tomorrow if none provided)
    if date:
        try:
            target_date = pd.to_datetime(date)
        except:
            return {"error": "Invalid date format. Use YYYY-MM-DD"}
    else:
        target_date = pd.Timestamp.now() + pd.Timedelta(days=1)

    forecast_month = target_date.month
    forecast_day = target_date.dayofweek
    
    # 2. Filter Registry
    current_registry = shelter_registry.copy()
    
    # If the app sends a sector (e.g., "Women"), filter the list
    if sector and sector.lower() != "all":
        current_registry = current_registry[current_registry['SECTOR'].str.contains(sector, case=False, na=False)]

    results = []
    
    for _, s in current_registry.iterrows():
        # Match the input features to what the model was trained on
        input_data = pd.DataFrame([{
            "month": forecast_month,
            "day_of_week": forecast_day,
            "capacity": s['CAPACITY_ACTUAL_BED'] if 'CAPACITY_ACTUAL_BED' in s else 50,
            "SECTOR_ENC": s['SECTOR_ENC'],
            "MODEL_ENC": s['MODEL_ENC'],
            "shelter_avg_avail": shelter_means.get(s['SHELTER_ID'], 0)
        }])
        
        raw_pred = final_model.predict(input_data)[0]
        predicted_beds = int(np.ceil(max(0, raw_pred - 0.1)))
        
        # Use your new status logic
        status = "🔴 FULL"
        if predicted_beds > 5: status = "🟢 HIGH CHANCE"
        elif predicted_beds > 0: status = "🟡 LIMITED"

        results.append({
            "shelter_id": int(s['SHELTER_ID']),
            "shelter_name": str(s['ORGANIZATION_NAME']), # Fixed column name
            "address": str(s['LOCATION_ADDRESS']),
            "sector": str(s['SECTOR']),
            "predicted_beds": predicted_beds,
            "status": status,
            "date": target_date.strftime('%Y-%m-%d')
        })
    # Sort so shelters with beds appear at the top
    return sorted(results, key=lambda x: x['predicted_beds'], reverse=True)

@app.get("/")
def home():
    return {"message": "Toronto Shelter API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)