"""
score.py — Airbnb Price Prediction Scoring Script
Loads saved model, predicts price for new rows in fixed.csv,
and saves results back so Tableau dashboard refreshes live.

Usage:
    python score.py
"""

import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

# ── 1. LOAD ARTIFACTS ──────────────────────────────────────
print("Loading model artifacts...")
model          = joblib.load("airbnb_price_model.pkl")
model_features = joblib.load("model_features.pkl")
medians        = joblib.load("airbnb_medians.pkl")
print("✓ All artifacts loaded\n")

# ── 2. READ DATA ────────────────────────────────────────────
df = pd.read_csv("fixed.csv")
print(f"Loaded fixed.csv: {df.shape[0]} rows, {df.shape[1]} columns")

# ── 3. FIND ROWS NEEDING PREDICTION ────────────────────────
new_rows = df["predicted_price"].isna()
print(f"Rows needing prediction: {new_rows.sum()}")

if new_rows.sum() == 0:
    print("No new rows to predict. Add a row with empty predicted_price to fixed.csv and run again.")
else:
    # ── 4. PREPROCESS ───────────────────────────────────────
    def preprocess(raw):
        df2 = raw.copy()

        # price/fee cleanup
        for col in ["price", "cleaning_fee"]:
            if col in df2.columns:
                df2[col] = (df2[col].astype(str)
                            .str.replace("$", "", regex=False)
                            .str.replace(",", "", regex=False)
                            .pipe(pd.to_numeric, errors="coerce"))
        df2["cleaning_fee"] = df2.get("cleaning_fee", pd.Series(0)).fillna(0)

        # percentage columns
        for col in ["host_response_rate", "host_acceptance_rate"]:
            if col in df2.columns:
                df2[col] = (df2[col].astype(str)
                            .str.replace("%", "", regex=False)
                            .pipe(pd.to_numeric, errors="coerce"))

        # boolean columns
        bool_cols = ["host_is_superhost", "instant_bookable",
                     "require_guest_profile_picture",
                     "require_guest_phone_verification",
                     "host_has_profile_pic", "host_identity_verified"]
        for col in bool_cols:
            if col in df2.columns:
                df2[col] = (df2[col] == "t").astype(int)

        # numeric conversions
        for col in ["bathrooms", "bedrooms", "beds", "accommodates"]:
            if col in df2.columns:
                df2[col] = pd.to_numeric(df2[col], errors="coerce").fillna(0).astype(int)

        # median imputation
        if isinstance(medians, dict):
            for col, med in medians.items():
                if col in df2.columns:
                    df2[col] = pd.to_numeric(df2[col], errors="coerce").fillna(med)

        df2 = df2.fillna(0)

        # one-hot encoding
        for col in ["property_type", "room_type", "bed_type"]:
            if col in df2.columns:
                df2 = pd.get_dummies(df2, columns=[col], drop_first=True)

        # location features
        seattle_lat, seattle_lng = 47.6062, -122.3321
        if "latitude" in df2.columns and "longitude" in df2.columns:
            df2["distance_from_center"] = np.sqrt(
                (df2["latitude"] - seattle_lat) ** 2 +
                (df2["longitude"] - seattle_lng) ** 2)
            df2["lat_zone"]  = pd.cut(df2["latitude"],  bins=5, labels=False)
            df2["lng_zone"]  = pd.cut(df2["longitude"], bins=5, labels=False)
            df2["near_water"] = (df2["latitude"] > 47.65).astype(int)
            df2["location_quality"] = df2["distance_from_center"] * df2.get("bedrooms", 0)

        # interaction features
        df2["host_since"] = pd.to_datetime(df2.get("host_since", pd.NaT), errors="coerce")
        df2["host_days_active"] = (pd.Timestamp.now() - df2["host_since"]).dt.days.fillna(0)
        df2["price_per_guest"]      = df2.get("price", 0) / (df2.get("guests_included", 0) + 1)
        df2["price_per_bed"]        = df2.get("price", 0) / (df2.get("beds", 0) + 1)
        df2["host_credibility"]     = df2.get("host_response_rate", 0) * df2.get("host_acceptance_rate", 0)
        df2["superhost_experience"] = df2.get("host_is_superhost", 0) * df2["host_days_active"]
        df2["property_quality"]     = df2.get("bedrooms", 0) * df2.get("cleaning_fee", 0)

        # align to model features (fill missing with 0)
        for col in model_features:
            if col not in df2.columns:
                df2[col] = 0

        return df2[model_features]

    X = preprocess(df[new_rows])
    predictions = model.predict(X)
    df.loc[new_rows, "predicted_price"] = predictions

    # ── 5. SAVE BACK ────────────────────────────────────────
    df.to_csv("fixed.csv", index=False)
    print("\n" + "="*50)
    print("PREDICTED PRICE(S)")
    print("="*50)
    for i, p in enumerate(predictions):
        print(f"  New listing {i+1}: ${p:.2f} / night")
    print("\n✓ fixed.csv updated! Refresh Tableau to see new predictions.")