"""
score.py — Airbnb Price Prediction Scoring Script
--------------------------------------------------
Loads saved model artifacts and predicts price for new raw listings.

Required files (must be in the same directory):
    - airbnb_price_model.pkl   : trained FLAML model
    - model_features.pkl       : list of feature names
    - airbnb_pca.pkl           : fitted PCA for text embeddings
    - airbnb_medians.pkl       : median values for imputation

Usage:
    python score.py                         # runs on built-in sample
    python score.py --input new_data.csv    # runs on a CSV file
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
from textblob import TextBlob
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# 1. LOAD SAVED ARTIFACTS
# ─────────────────────────────────────────────────────────────

print("Loading model artifacts...")
model         = joblib.load("airbnb_price_model.pkl")
model_features = joblib.load("model_features.pkl")
pca           = joblib.load("airbnb_pca.pkl")
medians       = joblib.load("airbnb_medians.pkl")
print("✓ All artifacts loaded\n")


# ─────────────────────────────────────────────────────────────
# 2. PREPROCESSING FUNCTION  (mirrors the training notebook)
# ─────────────────────────────────────────────────────────────

def preprocess(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    # --- price / fee columns ---
    for col in ["price", "cleaning_fee"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                       .str.replace("$", "", regex=False)
                       .str.replace(",", "", regex=False)
                       .pipe(pd.to_numeric, errors="coerce")
            )
    df["cleaning_fee"] = df.get("cleaning_fee", pd.Series(0, index=df.index)).fillna(0)

    # --- percentage columns ---
    for col in ["host_response_rate", "host_acceptance_rate"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                       .str.replace("%", "", regex=False)
                       .pipe(pd.to_numeric, errors="coerce")
            )

    # --- boolean columns ---
    bool_cols = [
        "host_is_superhost", "has_availability", "instant_bookable",
        "require_guest_profile_picture", "require_guest_phone_verification",
        "is_location_exact", "requires_license",
        "host_has_profile_pic", "host_identity_verified",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = (df[col] == "t").astype(int)

    # --- numeric conversions ---
    for col in ["bathrooms", "bedrooms", "beds", "accommodates"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # --- median imputation ---
    for col, med in medians.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(med)

    # --- reviews_per_month ---
    if "reviews_per_month" in df.columns:
        df["reviews_per_month"] = pd.to_numeric(df["reviews_per_month"], errors="coerce").fillna(0)

    # fill remaining numeric NaNs with 0
    df = df.fillna(0)

    # ── property_type grouping ──────────────────────────────
    if "property_type" in df.columns:
        small_types = ["Loft", "Bed & Breakfast", "Cabin"]
        df["property_type"] = df["property_type"].replace(small_types, "Other")

    # ── one-hot encoding ────────────────────────────────────
    for col in ["property_type", "room_type", "bed_type"]:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col], drop_first=True)

    # ── text features (TextBlob) ────────────────────────────
    df["description"] = df.get("description", pd.Series("", index=df.index)).fillna("").astype(str)
    df["space"]       = df.get("space",       pd.Series("", index=df.index)).fillna("").astype(str)

    df["desc_length"]    = df["description"].str.len()
    df["desc_word_count"] = df["description"].str.split().str.len()
    df["desc_sentiment"] = df["description"].apply(lambda x: TextBlob(x).sentiment.polarity)
    df["space_length"]   = df["space"].str.len()
    df["space_word_count"] = df["space"].str.split().str.len()

    # ── sentence embeddings + PCA ───────────────────────────
    print("  Generating text embeddings (this takes ~10 s)...")
    embedder = SentenceTransformer("all-mpnet-base-v2")
    combined = (df["description"] + " " + df["space"]).values
    embeddings = embedder.encode(combined, show_progress_bar=False)
    reduced    = pca.transform(embeddings)          # ← uses SAVED PCA (no re-fitting)
    for i in range(reduced.shape[1]):
        df[f"embed_pc{i}"] = reduced[:, i]

    # ── location features ───────────────────────────────────
    seattle_lat, seattle_lng = 47.6062, -122.3321
    if "latitude" in df.columns and "longitude" in df.columns:
        df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce").fillna(seattle_lat)
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce").fillna(seattle_lng)
        df["distance_from_center"] = np.sqrt(
            (df["latitude"] - seattle_lat) ** 2 +
            (df["longitude"] - seattle_lng) ** 2
        )
        df["lat_zone"]  = pd.cut(df["latitude"],  bins=5, labels=False)
        df["lng_zone"]  = pd.cut(df["longitude"], bins=5, labels=False)
        df["near_water"] = (df["latitude"] > 47.65).astype(int)
        df["location_quality"] = df["distance_from_center"] * df.get("bedrooms", 0)
    else:
        for col in ["distance_from_center", "lat_zone", "lng_zone", "near_water", "location_quality"]:
            df[col] = 0

    # ── interaction features ────────────────────────────────
    df["host_since"] = pd.to_datetime(df.get("host_since", pd.NaT), errors="coerce")
    df["host_days_active"] = (pd.Timestamp.now() - df["host_since"]).dt.days.fillna(0)

    df["price_per_guest"]      = df.get("price", 0) / (df.get("guests_included", 0) + 1)
    df["price_per_bed"]        = df.get("price", 0) / (df.get("beds", 0) + 1)
    df["superhost_listings"]   = df.get("host_is_superhost", 0) * df.get("host_listings_count", 0)
    df["host_credibility"]     = df.get("host_response_rate", 0) * df.get("host_acceptance_rate", 0)
    df["superhost_experience"] = df.get("host_is_superhost", 0) * df["host_days_active"]
    df["property_quality"]     = df.get("bedrooms", 0) * df.get("cleaning_fee", 0)

    # ── align to training feature set ──────────────────────
    for col in model_features:
        if col not in df.columns:
            df[col] = 0          # add any missing dummies as 0

    return df[model_features]


# ─────────────────────────────────────────────────────────────
# 3. PREDICT FUNCTION
# ─────────────────────────────────────────────────────────────

def predict_price(raw_df: pd.DataFrame) -> np.ndarray:
    """
    Takes a raw DataFrame (same columns as original listings.csv)
    and returns predicted price(s) in USD.
    """
    X = preprocess(raw_df)
    return model.predict(X)


# ─────────────────────────────────────────────────────────────
# 4. MAIN — demo or CSV input
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="Path to new raw CSV file")
    args = parser.parse_args()

    if args.input:
        print(f"Loading data from: {args.input}")
        raw = pd.read_csv(args.input)
    else:
        # Built-in sample (one listing)
        raw = pd.DataFrame([{
            "host_response_rate":    "100%",
            "host_acceptance_rate":  "95%",
            "host_is_superhost":     "t",
            "host_listings_count":   1,
            "host_has_profile_pic":  "t",
            "host_identity_verified":"t",
            "host_since":            "2015-06-01",
            "latitude":              47.6200,
            "longitude":            -122.3500,
            "accommodates":          4,
            "bathrooms":             1,
            "bedrooms":              2,
            "beds":                  2,
            "cleaning_fee":         "$50",
            "guests_included":       2,
            "minimum_nights":        2,
            "maximum_nights":        30,
            "instant_bookable":      "f",
            "require_guest_profile_picture": "f",
            "require_guest_phone_verification": "f",
            "availability_30":       10,
            "availability_365":      120,
            "property_type":         "House",
            "room_type":             "Entire home/apt",
            "bed_type":              "Real Bed",
            "description":           "Cozy 2-bedroom house in Capitol Hill, great location.",
            "space":                 "Open living room, full kitchen, backyard.",
            "review_scores_rating":  96,
            "number_of_reviews":     45,
            "reviews_per_month":     1.5,
        }])
        print("Running on built-in sample listing...\n")

    print("Preprocessing...")
    predictions = predict_price(raw)

    print("\n" + "="*50)
    print("PREDICTED PRICE(S)")
    print("="*50)
    for i, p in enumerate(predictions):
        print(f"  Listing {i+1}: ${p:.2f} / night")