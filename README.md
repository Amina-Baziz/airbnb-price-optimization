# Airbnb Price Optimization — Seattle

## Overview
An end-to-end Business Analytics and Machine Learning pipeline to predict and optimize Airbnb listing prices in Seattle, WA. Built as part of DSAI 4103 Business Analytics at UDST.

## Business Problem
Airbnb hosts often misprice their listings — either leaving money on the table or pricing themselves out of bookings. This project builds a predictive model to recommend optimal prices based on listing features, location, and host characteristics.

## Dataset
- Source: Seattle Airbnb Open Data (Kaggle)
- 3 files: listings, calendar, reviews
- 3,754 listings after cleaning

## Model Performance
- Algorithm: FLAML AutoML
- R² Score: 0.6645
- RMSE: $39.85

## Project Structure
- `BA_code.ipynb` — Full pipeline (EDA, feature engineering, modeling, SHAP, bias analysis)
- `score.py` — Scoring script for deployment (loads model without retraining)
- `airbnb_price_model.pkl` — Trained FLAML model
- `model_features.pkl` — Selected feature names
- `airbnb_medians.pkl` — Median values for imputation
- `airbnb_pca.pkl` — PCA for text embeddings
- `fixed.csv` — Cleaned dataset with predictions

## How to Run Scoring Script
```bash
pip install flaml joblib pandas numpy scikit-learn
python score.py
```

## Key Features
- AutoML with FLAML
- SHAP explainability
- Fairness/bias audit
- Live scoring pipeline
- Tableau dashboard with Pricing Efficiency Map

## Tools Used
Python, FLAML, SHAP, Pandas, Scikit-learn, Tableau