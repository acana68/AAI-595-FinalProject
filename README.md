# ERTime — Emergency Room Wait Time Predictor

**AAI 595 - Applied Machine Learning**  
Michael Lugo · Amoy Mowatt · Ardit Cana · Wesley Nabo

---

## Overview

ERTime predicts emergency room wait times using a **Random Forest Regressor** trained on ER patient data. The app helps users find the nearest ER with the shortest total time to care (travel + predicted wait), and includes a manual prediction tool for exploring how facility and patient factors affect wait times.

## Features

- **Find ER Near Me** — Enter your address or ZIP code to get ranked nearby ERs with predicted wait times and an interactive map
- **Manual Prediction** — Adjust urgency level, region, time of day, facility size, and staffing to instantly predict wait time
- **Model Comparison** — Random Forest vs. Linear Regression baseline (64% reduction in MAE)

## Project Structure

```
AAI595-Final/
├── app.py                  # Streamlit web application
├── ertime_randomforest.py  # Model training, evaluation, and visualizations
├── cleaned_data.csv        # Preprocessed ER dataset
└── erhallway.jpg           # Supporting image asset
```

## Model

We trained and compared two models on the same dataset and train/test split (80/20, `random_state=42`):

### Linear Regression (Baseline)
Linear Regression was tested first as a baseline to understand how well a simple model could predict ER wait times. It captured general trends but struggled with the non-linear relationships between features like urgency level and staffing ratios.

### Random Forest Regressor
The Random Forest model was then trained as the primary model. It uses 100 decision trees and was evaluated using the same features and split as the baseline for a fair comparison. Performance was further validated with 5-Fold Cross-Validation to ensure the results generalize beyond the test set.

| Metric | Linear Regression (Baseline) | Random Forest |
|--------|------------------------------|---------------|
| R²     | 0.5348                       | ~0.85+        |
| MAE    | 34.81 min                    | ~12.5 min     |

Random Forest reduced prediction error by **~64%** compared to the Linear Regression baseline.

### Testing Methodology
- **Train/Test Split:** 80% training, 20% testing
- **Cross-Validation:** 5-Fold CV with R² scoring to check for overfitting
- **Learning Curves:** Generated to diagnose bias vs. variance
- **Metrics reported:** MAE, MSE, RMSE, R²

**Features used (no data leakage — all available at patient arrival):**
- Region, Day of Week, Season, Time of Day, Urgency Level
- Nurse-to-Patient Ratio, Specialist Availability, Facility Size (Beds)

> Process durations (Time to Registration, Time to Triage, Time to Medical Professional) were intentionally excluded — they sum directly to the target variable and would cause data leakage.

## Setup

```bash
pip install streamlit pandas numpy scikit-learn folium streamlit-folium geopy requests matplotlib seaborn
```

## Run the App

```bash
streamlit run app.py
```

## Run the Model Script

```bash
python ertime_randomforest.py
```
