import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =========================
# Load saved files
# =========================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")
cm = joblib.load("conf_matrix.pkl")

# =========================
# Page setup
# =========================
st.set_page_config(page_title="Flight Delay Predictor", layout="wide")

st.title("Flight Delay Prediction Tool")
st.write("Enter airport traffic and weather conditions below to predict the likelihood of a major delay event.")

# =========================
# Sidebar inputs
# =========================
st.sidebar.header("Input Features")

# Start with all features as 0
input_data = {feature: 0.0 for feature in features}

# User-friendly inputs for key features
if "AAR_CAPACITY" in input_data:
    input_data["AAR_CAPACITY"] = st.sidebar.number_input("Arrival Capacity", value=40.0)

if "visibility" in input_data:
    input_data["visibility"] = st.sidebar.number_input("Visibility", value=10.0)

if "wind_speed" in input_data:
    input_data["wind_speed"] = st.sidebar.number_input("Wind Speed", value=10.0)

if "wind_gust" in input_data:
    input_data["wind_gust"] = st.sidebar.number_input("Wind Gust", value=15.0)

if "cross_winds" in input_data:
    input_data["cross_winds"] = st.sidebar.number_input("Crosswinds", value=5.0)

if "headtail_winds" in input_data:
    input_data["headtail_winds"] = st.sidebar.number_input("Head/Tail Winds", value=5.0)

if "hour" in input_data:
    input_data["hour"] = st.sidebar.slider("Hour of Day", 0, 23, 12)

if "dayofweek" in input_data:
    input_data["dayofweek"] = st.sidebar.slider("Day of Week", 0, 6, 2)

if "cld<500" in input_data:
    input_data["cld<500"] = st.sidebar.number_input("Clouds Below 500 ft", value=0.0)

if "cld 500-999" in input_data:
    input_data["cld 500-999"] = st.sidebar.number_input("Clouds 500–999 ft", value=0.0)

if "cld 1000-3000" in input_data:
    input_data["cld 1000-3000"] = st.sidebar.number_input("Clouds 1000–3000 ft", value=0.0)

if "COUNT_ARR_SCH" in input_data:
    input_data["COUNT_ARR_SCH"] = st.sidebar.number_input("Scheduled Arrivals", value=35.0)

# =========================
# Feature engineering
# =========================
# Make dataframe
input_df = pd.DataFrame([input_data])

# Recreate engineered features if they exist in training set
if "excess_demand" in input_df.columns and "AAR_CAPACITY" in input_df.columns and "COUNT_ARR_SCH" in input_df.columns:
    input_df["excess_demand"] = input_df["AAR_CAPACITY"] - input_df["COUNT_ARR_SCH"]

if "visibility_demand" in input_df.columns and "excess_demand" in input_df.columns and "visibility" in input_df.columns:
    input_df["visibility_demand"] = input_df["excess_demand"] / (input_df["visibility"] + 0.0001)

if "vissibility_demand" in input_df.columns and "excess_demand" in input_df.columns and "visibility" in input_df.columns:
    input_df["vissibility_demand"] = input_df["excess_demand"] / (input_df["visibility"] + 0.0001)

if "ceiling_sum" in input_df.columns:
    c1 = input_df["cld<500"] if "cld<500" in input_df.columns else 0
    c2 = input_df["cld 500-999"] if "cld 500-999" in input_df.columns else 0
    c3 = input_df["cld 1000-3000"] if "cld 1000-3000" in input_df.columns else 0
    input_df["ceiling_sum"] = c1 + c2 + c3

if "wind_gust_squared" in input_df.columns and "wind_gust" in input_df.columns:
    input_df["wind_gust_squared"] = input_df["wind_gust"] ** 2

for feature in ["cross_winds", "headtail_winds", "wind_speed"]:
    log_name = f"{feature}_log"
    if log_name in input_df.columns and feature in input_df.columns:
        input_df[log_name] = np.log1p(np.clip(input_df[feature], a_min=0, a_max=None))

# Make sure all expected columns exist and are in the right order
for col in features:
    if col not in input_df.columns:
        input_df[col] = 0.0

input_df = input_df[features]

# Final cleanup
input_df = input_df.replace([np.inf, -np.inf], np.nan).fillna(0)

# =========================
# Predict
# =========================
if st.button("Predict"):
    input_scaled = scaler.transform(input_df)
    prob = model.predict_proba(input_scaled)[0][1]
    pred = int(prob >= 0.5)

    st.subheader("Prediction Result")
    st.write(f"**Predicted probability of major delay:** {prob:.3f}")

    if pred == 1:
        st.error("High likelihood of a major delay event.")
    else:
        st.success("Low likelihood of a major delay event.")

# =========================
# Show confusion matrix
# =========================
st.subheader("Overall Model Confusion Matrix")

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, interpolation="nearest")
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["No Delay", "Delay"])
ax.set_yticklabels(["No Delay", "Delay"])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center")

st.pyplot(fig)