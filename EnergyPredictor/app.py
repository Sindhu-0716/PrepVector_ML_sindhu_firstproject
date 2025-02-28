import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import requests

# ✅ GitHub Raw URLs for Data
GITHUB_CSV_URL = "https://raw.githubusercontent.com/Sindhu-0716/PrepVector_ML_Sindhu/main/EnergyPredictor/PJME_hourly.csv"
GITHUB_FORECAST_URL = "https://raw.githubusercontent.com/Sindhu-0716/PrepVector_ML_Sindhu/main/EnergyPredictor/forecast.csv"

# ✅ Local file paths
csv_file = "PJME_hourly.csv"
forecast_file = "forecast.csv"

# ✅ Function to download files from GitHub
def download_file(url, save_path):
    """Downloads a file from a URL if it doesn't exist locally."""
    if not os.path.exists(save_path):
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(save_path, "wb") as file:
                file.write(response.content)
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Failed to download {url}: {e}")
            st.stop()

# ✅ Load Historical Data
def get_data():
    """Loads the raw energy demand data from GitHub."""
    download_file(GITHUB_CSV_URL, csv_file)
    try:
        df = pd.read_csv(csv_file)
        df.rename(columns={"PJME_MW": "y"}, inplace=True)
        df.rename(columns={df.columns[0]: 'ds'}, inplace=True)
        df["ds"] = pd.to_datetime(df["ds"], format='%Y-%m-%d %H:%M:%S')  # ✅ Ensure correct format
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# ✅ Load Precomputed Forecast Data
def get_forecast():
    """Loads the precomputed forecast data from GitHub."""
    download_file(GITHUB_FORECAST_URL, forecast_file)
    try:
        df = pd.read_csv(forecast_file)
        df["ds"] = pd.to_datetime(df["ds"])
        return df
    except Exception as e:
        st.error(f"Error loading forecast data: {e}")
        return None

# ✅ Set Page Configuration
st.set_page_config(
    page_title="Static Energy Demand Forecasting",
    page_icon=":bar_chart:",
    layout="wide",
)

# ✅ Title
st.markdown("<h1 style='text-align: center; color:#faa356;'>Static Energy Demand Forecasting 📉</h1>", unsafe_allow_html=True)

# ✅ Load & Show Raw Data
st.write("### 📊 Raw Energy Demand Data (2002 - 2018)")
raw_data = get_data()
if raw_data is None:
    st.stop()
st.dataframe(raw_data)

# ✅ Load & Show Forecast Data
st.write("### 🔮 Precomputed Forecast Data (2018 - 2023)")
forecast_data = get_forecast()
if forecast_data is None:
    st.stop()
st.dataframe(forecast_data)

# ✅ Plot Forecast
fig = go.Figure()

# 🔹 Plot actual data (showing full 2002-2018 range)
fig.add_trace(go.Scatter(
    x=raw_data["ds"],  # ✅ Show full range instead of limiting to last 1000 points
    y=raw_data["y"], 
    mode='lines', 
    name='Actual Demand', 
    line=dict(color='blue', width=2)
))

# 🔹 Plot precomputed forecast (full dataset)
fig.add_trace(go.Scatter(
    x=forecast_data["ds"], 
    y=forecast_data["yhat"], 
    mode='lines', 
    name='Forecast',  # ✅ Fixed syntax error
    line=dict(color='orange', width=2)
))

# 🔹 Confidence interval shading
fig.add_trace(go.Scatter(
    x=forecast_data["ds"].tolist() + forecast_data["ds"].tolist()[::-1], 
    y=forecast_data["yhat_upper"].tolist() + forecast_data["yhat_lower"].tolist()[::-1],
    fill='toself',
    fillcolor='rgba(255, 165, 0, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    name='Confidence Interval'
))

fig.update_layout(title="Energy Demand Forecast", xaxis_title="Date", yaxis_title="Demand (MW)")
st.plotly_chart(fig)