import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle
import os
import requests
from prophet import Prophet

# ✅ GitHub Raw URLs
GITHUB_CSV_URL = "https://raw.githubusercontent.com/Sindhu-0716/PrepVector_ML_Sindhu/main/EnergyPredictor/PJME_hourly.csv"
GITHUB_MODEL_URL = "https://raw.githubusercontent.com/Sindhu-0716/PrepVector_ML_Sindhu/main/EnergyPredictor/prophet_model.pkl"

# ✅ Define local file paths
csv_file = "PJME_hourly.csv"
model_file = "prophet_model.pkl"

# ✅ Function to download files
def download_file(url, save_path):
    if not os.path.exists(save_path):
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(save_path, "wb") as file:
                file.write(response.content)
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Failed to download {url}: {e}")
            st.stop()

# ✅ Load Data Function (Mini Version)
def get_data():
    download_file(GITHUB_CSV_URL, csv_file)  # Ensure CSV is available
    try:
        df = pd.read_csv(csv_file)
        df.rename(columns={"PJME_MW": "y"}, inplace=True)
        df.rename(columns={df.columns[0]: 'ds'}, inplace=True)
        df["ds"] = pd.to_datetime(df["ds"])
        return df.tail(5000)  # Load only last 5000 rows for efficiency
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# ✅ Set Page Configuration
st.set_page_config(
    page_title="Mini Demand Forecasting",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

# ✅ Title & Styling
st.markdown("<h1 style='text-align: center; color:#faa356;'>Mini Demand Forecasting 📉</h1>", unsafe_allow_html=True)

# ✅ Load & Show Raw Data
st.write("### Raw Data (Limited for Cloud)")
raw_data = get_data()
if raw_data is None:
    st.stop()
st.dataframe(raw_data.head())

# ✅ Download Model from GitHub (Mini Version)
download_file(GITHUB_MODEL_URL, model_file)

# ✅ Cache Model Loading to Reduce Memory Usage
@st.cache_resource
def load_model():
    with open(model_file, "rb") as file:
        return pickle.load(file)

# ✅ Load Prophet Model
model = load_model()

# ✅ Limited Forecasting (Mini Version)
days = st.slider("Select Forecasting Days", min_value=1, max_value=7, value=3)  # Limit to 7 days
forecast_hours = days * 24

future = model.make_future_dataframe(periods=forecast_hours, freq='H')
forecast = model.predict(future)

# ✅ Show Forecasted Data
st.write("### Forecasted Data (Mini Version)")
st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())

# ✅ Plot Forecast (Mini Version)
fig = go.Figure()

# 🔹 Plot actual data
fig.add_trace(go.Scatter(x=raw_data["ds"], y=raw_data["y"], mode='lines', name='Actual Demand', line=dict(color='blue', width=2)))

# 🔹 Plot forecast (Mini Version)
fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode='lines', name='Forecast', line=dict(color='orange', width=2)))

# 🔹 Confidence interval shading
fig.add_trace(go.Scatter(
    x=forecast["ds"].tolist() + forecast["ds"].tolist()[::-1], 
    y=forecast["yhat_upper"].tolist() + forecast["yhat_lower"].tolist()[::-1],
    fill='toself',
    fillcolor='rgba(255, 165, 0, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    name='Confidence Interval'
))

fig.update_layout(title="Mini Energy Demand Forecast", xaxis_title="Date", yaxis_title="Demand (MW)")
st.plotly_chart(fig)
