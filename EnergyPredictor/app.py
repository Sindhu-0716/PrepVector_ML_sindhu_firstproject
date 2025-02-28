import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle
import os
import requests
from prophet import Prophet

# ✅ GitHub Raw URL for the CSV file (Update if Repo Changes)
GITHUB_CSV_URL = "https://raw.githubusercontent.com/Sindhu-0716/PrepVector_ML_Sindhu/main/EnergyPredictor/PJME_hourly.csv"

# ✅ Define local file paths
csv_file = "PJME_hourly.csv"
model_file = "prophet_model.pkl"

# ✅ Function to download CSV if missing (for Streamlit Cloud)
def download_csv():
    if not os.path.exists(csv_file):
        st.info(f"📥 Downloading CSV from GitHub: {GITHUB_CSV_URL}")
        try:
            response = requests.get(GITHUB_CSV_URL)
            response.raise_for_status()  # Raise error if request fails
            with open(csv_file, "wb") as file:
                file.write(response.content)  # Save CSV locally
            st.success("✅ File downloaded successfully!")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Failed to download file: {e}")
            st.stop()

# ✅ Load Data Function
def get_data():
    if not os.path.exists(csv_file):
        download_csv()  # Download if missing

    try:
        data = pd.read_csv(csv_file)

        # Ensure correct column names
        if "PJME_MW" in data.columns:
            data.rename(columns={"PJME_MW": "y"}, inplace=True)

        # Convert first column to datetime
        if 'ds' in data.columns:
            data['ds'] = pd.to_datetime(data['ds'])
        else:
            data.rename(columns={data.columns[0]: 'ds'}, inplace=True)
            data['ds'] = pd.to_datetime(data['ds'])

        # Validate required columns
        if 'ds' not in data.columns or 'y' not in data.columns:
            st.error("❌ Missing required columns 'ds' and 'y'. Ensure CSV format is correct.")
            return None

        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# ✅ Set Page Configuration
st.set_page_config(
    page_title="Demand Forecasting",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

# ✅ Title & Styling
st.markdown("""
    <h1 style='text-align: center; color:#faa356;'>Demand Forecasting 📈</h1>
    <h2 style='color:#a2d2fb;'>Problem Statement</h2>
    <p>Energy demand forecasting is crucial for effective resource planning and management in the power sector. 
    This app provides a user-friendly interface for forecasting electricity demand.</p>
    """, unsafe_allow_html=True)

# ✅ Load & Show Raw Data
st.write("### Raw Data")
raw_data = get_data()

# 🛑 Stop if data is missing
if raw_data is None:
    st.stop()

# ✅ Display First Few Rows
st.dataframe(raw_data.head())

# ✅ Model Handling
st.markdown("<h2 style='color:#a2d2fb;'>Energy Demand Forecasting</h2>", unsafe_allow_html=True)

# ✅ Check if Model Exists
if not os.path.exists(model_file):
    st.warning(f"⚠️ Model file not found: {model_file}. Training a new model...")

    # ✅ Train & Save Model
    with st.spinner("Training model..."):
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
        model.fit(raw_data[['ds', 'y']])  # Fit only 'ds' and 'y' columns
        with open(model_file, "wb") as file:
            pickle.dump(model, file)
        st.success("✅ Model trained and saved successfully! Reloading...")

# ✅ Load Model
with open(model_file, "rb") as file:
    model = pickle.load(file)

# ✅ Future Forecasting
days = st.slider("Select Forecasting Days", min_value=1, max_value=60, value=7)
forecast_hours = days * 24

future = model.make_future_dataframe(periods=forecast_hours, freq='H')
forecast = model.predict(future)

# ✅ Show Forecasted Data
st.write("### Forecasted Data")
st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())

# ✅ Plot Forecast with Confidence Interval
fig = go.Figure()

# 🔹 Plot actual data
fig.add_trace(go.Scatter(x=raw_data["ds"], y=raw_data["y"], mode='lines', name='Actual Demand', line=dict(color='blue', width=2)))

# 🔹 Plot forecast
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

fig.update_layout(title="Energy Demand Forecast", xaxis_title="Date", yaxis_title="Demand (MW)")
st.plotly_chart(fig)
