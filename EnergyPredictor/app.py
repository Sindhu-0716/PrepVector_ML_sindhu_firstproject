import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle
import os
from prophet import Prophet

# Set page configuration
st.set_page_config(
    page_title="Demand Forecasting",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

# Title and Styling
st.markdown("""
    <h1 style='text-align: center; color:#faa356;'>Demand Forecasting 📈</h1>
    <h2 style='color:#a2d2fb;'>Problem Statement</h2>
    <p>Energy demand forecasting is crucial for effective resource planning and management in the power sector. 
    This app provides a user-friendly interface for forecasting electricity demand.</p>
    """, unsafe_allow_html=True)

# ✅ Data Loading Function
def get_data(file_name):
    try:
        if not os.path.exists(file_name):
            st.error(f"❌ File not found: {file_name}")
            return None  
        
        # Read CSV
        data = pd.read_csv(file_name, parse_dates=[0])

        # Ensure 'ds' column exists
        data = data.reset_index()  # Ensure datetime is a column, not an index
        data.rename(columns={"index": "ds"}, inplace=True)

        if 'ds' not in data.columns:
            st.error("❌ Missing required column 'ds' for Prophet.")
            return None

        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


# ✅ Load & Show Raw Data
st.write("### Raw Data")
file_path = "C:/Users/TeZZa/Downloads/PrepVector_ML_Sindhu/EnergyPredictor/PJME_hourly.csv"

raw_data = get_data(file_path)  
if raw_data is not None:
    st.dataframe(raw_data)


# ✅ Forecasting Model Selection
st.markdown("<h2 style='color:#a2d2fb;'>Energy Demand Forecasting</h2>", unsafe_allow_html=True)

model_path = "C:/Users/TeZZa/Downloads/PrepVector_ML_Sindhu/EnergyPredictor/prophet_model.pkl"

# Check if model exists
if not os.path.exists(model_path):
    st.error(f"❌ Model file not found: {model_path}. Please train and save the model first.")
    st.stop()

# ✅ Load Prophet Model
with open(model_path, "rb") as file:
    model = pickle.load(file)

# ✅ Future Forecasting
days = st.slider("Select Forecasting Days", min_value=1, max_value=60, value=7)
forecast_hours = days * 24
future = model.make_future_dataframe(periods=forecast_hours, freq='H')
forecast = model.predict(future)

# ✅ Plot Forecast
fig = go.Figure([go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode='lines', name='Forecast')])
st.plotly_chart(fig)
