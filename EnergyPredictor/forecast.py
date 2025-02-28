import pandas as pd
import pickle
from prophet import Prophet

# ✅ Load CSV File (Make sure PJME_hourly.csv is in the same folder)
file_path = "PJME_hourly.csv"
df = pd.read_csv(file_path)

# ✅ Rename and Format Data for Prophet
df.rename(columns={"PJME_MW": "y"}, inplace=True)  # Rename target column
df.rename(columns={df.columns[0]: "ds"}, inplace=True)  # Ensure first column is datetime
df["ds"] = pd.to_datetime(df["ds"])  # Convert 'ds' to datetime format

# ✅ Train Prophet Model
model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)  # Disable daily seasonality for efficiency
model.fit(df[['ds', 'y']])

# ✅ Make Future Predictions (Static Forecast)
future = model.make_future_dataframe(periods=30 * 24, freq='H')  # Forecast for the next 30 days (hourly)
forecast = model.predict(future)

# ✅ Save Forecasted Data as CSV
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv("forecast.csv", index=False)

print("✅ Forecast saved successfully as 'forecast.csv' in your local folder.")
