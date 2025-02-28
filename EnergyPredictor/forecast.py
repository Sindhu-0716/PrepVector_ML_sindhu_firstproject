import pandas as pd
from prophet import Prophet

# ✅ Load CSV File
file_path = "PJME_hourly.csv"
df = pd.read_csv(file_path)

# ✅ Rename and Format Data for Prophet
df.rename(columns={"PJME_MW": "y"}, inplace=True)  # Rename target column
df.rename(columns={df.columns[0]: "ds"}, inplace=True)  # Ensure first column is datetime
df["ds"] = pd.to_datetime(df["ds"])  # Convert 'ds' to datetime format

# ✅ Train Prophet Model
model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)  # Optimize model
model.fit(df[['ds', 'y']])

# ✅ Make Future Predictions (5-Year Forecast)
future = model.make_future_dataframe(periods=5 * 365 * 24, freq='H')  # 5 Years Forecast
forecast = model.predict(future)

# ✅ Save Forecasted Data as CSV
forecast_file_path = "forecast.csv"
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(forecast_file_path, index=False)

print("✅ Forecast saved successfully as 'forecast.csv'.")