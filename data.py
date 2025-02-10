# Import essential libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statistics
import math
from sklearn.impute import KNNImputer #for imputing missing values
from sklearn.model_selection import train_test_split


# Load Data and check Data Types
uber_data = pd.read_csv('raw_data.csv')
print(uber_data.head())
# Check data types
print(uber_data.dtypes)
print(uber_data.info())


# Analyze null values
# Check missing values for non-numeric columns
print(uber_data.describe(exclude=np.number).T)
# Check missing values for numeric columns
print(uber_data.select_dtypes(exclude=['object']).isnull().sum())
uber_data.replace("", np.nan, inplace=True)
uber_data.replace(" ", np.nan, inplace=True)
print(uber_data.isnull().sum())

# Unique values of some categorical columns
columns = ['Weather_conditions', 'Road_traffic_density',
       'Vehicle_condition', 'Type_of_order', 'Type_of_vehicle',
      'Multiple_deliveries', 'Festival', 'City_type']

for column in columns:
    unique_values = uber_data[column].unique().tolist()
    print(column, ":", unique_values)

# Convert String 'NaN' to np.nan
'''
It seems "NaN" strings have a space after them, convert both "NaN " and "NaN" to object Null values
So, we use regex approach.
'''
def convert_nan(df):
    df.replace('NaN', float(np.nan), regex=True,inplace=True)
convert_nan(uber_data)
# Check null values
uber_data.isnull().sum().sort_values(ascending=False)

# Data Cleaning
# 1. remove unecessary characters in columns, 
# Using Regex to remove non-numeric characters from column Time_taken
import re

# Remove non-numeric characters from the 'Time_taken' column
uber_data['Time_taken(min)'] = uber_data['Time_taken(min)'].astype(str).str.replace(r'[^0-9.]', '', regex=True)

# Convert back to float (or int if no decimals)
uber_data['Time_taken(min)'] = pd.to_numeric(uber_data['Time_taken(min)'], errors='coerce')

# Display the first few rows to verify changes
print(uber_data[['Time_taken(min)']].head())

# Update datatypes
def update_datatype(df):
    df['Delivery_person_Age'] = df['Delivery_person_Age'].astype('float64')
    df['Delivery_person_Ratings'] = df['Delivery_person_Ratings'].astype('float64')
    df['Multiple_deliveries'] = df['Multiple_deliveries'].astype('float64')
    df['Order_Date']=pd.to_datetime(df['Order_Date'],format="%d-%m-%Y")

update_datatype(uber_data)

# Feature engineering: Add distance column using Haversine formula
def haversine(coord1, coord2):
    lon1, lat1 = coord1
    lon2, lat2 = coord2
    R = 6371000  # Radius of Earth in meters
    phi_1 = math.radians(lat1)
    phi_2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0)**2 + math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda / 2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    meters = R * c
    km = meters / 1000.0
    return round(km, 3)

def calculate_distance(row):
    rest_coords = (row['Restaurant_latitude'], row['Restaurant_longitude'])
    delivery_coords = (row['Delivery_location_latitude'], row['Delivery_location_longitude'])
    return haversine(rest_coords, delivery_coords)

uber_data['distance_km'] = uber_data.apply(calculate_distance, axis=1)

# Perform exploratory data analysis (EDA)
selected_col = ['City_type', 'Festival', 'Multiple_deliveries', 'Vehicle_condition', 'Weatherconditions',
                'Type_of_order', 'Type_of_vehicle', 'Road_traffic_density', 'Delivery_person_Age']

# Create bar plots for categorical columns
num_cols = 2
num_rows = (len(selected_col) + 1) // num_cols
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 4 * num_rows))
axes = axes.flatten()

for i, col in enumerate(selected_col):
    uber_data[col].value_counts().plot(kind='bar', color='purple', edgecolor='black', ax=axes[i])
    axes[i].set_title(f"{col} Frequency")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")
    axes[i].tick_params(axis='x', rotation=45)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# Plot correlation heatmap for numeric factors
numeric_factors = ['Time_taken(min)', 'Delivery_person_Age', 'distance_km', 'Delivery_person_Ratings']
correlation_matrix = uber_data[numeric_factors].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Numeric Factors")
plt.show()

# Create boxplots to analyze categorical factors
categorical_factors = ['Type_of_vehicle', 'Weatherconditions', 'Road_traffic_density',
                       'Vehicle_condition', 'Multiple_deliveries', 'Festival', 'City_type',
                       'Delivery_person_Ratings']
for factor in categorical_factors:
    if factor in uber_data.columns:
        plt.figure(figsize=(5, 3))
        sns.boxplot(x=uber_data[factor], y=uber_data['Time_taken(min)'], palette="viridis")
        plt.title(f"Impact of {factor} on Delivery Time")
        plt.xlabel(factor)
        plt.ylabel("Time Taken (minutes)")
        plt.xticks(rotation=45)
        plt.show()
    else:
        print(f"Column '{factor}' not found in dataset.")

# Extract the year from 'Order_Date'
def extract_year(order_date):
    try:
        return pd.to_datetime(order_date, errors='coerce').year
    except Exception as e:
        return None

uber_data['Order_Year'] = uber_data['Order_Date'].apply(extract_year)
print("Years in the data:", uber_data['Order_Year'].unique())

# Split the data into train and test sets
train_data, test_data = train_test_split(uber_data, test_size=0.2, random_state=42)

# Display the split
print("Training data shape:", train_data.shape)
print("Testing data shape:", test_data.shape)
# Null value imputation on train_data
# Convert empty strings to NaN for consistency
train_data.replace("", np.nan, inplace=True)
train_data.replace(" ", np.nan, inplace=True)

# Impute missing values in 'Time_Ordered' using KNN Imputer
if 'Time_Ordered' in train_data.columns:
    def time_to_seconds(time_str):
        if pd.isnull(time_str) or not isinstance(time_str, str):
            return None
        try:
            h, m, s = map(int, time_str.split(':'))
            return h * 3600 + m * 60 + s
        except ValueError:
            return None

    train_data['Time_Ordered_seconds'] = train_data['Time_Ordered'].apply(time_to_seconds)

    # Select numeric columns for imputation
    numeric_features = train_data.select_dtypes(include=['number']).columns.tolist()

    if len(numeric_features) < 2:
        raise ValueError("KNN Imputation requires at least one numeric feature besides 'Time_Ordered_seconds'!")

    imputer = KNNImputer(n_neighbors=3)
    train_data[numeric_features] = imputer.fit_transform(train_data[numeric_features])

    def seconds_to_time(seconds):
        if pd.isnull(seconds):
            return None
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02}:{m:02}:{s:02}"

    train_data['Time_Ordered'] = train_data['Time_Ordered_seconds'].apply(seconds_to_time)
    train_data.drop(columns=['Time_Ordered_seconds'], inplace=True)
    print("KNN Imputation completed for 'Time_Ordered' in train_data.")
else:
    print("The column 'Time_Ordered' does not exist in train_data.")

# Impute missing values for other categorical and numeric columns in train_data
train_data['Type_of_vehicle'].fillna(train_data['Type_of_vehicle'].mode()[0], inplace=True)
train_data['City_type'].fillna(train_data['City_type'].mode()[0], inplace=True)
train_data['Festival'].fillna('No', inplace=True)
train_data['Multiple_deliveries'].fillna(0, inplace=True)
train_data['Delivery_person_Age'].fillna(train_data['Delivery_person_Age'].mean(), inplace=True)
train_data['Weatherconditions'].fillna(train_data['Weatherconditions'].mode()[0], inplace=True)
train_data['Road_traffic_density'].fillna('Low', inplace=True)

# Verify missing values in train_data
print("Remaining missing values in train_data:")
print(train_data.isnull().sum())
