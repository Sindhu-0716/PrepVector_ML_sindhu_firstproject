# Import essential libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.impute import KNNImputer #for imputing missing values
from sklearn.model_selection import train_test_split

# Load Data and check Data Types
uber_data = pd.read_csv('C:/Users/TeZZa/UberETA/PrepVector_ML_sindhu_firstproject/uber_data.csv')
print(uber_data.head())
# Check data types
print(uber_data.dtypes)
print(uber_data.info())

# Unique values of some categorical columns of df_train
columns = ['Weatherconditions', 'Road_traffic_density',
       'Vehicle_condition', 'Type_of_order', 'Type_of_vehicle',
      'Multiple_deliveries', 'Festival', 'City_type']
for column in columns:
    unique_values = uber_data[column].unique().tolist()
    print(column, ":", unique_values)

# Analyze null values
# Check missing values for non-numeric columns
print(uber_data.describe(exclude=np.number).T)
# Check missing values for numeric columns
print(uber_data.select_dtypes(exclude=['object']).isnull().sum())
uber_data.replace("", np.nan, inplace=True)
uber_data.replace(" ", np.nan, inplace=True)
print(uber_data.isnull().sum())
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

# Define columns to drop
columns_to_drop = ['ID', 'Delivery_person_ID', 'Restaurant_latitude',
                   'Restaurant_longitude','Delivery_location_latitude',
                   'Delivery_location_longitude']  # Replace with actual column names
# Drop columns from train_data
uber_data.drop(columns=columns_to_drop, inplace=True)


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
                       'Delivery_person_Ratings','Type_of_order']
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


# Split the data into train and test sets
train_data, test_data = train_test_split(uber_data, test_size=0.2, random_state=42)
# Display the split
print("Training data shape:", train_data.shape)
print("Testing data shape:", test_data.shape)
# Null value imputation on train_data
# Convert empty strings to NaN for consistency
train_data.replace("", np.nan, inplace=True)
train_data.replace(" ", np.nan, inplace=True)
train_data.isnull().sum()

# Impute missing values for other categorical and numeric columns in train_data
train_data['Type_of_vehicle'].fillna(train_data['Type_of_vehicle'].mode()[0], inplace=True)
train_data['City_type'].fillna(train_data['City_type'].mode()[0], inplace=True)
train_data['Festival'].fillna('No', inplace=True)
train_data['Multiple_deliveries'].fillna(0, inplace=True)
train_data['Delivery_person_Age'].fillna(train_data['Delivery_person_Age'].mean(), inplace=True)
train_data['Delivery_person_Ratings'].fillna(train_data['Delivery_person_Ratings'].median(), inplace=True)
train_data['Weatherconditions'].fillna(train_data['Weatherconditions'].mode()[0], inplace=True)
train_data['Road_traffic_density'].fillna('Low', inplace=True)

# Verify missing values in train_data
print("Remaining missing values in train_data:")
print(train_data.isnull().sum())

#converting time_ordered to multiple values
def extract_date_features(data):
    '''
    Create date intervals as; is weekend or not, periods of month and quarters of year
    '''
    
    # Ensure Order_Date is in datetime format
    data["Order_Date"] = pd.to_datetime(data["Order_Date"])

    data["is_weekend"] = data["Order_Date"].dt.day_of_week > 4

    data["month_intervals"] = data["Order_Date"].apply(lambda x: "start_month" if x.day <=10
                                                   else ("middle_month" if x.day <= 20 else "end_month"))

    data["year_quarter"] = data["Order_Date"].apply(lambda x: x.quarter)

extract_date_features(train_data)
extract_date_features(test_data)

# Find the difference between ordered time & picked time

def calculate_time_diff(df):
    # Convert time columns to timedelta
    df['Time_Ordered'] = pd.to_timedelta(df['Time_Ordered'])
    df['Time_Order_picked'] = pd.to_timedelta(df['Time_Order_picked'])
    #print(df['Time_Ordered'])
    #print(df['Time_Order_picked'])

    # Calculate formatted pickup time considering cases where pickup time is on the next day
    df['Time_Order_picked_formatted'] = df['Order_Date'] + pd.to_timedelta(np.where(df['Time_Order_picked'] < df['Time_Ordered'], 1, 0), unit='D') + df['Time_Order_picked']
    # print(df['Time_Order_picked_formatted'])

    # Calculate formatted order time
    df['Time_Ordered_formatted'] = df['Order_Date'] + df['Time_Ordered']
    # print(df['Time_Ordered_formatted'])

    # Calculate time difference in minutes
    df['order_prepare_time'] = (df['Time_Order_picked_formatted'] - df['Time_Ordered_formatted']).dt.total_seconds() / 60
    # print(df['order_prepare_time'])

    # Handle null values by filling with the median
    df['order_prepare_time'].fillna(df['order_prepare_time'].median(), inplace=True)
    # print(df['order_prepare_time'])

    # Drop all the time & date related columns
    df.drop(['Time_Ordered', 'Time_Order_picked', 'Time_Ordered_formatted', 'Time_Order_picked_formatted', 'Order_Date'], axis=1, inplace=True)

calculate_time_diff(train_data)
calculate_time_diff(test_data)

# Display the split
print("Training data shape:", train_data.shape)
print("Testing data shape:", test_data.shape)
# Save train and test datasets as separate CSV files
train_data.to_csv('C:/Users/TeZZa/UberETA/PrepVector_ML_sindhu_firstproject/train.csv', index=False)
test_data.to_csv('C:/Users/TeZZa/UberETA/PrepVector_ML_sindhu_firstproject/test.csv', index=False)

# Define target variable
target_column = 'Time_taken(min)'

# Separate features (X) and target (Y)
X_train = train_data.drop(columns=[target_column])
Y_train = train_data[target_column].fillna(train_data[target_column].median())  # Handle missing values in Y_train
X_test = test_data.drop(columns=[target_column])
Y_test = test_data[target_column].fillna(test_data[target_column].median()) 


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
# Handle missing values using SimpleImputer
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

# Separate numerical and categorical columns
num_cols = X_train.select_dtypes(include=['number']).columns.tolist()
cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

# Apply imputation
X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
X_test[num_cols] = num_imputer.transform(X_test[num_cols])
X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

# Apply Label Encoding to categorical variables
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])
    label_encoders[col] = le

# Display shapes
print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score

# Apply different regression models and evaluate R2 score
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

best_model = None
best_r2 = float('-inf')

for name, model in models.items():
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(Y_test, y_pred)
    print(f"{name} R2 Score: {r2:.4f}")
    
    if r2 > best_r2:
        best_r2 = r2
        best_model = name

print(f"Best Model: {best_model} with R2 Score: {best_r2:.4f}")

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# Train the best model (Random Forest) and evaluate
rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
rf_model.fit(X_train, Y_train)
y_pred = rf_model.predict(X_test)

# Calculate evaluation metrics
r2 = r2_score(Y_test, y_pred)
mae = mean_absolute_error(Y_test, y_pred)
mse = mean_squared_error(Y_test, y_pred)
rmse = np.sqrt(mse)

# Print metrics
print(f"Random Forest R2 Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
