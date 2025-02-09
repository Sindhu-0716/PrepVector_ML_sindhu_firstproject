# %% [markdown]
# #### **IMPORT ESSENTIAL LIBRARIES..**

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statistics
import math

# %% [markdown]
# #### **DATA LOADING ~~**  *checking Data Types*

# %%
uber_data = pd.read_csv('raw_data.csv')
print(uber_data.head())
print(uber_data.info())

# %% [markdown]
# ### **DATA CLEANING**

# %%
# checking missing values, NaN, empty strings and Nulls
#  Use describe if a column looks numeric but stored as string because isnull() cant detect NaN in such cases
uber_data.describe(exclude= np.number).T # checking nulls in object type


# %% [markdown]
# ##### *Time_ordered has NaN values but cant use forward fill, backward fill or Median/Mean as Time_ordered to Time_picked the difference to be 15 mins.*

# %%
uber_data.select_dtypes(exclude=['object']).isnull().sum() #checking nulls in float, int types

# %%
# making sure NaN's in strings are checked correctly as 'describe' ignores empty strings
uber_data.replace("", np.nan, inplace=True)  # Convert empty strings
uber_data.replace(" ", np.nan, inplace=True)  # Convert spaces
print(uber_data.isnull().sum())  # Now NaNs should be detected


# %% [markdown]
# ##### *can't use forward fill, backward fill, median imputaion for time_ordered as the data has different patterns as median prep time for snacks and meal is different -- better to use KNN imputation*
# 

# %% [markdown]
# ##### **EXPLORATORY DATA ANALYSIS**

# %%
# checking box plots for categorical type columns just to check the distributions
selected_col = ['City_type','Festival','Multiple_deliveries','Vehicle_condition','Weatherconditions',
                'Type_of_order','Type_of_vehicle','Road_traffic_density','Delivery_person_Age'] #enter column names

# display 2 plots per row
num_cols = 2
num_rows = (len(selected_col) + 1) // num_cols

# Create subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 4 * num_rows))

# Flatten axes for easy iteration (handles both single and multiple rows)
axes = axes.flatten()

# Plot each column
for i, col in enumerate(selected_col):
    uber_data[col].value_counts().plot(kind='bar', color='purple', edgecolor='black', ax=axes[i])
    axes[i].set_title(f"{col} Frequency")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")
    axes[i].tick_params(axis='x', rotation=45)

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# %% [markdown]
# ##### *Few columns have NaNs as shown above like Delivery_Person_Age, Road_traffic_Density,* *WeatherConditions, Festival , Multiple Deliveries, City Type*

# %%
# Calculate the difference time using Haversine Distance to add new column that shows Distance in KM between restaurant and delivery location
def haversine(coord1: object, coord2: object):
    import math

    # Coordinates in decimal degrees (e.g. 2.89078, 12.79797)
    lon1, lat1 = coord1
    lon2, lat2 = coord2

    R = 6371000  # radius of Earth in meters
    phi_1 = math.radians(lat1)
    phi_2 = math.radians(lat2)

    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda / 2.0) ** 2
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    meters = R * c  # output distance in meters
    km = meters / 1000.0  # output distance in kilometers

    meters = round(meters, 3)
    km = round(km, 3)

    return km
   

# %%
# Function to calculate Haversine distance
def calculate_distance(row):
    rest_coords = (float(row['Restaurant_latitude']), float(row['Restaurant_longitude']))
    delivery_coords = (float(row['Delivery_location_latitude']), float(row['Delivery_location_longitude']))
    return haversine(rest_coords, delivery_coords)

# Add distance column to the dataset
uber_data['distance_km'] = uber_data.apply(calculate_distance, axis=1)

# Add 'distance_km' to selected columns if not already included
if 'distance_km' not in selected_col:
    selected_col.append('distance_km')


# %%
# Using Regex to remove non-numeric characters from column Time_taken
import re

# Remove non-numeric characters from the 'Time_taken' column
uber_data['Time_taken(min)'] = uber_data['Time_taken(min)'].astype(str).str.replace(r'[^0-9.]', '', regex=True)

# Convert back to float (or int if no decimals)
uber_data['Time_taken(min)'] = pd.to_numeric(uber_data['Time_taken(min)'], errors='coerce')

# Display the first few rows to verify changes
print(uber_data[['Time_taken(min)']].head())


# %%
uber_data.dtypes

# %%
# Analyze correlation of numeric factors with the Delivery_Time
numeric_factors = ['Time_taken(min)', 'Delivery_person_Age', 'distance_km','Delivery_person_Ratings']
correlation_matrix = uber_data[numeric_factors].corr()

# Plot the correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Numeric Factors on Delivery Time")
plt.show()

# %% [markdown]
# ##### *Looks like Delivery_Person Age is positively correlated with Time Taken to Deliver*

# %%
# Boxplot Analysis: Checkin how Categorical Factors might influence Time Taken to Deliver
categorical_factors = ['Type_of_vehicle', 'Weatherconditions', 'Road_traffic_density',
                       'Vehicle_condition', 'Multiple_deliveries','Festival','City_type',
                       'Type_of_vehicle','Delivery_person_Ratings']
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

# %% [markdown]
# ###### *Some imp insights for **BOX PLOTS** 
# 
# ###### 'Type_of_vehicle', 'Weatherconditions' , 'Road_traffic_density' , 'Vehicle_condition' , 'Multiple_deliveries' , 'Festival' , 'City_type' , 'Delivery_person_Ratings' 
# 
# ###### 1. Delivery persons with higher ratings (e.g., 4.8, 4.9) generally complete deliveries faster, as shown by lower median times.
# ###### 2. The NaN group shows a wide box and many outliers, suggesting inconsistent and possibly longer delivery times for delivery persons with missing ratings.
# ###### 3.Deliveries in Urban areas are the fastest, while Semi-Urban areas experience the slowest deliveries.Metropolitan areas have high variability may be due to factors like traffic density.
# ###### 4. Festivals significantly increase delivery time due to higher demand and traffic congestion.
# ###### 5. Traffic jams significantly increase delivery time, while low-traffic conditions enable faster and more predictable deliveries.
# ###### 6. Adverse weather conditions like stormy and sandstorms increase delivery time and variability.
# 

# %% [markdown]
# 


