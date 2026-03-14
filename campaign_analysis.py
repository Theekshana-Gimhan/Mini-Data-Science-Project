import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA

# 1. Load the data
df = pd.read_csv('nykaa_campaign_data.csv')

# 2. Data Cleaning & Preprocessing
print("--- Starting Data Cleaning ---")

# Convert Date to datetime and extract Month/Day of Week
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Month'] = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.dayofweek

# Drop unique identifier and original Date
df_cleaned = df.drop(['Campaign_ID', 'Date'], axis=1)

# Handle 'Channel_Used' (It has multiple values, let's take the first one for simplicity or count them)
# For this study, we'll count how many channels were used as a feature
df_cleaned['Num_Channels'] = df['Channel_Used'].apply(lambda x: len(x.split(',')))
df_cleaned = df_cleaned.drop(['Channel_Used'], axis=1)

# Categorical Encoding
# Using get_dummies for multi-category columns (One-Hot Encoding)
categorical_cols = ['Campaign_Type', 'Target_Audience', 'Language', 'Customer_Segment']
df_final = pd.get_dummies(df_cleaned, columns=categorical_cols, drop_first=True)

print(f"Original shape: {df.shape}")
print(f"Final processed shape: {df_final.shape}")

# 3. Exploratory Data Analysis (EDA) - Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df_cleaned.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap of Numerical Features")
plt.savefig('correlation_heatmap.png')
print("\nCorrelation heatmap saved as 'correlation_heatmap.png'")

# 4. Feature Scaling & PCA (following the example methodology)
# Select numerical columns for PCA (exclude target ROI)
num_features = df_cleaned.select_dtypes(include=[np.number]).drop(['ROI'], axis=1)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(num_features)

pca = PCA(n_components=0.95) # Retain 95% of variance
pca_result = pca.fit_transform(scaled_features)
print(f"\nPCA reduced features from {num_features.shape[1]} to {pca_result.shape[1]} components.")

# 5. Save the cleaned dataset for the next step
df_final.to_csv('cleaned_campaign_data.csv', index=False)
print("\nCleaned data saved to 'cleaned_campaign_data.csv'")
