import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# 1. Load the data
df = pd.read_csv('nykaa_campaign_data.csv')

# 2. Data Cleaning
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df_cleaned = df.drop(['Campaign_ID'], axis=1)
df_cleaned['Num_Channels'] = df['Channel_Used'].apply(lambda x: len(x.split(',')))

# --- NEW: EDA VISUALS FOR REPORT ---

# 2a. ROI Distribution (Histogram + Boxplot)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['ROI'], bins=30, kde=True)
plt.title('Distribution of ROI')
plt.subplot(1, 2, 2)
sns.boxplot(x=df['ROI'])
plt.title('Boxplot of ROI')
plt.savefig('roi_distribution.png')

# 2b. Average ROI by Campaign Type
plt.figure(figsize=(10, 6))
df.groupby('Campaign_Type')['ROI'].mean().sort_values().plot(kind='barh')
plt.title('Average ROI per Campaign Type')
plt.xlabel('ROI')
plt.savefig('avg_roi_campaign.png')

# 2c. Number of Campaigns per Audience
plt.figure(figsize=(10, 6))
sns.countplot(y='Target_Audience', data=df, order=df['Target_Audience'].value_counts().index)
plt.title('Number of Campaigns per Target Audience')
plt.savefig('campaign_counts.png')

# 2d. Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.savefig('correlation_heatmap.png')

# 3. PCA & Scree Plot
num_features = df.select_dtypes(include=[np.number]).drop(['ROI'], axis=1)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(num_features)

pca = PCA()
pca.fit(scaled_features)
exp_var = pca.explained_variance_ratio_

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(exp_var)+1), exp_var, 'o-')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.axhline(y=0.1, color='r', linestyle='--')
plt.savefig('scree_plot.png')

# 4. Final Processing for Modeling
categorical_cols = ['Campaign_Type', 'Target_Audience', 'Language', 'Customer_Segment']
df_final = pd.get_dummies(df_cleaned.drop(['Date', 'Channel_Used'], axis=1), columns=categorical_cols, drop_first=True)

X = df_final.drop(['ROI'], axis=1)
y = df_final['ROI']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=88)

scaler_model = StandardScaler()
X_train_scaled = scaler_model.fit_transform(X_train)
X_test_scaled = scaler_model.transform(X_test)

# 5. Modeling
# Linear Regression
lr = LinearRegression().fit(X_train_scaled, y_train)
# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=88).fit(X_train, y_train)
# Neural Network
nn = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(1)
])
nn.compile(optimizer='adam', loss='mse')
nn.fit(X_train_scaled, y_train, epochs=20, batch_size=32, verbose=0)

# Save results for LaTeX
print("Modeling complete. All visuals saved.")
