import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# 1. Load the cleaned data
df = pd.read_csv('cleaned_campaign_data.csv')

# 2. Prepare features and target
# Target is ROI
X = df.drop(['ROI'], axis=1)
y = df['ROI']

# 3. Train-Test Split (70/30 as per example)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=88)

# 4. Feature Scaling (Crucial for Neural Networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n--- {name} Performance ---")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    return mae, mse, r2

results = []

# --- Model 1: Linear Regression ---
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
results.append(('Linear Regression', *evaluate_model("Linear Regression", y_test, y_pred_lr)))

# --- Model 2: Random Forest ---
rf = RandomForestRegressor(n_estimators=100, random_state=88)
rf.fit(X_train, y_train) # Random forest handles unscaled data well
y_pred_rf = rf.predict(X_test)
results.append(('Random Forest', *evaluate_model("Random Forest", y_test, y_pred_rf)))

# --- Model 3: Neural Network (ANN) ---
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1) # Output layer for regression
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("\n--- Training Neural Network ---")
history = model.fit(X_train_scaled, y_train, 
                    validation_split=0.2, 
                    epochs=50, # Reduced from 100 for speed, adjust as needed
                    batch_size=16, 
                    verbose=0)

y_pred_nn = model.predict(X_test_scaled).flatten()
results.append(('Neural Network', *evaluate_model("Neural Network", y_test, y_pred_nn)))

# 5. Model Comparison Visualization
results_df = pd.DataFrame(results, columns=['Model', 'MAE', 'MSE', 'R2'])
print("\n--- Model Comparison Table ---")
print(results_df)

plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='R2', data=results_df)
plt.title('Model R2 Score Comparison')
plt.savefig('model_comparison.png')
print("\nComparison plot saved as 'model_comparison.png'")
