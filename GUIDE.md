# Comprehensive Data Science Case Study Guide 🚀

This guide is designed to help you complete a Data Science assignment from scratch, even if you have never coded before. It follows a professional methodology: **Research -> Strategy -> Execution -> Validation**.

---

## 🏗️ Step 1: Setting Up Your Environment

Before you start, you need to install the tools required to run Python and manage your project.

### 1.1 Install Python
1. Go to [python.org](https://www.python.org/downloads/).
2. Download the latest version for Windows.
3. **CRITICAL:** When installing, check the box that says **"Add Python to PATH"**.
4. Click "Install Now".

### 1.2 Open Your Terminal
We will use **PowerShell** (search for it in your Start menu). To check if Python installed correctly, type:
```powershell
python --version
```

### 1.3 Create a Project Folder
Create a folder on your computer where all your work will live (e.g., `Data_Science_Project`). Open that folder in your terminal:
```powershell
cd "path/to/your/folder"
```

---

## 📂 Step 2: Choosing Your Dataset

The assignment requires an **original** real-world problem. Do not use the fast food example from class.

1. Go to [Kaggle.com](https://www.kaggle.com/datasets).
2. Search for a topic you like (e.g., "Healthcare", "E-commerce", "Sports", "Finance").
3. Download a dataset in **CSV format**.
4. Move that CSV file into your project folder.

---

## 🐍 Step 3: Setting Up a Virtual Environment

A virtual environment keeps your project clean and avoids library conflicts.

1. **Create the environment:**
   ```powershell
   python -m venv my_env
   ```
2. **Activate it:**
   ```powershell
   .\my_env\Scripts\activate
   ```
3. **Install required libraries:**
   ```powershell
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
   ```

---

## 🧪 Step 4: Data Cleaning & Analysis (The "Script")

Create a file named `analysis.py` in your folder. You can use any text editor (Notepad, VS Code, etc.). Copy and adapt this general template:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. LOAD YOUR DATA
df = pd.read_csv('your_dataset_name.csv') # CHANGE THIS to your file name

# 2. INSPECT DATA
print(df.info())
print(df.describe())

# 3. CLEANING (Handling missing values)
df = df.fillna(0) # Fills empty spots with 0

# 4. VISUALIZATION (Create a correlation heatmap)
plt.figure(figsize=(10, 8))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True)
plt.savefig('correlation.png')
plt.show()
```

---

## 🤖 Step 5: Machine Learning (Regression)

In your `analysis.py`, add the modeling part. You are required to compare three models:

1. **Linear Regression:** Good for simple relationships.
2. **Random Forest:** Very powerful for complex, structured data.
3. **Neural Networks:** Deep learning approach.

### Code Template for Modeling:
```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Define X (Inputs) and y (What you want to predict)
X = df.drop(['Target_Column'], axis=1) # CHANGE Target_Column
y = df['Target_Column']

# Split data (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=88)

# Example: Training Random Forest
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)
print(f"R2 Score: {r2_score(y_test, predictions)}")
```

---

## 📊 Step 6: Evaluation Metrics

You must report three main metrics for every model:
1. **MAE (Mean Absolute Error):** How much the prediction is off on average.
2. **MSE (Mean Squared Error):** Penalizes larger errors more.
3. **R² Score:** Overall accuracy (closer to 1.0 is better).

---

## 📝 Step 7: The Final Report

Your report should be structured like this:
1. **Introduction:** Why is this problem important?
2. **Problem Formulation:** What is your research question?
3. **Data Acquisition:** Where did you get the data?
4. **Data Analysis:** Show your Heatmap and Scree Plots (PCA).
5. **ML Methods:** Explain the 3 models you used.
6. **Results:** Compare the R² scores in a table.
7. **Reflection:** What did you learn? Which model was best?

---

## 💡 Top Tips for Success
*   **Unique Question:** Don't just "predict price". Try "predicting the probability of disease" or "predicting customer retention".
*   **Visualize Early:** Look at your graphs before you start modeling. They tell you if the data is messy.
*   **Random State:** Always use `random_state=88` (or any fixed number) so your results stay consistent every time you run the code.
*   **Virtual Env:** If you get a "ModuleNotFoundError", it means you haven't activated your environment or installed the library using `pip`.

Good luck! 🍀
