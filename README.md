# Nykaa Marketing Campaign Performance Analysis 💄📈

## Project Overview
This project focuses on analyzing real-world marketing campaign data from Nykaa to predict and optimize campaign success. As part of **MAD LABS'** transition into a SaaS-focused IT company, this study aims to build a robust predictive model for **Return on Investment (ROI)** and **Conversion Rates**.

By understanding which channels, target audiences, and campaign types yield the highest returns, we can provide data-driven insights to optimize marketing spend.

## Research Question
*“Which marketing factors (Channel, Audience, Duration, etc.) most significantly impact the ROI of a digital campaign, and can we accurately predict campaign performance using Machine Learning?”*

---

## Project Roadmap

### Phase 1: Setup & Data Inspection ✅
- [x] Create project workspace and Git repository.
- [x] Download and load the Nykaa Marketing Dataset.
- [x] Perform initial data inspection (Shape, Info, Missing Values).

### Phase 2: Data Preprocessing & Cleaning 🔄
- [ ] Convert `Date` columns to datetime objects and extract seasonal features (Month, Day).
- [ ] Feature Engineering: Calculate `Num_Channels` from the comma-separated strings.
- [ ] Encode categorical variables using One-Hot Encoding.
- [ ] Handle outliers and scale numerical features for machine learning.

### Phase 3: Exploratory Data Analysis (EDA) 📊
- [ ] Generate correlation heatmaps to identify key relationships.
- [ ] Visualize ROI distribution across different `Campaign_Type` and `Target_Audience`.
- [ ] Perform **Principal Component Analysis (PCA)** to reduce dimensionality while retaining 95% variance.

### Phase 4: Machine Learning Modeling 🤖
Develop and compare three regression models:
1. **Linear Regression:** Establishing a baseline performance.
2. **Random Forest Regressor:** Capturing non-linear relationships and feature importance.
3. **Neural Networks (MLP):** Using deep learning to optimize predictions.

### Phase 5: Evaluation & Validation 📏
- [ ] Evaluate models using metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² Score.
- [ ] Compare performance across all three models to identify the best predictor.

### Phase 6: Reporting & Reflection 📝
- [ ] Compile findings into a final case study report.
- [ ] Reflect on the business impact for **MAD LABS** and SaaS products like Simpala HR.

---

## Tech Stack
- **Language:** Python 3.x
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **Version Control:** Git & GitHub
