# Crop Yield Prediction using Machine Learning

This project aims to predict **crop yield per hectare** using machine learning models based on environmental and agricultural features like rainfall, pesticide usage, temperature, and crop type.

## 📁 Dataset

- Source: `yield_df.csv`
- Contains columns such as:
  - `Area`: Region or country
  - `Item`: Crop type
  - `Year`
  - `average_rain_fall_mm_per_year`
  - `pesticides_tonnes`
  - `avg_temp`
  - `hg/ha_yield` (Target variable)

## 🧹 Data Preprocessing

- Removed duplicates and handled missing/inconsistent values
- Dropped unnecessary columns (e.g., unnamed index columns)
- Converted string-based numeric columns to proper numerical types

## 📊 Exploratory Data Analysis

- Visualized distribution of `Area` and `Item`
- Analyzed correlation between features and yield

## 🧠 Machine Learning Pipeline

### Feature Engineering
- One-Hot Encoding for categorical variables: `Area`, `Item`
- Standard Scaling for numerical columns: `Year`, `Rainfall`, `Pesticides`, `Temperature`

### Model Training & Evaluation

Models used:
- Linear Regression
- Lasso Regression
- Ridge Regression
- Decision Tree Regressor ✅ (Best Performer)

Evaluation Metrics:
- **Mean Absolute Error (MAE)**
- **R² Score**

```python
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name} - MAE: {mae}, R² Score: {r2}")
