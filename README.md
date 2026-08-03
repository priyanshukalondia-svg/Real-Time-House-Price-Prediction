# 🏠 Real-Time House Price Prediction System

A Machine Learning web application that predicts residential house prices using the **Ames Housing Dataset**. The project follows a complete end-to-end ML workflow, including data preprocessing, feature engineering, model selection, hyperparameter tuning, and deployment using **Streamlit**.

---

## 📌 Project Overview

The goal of this project is to estimate the selling price of a house based on its characteristics such as location, quality, size, garage capacity, basement area, and several other features.

The application allows users to enter property details through an interactive web interface and instantly receive a predicted house price.

---

## 🚀 Live Demo

**Streamlit App:** *https://house-predictor-by-priyanshu.streamlit.app/*

---

## 📂 Project Structure

```
Real-Time-House-Price-Prediction/
│
├── app/
│   └── app.py
│
├── data/
│   └── AmesHousing.csv
│
├── models/
│   └── house_price_pipeline.joblib
│
├── notebook/
│   └── House_Price_Prediction.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 📊 Dataset

**Dataset:** Ames Housing Dataset

The dataset contains detailed information about residential properties including:

- Lot Size
- Overall Quality
- Basement Area
- Garage Information
- Living Area
- Construction Year
- Exterior Features
- Neighborhood
- Bathrooms
- Bedrooms
- Sale Price

---

# ⚙️ Machine Learning Workflow

### 1. Data Cleaning

- Missing value handling
- Duplicate removal
- Data consistency checks

---

### 2. Exploratory Data Analysis

- Distribution Analysis
- Correlation Analysis
- Outlier Detection
- Feature Relationships

---

### 3. Feature Engineering

The following custom features were created:

- HouseAge
- RemodAge
- TotalBathrooms
- TotalSF
- TotalPorchSF

---

### 4. Data Preprocessing

- Pipeline
- ColumnTransformer
- StandardScaler
- OneHotEncoder

---

### 5. Models Evaluated

- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

---

### 6. Model Selection

After comparing multiple regression models, the **Gradient Boosting Regressor** achieved the best performance and was selected as the final model.

---

## 📈 Final Model Performance

| Metric | Value |
|---------|-------|
| Model | Gradient Boosting Regressor |
| MAE | 13,482 |
| RMSE | 22,945 |
| R² Score | **0.9343** |

---

## 🔍 Model Optimization

The final model was optimized using:

- Cross Validation
- GridSearchCV
- Hyperparameter Tuning

Best Parameters:

```python
{
    'learning_rate': 0.1,
    'max_depth': 4,
    'n_estimators': 200,
    'subsample': 0.8
}
```

---

## 📌 Feature Importance

The most influential features were:

- Overall Quality
- Total Square Footage
- Total Bathrooms
- Garage Capacity
- Second Floor Area
- House Age
- Lot Area
- Basement Finished Area
- Living Area
- Basement Quality

Permutation Feature Importance was also performed to validate the model's feature rankings.

---

## 💻 Technologies Used

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Joblib
- Streamlit

---

## 🎯 Features

- Interactive web interface
- Real-time price prediction
- Automatic feature engineering
- Machine Learning pipeline
- Clean and responsive UI
- Model serialization using Joblib

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Real-Time-House-Price-Prediction.git
```

Move into the project directory:

```bash
cd Real-Time-House-Price-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app/app.py
```

---



## 📈 Future Improvements

- Interactive price visualization
- Neighborhood comparison
- Price confidence interval
- Explainable AI (SHAP or similar)
- Model retraining with new data
- Cloud database integration

---

## 👨‍💻 Author

**Priyanshu Kalondia**

Aspiring Machine Learning Engineer


## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.
