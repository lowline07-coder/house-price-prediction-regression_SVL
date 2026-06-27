# house-price-prediction-regression_SVL
A machine learning project that predicts residential house prices in INR using real estate data. Implements and compares multiple regression algorithms including Ridge, Lasso, Decision Tree, Random Forest, and SVR to identify key property price drivers such as area, location score, and crime rate.
# Robust Regression Pipeline – House Price Prediction (INR)

## Project Overview
This project builds a complete regression pipeline to predict house prices (in INR)
using the PR2 real estate dataset. It covers regularized linear models, tree-based
models, support vector regression, and multiple cross-validation strategies.

---

## Dataset
- **File:** `pr2 dataset.csv`
- **Rows:** 3800 | **Columns:** 12
- **Target:** `house_price_inr`
- **Features:**
  | Column | Description |
  |---|---|
  | area_sqft | Property size in square feet |
  | bedrooms | Number of bedrooms |
  | bathrooms | Number of bathrooms |
  | location_score | Location quality score (1–10) |
  | property_age | Age of property in years |
  | distance_city_km | Distance to city center in km |
  | near_school | Near a school (1=Yes, 0=No) |
  | near_metro | Near a metro station (1=Yes, 0=No) |
  | crime_rate_index | Neighborhood crime rate index |
  | sale_date | Date of property sale |

---

## Project Structure

Data Loading → EDA → Cleaning → Feature Engineering

→ Train-Test Split → Scaling → Model Training

→ Cross-Validation → Hyperparameter Tuning → Evaluation → Reporting

---

## 🧠 Models Implemented

| Model | Type | Scaling Required |
|---|---|---|
| Ridge Regression (L2) | Linear | ✅ Yes |
| Lasso Regression (L1) | Linear | ✅ Yes |
| Decision Tree (unpruned) | Tree-based | ❌ No |
| Decision Tree (pruned) | Tree-based | ❌ No |
| Random Forest | Ensemble | ❌ No |
| SVR – Linear Kernel | SVM | ✅ Yes |
| SVR – RBF Kernel | SVM | ✅ Yes |

---

## 🔁 Cross-Validation Strategies

| Strategy | Description |
|---|---|
| K-Fold (k=5) | Standard 5-fold cross-validation |
| Stratified K-Fold | Preserves target distribution in each fold |
| LOOCV | Leave-One-Out on 500-sample subset |
| Time Series Split | Respects temporal order of sale dates |

---

## 📈 Evaluation Metrics

- **MSE** – Mean Squared Error
- **MAE** – Mean Absolute Error
- **RMSE** – Root Mean Squared Error
- **R² Score** – Coefficient of Determination

---

## 🏆 Results Summary

| Model | Test RMSE | Test R² |
|---|---|---|
| Ridge Regression | ~3.2M | ~0.86 |
| Lasso Regression | ~3.2M | ~0.86 |
| Decision Tree (pruned) | ~2.8M | ~0.89 |
| SVR (RBF) | ~2.5M | ~0.91 |
| **Random Forest** | **~2.1M** | **~0.94** |

> ✅ **Best Model: Random Forest** — highest R² and lowest RMSE

---

## 🔍 Key Findings

- `area_sqft`, `location_score`, and `crime_rate_index` are the strongest predictors of house price
- Random Forest outperforms all other models due to bagging and feature randomization
- Regularization (Ridge/Lasso) is essential for linear models to prevent overfitting
- Unpruned Decision Tree severely overfits — pruning and ensemble methods resolve this
- Cross-validation ensured reliable hyperparameter tuning without data leakage

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.x | Core language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Matplotlib / Seaborn | Data visualization |
| Scikit-learn | ML models, preprocessing, evaluation |
| Jupyter Notebook | Development environment |

---

## ⚙️ Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/your-username/house-price-prediction-ml.git
cd house-price-prediction-ml
```

2. **Install dependencies**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

3. **Run the notebook**
- Open `insurance_regression_model.ipynb` in Jupyter Notebook or VS Code
- Update the dataset path in Step 2 to match your local file location
- Run all cells from top to bottom

---

## 📌 Project Structure

house-price-prediction-ml/

│

├── pr2 dataset.csv                   # Raw dataset

├── insurance_regression_model.ipynb  # Full ML pipeline notebook

└── README.md                         # Project documentation
