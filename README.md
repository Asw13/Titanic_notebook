# 🚢 Titanic Survival Prediction

This project aims to predict which passengers survived the Titanic shipwreck using machine learning techniques. It is based on the classic Kaggle dataset: [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic).

---

## 📌 Project Overview

- Perform **Exploratory Data Analysis (EDA)** to understand data distributions and relationships.
- Handle missing values using **KNN Imputation** and simple imputation techniques.
- Apply **feature engineering** and **one-hot encoding** for categorical data.
- Train and compare multiple ML models:
  - Logistic Regression
  - Support Vector Classifier (SVC)
  - Gradient Boosting Classifier
  - XGBoost Classifier
  - **Random Forest Classifier** (best performing)
- Evaluate models using:
  - Accuracy
  - ROC-AUC Score
  - R² Score
- Optimize the best model using **GridSearchCV**.
- Generate predictions on unseen test data.

---

## 📁 Dataset Description

| Column       | Description                                          |
|--------------|------------------------------------------------------|
| PassengerId  | Unique ID of the passenger                          |
| Survived     | Survival status (0 = No, 1 = Yes)                   |
| Pclass       | Ticket class (1 = Upper, 2 = Middle, 3 = Lower)     |
| Name         | Full name                                           |
| Sex          | Gender                                              |
| Age          | Age in years (missing values imputed)              |
| SibSp        | # of siblings / spouses aboard                     |
| Parch        | # of parents / children aboard                     |
| Ticket       | Ticket number                                       |
| Fare         | Passenger fare                                      |
| Cabin        | Cabin number (missing → "No_Class")                |
| Embarked     | Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

---

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn (for EDA)
- Scikit-learn
- XGBoost
- Jupyter Notebook

---

## ⚙️ Model Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Define pipeline steps...
