# 🏦 Loan Approval Prediction using XGBoost + SHAP + KMeans clustering

This project is a comprehensive machine learning pipeline developed in Python to predict loan approval status based on applicant data. It includes meticulous preprocessing, robust modeling using XGBoost, hyperparameter optimization via 4-vector strategies, and post-hoc interpretability using SHAP to explain individual predictions. Also applying KMeans clustering, identifying natural groupings of loan applicants.

---

## 🚀 Project Objectives

- Predict whether a loan will be **Approved** or **Rejected**
- Build a robust XGBoost model with hyperparameter optimization
- Explain the predictions using SHAP visualizations and custom textual summaries
- KMeans clustering, Helped in identifying natural groupings of loan applicants
- Provide actionable feedback for rejected applications

------

## 📚 Dataset

- Format: CSV (`;` delimited)
- Dataset shape : Records : 45000, Columns : 14

- Credit_History
  - **Target**: Loan_Status (Y/N)

----------------- 

## 🧱 Workflow Overview

### 1. 📊 Data Cleaning & Exploration
- Dropped duplicate rows
- Visualized distributions using pie and bar charts
- Confirmed datatype consistency and field coverage

### 2. 🧼 Feature Engineering
- Encoded categorical variables using LabelEncoder
- Removed uninformative features (`Home_Ownership`, `Loan_Purpose`, etc.)
- Split dataset: 70% train / 15% validation / 15% test

### 3. 🧠 Model Building: XGBoost
- Baseline model trained using default XGBoost

### 4. 🧬 Optimization via 4-Vector Search
A novel 4-dimensional vector technique was used to search the hyperparameter space. This method iteratively tested combinations of four continuous variables to identify the most performant XGBoost setup based on validation accuracy.

### 5. 🔍 Explainability with SHAP
- **SHAP TreeExplainer** used to compute feature contributions

- Visuals:
  - Summary plots
  - Force plots for individual predictions

- Custom explanation DataFrame:
  - `Loan_ID`, outcome, top contributing reasons, and suggested advice
  - Example:
    - **Rejected** due to low `Credit_History` and high `LoanAmount`
    - **Advice**: "Try applying for a smaller loan or improving your credit history"

### 6. 🔵 Customer Segmentation with KMeans

- Performed KMeans clustering on numeric features (e.g., ApplicantIncome, LoanAmount)
- Assigned cluster labels as a new feature to enhance predictive power
- Helped in identifying natural groupings of loan applicants

---

## 📈 Visualizations

- 📊 Stacked bar plots for gender-wise approval rates
- 🥧 Pie chart of approval/rejection proportions
- 🧠 Interactive SHAP plots (summary, dependence, force)

---

## 🛠 Requirements

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn

🧪 Optional (Development & Notebook Use)
pip install jupyter notebook

