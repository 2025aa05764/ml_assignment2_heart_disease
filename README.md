# ml_assignment2_heart_disease
## 1. Problem Statement

This project focuses on building a machine learning model to predict whether a patient diagnosed with heart failure is likely to experience a death event during the follow-up period. By identifying high-risk individuals early, healthcare providers can prioritize timely interventions and monitoring.

The classification task is binary:
- 0 → Survived (No Death Event)
- 1 → Not Survived (Death Event)

Multiple algorithms are evaluated to determine which model provides the most reliable predictions.

## 2. Dataset information

The dataset used in this study is the Heart Failure Clinical Records dataset, available on Kaggle: Dataset Link (kaggle.com in Bing). It contains real-world patient data, including demographic details, medical history, and key laboratory/clinical measurements. These features are crucial for assessing mortality risk in heart failure patients.
Dataset Source

The dataset was collected from Kaggle: https://www.kaggle.com/datasets/aadarshvelu/heart-failure-prediction-clinical-records


### Dataset Overview

⦁	Total Records: 5000
⦁	Total Column: 13
⦁	Total Features: 12 Predictors
⦁	Target Column: Death_Event


### Feature Descriptions

- age: Patient’s age (years)
- anaemia: Presence of anemia (boolean)
- creatinine phosphokinase (CPK): CPK enzyme level in blood (mcg/L)
- diabetes: Diabetes status (boolean)
- ejection fraction: Percentage of blood pumped out per heartbeat
- high blood pressure: Hypertension status (boolean)
- platelets: Platelet count (kilo platelets/mL)
- sex: Male/Female (binary)
- serum creatinine: Creatinine level in blood (mg/dL)
- serum sodium: Sodium level in blood (mEq/L)
- smoking: Smoking status (binary)
- time: Follow-up period (days)
- DEATH_EVENT: Mortality outcome (boolean)

## 3. Models Used:

## Model Performance Comparison

| ML Model Name | Accuracy (%) | AUC | Precision | Recall | F1 Score | MCC |
|-------------|--------------|-----|----------|--------|---------|-----|
| Logistic Regression | 0.836 | 0.835 | 0.848 | 0.848 | 0.848 | 0.670 |
| Decision Tree Classifier | 0.738 | 0.739 | 0.774 | 0.727 | 0.750 | 0.476 |
| K-Nearest Neighbor Classifier | 0.836 | 0.838 | 0.871 | 0.818 | 0.844 | 0.673 |
| Naive Bayes (Gaussian) | 0.803 | 0.802 | 0.818 | 0.818 | 0.818 | 0.604 |
| Random Forest | 0.803 | 0.802 | 0.818 | 0.818 | 0.818 | 0.604 |
| XGBoost | 0.754 | 0.754 | 0.781 | 0.758 | 0.769 | 0.506 |

ML Model Name	Accuracy (%)	AUC	Precision	Recall	F1 Score	MCC
Logistic Regression	0.853	0.882	0.792	0.701	0.744	0.642
Decision Tree Classifier	0.948	0.942	0.897	0.931	0.913	0.879
K-Nearest Neighbor Classifier	0.874	0.931	0.881	0.689	0.771	0.695
Naive Bayes (Gaussian)	0.804	0.846	0.709	0.565	0.629	0.503
Random Forest	0.962	0.983	0.957	0.919	0.938	0.913
XGBoost	0.951	0.979	0.942	0.894	0.917	0.885


# Heart Disease Prediction Models - Performance Summary

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Provides a solid baseline with good accuracy and AUC. However, recall is moderate, meaning some death events may be missed. |
| Decision Tree Classifier | High accuracy and recall, but single trees risk overfitting compared to ensemble methods. |
| K-Nearest Neighbor Classifier | Strong precision and AUC, but recall is lower, making it better at avoiding false positives than capturing all true positives. |
| Naive Bayes (Gaussian) | Weakest performance overall, likely due to its independence assumption not fitting well with clinical data |
| Random Forest | Best overall performer, achieving the highest accuracy, AUC, F1, and MCC. Balanced and reliable across metrics |
| XGBoost | Nearly as strong as Random Forest, with excellent precision-recall balance and competitive overall performance|







