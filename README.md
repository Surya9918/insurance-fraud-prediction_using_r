# 🚗 Insurance Claim Risk Prediction & Fraud Pattern Analysis

> A reproducible R machine-learning pipeline for detecting fraudulent auto insurance claims while addressing severe class imbalance, high-cardinality categorical variables, and the Accuracy Paradox.

---

## 📌 Overview

Insurance fraud is a major challenge for insurers because fraudulent claims are often hidden among a much larger number of legitimate claims.

This project develops an end-to-end **machine learning pipeline in R** to identify potentially fraudulent auto insurance claims.

The project compares three models:

- **Logistic Regression** — interpretable statistical baseline
- **XGBoost** — advanced nonlinear gradient-boosting model
- **Decision Tree** — interpretable model used for decision-rule visualization

The project specifically focuses on the **Accuracy Paradox** associated with imbalanced classification problems.

Instead of relying primarily on accuracy, model performance is evaluated using:

- **Precision**
- **Recall / Sensitivity**
- **PR-AUC**
- **F1 Score**
- **Confusion Matrix**
- **ROC-AUC** as a supplementary metric

---

# 🎯 Objectives

The main objectives of this project are:

1. Clean and preprocess insurance claim data.
2. Handle missing values represented by `?`.
3. Identify and remove non-predictive or leakage-prone variables.
4. Handle rare categorical levels.
5. Convert categorical variables into numerical features.
6. Address class imbalance using **SMOTE**.
7. Train and compare multiple classification models.
8. Evaluate models using metrics appropriate for imbalanced data.
9. Analyze the variables contributing to fraud predictions.
10. Select an appropriate fraud-detection model and operating threshold.
11. Produce a completely reproducible R Markdown analysis.

---

# 📊 Dataset

### Dataset

**Auto Insurance Fraud Dataset**

Source:

**brijbhushannanda1979 via Kaggle**

Expected dataset characteristics:

- Approximately **1,000 insurance claims**
- Approximately **39 variables**
- Binary target variable:

```text
fraud_reported
