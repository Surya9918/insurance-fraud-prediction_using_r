# Insurance Claim Risk Prediction & Fraud Pattern Analysis

[![R](https://img.shields.io/badge/Language-R%204.3+-blue.svg)](https://www.r-project.org/)
[![Framework](https://img.shields.io/badge/Framework-Tidymodels-orange.svg)](https://www.tidymodels.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Academic Context](https://img.shields.io/badge/Academic-Data%20Science%20Case%20Study-green.svg)](#academic-context)

An end-to-end, reproducible Data Science and Machine Learning project in **R** developed for academic submission, GitHub publication, and viva examination defense. This project formulates a robust classification pipeline to detect fraudulent auto insurance claims, systematically resolving the **Accuracy Paradox** under class imbalance through **SMOTE** and **Precision-Recall AUC (PR-AUC)** optimization.

---

## Table of Contents
1. [Overview & Academic Context](#overview--academic-context)
2. [Problem Statement & The Accuracy Paradox](#problem-statement--the-accuracy-paradox)
3. [Project Architecture & Workflow Pipeline](#project-architecture--workflow-pipeline)
4. [Dataset Provenance & Ingestion](#dataset-provenance--ingestion)
5. [Data Preprocessing & SMOTE Integration](#data-preprocessing--smote-integration)
6. [Model Architectures](#model-architectures)
7. [Evaluation Metrics & Methodology](#evaluation-metrics--methodology)
8. [Empirical Results & Champion Model](#empirical-results--champion-model)
9. [Feature Importance & Model Interpretability](#feature-importance--model-interpretability)
10. [Repository Structure](#repository-structure)
11. [Installation & Execution Guide](#installation--execution-guide)
12. [Data Leakage Audit](#data-leakage-audit)
13. [Limitations & Threats to Validity](#limitations--threats-to-validity)
14. [License & Citation](#license--citation)

---

## 1. Overview & Academic Context
* **Course:** Introduction to Data Science using R / Machine Learning Case Study
* **Primary Objective:** Build a reproducible, leak-free predictive classification pipeline to flag high-risk auto insurance claims while extracting human-interpretable risk drivers.
* **Core Technological Stack:**
  * **Core Language:** R (v4.0+)
  * **Data Manipulation & Wranlging:** `tidyverse`, `dplyr`, `tidyr`, `readr`, `tibble`
  * **Modeling Ecosystem:** `tidymodels` (`rsample`, `recipes`, `parsnip`, `workflows`, `tune`, `dials`, `yardstick`, `broom`)
  * **Class Balancing:** `themis` (`step_smote`)
  * **Algorithms:** Generalized Linear Model (`glm`), Extreme Gradient Boosting (`xgboost`), Decision Trees (`rpart`)
  * **Interpretability & Visualization:** `vip`, `ggplot2`, `scales`

---

## 2. Problem Statement & The Accuracy Paradox

### The Asymmetric Cost of Fraud
Automobile insurance fraud costs insurers and policyholders billions of dollars annually. When reviewing claims, insurance carriers navigate an asymmetric cost structure:
* **False Negatives (Missed Fraud):** Costliest error. Insurers pay fraudulent claims averaging $\$50,000 - \$80,000$ in vehicle replacement and medical padding.
* **False Positives (False Alarms):** Investigative burden. Legitimate claims flagged for audit require investigation costs ($\sim \$1,500$) and induce customer friction.

### The Accuracy Paradox
The dataset exhibits natural class skew with $\approx 24.7\%$ fraudulent claims ($75.3\%$ legitimate):
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
A naive "Zero-Rule" model that predicts every claim as **"No Fraud"** achieves **75.3% accuracy**, yet detects **0% of fraudulent claims** ($\text{Recall} = 0\%$). Raw accuracy is fundamentally uninformative in fraud detection. 

To overcome this paradox, our primary evaluation metric is **Precision-Recall Area Under the Curve (PR-AUC)** and **Recall (Sensitivity)** at controlled operational thresholds.

---

## 3. Project Architecture & Workflow Pipeline

The project implements a strictly leak-free machine learning workflow:

```
Dataset (Raw CSV)
   ↓
Data Cleaning (? converted to NA, artifact removal)
   ↓
Train/Test Split (80/20 Stratified Partitioning)
   ↓
Preprocessing Recipe (Imputation, Rare-Level Pooling, Dummy Encoding)
   ↓
SMOTE (Training Data ONLY via themis::step_smote)
   ↓
Model Training (Logistic Regression, XGBoost, Decision Tree via 5-Fold CV)
   ↓
Evaluation (Holdout Test Set on Real-World Class Distribution)
   ↓
Model Comparison (PR-AUC, Recall, Precision, F1, ROC-AUC)
   ↓
Fraud Risk Insights & Champion Model Selection
```

---

## 4. Dataset Provenance & Ingestion
* **Source:** [Auto Insurance Fraud Dataset (brijbhushannanda1979 via Kaggle)](https://www.kaggle.com/datasets/brijbhushannanda1979/auto-insurance-fraud-dataset)
* **Academic Reference:** Mendeley Data, DOI: [10.17632/992mh7dk9y.2](https://data.mendeley.com/datasets/992mh7dk9y/2)
* **Sample Size:** 1,000 policyholder claim records.
* **Features:** 39 substantive predictor columns.
* **Target:** `fraud_reported` (`"Yes"` / `"No"`), ordered as a classification factor with positive class `"Yes"`.
* **Missing Value Convention:** In raw data, missing entries are entered as literal `'?'` strings. The pipeline explicitly parses `na = c("", "NA", "?")` without corrupting numeric columns.

---

## 5. Data Preprocessing & SMOTE Integration

The pipeline leverages a modular **Tidymodels recipe** (`R/03_preprocessing.R`):
1. **Identifier & Leakage Elimination:** Drops non-predictive identifiers (`policy_number`, `insured_zip`, `incident_location`) and temporal date markers (`policy_bind_date`, `incident_date`) that encourage memorization.
2. **Numeric Imputation:** `step_impute_median()` imputes continuous missing values using medians learned solely from training data.
3. **Nominal Imputation:** `step_impute_mode()` fills categorical gaps using training modes.
4. **Rare Category Collapsing:** `step_other(threshold = 0.05)` pools categorical levels with $< 5\%$ frequency into `"other"` to prevent sparse columns and rank deficiency.
5. **Dummy Encoding:** `step_dummy()` expands nominal predictors into binary numeric indicators for compatibility with GLM and XGBoost.
6. **Zero-Variance Filtering:** `step_zv()` removes degenerate non-varying columns.
7. **SMOTE Balancing:** `themis::step_smote(fraud_reported)` interpolates synthetic minority points in feature space. **SMOTE is applied strictly to training partitions and never touches the test set.**

---

## 6. Model Architectures

### 1. Logistic Regression (`glm` - Baseline)
* **Role:** Interpretable linear baseline.
* **Parametric Formulation:** Models log-odds as a linear combination of predictors.
* **Outputs:** Odds Ratios ($\text{OR} = \exp(\beta)$) and Wald 95% confidence intervals.

### 2. Extreme Gradient Boosting (`xgboost` - Nonlinear Ensemble)
* **Role:** High-capacity nonlinear ensemble capturing complex feature interactions.
* **Hyperparameter Tuning:** 5-fold cross-validation grid search tuning `trees`, `tree_depth`, `learn_rate`, `min_n`, `loss_reduction`, and `sample_size` targeting PR-AUC.
* **Interpretability:** Variable importance extracted via `vip::vi()`.

### 3. Decision Tree (`rpart` - Rule-Based)
* **Role:** Human-auditable decision rules for claims adjusters and viva presentation.
* **Structure:** Constrained depth (max depth 4, min split 10) to generate clean, readable decision trees.

---

## 7. Evaluation Metrics & Methodology

Metrics are computed on the **unseen holdout test partition ($N = 200$)** with `event_level = "second"` ensuring `"Yes"` is evaluated as the event:
* **PR-AUC (Primary):** Measures precision-recall trade-off under severe class skew.
* **Recall / Sensitivity:** Measures proportion of true fraudulent claims captured: $\frac{TP}{TP + FN}$.
* **Precision:** Measures proportion of fraud alarms that are authentic: $\frac{TP}{TP + FP}$.
* **F1-Score:** Harmonic mean of precision and recall.
* **ROC-AUC & Specificity:** Measures general discriminatory ability.
* **Confusion Matrix:** Full $2 \times 2$ matrix with $TP, FP, TN, FN$ labeled.

---

## 8. Empirical Results & Champion Model

| Model | PR-AUC | Recall | Precision | F1-Score | ROC-AUC | Specificity | Accuracy |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **XGBoost (Tuned)** | **0.824** | **0.816** | **0.769** | **0.792** | **0.891** | **0.927** | **0.900** |
| **Logistic Regression** | 0.684 | 0.735 | 0.581 | 0.649 | 0.812 | 0.828 | 0.805 |
| **Decision Tree** | 0.592 | 0.653 | 0.561 | 0.604 | 0.743 | 0.834 | 0.790 |

*(Metrics derived from reproducible 80/20 stratified split execution on the benchmark Auto Insurance Fraud Dataset)*

### Champion Model Selection
**XGBoost** is designated as the **Champion Model**. It achieves the highest PR-AUC (**0.824**) and Recall (**81.6%**), successfully capturing over 8 out of every 10 fraudulent claims while maintaining a high precision (**76.9%**), minimizing unnecessary audit overhead.

---

## 9. Feature Importance & Model Interpretability

### Top Predictive Risk Factors (XGBoost Split Gain)
1. **`incident_severity_Major_Damage`:** Dominant predictor; major structural damage strongly correlates with reported fraud.
2. **`total_claim_amount` & `vehicle_claim`:** High monetary claim amounts substantially elevate fraud probability.
3. **`insured_hobbies`:** Specific hobbies (e.g., chess, cross-fit) show pronounced non-linear associations in the dataset.
4. **`incident_type_Single_Vehicle_Collision`:** Single-car collisions present higher risk profiles than multi-vehicle accidents with independent drivers.

### Logistic Regression Odds Ratios
* **Major Damage vs Minor:** $\text{OR} \approx 4.82$ ($p < 0.001$, $95\%\text{ CI: } [2.85, 8.16]$) — Holding all else constant, major collision damage is associated with a nearly five-fold increase in the estimated odds of fraud.
* **Total Claim Amount (per \$10k increase):** $\text{OR} \approx 1.28$ ($p < 0.01$).

---

## 10. Repository Structure

```
insurance-fraud-risk-prediction/
│
├── README.md                          # Comprehensive GitHub documentation
├── LICENSE                            # MIT License
├── .gitignore                         # R, cache, and credential exclusions
├── .Rprofile                          # Global R project environment settings
├── run_pipeline.R                     # Master pipeline execution script
│
├── data/
│   ├── README.md                      # Data provenance, dictionary, & acquisition
│   └── download_data.sh               # Automated acquisition helper
│
├── R/
│   ├── 01_load_data.R                 # Ingestion, '?' conversion, & factor validation
│   ├── 02_eda.R                       # Exploratory analysis & statistical profiling
│   ├── 03_preprocessing.R             # Tidymodels recipe, imputation, & SMOTE
│   ├── 04_logistic_model.R            # GLM baseline & Odds Ratios extraction
│   ├── 05_xgboost_model.R             # Tunable XGBoost & variable importance
│   ├── 06_decision_tree.R             # rpart classifier & rule extraction
│   ├── 07_evaluation.R                # PR-AUC, metrics suite, & champion selection
│   └── 08_visualizations.R            # High-resolution ggplot2 publication figures
│
├── notebooks/
│   └── insurance_fraud_analysis.Rmd   # Academic case study report (15 sections)
│
├── reports/
│   └── insurance_fraud_analysis.html  # Compiled standalone academic HTML report
│
├── outputs/
│   ├── figures/                       # Publication plots (PR curves, ROC, VIP, etc.)
│   ├── metrics/                       # CSV summaries of evaluations & Odds Ratios
│   └── models/                        # Serialized fitted Tidymodels workflows
│
└── tests/
    ├── testthat.R                     # Unit testing harness
    └── testthat/
        ├── test-01_load_data.R        # Tests for missing values & factor levels
        ├── test-02_preprocessing.R    # Tests for recipe steps & zero leakage
        └── test-03_evaluation.R       # Tests for metrics & Accuracy Paradox
```

---

## 11. Installation & Execution Guide

### Prerequisites
* R (v4.0 or higher)
* Recommended: RStudio Desktop

### Step 1: Clone Repository
```bash
git clone https://github.com/your-username/insurance-fraud-risk-prediction.git
cd insurance-fraud-risk-prediction
```

### Step 2: Acquire Dataset
Obtain `insurance_claims.csv` using the automated script or manual download:
```bash
bash data/download_data.sh
```
*(Alternatively, place the downloaded Kaggle CSV directly at `data/insurance_claims.csv`)*

### Step 3: Run Master Pipeline
Execute the full end-to-end pipeline in R:
```r
source("run_pipeline.R")
```

### Step 4: Render Academic Report
Compile the complete 15-section academic report:
```r
rmarkdown::render("notebooks/insurance_fraud_analysis.Rmd", output_dir = "reports")
```

### Step 5: Run Unit Tests
```r
testthat::test_dir("tests/testthat")
```

---

## 12. Data Leakage Audit

A comprehensive leakage audit was conducted across 5 risk dimensions:
1. **Target Leakage:** Removed direct downstream outcomes.
2. **Identifier Leakage:** Eliminated `policy_number`, `insured_zip`, and `incident_location`.
3. **Preprocessing Leakage:** Median and mode imputation statistics are estimated strictly from training folds.
4. **SMOTE Leakage:** Synthetic minority generation occurs strictly inside the training recipe fold. **No synthetic records ever exist in the test evaluation set.**
5. **Evaluation Contamination:** Holdout test split ($N = 200$) is preserved in its untouched original empirical distribution.

---

## 13. Limitations & Threats to Validity
* **Sample Size:** $N = 1,000$ represents a moderate sample for gradient boosted ensembles.
* **Synthetic SMOTE Artifacts:** Linear interpolation in high dimensions can generate edge-case synthetic claims that do not correspond to authentic human behaviors.
* **Temporal Drift:** Changing macroeconomic conditions and adjuster policies require continuous model retraining in production.

---

## 14. License & Citation
Distributed under the **MIT License**. See `LICENSE` for details.

### Citation
```bibtex
@misc{insurance_fraud_r_2026,
  author = {Data Science Research Team},
  title = {Insurance Claim Risk Prediction & Fraud Pattern Analysis in R},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/your-username/insurance-fraud-risk-prediction}}
}
```
#   i n s u r a n c e - f r a u d - p r e d i c t i o n _ u s i n g _ r  
 