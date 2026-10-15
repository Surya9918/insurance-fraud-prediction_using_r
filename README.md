# Insurance Claim Risk Prediction & Fraud Pattern Analysis in R

[![R](https://img.shields.io/badge/Language-R%204.3+-blue.svg)](https://www.r-project.org/)
[![Framework](https://img.shields.io/badge/Framework-Tidymodels-orange.svg)](https://www.tidymodels.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![UI: Shiny](https://img.shields.io/badge/UI-R%20Shiny%20Enterprise-teal.svg)](app.R)
[![Academic Context](https://img.shields.io/badge/Academic-Data%20Science%20Case%20Study-green.svg)](#1-overview--academic-context)

> An end-to-end, reproducible Data Science and Machine Learning project in **R** for detecting fraudulent auto insurance claims while addressing severe class imbalance, high-cardinality categorical variables, and the Accuracy Paradox. Features a leak-free Tidymodels training pipeline, calibrated ensemble models, and an interactive **ShieldClaims Risk Intelligence** web dashboard.

---

## Overview

Insurance fraud is a major challenge for insurers because fraudulent claims are often hidden among a much larger number of legitimate claims. This project develops an end-to-end **machine learning pipeline in R** to identify potentially fraudulent auto insurance claims.

The project formulates, tunes, and compares three classification models:
- **Logistic Regression** -- interpretable statistical baseline with odds ratio estimation
- **XGBoost (Calibrated)** -- advanced nonlinear gradient-boosted ensemble
- **Decision Tree (`rpart`)** -- interpretable rule-based classifier for visual adjuster auditing

The project specifically focuses on resolving the **Accuracy Paradox** associated with imbalanced classification problems (24.7% fraud prevalence). Rather than relying on misleading raw accuracy, model performance is evaluated using:
- **PR-AUC** (Precision-Recall AUC - Primary Metric)
- **Recall / Sensitivity** (capturing true fraud losses)
- **Precision** (minimizing audit false alarms)
- **F1-Score** (harmonic balance)
- **ROC-AUC & Specificity**
- **Confusion Matrix Breakdown**

### Key Objectives
1. Clean and preprocess raw auto insurance claims, handling missing values denoted by `'?'`.
2. Audit and eliminate target, identifier, and temporal leakage vectors.
3. Handle rare categorical levels and class imbalance via training-fold **SMOTE**.
4. Train and tune multiple classification models using 5-fold cross-validation.
5. Calibrate operating decision thresholds ($\theta = 0.18 - 0.20$) to eliminate zero-rule false alarm saturation.
6. Extract and visualize feature importance (gain, odds ratios, decision tree paths).
7. Provide an interactive enterprise R Shiny application (**ShieldClaims Risk Intelligence**) for real-time simulation and batch audit.
8. Deliver a 100% reproducible academic R Markdown case study report.

---

## Table of Contents
1. [Overview & Academic Context](#1-overview--academic-context)
2. [Problem Statement & The Accuracy Paradox](#2-problem-statement--the-accuracy-paradox)
3. [Project Architecture & Workflow Pipeline](#3-project-architecture--workflow-pipeline)
4. [Dataset Provenance & Ingestion](#4-dataset-provenance--ingestion)
5. [Data Preprocessing & SMOTE Integration](#5-data-preprocessing--smote-integration)
6. [Model Architectures](#6-model-architectures)
7. [Evaluation Metrics & Methodology](#7-evaluation-metrics--methodology)
8. [Empirical Results & Model Comparison](#8-empirical-results--model-comparison)
9. [Feature Importance & Model Interpretability](#9-feature-importance--model-interpretability)
10. [Repository Structure](#10-repository-structure)
11. [Installation & Execution Guide](#11-installation--execution-guide)
12. [Data Leakage Audit](#12-data-leakage-audit)
13. [Limitations & Threats to Validity](#13-limitations--threats-to-validity)
14. [License & Citation](#14-license--citation)

---

## 1. Overview & Academic Context
* **Course:** Introduction to Data Science using R / Machine Learning Case Study
* **Primary Objective:** Build a reproducible, leak-free predictive classification pipeline to flag high-risk auto insurance claims while extracting human-interpretable risk drivers.
* **Core Technological Stack:**
  * **Core Language:** R (v4.0+)
  * **Data Manipulation & Wrangling:** `tidyverse`, `dplyr`, `tidyr`, `readr`, `tibble`
  * **Modeling Ecosystem:** `tidymodels` (`rsample`, `recipes`, `parsnip`, `workflows`, `tune`, `dials`, `yardstick`, `broom`)
  * **Class Balancing:** `themis` (`step_smote`)
  * **Algorithms:** Generalized Linear Model (`glm`), Extreme Gradient Boosting (`xgboost`), Decision Trees (`rpart`)
  * **Interpretability & Visualization:** `vip`, `ggplot2`, `scales`
  * **Interactive Web Platform:** `shiny`, `bslib`, `DT`, `plotly`

---

## 2. Problem Statement & The Accuracy Paradox

### The Asymmetric Cost of Fraud
Automobile insurance fraud costs insurers and policyholders billions of dollars annually. When reviewing claims, insurance carriers navigate an asymmetric cost structure:
* **False Negatives (Missed Fraud):** Costliest error. Insurers pay fraudulent claims averaging \$50,000 - \$80,000 in vehicle replacement and medical padding.
* **False Positives (False Alarms):** Investigative burden. Legitimate claims flagged for audit require investigation costs (~ \$1,500) and induce customer friction.

### The Accuracy Paradox
The dataset exhibits natural class skew with ~24.7% fraudulent claims (75.3% legitimate):
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
A naive "Zero-Rule" model that predicts every claim as **"No Fraud"** achieves **75.3% accuracy**, yet detects **0% of fraudulent claims** ($\text{Recall} = 0\%$). Raw accuracy is fundamentally uninformative in fraud detection. 

To overcome this paradox, our primary evaluation metric is **Precision-Recall Area Under the Curve (PR-AUC)** and **Recall (Sensitivity)** at controlled operational thresholds.

---

## 3. Project Architecture & Workflow Pipeline

The project implements a strictly leak-free machine learning workflow:

```
Dataset (Raw CSV)
   │
   ▼
Data Cleaning ('?' converted to NA, artifact removal)
   │
   ▼
Train/Test Split (80/20 Stratified Partitioning, Seed = 123)
   │
   ▼
Preprocessing Recipe (Median/Mode Imputation, step_other, step_dummy)
   │
   ▼
SMOTE Balancing (Training Data ONLY via themis::step_smote)
   │
   ▼
Model Training (Logistic Regression, XGBoost, Decision Tree via 5-Fold CV)
   │
   ▼
Probability Calibration & Threshold Tuning (θ = 0.18 - 0.20)
   │
   ▼
Evaluation (Holdout Test Set N = 201 on Real-World Class Distribution)
   │
   ▼
Model Comparison (PR-AUC, Recall, Precision, F1, ROC-AUC)
   │
   ▼
Deployment & Interactive Web UI (ShieldClaims Risk Intelligence Dashboard)
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
4. **Rare Category Collapsing:** `step_other(threshold = 0.05)` pools categorical levels with < 5% frequency into `"other"` to prevent sparse columns and rank deficiency.
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
* **Calibration & Threshold Tuning:** Scale-positive weight balancing combined with operational threshold tuning ($\theta = 0.18 - 0.20$) matching the natural fraud prevalence.
* **Interpretability:** Variable importance extracted via `vip::vi()`.

### 3. Decision Tree (`rpart` - Rule-Based)
* **Role:** Human-auditable decision rules for claims adjusters and viva presentation.
* **Structure:** Constrained depth (max depth 4, min split 10) to generate clean, readable decision trees.

---

## 7. Evaluation Metrics & Methodology

Metrics are computed on the **unseen holdout test partition ($N = 201$)** with `event_level = "second"` ensuring `"Yes"` is evaluated as the event:
* **PR-AUC (Primary):** Measures precision-recall trade-off under severe class skew.
* **Recall / Sensitivity:** Measures proportion of true fraudulent claims captured: $\frac{TP}{TP + FN}$.
* **Precision:** Measures proportion of fraud alarms that are authentic: $\frac{TP}{TP + FP}$.
* **F1-Score:** Harmonic mean of precision and recall.
* **ROC-AUC & Specificity:** Measures general discriminatory ability across thresholds.
* **Confusion Matrix:** Full $2 \times 2$ matrix with $TP, FP, TN, FN$ labeled.

---

## 8. Empirical Results & Model Comparison

| Model | PR-AUC | Recall | Precision | F1-Score | ROC-AUC | Specificity | Accuracy | Decision Threshold |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **XGBoost (Calibrated)** | 0.634 | **0.700** | **0.700** | **0.700** | **0.829** | 0.901 | **0.851** | **0.18 - 0.20** |
| **Decision Tree** | **0.662** | 0.640 | 0.696 | 0.667 | 0.731 | **0.907** | 0.841 | 0.50 |
| **Logistic Regression** | 0.616 | 0.680 | 0.567 | 0.618 | 0.815 | 0.828 | 0.791 | 0.50 |

*(Metrics derived from reproducible 80/20 stratified split execution on the benchmark Auto Insurance Fraud Dataset, N = 201 holdout test set)*

### Confusion Matrix Breakdown (Test Set N = 201)

| Model | True Positives (Fraud Caught) | False Positives (False Alarms) | True Negatives (Legit Approved) | False Negatives (Missed Fraud) | Decision Threshold |
|:---|:---:|:---:|:---:|:---:|:---:|
| **XGBoost (Calibrated)** | **35** | **15** | **136** | **15** | **0.18 - 0.20** |
| **Decision Tree** | 32 | 14 | 137 | 18 | 0.50 |
| **Logistic Regression** | 34 | 26 | 125 | 16 | 0.50 |

### Champion Model Selection & Threshold Optimization
* **XGBoost** is designated as the **Overall Operational Champion**. When decision thresholds are calibrated to $\theta = 0.18 - 0.20$ (aligning with empirical fraud prevalence and eliminating zero-rule false alarm saturation), XGBoost achieves the **highest overall accuracy (85.1%)**, **highest F1-score (0.700)**, and **highest ROC-AUC (0.829)**, capturing **7 out of 10 frauds (70.0% Recall)** while restricting false alarms to just 15 out of 151 legitimate claims (**70.0% Precision**, **90.1% Specificity**).
* **Decision Tree** serves as the **Auditable Rules Champion** ($\text{PR-AUC} = 0.662$, $\text{Accuracy} = 84.1\%$), providing human-interpretable IF-THEN paths for claims adjusters and viva compliance defense.

---

## 9. Feature Importance & Model Interpretability

### Top Predictive Risk Factors (XGBoost Split Gain)
1. **`incident_severity_Major_Damage`:** Dominant predictor; major structural damage strongly correlates with reported fraud.
2. **`total_claim_amount` & `vehicle_claim`:** High monetary claim amounts substantially elevate fraud probability.
3. **`insured_hobbies`:** Specific hobbies (e.g., chess, cross-fit) show pronounced non-linear associations in the dataset.
4. **`incident_type_Single_Vehicle_Collision`:** Single-car collisions present higher risk profiles than multi-vehicle accidents with independent drivers.

### Logistic Regression Odds Ratios
* **Major Damage vs Minor:** $\text{OR} \approx 4.82$ ($p < 0.001$, $95\%\text{ CI: } [2.85, 8.16]$) -- Holding all else constant, major collision damage is associated with a nearly five-fold increase in the estimated odds of fraud.
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
├── run_app.R                          # Standalone launcher for Shiny enterprise app
├── app.R                              # Full interactive R Shiny application
│
├── data/
│   ├── README.md                      # Data provenance, dictionary, & acquisition
│   ├── download_data.sh               # Automated acquisition helper
│   └── test_claims.csv                # Holdout test claims partition for batch auditing
│
├── R/
│   ├── 01_load_data.R                 # Ingestion, '?' conversion, & factor validation
│   ├── 02_eda.R                       # Exploratory analysis & statistical profiling
│   ├── 03_preprocessing.R             # Tidymodels recipe, imputation, & SMOTE
│   ├── 04_logistic_model.R            # GLM baseline & Odds Ratios extraction
│   ├── 05_xgboost_model.R             # Tunable XGBoost & variable importance
│   ├── 06_decision_tree.R             # rpart classifier & rule extraction
│   ├── 07_evaluation.R                # PR-AUC, metrics suite, & champion selection
│   ├── 08_visualizations.R            # High-resolution ggplot2 publication figures
│   └── shiny_helpers.R                # UI helper functions, metrics loader, batch scorer
│
├── www/
│   └── styles.css                     # Custom enterprise CSS styling for Shiny app
│
├── scratch/                           # Model calibration and threshold tuning experiments
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
git clone https://github.com/Surya9918/insurance-fraud-prediction_using_r.git
cd insurance-fraud-prediction_using_r
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

### Step 6: Launch Interactive Web UI (ShieldClaims Risk Intelligence)
Launch the full interactive R Shiny enterprise dashboard:
```r
# From R console or RStudio:
shiny::runApp()

# Or from system terminal / PowerShell:
# Rscript run_app.R
```
The dashboard will start at `http://127.0.0.1:7860` featuring:
* **Real-Time Claim Scorer:** Interactive claim simulator with 1-click presets (High-Risk, Clean, Borderline), dynamic fraud probability meter, and multi-model consensus.
* **Executive Analytics:** Class imbalance visualizer, Accuracy Paradox metrics, and categorical risk profiles.
* **Model Benchmark:** Live comparison scorecard, PR-AUC / ROC curves, and confusion matrices.
* **Batch Claim Audit Engine:** Scorer for preloaded holdout claims or custom CSV uploads with audit triage and CSV download.
* **Decision Tree Auditor:** Transparent, auditable decision rules for adjusters and viva presentation.
* **Methodology & Viva Guide:** Comprehensive defense Q&A and pipeline documentation.

---

## 12. Data Leakage Audit

A comprehensive leakage audit was conducted across 5 risk dimensions:
1. **Target Leakage:** Removed direct downstream outcomes.
2. **Identifier Leakage:** Eliminated `policy_number`, `insured_zip`, and `incident_location`.
3. **Preprocessing Leakage:** Median and mode imputation statistics are estimated strictly from training folds.
4. **SMOTE Leakage:** Synthetic minority generation occurs strictly inside the training recipe fold. **No synthetic records ever exist in the test evaluation set.**
5. **Evaluation Contamination:** Holdout test split ($N = 201$) is preserved in its untouched original empirical distribution.

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
  howpublished = {\url{https://github.com/Surya9918/insurance-fraud-prediction_using_r}}
}
```
