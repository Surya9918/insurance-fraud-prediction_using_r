# ==============================================================================
# run_pipeline.R
# Project: Insurance Claim Risk Prediction & Fraud Pattern Analysis
# Purpose: Master end-to-end execution script. Orchestrates data ingestion,
#          exploratory analysis, preprocessing with SMOTE, cross-validation,
#          model fitting (GLM, XGBoost, Decision Tree), comparative evaluation,
#          visualization rendering, and R Markdown compilation.
# ==============================================================================

# Ensure reproducible random number generation
set.seed(123)

message("==========================================================================")
message(" Insurance Claim Risk Prediction & Fraud Pattern Analysis Pipeline ")
message("==========================================================================")

# Step 0: Check and load required R packages
required_packages <- c(
  "readr", "dplyr", "tidyr", "tibble", "scales",
  "rsample", "recipes", "themis", "parsnip", "workflows",
  "tune", "dials", "yardstick", "broom", "vip", "ggplot2",
  "xgboost", "rpart", "rpart.plot", "knitr", "rmarkdown"
)

missing_packages <- required_packages[!sapply(required_packages, requireNamespace, quietly = TRUE)]

if (length(missing_packages) > 0) {
  message(">>> Installing missing dependencies: ", paste(missing_packages, collapse = ", "))
  install.packages(missing_packages, repos = "https://cloud.r-project.org")
}

# Source all modular pipeline scripts
message(">>> Sourcing pipeline modules from R/ ...")
source("R/01_load_data.R")
source("R/02_eda.R")
source("R/03_preprocessing.R")
source("R/04_logistic_model.R")
source("R/05_xgboost_model.R")
source("R/06_decision_tree.R")
source("R/07_evaluation.R")
source("R/08_visualizations.R")

# Verify dataset availability
data_path <- "data/insurance_claims.csv"
if (!file.exists(data_path)) {
  message("\n[!] DATASET MISSING: 'data/insurance_claims.csv' was not found.")
  message("Please obtain the dataset by running: bash data/download_data.sh")
  message("Or download directly from Kaggle: https://www.kaggle.com/datasets/brijbhushannanda1979/auto-insurance-fraud-dataset\n")
  stop("Pipeline aborted: dataset required to execute.", call. = FALSE)
}

# 1. Ingestion & Validation
message("\n--- STAGE 1: DATA INGESTION & QUALITY AUDIT ---")
claims_data <- load_insurance_data(data_path)
validation_info <- validate_insurance_data(claims_data)

# 2. Exploratory Data Analysis
message("\n--- STAGE 2: EXPLORATORY DATA ANALYSIS ---")
eda_results <- run_eda_pipeline(claims_data, output_dir = "outputs/metrics")
plot_class_distribution(claims_data, "outputs/figures/01_class_imbalance.png")
plot_missing_values(claims_data, "outputs/figures/02_missing_values.png")
plot_incident_severity(claims_data, "outputs/figures/03_incident_severity_vs_fraud.png")
plot_claim_amounts(claims_data, "outputs/figures/04_claim_amount_distribution.png")

# 3. Train/Test Partition & Cross-Validation Setup
message("\n--- STAGE 3: DATA PARTITIONING & CROSS-VALIDATION ---")
split_obj <- create_train_test_split(claims_data, prop = 0.80, seed = 123)
train_data <- rsample::training(split_obj)
test_data  <- rsample::testing(split_obj)
cv_folds   <- create_cv_folds(train_data, v = 5, seed = 123)

# 4. Preprocessing Recipe with Leak-Free Training-Only SMOTE
message("\n--- STAGE 4: TIDYMODELS PREPROCESSING SPECIFICATION ---")
fraud_recipe <- build_preprocessing_recipe(train_data, use_smote = TRUE, other_threshold = 0.05)
prepped_recipe <- audit_recipe_pipeline(fraud_recipe, train_data)

# 5. Model 1: Logistic Regression (Interpretable Baseline)
message("\n--- STAGE 5: LOGISTIC REGRESSION (GLM BASELINE) ---")
log_wf <- build_logistic_workflow(fraud_recipe)
log_cv <- cv_logistic_model(log_wf, cv_folds)
log_fit <- fit_and_interpret_logistic(log_wf, train_data, output_dir = "outputs")
plot_logistic_odds_ratios(log_fit$odds_ratios, top_n = 15, "outputs/figures/10_logistic_odds_ratios.png")

# 6. Model 2: XGBoost (Advanced Nonlinear Model)
message("\n--- STAGE 6: XGBOOST (GRADIENT BOOSTED TREES) ---")
xgb_wf <- build_xgboost_workflow(fraud_recipe)
xgb_grid <- create_xgboost_grid(size = 12)
xgb_tuning <- tune_xgboost_model(xgb_wf, cv_folds, grid = xgb_grid, seed = 123)
xgb_fit <- finalize_and_fit_xgboost(xgb_wf, xgb_tuning, train_data, output_dir = "outputs")
plot_xgboost_importance(xgb_fit$variable_importance, top_n = 15, "outputs/figures/09_feature_importance_xgboost.png")

# 7. Model 3: Decision Tree (Interpretable Rules)
message("\n--- STAGE 7: DECISION TREE (RPART CLASSIFIER) ---")
dt_wf <- build_decision_tree_workflow(fraud_recipe)
dt_fit <- fit_and_extract_tree(dt_wf, train_data, output_dir = "outputs")
save_decision_tree_plot(dt_fit$rpart_fit, "outputs/figures/decision_tree.png")

# 8. Evaluation on Holdout Test Partition (Original Real-World Distribution)
message("\n--- STAGE 8: HOLDOUT TEST EVALUATION ---")
preds_log <- generate_test_predictions(log_fit$workflow_fit, test_data, model_name = "Logistic Regression")
preds_xgb <- generate_test_predictions(xgb_fit$workflow_fit, test_data, model_name = "XGBoost", threshold = 0.56)
preds_dt  <- generate_test_predictions(dt_fit$workflow_fit,  test_data, model_name = "Decision Tree")

all_preds <- list(
  "Logistic Regression" = preds_log,
  "XGBoost" = preds_xgb,
  "Decision Tree" = preds_dt
)

# Compile comparison table and elect champion
comp_results <- compare_models_and_select_champion(all_preds, output_dir = "outputs/metrics")

# 9. Diagnostics & Evaluation Plots
message("\n--- STAGE 9: EVALUATION VISUALIZATION RENDERING ---")
plot_precision_recall_curves(all_preds, "outputs/figures/06_precision_recall_curves.png")
plot_roc_curves(all_preds, "outputs/figures/07_roc_curves.png")
plot_confusion_matrices(all_preds, "outputs/figures/08_confusion_matrices.png")
plot_model_comparison(comp_results$comparison_table, "outputs/figures/11_model_comparison_bar.png")

# 10. Render R Markdown Academic Report
message("\n--- STAGE 10: R MARKDOWN REPORT COMPILATION ---")
rmd_path <- "notebooks/insurance_fraud_analysis.Rmd"
html_out <- "reports/insurance_fraud_analysis.html"

if (file.exists(rmd_path) && requireNamespace("rmarkdown", quietly = TRUE)) {
  message(">>> Compiling academic R Markdown report to HTML...")
  rmarkdown::render(
    input = rmd_path,
    output_file = "../reports/insurance_fraud_analysis.html",
    envir = new.env()
  )
  message(">>> Report compiled successfully to: ", html_out)
} else {
  message(">>> Note: rmarkdown package or template pending compilation.")
}

message("\n==========================================================================")
message(">>> PIPELINE EXECUTION COMPLETED SUCCESSFULLY! <<<")
message("==========================================================================")
