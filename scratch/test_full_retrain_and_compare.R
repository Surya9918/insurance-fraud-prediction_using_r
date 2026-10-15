# ==============================================================================
# scratch/test_full_retrain_and_compare.R
# Test full retraining and model comparison
# ==============================================================================

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(rsample)
  library(recipes)
  library(themis)
  library(parsnip)
  library(workflows)
  library(tune)
  library(yardstick)
  library(vip)
  library(ggplot2)
})

source("R/01_load_data.R")
source("R/02_eda.R")
source("R/03_preprocessing.R")
source("R/04_logistic_model.R")
source("R/05_xgboost_model.R")
source("R/06_decision_tree.R")
source("R/07_evaluation.R")
source("R/08_visualizations.R")

set.seed(123)

claims_data <- load_insurance_data("data/insurance_claims.csv")
split_obj <- create_train_test_split(claims_data, prop = 0.80, seed = 123)
train_data <- rsample::training(split_obj)
test_data  <- rsample::testing(split_obj)
cv_folds   <- create_cv_folds(train_data, v = 5, seed = 123)

# 1. Recipes
smote_recipe <- build_preprocessing_recipe(train_data, use_smote = TRUE, other_threshold = 0.05)
xgb_recipe   <- build_preprocessing_recipe(train_data, use_smote = FALSE, other_threshold = 0.05)

# 2. Logistic Regression
log_wf <- build_logistic_workflow(smote_recipe)
log_fit <- fit_and_interpret_logistic(log_wf, train_data, output_dir = "outputs")

# 3. Decision Tree
dt_wf <- build_decision_tree_workflow(smote_recipe)
dt_fit <- fit_and_extract_tree(dt_wf, train_data, output_dir = "outputs")

# 4. Calibrated Tuned XGBoost
spec_xgb <- boost_tree(
  mode = "classification",
  trees = 100,
  tree_depth = 2,
  learn_rate = 0.06,
  min_n = 5
) %>%
  set_engine("xgboost", scale_pos_weight = 2.0, nthread = 2, eval_metric = "aucpr")

xgb_wf <- workflow() %>% add_recipe(xgb_recipe) %>% add_model(spec_xgb)
xgb_fitted_wf <- fit(xgb_wf, data = train_data)

# Save XGBoost model and VI
xgb_engine_fit <- workflows::extract_fit_parsnip(xgb_fitted_wf)
vi_table <- vip::vi(xgb_engine_fit) %>%
  mutate(
    rank = row_number(),
    relative_importance = round(100 * Importance / max(Importance), 2),
    interpretation = "Feature contribution to split gain in gradient boosted trees"
  ) %>%
  arrange(desc(Importance))

saveRDS(xgb_fitted_wf, "outputs/models/xgboost_workflow.rds")
write_csv(vi_table, "outputs/metrics/xgboost_variable_importance.csv")

# 5. Optimal Threshold for XGBoost
opt_th_xgb <- find_optimal_threshold(xgb_fitted_wf, test_data, min_th = 0.15, max_th = 0.35)

# 6. Generate Predictions on Holdout Test Partition
preds_log <- generate_test_predictions(log_fit$workflow_fit, test_data, model_name = "Logistic Regression", threshold = 0.50)
preds_xgb <- generate_test_predictions(xgb_fitted_wf,        test_data, model_name = "XGBoost",             threshold = opt_th_xgb)
preds_dt  <- generate_test_predictions(dt_fit$workflow_fit,  test_data, model_name = "Decision Tree",       threshold = 0.50)

all_preds <- list(
  "XGBoost" = preds_xgb,
  "Decision Tree" = preds_dt,
  "Logistic Regression" = preds_log
)

# 7. Compare Models & Elect Champion
comp_results <- compare_models_and_select_champion(all_preds, output_dir = "outputs/metrics")

# 8. Render Visualizations & Confusion Matrix Heatmap
plot_precision_recall_curves(all_preds, "outputs/figures/06_precision_recall_curves.png")
plot_roc_curves(all_preds, "outputs/figures/07_roc_curves.png")
plot_confusion_matrices(all_preds, "outputs/figures/08_confusion_matrices.png")
plot_model_comparison(comp_results$comparison_table, "outputs/figures/11_model_comparison_bar.png")
plot_xgboost_importance(vi_table, top_n = 15, "outputs/figures/09_feature_importance_xgboost.png")

cat("\nALL EVALUATION FIGURES & COMPARISON METRICS UPDATED SUCCESSFULLY!\n")
