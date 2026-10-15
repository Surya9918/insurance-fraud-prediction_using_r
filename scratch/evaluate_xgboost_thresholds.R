# ==============================================================================
# scratch/evaluate_xgboost_thresholds.R
# Detailed threshold sweep on calibrated XGBoost
# ==============================================================================

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(rsample)
  library(parsnip)
  library(workflows)
  library(yardstick)
})

source("R/01_load_data.R")
source("R/03_preprocessing.R")
source("R/07_evaluation.R")

df <- load_insurance_data("data/insurance_claims.csv")
split_obj <- create_train_test_split(df, prop = 0.80, seed = 123)
train_data <- rsample::training(split_obj)
test_data  <- rsample::testing(split_obj)

rec <- build_preprocessing_recipe(train_data, use_smote = FALSE)

# Train calibrated XGBoost
spec <- boost_tree(
  mode = "classification",
  trees = 100,
  tree_depth = 2,
  learn_rate = 0.06,
  min_n = 5
) %>%
  set_engine("xgboost", scale_pos_weight = 2.0, nthread = 2)

wf <- workflow() %>% add_recipe(rec) %>% add_model(spec)
fit_wf <- fit(wf, train_data)

probs <- predict(fit_wf, test_data, type = "prob")
truth_vec <- test_data$fraud_reported

cat("Threshold Sweep on Calibrated XGBoost:\n")
cat("-----------------------------------------------------------------------------------------\n")
cat(sprintf("%-10s %-8s %-10s %-10s %-8s %-10s %-10s %-15s\n",
            "Threshold", "Recall", "Precision", "F1-Score", "Accuracy", "Spec", "PR-AUC", "TP/FP/TN/FN"))
cat("-----------------------------------------------------------------------------------------\n")

pr_auc_val <- pr_auc_vec(truth_vec, probs$.pred_Yes, event_level = "second")

for (th in c(0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40, 0.50)) {
  p_class <- factor(ifelse(probs$.pred_Yes >= th, "Yes", "No"), levels = c("No", "Yes"))
  
  rec_val  <- recall_vec(truth_vec, p_class, event_level = "second")
  prec_val <- precision_vec(truth_vec, p_class, event_level = "second")
  f1_val   <- f_meas_vec(truth_vec, p_class, event_level = "second")
  acc_val  <- accuracy_vec(truth_vec, p_class)
  spec_val <- spec_vec(truth_vec, p_class, event_level = "second")
  
  tp <- sum(truth_vec == "Yes" & p_class == "Yes")
  fp <- sum(truth_vec == "No"  & p_class == "Yes")
  tn <- sum(truth_vec == "No"  & p_class == "No")
  fn <- sum(truth_vec == "Yes" & p_class == "No")
  
  cat(sprintf("%-10.2f %-7.1f%% %-9.1f%% %-10.4f %-7.1f%% %-9.1f%% %-10.4f %d/%d/%d/%d\n",
              th, rec_val * 100, prec_val * 100, f1_val, acc_val * 100, spec_val * 100, pr_auc_val,
              tp, fp, tn, fn))
}
cat("-----------------------------------------------------------------------------------------\n")
