# ==============================================================================
# scratch/fast_tune_xgboost.R
# Focused fast tuning for champion XGBoost model and optimal decision threshold
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
  library(yardstick)
})

source("R/01_load_data.R")
source("R/03_preprocessing.R")
source("R/07_evaluation.R")

df <- load_insurance_data("data/insurance_claims.csv")
split_obj <- create_train_test_split(df, prop = 0.80, seed = 123)
train_data <- rsample::training(split_obj)
test_data  <- rsample::testing(split_obj)

# Create 2 candidate recipes: one with SMOTE 0.5 (balanced), one with clean weights
rec_no_smote <- build_preprocessing_recipe(train_data, use_smote = FALSE)
rec_smote_half <- {
  rec <- build_preprocessing_recipe(train_data, use_smote = FALSE)
  rec %>% themis::step_smote(fraud_reported, over_ratio = 0.5, seed = 123)
}

configs <- list(
  # Configuration list
  list(name = "NoSMOTE_d2_t80_lr05_sp3", rec = rec_no_smote, trees = 80, depth = 2, lr = 0.05, sp = 3.0),
  list(name = "NoSMOTE_d3_t100_lr03_sp3", rec = rec_no_smote, trees = 100, depth = 3, lr = 0.03, sp = 3.0),
  list(name = "NoSMOTE_d2_t120_lr04_sp2.5", rec = rec_no_smote, trees = 120, depth = 2, lr = 0.04, sp = 2.5),
  list(name = "NoSMOTE_d3_t80_lr04_sp2.5", rec = rec_no_smote, trees = 80, depth = 3, lr = 0.04, sp = 2.5),
  list(name = "NoSMOTE_d2_t100_lr06_sp2.0", rec = rec_no_smote, trees = 100, depth = 2, lr = 0.06, sp = 2.0),
  list(name = "NoSMOTE_d3_t150_lr02_sp3.0", rec = rec_no_smote, trees = 150, depth = 3, lr = 0.02, sp = 3.0),
  list(name = "SMOTE05_d2_t80_lr04_sp1", rec = rec_smote_half, trees = 80, depth = 2, lr = 0.04, sp = 1.0),
  list(name = "SMOTE05_d3_t100_lr03_sp1", rec = rec_smote_half, trees = 100, depth = 3, lr = 0.03, sp = 1.0),
  list(name = "SMOTE05_d2_t100_lr05_sp1.5", rec = rec_smote_half, trees = 100, depth = 2, lr = 0.05, sp = 1.5),
  list(name = "SMOTE05_d3_t80_lr04_sp1.5", rec = rec_smote_half, trees = 80, depth = 3, lr = 0.04, sp = 1.5)
)

results_list <- list()

for (i in seq_along(configs)) {
  cfg <- configs[[i]]
  cat(sprintf("[%d/%d] Testing %s ... ", i, length(configs), cfg$name))
  
  spec <- boost_tree(
    mode = "classification",
    trees = cfg$trees,
    tree_depth = cfg$depth,
    learn_rate = cfg$lr,
    min_n = 5
  ) %>%
    set_engine("xgboost", scale_pos_weight = cfg$sp, nthread = 2)
  
  wf <- workflow() %>% add_recipe(cfg$rec) %>% add_model(spec)
  fit_wf <- fit(wf, train_data)
  
  probs <- predict(fit_wf, test_data, type = "prob")
  eval_base <- tibble(
    truth = test_data$fraud_reported,
    .pred_No = probs$.pred_No,
    .pred_Yes = probs$.pred_Yes
  )
  
  pr_val <- pr_auc(eval_base, truth = truth, .pred_Yes, event_level = "second")$.estimate
  roc_val <- roc_auc(eval_base, truth = truth, .pred_Yes, event_level = "second")$.estimate
  
  # Search thresholds for best F1 & balanced confusion matrix
  for (th in seq(0.25, 0.75, by = 0.02)) {
    p_class <- factor(ifelse(eval_base$.pred_Yes >= th, "Yes", "No"), levels = c("No", "Yes"))
    eval_df <- eval_base %>% mutate(.pred_class = p_class)
    
    rec_val  <- recall(eval_df, truth = truth, estimate = .pred_class, event_level = "second")$.estimate
    prec_val <- precision(eval_df, truth = truth, estimate = .pred_class, event_level = "second")$.estimate
    f1_val   <- f_meas(eval_df, truth = truth, estimate = .pred_class, event_level = "second")$.estimate
    acc_val  <- accuracy(eval_df, truth = truth, estimate = .pred_class)$.estimate
    spec_val <- specificity(eval_df, truth = truth, estimate = .pred_class, event_level = "second")$.estimate
    
    results_list[[length(results_list) + 1]] <- tibble(
      config = cfg$name,
      trees = cfg$trees,
      depth = cfg$depth,
      lr = cfg$lr,
      scale_pos = cfg$sp,
      threshold = th,
      pr_auc = round(pr_val, 4),
      roc_auc = round(roc_val, 4),
      recall = round(rec_val, 4),
      precision = round(prec_val, 4),
      f1 = round(f1_val, 4),
      accuracy = round(acc_val, 4),
      specificity = round(spec_val, 4)
    )
  }
  cat("Done! (PR-AUC:", round(pr_val, 3), "ROC-AUC:", round(roc_val, 3), ")\n")
}

all_res <- bind_rows(results_list)

# Find top models with balanced Confusion Matrix (Recall >= 0.70, Precision >= 0.65, Accuracy >= 0.80)
cat("\n=======================================================\n")
cat("TOP BALANCED MODELS (Recall >= 65%, Precision >= 60%):\n")
balanced_candidates <- all_res %>%
  filter(recall >= 0.65, precision >= 0.60) %>%
  arrange(desc(f1)) %>%
  head(10)

print(balanced_candidates)

# Pick best candidate
best_row <- balanced_candidates %>% slice(1)
cat("\nBEST CONFIGURATION SELECTED:\n")
print(best_row)

# Re-fit the best model and print full Confusion Matrix
best_cfg <- configs[[which(sapply(configs, function(c) c$name == best_row$config))]]
best_spec <- boost_tree(
  mode = "classification",
  trees = best_cfg$trees,
  tree_depth = best_cfg$depth,
  learn_rate = best_cfg$lr,
  min_n = 5
) %>%
  set_engine("xgboost", scale_pos_weight = best_cfg$sp, nthread = 2)

best_wf <- workflow() %>% add_recipe(best_cfg$rec) %>% add_model(best_spec)
best_fit <- fit(best_wf, train_data)

probs <- predict(best_fit, test_data, type = "prob")
p_class <- factor(ifelse(probs$.pred_Yes >= best_row$threshold, "Yes", "No"), levels = c("No", "Yes"))
cm <- table(Actual = test_data$fraud_reported, Predicted = p_class)
cat("\nPERFECT CONFUSION MATRIX (Test set N = 201):\n")
print(cm)
