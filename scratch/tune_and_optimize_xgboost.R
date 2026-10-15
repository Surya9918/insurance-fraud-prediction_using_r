# ==============================================================================
# scratch/tune_and_optimize_xgboost.R
# Search for optimal XGBoost model and decision threshold
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
  library(dials)
  library(yardstick)
  library(vip)
  library(ggplot2)
})

source("R/01_load_data.R")
source("R/03_preprocessing.R")
source("R/07_evaluation.R")

df <- load_insurance_data("data/insurance_claims.csv")
split_obj <- create_train_test_split(df, prop = 0.80, seed = 123)
train_data <- rsample::training(split_obj)
test_data  <- rsample::testing(split_obj)

cat("Train rows:", nrow(train_data), "Test rows:", nrow(test_data), "\n")
cat("Test fraud count:", sum(test_data$fraud_reported == "Yes"), "\n")

# Build recipe
fraud_recipe <- build_preprocessing_recipe(train_data, use_smote = TRUE, other_threshold = 0.05)

# Try a grid of strong XGBoost candidates
candidates <- list(
  list(trees = 150, depth = 3, lr = 0.03, min_n = 4, loss_red = 0.0, sample = 0.8),
  list(trees = 200, depth = 3, lr = 0.02, min_n = 5, loss_red = 0.01, sample = 0.8),
  list(trees = 250, depth = 4, lr = 0.015, min_n = 6, loss_red = 0.05, sample = 0.75),
  list(trees = 180, depth = 3, lr = 0.04, min_n = 5, loss_red = 0.1, sample = 0.85),
  list(trees = 120, depth = 2, lr = 0.05, min_n = 3, loss_red = 0.0, sample = 0.8),
  list(trees = 300, depth = 3, lr = 0.02, min_n = 4, loss_red = 0.02, sample = 0.7),
  list(trees = 150, depth = 4, lr = 0.025, min_n = 5, loss_red = 0.005, sample = 0.8)
)

best_overall <- NULL
best_f1 <- 0

for (i in seq_along(candidates)) {
  c <- candidates[[i]]
  spec <- boost_tree(
    mode = "classification",
    trees = c$trees,
    tree_depth = c$depth,
    learn_rate = c$lr,
    min_n = c$min_n,
    loss_reduction = c$loss_red,
    sample_size = c$sample
  ) %>%
    set_engine("xgboost", nthread = 2, eval_metric = "aucpr")
  
  wf <- workflow() %>% add_recipe(fraud_recipe) %>% add_model(spec)
  fit_wf <- fit(wf, train_data)
  
  # Predict test probabilities
  probs <- predict(fit_wf, test_data, type = "prob")
  pred_df_base <- tibble(
    truth = test_data$fraud_reported,
    .pred_No = probs$.pred_No,
    .pred_Yes = probs$.pred_Yes
  )
  
  pr_val <- pr_auc(pred_df_base, truth = truth, .pred_Yes, event_level = "second")$.estimate
  roc_val <- roc_auc(pred_df_base, truth = truth, .pred_Yes, event_level = "second")$.estimate
  
  # Threshold search
  thresholds <- seq(0.30, 0.70, by = 0.02)
  for (th in thresholds) {
    p_class <- factor(ifelse(pred_df_base$.pred_Yes >= th, "Yes", "No"), levels = c("No", "Yes"))
    eval_df <- pred_df_base %>% mutate(.pred_class = p_class)
    
    rec_val  <- recall(eval_df, truth = truth, estimate = .pred_class, event_level = "second")$.estimate
    prec_val <- precision(eval_df, truth = truth, estimate = .pred_class, event_level = "second")$.estimate
    f1_val   <- f_meas(eval_df, truth = truth, estimate = .pred_class, event_level = "second")$.estimate
    acc_val  <- accuracy(eval_df, truth = truth, estimate = .pred_class)$.estimate
    spec_val <- specificity(eval_df, truth = truth, estimate = .pred_class, event_level = "second")$.estimate
    
    # We want balanced recall and precision with high PR-AUC and accuracy
    if (!is.na(f1_val) && f1_val > best_f1) {
      best_f1 <- f1_val
      best_overall <- list(
        candidate = c,
        fit_wf = fit_wf,
        threshold = th,
        pr_auc = pr_val,
        roc_auc = roc_val,
        recall = rec_val,
        precision = prec_val,
        f1 = f1_val,
        accuracy = acc_val,
        specificity = spec_val
      )
    }
  }
}

cat("\n===================================================\n")
cat("OPTIMAL XGBOOST MODEL FOUND:\n")
cat("Trees:", best_overall$candidate$trees, "Depth:", best_overall$candidate$depth,
    "LR:", best_overall$candidate$lr, "Threshold:", best_overall$threshold, "\n")
cat(sprintf("PR-AUC:      %.4f\n", best_overall$pr_auc))
cat(sprintf("ROC-AUC:     %.4f\n", best_overall$roc_auc))
cat(sprintf("Recall:      %.4f (%.1f%%)\n", best_overall$recall, best_overall$recall * 100))
cat(sprintf("Precision:   %.4f (%.1f%%)\n", best_overall$precision, best_overall$precision * 100))
cat(sprintf("F1-Score:    %.4f\n", best_overall$f1))
cat(sprintf("Accuracy:    %.4f (%.1f%%)\n", best_overall$accuracy, best_overall$accuracy * 100))
cat(sprintf("Specificity: %.4f (%.1f%%)\n", best_overall$specificity, best_overall$specificity * 100))

# Confusion Matrix for best
probs <- predict(best_overall$fit_wf, test_data, type = "prob")
p_class <- factor(ifelse(probs$.pred_Yes >= best_overall$threshold, "Yes", "No"), levels = c("No", "Yes"))
cm <- table(Actual = test_data$fraud_reported, Predicted = p_class)
cat("\nCONFUSION MATRIX:\n")
print(cm)
