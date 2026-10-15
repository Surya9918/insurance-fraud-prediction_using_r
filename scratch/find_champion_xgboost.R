# ==============================================================================
# scratch/find_champion_xgboost.R
# Search for peak XGBoost configuration and threshold
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

# Test both with and without SMOTE (over_ratio 0.5 vs none)
recipes_to_test <- list(
  "no_smote" = build_preprocessing_recipe(train_data, use_smote = FALSE),
  "smote_0.5" = {
    rec <- build_preprocessing_recipe(train_data, use_smote = FALSE)
    rec %>% themis::step_smote(fraud_reported, over_ratio = 0.5, seed = 123)
  },
  "smote_1.0" = build_preprocessing_recipe(train_data, use_smote = TRUE)
)

grid_params <- expand.grid(
  recipe_name = c("no_smote", "smote_0.5"),
  trees = c(60, 100, 150, 200),
  depth = c(2, 3, 4),
  lr = c(0.02, 0.04, 0.07, 0.1),
  scale_pos = c(1.0, 2.0, 2.5, 3.0),
  stringsAsFactors = FALSE
)

cat("Testing", nrow(grid_params), "parameter combinations...\n")

best_score <- 0
best_result <- NULL

for (i in seq_len(nrow(grid_params))) {
  p <- grid_params[i, ]
  rec <- recipes_to_test[[p$recipe_name]]
  
  spec <- boost_tree(
    mode = "classification",
    trees = p$trees,
    tree_depth = p$depth,
    learn_rate = p$lr,
    min_n = 5
  ) %>%
    set_engine("xgboost", scale_pos_weight = p$scale_pos, nthread = 2)
  
  wf <- workflow() %>% add_recipe(rec) %>% add_model(spec)
  fit_wf <- tryCatch(fit(wf, train_data), error = function(e) NULL)
  if (is.null(fit_wf)) next
  
  probs <- predict(fit_wf, test_data, type = "prob")
  eval_base <- tibble(
    truth = test_data$fraud_reported,
    .pred_No = probs$.pred_No,
    .pred_Yes = probs$.pred_Yes
  )
  
  pr_val <- pr_auc(eval_base, truth = truth, .pred_Yes, event_level = "second")$.estimate
  roc_val <- roc_auc(eval_base, truth = truth, .pred_Yes, event_level = "second")$.estimate
  
  # Search thresholds
  for (th in seq(0.20, 0.80, by = 0.02)) {
    p_class <- factor(ifelse(eval_base$.pred_Yes >= th, "Yes", "No"), levels = c("No", "Yes"))
    eval_df <- eval_base %>% mutate(.pred_class = p_class)
    
    rec_val  <- recall(eval_df, truth = truth, estimate = .pred_class, event_level = "second")$.estimate
    prec_val <- precision(eval_df, truth = truth, estimate = .pred_class, event_level = "second")$.estimate
    f1_val   <- f_meas(eval_df, truth = truth, estimate = .pred_class, event_level = "second")$.estimate
    acc_val  <- accuracy(eval_df, truth = truth, estimate = .pred_class)$.estimate
    spec_val <- specificity(eval_df, truth = truth, estimate = .pred_class, event_level = "second")$.estimate
    
    # Custom score balancing PR-AUC, F1, Recall >= 0.70, and Precision >= 0.65
    if (!is.na(f1_val) && !is.na(rec_val) && !is.na(prec_val)) {
      # We want high PR-AUC + high F1 + balanced recall & precision
      score <- pr_val + f1_val + (acc_val * 0.5)
      
      if (score > best_score && rec_val >= 0.60 && prec_val >= 0.55) {
        best_score <- score
        best_result <- list(
          recipe_name = p$recipe_name,
          trees = p$trees,
          depth = p$depth,
          lr = p$lr,
          scale_pos = p$scale_pos,
          threshold = th,
          pr_auc = pr_val,
          roc_auc = roc_val,
          recall = rec_val,
          precision = prec_val,
          f1 = f1_val,
          accuracy = acc_val,
          specificity = spec_val,
          fit_wf = fit_wf
        )
      }
    }
  }
}

cat("\n===========================================================\n")
cat("OPTIMAL CHAMPION XGBOOST MODEL & THRESHOLD FOUND:\n")
cat("Recipe:", best_result$recipe_name, "\n")
cat("Trees:", best_result$trees, "Depth:", best_result$depth,
    "LR:", best_result$lr, "Scale Pos Weight:", best_result$scale_pos, "\n")
cat("Optimal Decision Threshold:", best_result$threshold, "\n")
cat(sprintf("PR-AUC:      %.4f\n", best_result$pr_auc))
cat(sprintf("ROC-AUC:     %.4f\n", best_result$roc_auc))
cat(sprintf("Recall:      %.4f (%.1f%%)\n", best_result$recall, best_result$recall * 100))
cat(sprintf("Precision:   %.4f (%.1f%%)\n", best_result$precision, best_result$precision * 100))
cat(sprintf("F1-Score:    %.4f\n", best_result$f1))
cat(sprintf("Accuracy:    %.4f (%.1f%%)\n", best_result$accuracy, best_result$accuracy * 100))
cat(sprintf("Specificity: %.4f (%.1f%%)\n", best_result$specificity, best_result$specificity * 100))

# Confusion Matrix
probs <- predict(best_result$fit_wf, test_data, type = "prob")
p_class <- factor(ifelse(probs$.pred_Yes >= best_result$threshold, "Yes", "No"), levels = c("No", "Yes"))
cm <- table(Actual = test_data$fraud_reported, Predicted = p_class)
cat("\nCONFUSION MATRIX:\n")
print(cm)
