# ==============================================================================
# R/07_evaluation.R
# Project: Insurance Claim Risk Prediction & Fraud Pattern Analysis
# Purpose: Rigorous evaluation on real-world test partition, PR-AUC calculation,
#          confusion matrices, Accuracy Paradox mitigation, and Champion Model selection.
# ==============================================================================

suppressPackageStartupMessages({
  library(yardstick)
  library(dplyr)
  library(tibble)
  library(readr)
})

#' Find Optimal Classification Decision Threshold
#'
#' Sweeps decision thresholds across candidate values to select the cutoff that
#' optimizes F1-Score (harmonic mean of Precision and Recall), avoiding degenerate
#' all-positive or all-negative predictions under class imbalance.
#'
#' @param fit_wf Trained workflow.
#' @param eval_data Evaluation data frame with `fraud_reported`.
#' @param min_th Numeric. Minimum threshold candidate (default 0.15).
#' @param max_th Numeric. Maximum threshold candidate (default 0.65).
#' @param step Numeric. Step size for candidate evaluation (default 0.01).
#' @return Numeric. Optimal decision threshold.
#' @export
find_optimal_threshold <- function(fit_wf, eval_data, min_th = 0.15, max_th = 0.65, step = 0.01) {
  pred_prob <- predict(fit_wf, eval_data, type = "prob")
  truth_vec <- eval_data$fraud_reported
  truth_levels <- levels(truth_vec)

  thresholds <- seq(min_th, max_th, by = step)
  best_th <- 0.50
  best_score <- -1

  for (th in thresholds) {
    p_class <- factor(ifelse(pred_prob$.pred_Yes >= th, "Yes", "No"), levels = truth_levels)
    rec_val <- yardstick::recall_vec(truth_vec, p_class, event_level = "second")
    prec_val <- yardstick::precision_vec(truth_vec, p_class, event_level = "second")
    f1_val <- yardstick::f_meas_vec(truth_vec, p_class, event_level = "second")

    if (!is.na(f1_val) && !is.na(rec_val) && !is.na(prec_val)) {
      if (f1_val > best_score && rec_val >= 0.50) {
        best_score <- f1_val
        best_th <- th
      }
    }
  }

  message(sprintf(">>> Optimized decision threshold selected: %.2f (Peak F1: %.4f)", best_th, best_score))
  return(best_th)
}

#' Predict Class and Probabilities on Holdout Test Data
#'
#' Generates hard classification labels and calibrated posterior probabilities
#' for an unseen evaluation dataset using a trained Tidymodels workflow.
#'
#' @param fit_wf Trained workflow.
#' @param test_data Unseen test partition (original class distribution, NO SMOTE).
#' @param model_name Character. Label identifying the model.
#' @param threshold Numeric. Decision cutoff for positive 'Yes' classification.
#' @return A tibble containing ground truth, predictions, and class probabilities.
#' @export
generate_test_predictions <- function(fit_wf, test_data, model_name = "Model", threshold = 0.5) {
  stopifnot("fraud_reported" %in% names(test_data))

  # Generate predicted probabilities first, then apply the project-specific risk threshold.
  pred_prob <- predict(fit_wf, test_data, type = "prob")
  truth_levels <- levels(test_data$fraud_reported)
  pred_class <- tibble(
    .pred_class = factor(
      ifelse(pred_prob$.pred_Yes >= threshold, "Yes", "No"),
      levels = truth_levels
    )
  )

  bind_cols(
    tibble(
      model = model_name,
      truth = test_data$fraud_reported
    ),
    pred_class,
    pred_prob
  )
}

#' Calculate Comprehensive Classification Metrics Suite
#'
#' Evaluates test predictions across the full spectrum of imbalanced classification metrics.
#' CRITICAL: `event_level = "second"` designates "Yes" (Fraud) as the positive class.
#'
#' Primary Metrics:
#' - PR-AUC (`pr_auc`): Precision-Recall Area Under Curve (robust to severe class skew)
#' - Recall (`recall`): Sensitivity / True Positive Rate (proportion of actual frauds flagged)
#' - Precision (`precision`): Positive Predictive Value (proportion of flagged claims truly fraudulent)
#'
#' Supporting Metrics:
#' - F1-Score (`f_meas`): Harmonic mean of precision and recall
#' - ROC-AUC (`roc_auc`): Discrimination across all classification thresholds
#' - Specificity (`spec`): True Negative Rate (correct non-fraud retention)
#' - Accuracy (`accuracy`): Included for comparison; documented under the Accuracy Paradox
#'
#' @param pred_df Output from `generate_test_predictions`.
#' @return A tidy tibble of calculated metric values.
#' @export
calculate_model_metrics <- function(pred_df) {
  model_label <- unique(pred_df$model)

  pr_auc_val    <- yardstick::pr_auc(pred_df, truth = truth, .pred_Yes, event_level = "second")$.estimate
  roc_auc_val   <- yardstick::roc_auc(pred_df, truth = truth, .pred_Yes, event_level = "second")$.estimate
  recall_val    <- yardstick::recall(pred_df, truth = truth, estimate = .pred_class, event_level = "second")$.estimate
  precision_val <- yardstick::precision(pred_df, truth = truth, estimate = .pred_class, event_level = "second")$.estimate
  f1_val        <- yardstick::f_meas(pred_df, truth = truth, estimate = .pred_class, event_level = "second")$.estimate
  spec_val      <- yardstick::spec(pred_df, truth = truth, estimate = .pred_class, event_level = "second")$.estimate
  accuracy_val  <- yardstick::accuracy(pred_df, truth = truth, estimate = .pred_class)$.estimate

  tibble(
    model = model_label,
    pr_auc = round(pr_auc_val, 4),
    recall = round(recall_val, 4),
    precision = round(precision_val, 4),
    f1_score = round(f1_val, 4),
    roc_auc = round(roc_auc_val, 4),
    specificity = round(spec_val, 4),
    accuracy = round(accuracy_val, 4)
  )
}

#' Extract Labeled Confusion Matrix Summary
#'
#' Computes the 2x2 contingency table with explicitly labeled True Positives,
#' False Positives, True Negatives, and False Negatives along with insurance cost impact.
#'
#' @param pred_df Output from `generate_test_predictions`.
#' @return A list containing the confusion matrix table and derived financial counts.
#' @export
compute_confusion_matrix_breakdown <- function(pred_df) {
  model_label <- unique(pred_df$model)

  cm <- yardstick::conf_mat(pred_df, truth = truth, estimate = .pred_class)
  cm_tbl <- as.data.frame(cm$table)

  # In yardstick confusion matrices:
  # Rows: Prediction (.pred_class)
  # Columns: Truth
  # Levels: No (level 1), Yes (level 2)

  tn <- cm_tbl$Freq[cm_tbl$Prediction == "No" & cm_tbl$Truth == "No"]
  fp <- cm_tbl$Freq[cm_tbl$Prediction == "Yes" & cm_tbl$Truth == "No"]
  fn <- cm_tbl$Freq[cm_tbl$Prediction == "No" & cm_tbl$Truth == "Yes"]
  tp <- cm_tbl$Freq[cm_tbl$Prediction == "Yes" & cm_tbl$Truth == "Yes"]

  breakdown <- tibble(
    model = model_label,
    true_positive = tp,
    false_positive = fp,
    true_negative = tn,
    false_negative = fn,
    total_test_samples = tp + fp + tn + fn,
    fraud_detection_rate_recall = round(tp / (tp + fn), 4),
    false_alarm_rate = round(fp / (fp + tn), 4)
  )

  list(
    conf_mat_obj = cm,
    summary_table = breakdown
  )
}

#' Compare Multiple Models and Elect Champion
#'
#' Synthesizes evaluation metrics across all competing models, generates the master
#' comparative evaluation table, and programmatically designates the Champion Model
#' based on maximum PR-AUC and Recall.
#'
#' @param predictions_list Named list of prediction data frames from each model.
#' @param output_dir Output folder path for exporting summary tables.
#' @return A list containing the comparison table and champion model summary.
#' @export
compare_models_and_select_champion <- function(predictions_list, output_dir = "outputs/metrics") {
  message(">>> Compiling multi-model comparative performance table...")

  metrics_rows <- lapply(predictions_list, calculate_model_metrics)
  comparison_tbl <- bind_rows(metrics_rows) %>%
    arrange(desc(pr_auc), desc(recall))

  # Programmatically designate champion: Highest PR-AUC
  champion_row <- comparison_tbl[1, ]

  message("==========================================================================")
  message(">>> MODEL COMPARISON RESULTS:")
  print(comparison_tbl)
  message("--------------------------------------------------------------------------")
  message(sprintf(">>> ELECTED CHAMPION MODEL: %s", champion_row$model))
  message(sprintf(">>> Champion PR-AUC:   %.4f", champion_row$pr_auc))
  message(sprintf(">>> Champion Recall:   %.4f (Detects %.1f%% of all fraud cases)",
                  champion_row$recall, 100 * champion_row$recall))
  message(sprintf(">>> Champion Precision:%.4f (%.1f%% of fraud alerts are correct)",
                  champion_row$precision, 100 * champion_row$precision))
  message(sprintf(">>> Champion F1-Score: %.4f", champion_row$f1_score))
  message(sprintf(">>> Champion ROC-AUC:  %.4f", champion_row$roc_auc))
  message("==========================================================================")

  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  readr::write_csv(comparison_tbl, file.path(output_dir, "model_comparison_metrics.csv"))

  list(
    comparison_table = comparison_tbl,
    champion_model = champion_row$model,
    champion_metrics = champion_row
  )
}
