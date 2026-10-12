# tests/testthat/test-03_evaluation.R

test_that("evaluation metrics correctly identify positive class and expose Accuracy Paradox", {
  source("../../R/07_evaluation.R", local = TRUE)

  # Construct synthetic prediction scenario demonstrating the Accuracy Paradox
  # Ground truth: 75 'No', 25 'Yes'
  truth_vec <- factor(c(rep("No", 75), rep("Yes", 25)), levels = c("No", "Yes"))

  # Model A: Naive Predictor (Predicts 'No' for every claim)
  naive_preds <- tibble(
    model = "Naive Baseline",
    truth = truth_vec,
    .pred_class = factor(rep("No", 100), levels = c("No", "Yes")),
    .pred_No = rep(1.0, 100),
    .pred_Yes = rep(0.0, 100)
  )

  # Model B: Informed Balanced Predictor
  # Catches 20 of 25 frauds, with 10 false alarms
  pred_class_b <- c(
    rep("No", 65), rep("Yes", 10), # 10 False Positives
    rep("No", 5),  rep("Yes", 20)  # 20 True Positives, 5 False Negatives
  )
  pred_prob_yes_b <- c(runif(65, 0.05, 0.35), runif(10, 0.55, 0.85),
                       runif(5, 0.25, 0.45), runif(20, 0.65, 0.95))

  informed_preds <- tibble(
    model = "Informed Model",
    truth = truth_vec,
    .pred_class = factor(pred_class_b, levels = c("No", "Yes")),
    .pred_No = 1 - pred_prob_yes_b,
    .pred_Yes = pred_prob_yes_b
  )

  # Calculate metrics for Naive Model
  naive_metrics <- calculate_model_metrics(naive_preds)

  # The Accuracy Paradox in action:
  # Accuracy is 75%, but Recall is 0% and Precision is NA or 0!
  expect_equal(naive_metrics$accuracy, 0.75)
  expect_equal(naive_metrics$recall, 0.0)

  # Calculate metrics for Informed Model
  informed_metrics <- calculate_model_metrics(informed_preds)
  expect_equal(informed_metrics$recall, 0.80) # 20 / 25 = 0.80
  expect_equal(round(informed_metrics$precision, 4), round(20 / 30, 4)) # 20 / 30 = 0.6667
  expect_true(informed_metrics$pr_auc > 0.5)

  # Test Confusion Matrix breakdown
  cm_res <- compute_confusion_matrix_breakdown(informed_preds)
  expect_equal(cm_res$summary_table$true_positive, 20)
  expect_equal(cm_res$summary_table$false_positive, 10)
  expect_equal(cm_res$summary_table$true_negative, 65)
  expect_equal(cm_res$summary_table$false_negative, 5)

  # Test Model Comparison and Champion selection
  comp <- compare_models_and_select_champion(list("Naive" = naive_preds, "Informed" = informed_preds))
  expect_equal(comp$champion_model, "Informed")
})
