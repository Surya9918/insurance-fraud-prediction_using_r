# ==============================================================================
# R/04_logistic_model.R
# Project: Insurance Claim Risk Prediction & Fraud Pattern Analysis
# Purpose: Interpretable baseline Generalized Linear Model (Logistic Regression),
#          odds-ratio extraction, confidence intervals, and cross-validation.
# ==============================================================================

suppressPackageStartupMessages({
  library(parsnip)
  library(workflows)
  library(tune)
  library(yardstick)
  library(broom)
  library(dplyr)
  library(tibble)
  library(readr)
})

#' Specify Logistic Regression Model
#'
#' Instantiates a standard binary logistic regression classifier using the `glm` engine.
#'
#' @return A `parsnip` model specification.
#' @export
specify_logistic_model <- function() {
  parsnip::logistic_reg(mode = "classification") %>%
    parsnip::set_engine("glm")
}

#' Build Logistic Regression Workflow
#'
#' Pairs the preprocessing recipe (including SMOTE and dummy encoding) with
#' the logistic regression model specification.
#'
#' @param recipe_spec Recipe object from `build_preprocessing_recipe`.
#' @param model_spec Model specification from `specify_logistic_model`.
#' @return A `workflows::workflow` object.
#' @export
build_logistic_workflow <- function(recipe_spec, model_spec = specify_logistic_model()) {
  workflows::workflow() %>%
    workflows::add_recipe(recipe_spec) %>%
    workflows::add_model(model_spec)
}

#' Cross-Validate Logistic Regression Workflow
#'
#' Evaluates the logistic regression workflow across 5-fold stratified cross-validation.
#'
#' @param wf Workflow object.
#' @param folds `vfold_cv` splits from `create_cv_folds`.
#' @param metrics Metric set for evaluation.
#' @return A `resample_results` tuning tibble.
#' @export
cv_logistic_model <- function(wf, folds, metrics = NULL) {
  if (is.null(metrics)) {
    metrics <- yardstick::metric_set(
      yardstick::pr_auc,
      yardstick::roc_auc,
      yardstick::recall,
      yardstick::precision,
      yardstick::f_meas,
      yardstick::accuracy
    )
  }

  message(">>> Running 5-fold cross-validation for Logistic Regression (Baseline)...")
  cv_results <- tune::fit_resamples(
    wf,
    resamples = folds,
    metrics = metrics,
    control = tune::control_resamples(
      save_pred = TRUE,
      event_level = "second" # 'Yes' is the positive class
    )
  )

  message(">>> CV Metrics for Logistic Regression:")
  print(tune::collect_metrics(cv_results))

  return(cv_results)
}

#' Fit Final Logistic Regression and Extract Odds Ratios
#'
#' Fits the workflow on the entire training partition, extracts model parameters,
#' computes Odds Ratios with 95% Wald confidence intervals, and exports an
#' interpretation summary table.
#'
#' @param wf Workflow object.
#' @param train_data Training data frame.
#' @param output_dir Directory for storing model and metric artifacts.
#' @return A list with the fitted workflow and odds ratio table.
#' @export
fit_and_interpret_logistic <- function(wf, train_data, output_dir = "outputs") {
  message(">>> Fitting Logistic Regression on full training partition...")
  fit_wf <- parsnip::fit(wf, data = train_data)

  # Extract underlying glm object for coefficient inspection
  glm_fit <- workflows::extract_fit_parsnip(fit_wf)$fit

  # Tidy coefficients with exponentiation to obtain Odds Ratios: OR = exp(beta)
  coef_summary <- broom::tidy(glm_fit, conf.int = TRUE, exponentiate = FALSE) %>%
    mutate(
      odds_ratio = exp(estimate),
      conf_low   = exp(conf.low),
      conf_high  = exp(conf.high),
      significance = case_when(
        p.value < 0.001 ~ "***",
        p.value < 0.01  ~ "**",
        p.value < 0.05  ~ "*",
        p.value < 0.1   ~ ".",
        TRUE ~ "ns"
      ),
      # Plain-language interpretation
      interpretation = case_when(
        term == "(Intercept)" ~ "Baseline log-odds of fraud",
        odds_ratio > 1.05 & p.value < 0.05 ~ paste0("Increases estimated odds of fraud by ", round((odds_ratio - 1) * 100, 1), "%"),
        odds_ratio < 0.95 & p.value < 0.05 ~ paste0("Decreases estimated odds of fraud by ", round((1 - odds_ratio) * 100, 1), "%"),
        p.value < 0.05 ~ "Statistically significant modest effect",
        TRUE ~ "No statistically significant difference at alpha = 0.05"
      )
    ) %>%
    select(term, estimate, std.error, statistic, p.value, odds_ratio, conf_low, conf_high, significance, interpretation) %>%
    arrange(p.value)

  # Export outputs
  if (!dir.exists(file.path(output_dir, "models"))) dir.create(file.path(output_dir, "models"), recursive = TRUE)
  if (!dir.exists(file.path(output_dir, "metrics"))) dir.create(file.path(output_dir, "metrics"), recursive = TRUE)

  saveRDS(fit_wf, file.path(output_dir, "models", "logistic_workflow.rds"))
  readr::write_csv(coef_summary, file.path(output_dir, "metrics", "logistic_odds_ratios.csv"))

  message(">>> Logistic Regression artifacts successfully saved.")

  list(
    workflow_fit = fit_wf,
    odds_ratios = coef_summary
  )
}
