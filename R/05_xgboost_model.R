# ==============================================================================
# R/05_xgboost_model.R
# Project: Insurance Claim Risk Prediction & Fraud Pattern Analysis
# Purpose: Advanced nonlinear Extreme Gradient Boosting (XGBoost) model,
#          hyperparameter tuning optimized for PR-AUC, and variable importance.
# ==============================================================================

suppressPackageStartupMessages({
  library(parsnip)
  library(workflows)
  library(tune)
  library(dials)
  library(yardstick)
  library(vip)
  library(dplyr)
  library(tibble)
  library(readr)
})

#' Specify Tunable XGBoost Model
#'
#' Sets up a tunable gradient boosted tree specification via the `xgboost` engine.
#'
#' @return A `parsnip` model specification with tunable parameters.
#' @export
specify_xgboost_model <- function() {
  parsnip::boost_tree(
    mode = "classification",
    trees = tune::tune(),
    tree_depth = tune::tune(),
    learn_rate = tune::tune(),
    min_n = tune::tune(),
    loss_reduction = tune::tune(),
    sample_size = tune::tune()
  ) %>%
    parsnip::set_engine("xgboost", nthread = 2, eval_metric = "aucpr")
}

#' Build XGBoost Workflow
#'
#' Pairs the preprocessing recipe with the tunable XGBoost model specification.
#'
#' @param recipe_spec Recipe object from `build_preprocessing_recipe`.
#' @param model_spec Model specification from `specify_xgboost_model`.
#' @return A `workflows::workflow` object.
#' @export
build_xgboost_workflow <- function(recipe_spec, model_spec = specify_xgboost_model()) {
  workflows::workflow() %>%
    workflows::add_recipe(recipe_spec) %>%
    workflows::add_model(model_spec)
}

#' Define Computationally Efficient Hyperparameter Tuning Grid
#'
#' Constructs a focused parameter grid designed for imbalanced tabular fraud detection.
#'
#' @param size Integer. Grid size if generating space-filling design (default 15).
#' @return A tibble parameter grid.
#' @export
create_xgboost_grid <- function(size = 15) {
  # Customized dial ranges suited for n = 800 training observations
  dials::grid_latin_hypercube(
    dials::trees(range = c(100L, 500L)),
    dials::tree_depth(range = c(3L, 6L)),
    dials::learn_rate(range = c(-2.3, -1.0), trans = scales::log10_trans()), # 0.005 to 0.1
    dials::min_n(range = c(3L, 12L)),
    dials::loss_reduction(range = c(0, 5)),
    dials::sample_prop(range = c(0.6, 0.95)),
    size = size
  )
}

#' Tune XGBoost Hyperparameters using Cross-Validation
#'
#' Runs grid tuning across 5 cross-validation folds, targeting Precision-Recall AUC (PR-AUC).
#'
#' @param wf XGBoost workflow object.
#' @param folds Stratified CV folds from `create_cv_folds`.
#' @param grid Parameter grid from `create_xgboost_grid`.
#' @param seed Integer. Seed for tuning execution.
#' @return A `tune_results` object.
#' @export
tune_xgboost_model <- function(wf, folds, grid = NULL, seed = 123) {
  set.seed(seed)
  if (is.null(grid)) grid <- create_xgboost_grid(size = 15)

  metrics <- yardstick::metric_set(
    yardstick::pr_auc,
    yardstick::roc_auc,
    yardstick::recall,
    yardstick::precision,
    yardstick::f_meas
  )

  message(sprintf(">>> Tuning XGBoost across %d parameter candidates using 5-fold CV (metric = pr_auc)...", nrow(grid)))

  tune_results <- tune::tune_grid(
    wf,
    resamples = folds,
    grid = grid,
    metrics = metrics,
    control = tune::control_grid(
      save_pred = FALSE,
      event_level = "second", # 'Yes' is the positive class
      verbose = FALSE
    )
  )

  message(">>> Top 5 XGBoost hyperparameter configurations by PR-AUC:")
  top_models <- tune::show_best(tune_results, metric = "pr_auc", n = 5)
  print(top_models)

  return(tune_results)
}

#' Finalize and Fit Best XGBoost Model
#'
#' Selects the optimal hyperparameter combination by PR-AUC, finalizes the workflow,
#' fits the model on the full training set, and computes variable importance.
#'
#' @param wf XGBoost workflow.
#' @param tune_results Tuning results from `tune_xgboost_model`.
#' @param train_data Full training data.
#' @param output_dir Output folder path.
#' @return A list containing the fitted workflow, best parameters, and variable importance.
#' @export
finalize_and_fit_xgboost <- function(wf, tune_results, train_data, output_dir = "outputs") {
  # Select best parameters prioritizing PR-AUC
  best_params <- tune::select_best(tune_results, metric = "pr_auc")
  message(">>> Optimal XGBoost hyperparameters selected (max PR-AUC):")
  print(best_params)

  # Finalize workflow with best hyperparameters
  final_wf <- tune::finalize_workflow(wf, best_params)

  message(">>> Fitting finalized XGBoost model on full training partition...")
  fit_final_wf <- parsnip::fit(final_wf, data = train_data)

  # Extract fitted model engine for variable importance
  xgb_engine_fit <- workflows::extract_fit_parsnip(fit_final_wf)

  # Compute variable importance table via vip
  vi_table <- vip::vi(xgb_engine_fit) %>%
    mutate(
      rank = row_number(),
      relative_importance = round(100 * Importance / max(Importance), 2),
      interpretation = "Feature contribution to split gain in gradient boosted trees"
    ) %>%
    arrange(desc(Importance))

  # Save outputs
  if (!dir.exists(file.path(output_dir, "models"))) dir.create(file.path(output_dir, "models"), recursive = TRUE)
  if (!dir.exists(file.path(output_dir, "metrics"))) dir.create(file.path(output_dir, "metrics"), recursive = TRUE)

  saveRDS(fit_final_wf, file.path(output_dir, "models", "xgboost_workflow.rds"))
  saveRDS(best_params, file.path(output_dir, "models", "xgboost_best_params.rds"))
  readr::write_csv(vi_table, file.path(output_dir, "metrics", "xgboost_variable_importance.csv"))

  message(">>> XGBoost model and variable importance table saved.")

  list(
    workflow_fit = fit_final_wf,
    best_params = best_params,
    variable_importance = vi_table
  )
}
