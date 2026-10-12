# ==============================================================================
# R/06_decision_tree.R
# Project: Insurance Claim Risk Prediction & Fraud Pattern Analysis
# Purpose: Interpretable Decision Tree classifier (rpart) for visual rule
#          extraction, presentation slides, and qualitative decision paths.
# ==============================================================================

suppressPackageStartupMessages({
  library(parsnip)
  library(workflows)
  library(yardstick)
  library(dplyr)
  library(tibble)
  library(readr)
})

#' Specify Interpretable Decision Tree Model
#'
#' Configures a shallow classification tree via the `rpart` engine designed specifically
#' for executive interpretability and presentation readability (depth 4, min_n 10).
#'
#' @param tree_depth Integer. Maximum depth of tree splits (default 4).
#' @param min_n Integer. Minimum observations required in a node to split (default 10).
#' @param cost_complexity Numeric. Complexity parameter pruning threshold (default 0.01).
#' @return A `parsnip` model specification.
#' @export
specify_decision_tree <- function(tree_depth = 4, min_n = 10, cost_complexity = 0.01) {
  parsnip::decision_tree(
    mode = "classification",
    tree_depth = tree_depth,
    min_n = min_n,
    cost_complexity = cost_complexity
  ) %>%
    parsnip::set_engine("rpart")
}

#' Build Decision Tree Workflow
#'
#' Connects the preprocessing recipe with the decision tree specification.
#'
#' @param recipe_spec Recipe object from `build_preprocessing_recipe`.
#' @param model_spec Model specification from `specify_decision_tree`.
#' @return A `workflows::workflow` object.
#' @export
build_decision_tree_workflow <- function(recipe_spec, model_spec = specify_decision_tree()) {
  workflows::workflow() %>%
    workflows::add_recipe(recipe_spec) %>%
    workflows::add_model(model_spec)
}

#' Fit Decision Tree Model and Extract Rules
#'
#' Fits the tree workflow on the training partition and extracts the underlying `rpart` object.
#'
#' @param wf Decision tree workflow.
#' @param train_data Training data frame.
#' @param output_dir Output folder path.
#' @return A list containing the fitted workflow and underlying rpart fit.
#' @export
fit_and_extract_tree <- function(wf, train_data, output_dir = "outputs") {
  message(">>> Fitting Decision Tree classifier on training partition...")
  fit_wf <- parsnip::fit(wf, data = train_data)

  # Extract underlying rpart model object
  rpart_fit <- workflows::extract_fit_engine(fit_wf)

  # Save artifacts
  if (!dir.exists(file.path(output_dir, "models"))) dir.create(file.path(output_dir, "models"), recursive = TRUE)
  saveRDS(fit_wf, file.path(output_dir, "models", "decision_tree_workflow.rds"))
  saveRDS(rpart_fit, file.path(output_dir, "models", "rpart_engine_fit.rds"))

  message(">>> Decision Tree model fitted and exported.")

  list(
    workflow_fit = fit_wf,
    rpart_fit = rpart_fit
  )
}

#' Render and Save Presentation-Ready Decision Tree Plot
#'
#' Uses `rpart.plot` (if available) or base graphics to export a clean, high-resolution
#' visual diagram of the decision rules suitable for executive slides and viva examinations.
#'
#' @param rpart_fit Underlying `rpart` object.
#' @param output_file File path for output PNG.
#' @param width Width in pixels or inches.
#' @param height Height in pixels or inches.
#' @export
save_decision_tree_plot <- function(rpart_fit, output_file = "outputs/figures/decision_tree.png",
                                    width = 1200, height = 900) {
  if (!dir.exists(dirname(output_file))) dir.create(dirname(output_file), recursive = TRUE)

  message(">>> Generating presentation-ready Decision Tree visualization: ", output_file)

  # Check if rpart.plot is installed, otherwise fallback gracefully to base plot
  if (requireNamespace("rpart.plot", quietly = TRUE)) {
    png(output_file, width = width, height = height, res = 150)
    rpart.plot::rpart.plot(
      rpart_fit,
      type = 2,
      extra = 104, # Probability of class + percentage of observations
      under = TRUE,
      fallen.leaves = TRUE,
      box.palette = "BuGn",
      main = "Auto Insurance Fraud Decision Tree Rules",
      sub = "Node: Predicted Class | P(Fraud) | % of Sample",
      roundint = FALSE,
      cex = 0.8
    )
    dev.off()
  } else {
    png(output_file, width = width, height = height, res = 150)
    plot(rpart_fit, uniform = TRUE, margin = 0.1, main = "Decision Tree Rules")
    text(rpart_fit, use.n = TRUE, all = TRUE, cex = 0.7)
    dev.off()
  }

  message(">>> Decision Tree diagram successfully rendered.")
}
