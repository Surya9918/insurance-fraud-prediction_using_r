# ==============================================================================
# R/03_preprocessing.R
# Project: Insurance Claim Risk Prediction & Fraud Pattern Analysis
# Purpose: Reproducible data splitting, cross-validation setup, and tidymodels
#          recipes with median/mode imputation, rare-level pooling, dummy
#          encoding, and leak-free training-only SMOTE oversampling.
# ==============================================================================

suppressPackageStartupMessages({
  library(rsample)
  library(recipes)
  library(themis)
  library(dplyr)
})

#' Partition Dataset into Train and Test Sets
#'
#' Executes an 80/20 stratified split based on the target variable `fraud_reported`.
#' Setting `strata = fraud_reported` guarantees that both splits preserve the original
#' empirical class distribution (~75.3% No, ~24.7% Yes).
#'
#' @param df Cleaned data frame from `load_insurance_data`.
#' @param prop Numeric. Proportion of records allocated to training (default 0.80).
#' @param seed Integer. Random seed for deterministic reproducibility.
#' @return An `rsplit` object containing training and test subsets.
#' @export
create_train_test_split <- function(df, prop = 0.80, seed = 123) {
  set.seed(seed)
  message(sprintf(">>> Creating stratified train/test split (prop = %.2f, seed = %d)...", prop, seed))

  split_obj <- rsample::initial_split(
    df,
    prop = prop,
    strata = fraud_reported
  )

  train_data <- rsample::training(split_obj)
  test_data  <- rsample::testing(split_obj)

  message(sprintf(">>> Train set: %d observations (Fraud: %d, %.2f%%)",
                  nrow(train_data),
                  sum(train_data$fraud_reported == "Yes"),
                  100 * mean(train_data$fraud_reported == "Yes")))
  message(sprintf(">>> Test set:  %d observations (Fraud: %d, %.2f%%)",
                  nrow(test_data),
                  sum(test_data$fraud_reported == "Yes"),
                  100 * mean(test_data$fraud_reported == "Yes")))

  return(split_obj)
}

#' Generate Stratified Cross-Validation Folds
#'
#' Constructs 5-fold cross-validation folds stratified on `fraud_reported`.
#'
#' @param train_data Training data frame.
#' @param v Integer. Number of cross-validation partitions (default 5).
#' @param seed Integer. Random seed for fold assignment.
#' @return A `vfold_cv` object.
#' @export
create_cv_folds <- function(train_data, v = 5, seed = 123) {
  set.seed(seed)
  message(sprintf(">>> Generating %d-fold stratified cross-validation folds (seed = %d)...", v, seed))

  rsample::vfold_cv(
    train_data,
    v = v,
    strata = fraud_reported
  )
}

#' Build Tidymodels Preprocessing Recipe
#'
#' Defines the end-to-end transformation specification adhering strictly to leak-free
#' machine learning engineering principles:
#'
#' 1. Identifier & Leakage Removal: Drops high-cardinality, non-predictive identifiers
#'    (`policy_number`, `insured_zip`, `incident_location`) and exact date timestamps
#'    (`policy_bind_date`, `incident_date`) that encourage memorization over generalization.
#' 2. Numeric Imputation: `step_impute_median()` imputes numeric features using medians
#'    computed solely from the training partition.
#' 3. Nominal Imputation: `step_impute_mode()` replaces missing categorical values with the mode.
#' 4. High-Cardinality Pooling: `step_other()` collapses sparse levels (< 5% prevalence)
#'    into a unified `"other"` bucket to prevent sparse dummy expansion and rank deficiency.
#' 5. Dummy Encoding: `step_dummy()` generates indicator columns for all nominal predictors.
#' 6. Zero-Variance Filtering: `step_zv()` removes any degenerate columns with zero variance.
#' 7. SMOTE Balancing: When `use_smote = TRUE`, `themis::step_smote()` synthesizes minority
#'    instances in feature space. CRITICAL: SMOTE is encapsulated within the tidymodels
#'    recipe so it is executed ONLY on training partitions and NEVER touches the test set!
#'
#' @param train_data Training data frame for formula and schema initialization.
#' @param use_smote Logical. Whether to include SMOTE oversampling in the pipeline.
#' @param other_threshold Numeric. Threshold for pooling rare categorical levels (default 0.05).
#' @return An untrained `recipe` specification.
#' @export
build_preprocessing_recipe <- function(train_data, use_smote = TRUE, other_threshold = 0.05) {
  message(">>> Assembling Tidymodels preprocessing recipe...")

  # Define columns explicitly flagged for removal based on domain audit
  cols_to_remove <- c(
    "policy_number",     # Arbitrary unique identifier; risk of memorization
    "policy_bind_date",  # Exact date with near-unique values
    "incident_date",     # Exact incident timestamp; risk of temporal overfitting
    "incident_location", # Granular street address (~1000 unique values)
    "insured_zip"        # High cardinality postal identifier (~995 distinct values)
  )

  # Retain only removal columns that actually exist in the data
  cols_to_remove <- intersect(cols_to_remove, names(train_data))

  rec <- recipes::recipe(fraud_reported ~ ., data = train_data)

  # Step 1: Remove uninformative identifiers and date stamps
  if (length(cols_to_remove) > 0) {
    rec <- rec %>%
      recipes::step_rm(all_of(cols_to_remove))
  }

  # Step 2: Impute missing values
  # Numeric predictors via median; nominal predictors via mode
  rec <- rec %>%
    recipes::step_impute_median(recipes::all_numeric_predictors()) %>%
    recipes::step_impute_mode(recipes::all_nominal_predictors(), -recipes::all_outcomes())

  # Step 3: Handle high-cardinality & rare categories (< 5% frequency)
  # Prevents explosive dummy dimensionality and unseen factor level failure
  rec <- rec %>%
    recipes::step_other(
      recipes::all_nominal_predictors(),
      -recipes::all_outcomes(),
      threshold = other_threshold,
      other = "other"
    )

  # Step 4: Dummy / One-Hot Encoding for nominal predictors
  rec <- rec %>%
    recipes::step_dummy(
      recipes::all_nominal_predictors(),
      -recipes::all_outcomes(),
      one_hot = FALSE
    )

  # Step 5: Filter zero-variance columns created during dummy encoding
  rec <- rec %>%
    recipes::step_zv(recipes::all_predictors())

  # Step 6: Address class imbalance with SMOTE (Training data ONLY)
  if (use_smote) {
    message(">>> Integrating themis::step_smote() into training recipe pipeline.")
    rec <- rec %>%
      themis::step_smote(
        fraud_reported,
        over_ratio = 1.0, # Equalize minority to majority class count
        seed = 123
      )
  }

  return(rec)
}

#' Inspect Transformed Feature Matrix
#'
#' Preps and bakes the recipe on training data to inspect the transformed feature dimensions
#' and verify that no missing values or degenerate columns remain.
#'
#' @param recipe_spec Recipe object from `build_preprocessing_recipe`.
#' @param train_data Training data frame.
#' @return A prepped recipe object.
#' @export
audit_recipe_pipeline <- function(recipe_spec, train_data) {
  message(">>> Prepping recipe on training data to audit feature transformations...")
  prepped_rec <- recipes::prep(recipe_spec, training = train_data, verbose = FALSE)
  baked_train <- recipes::bake(prepped_rec, new_data = NULL)

  message(sprintf(">>> Prepped features: %d columns generated.", ncol(baked_train)))
  message(sprintf(">>> Post-recipe class distribution: No = %d, Yes = %d",
                  sum(baked_train$fraud_reported == "No"),
                  sum(baked_train$fraud_reported == "Yes")))

  na_remaining <- sum(is.na(baked_train))
  if (na_remaining > 0) {
    warning(sprintf("Baked training data contains %d remaining NA values!", na_remaining))
  } else {
    message(">>> Data integrity verified: 0 missing values in transformed training set.")
  }

  return(prepped_rec)
}
