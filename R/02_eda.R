# ==============================================================================
# R/02_eda.R
# Project: Insurance Claim Risk Prediction & Fraud Pattern Analysis
# Purpose: Exploratory Data Analysis functions, class imbalance assessment,
#          missingness characterization, and bivariate relationship profiling.
# ==============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(tibble)
  library(readr)
})

#' Compute Comprehensive Dataset Overview
#'
#' Evaluates structural properties, column data types, duplicates, and missingness.
#'
#' @param df Data frame or tibble loaded by `load_insurance_data`.
#' @return A list containing overview tables and structural metrics.
#' @export
compute_dataset_overview <- function(df) {
  stopifnot(is.data.frame(df))

  n_rows <- nrow(df)
  n_cols <- ncol(df)
  n_duplicates <- sum(duplicated(df))

  # Column type census
  col_types <- tibble(
    column = names(df),
    data_type = sapply(df, function(x) class(x)[1]),
    n_unique = sapply(df, function(x) length(unique(x))),
    n_missing = sapply(df, function(x) sum(is.na(x))),
    pct_missing = sapply(df, function(x) round(100 * sum(is.na(x)) / n_rows, 2))
  ) %>%
    arrange(desc(n_missing))

  type_summary <- col_types %>%
    group_by(data_type) %>%
    summarise(count = n(), .groups = "drop")

  overview <- list(
    dimensions = c(rows = n_rows, columns = n_cols),
    duplicates = n_duplicates,
    type_summary = type_summary,
    missingness_profile = col_types %>% filter(n_missing > 0),
    column_census = col_types
  )

  return(overview)
}

#' Compute Target Variable Distribution and Class Imbalance Metrics
#'
#' Calculates raw frequencies, class proportions, and the class imbalance ratio
#' to quantify the severity of the Accuracy Paradox.
#'
#' @param df Data frame containing `fraud_reported`.
#' @return A tibble with target distribution metrics.
#' @export
compute_target_analysis <- function(df) {
  stopifnot("fraud_reported" %in% names(df))

  target_tbl <- df %>%
    group_by(fraud_reported) %>%
    summarise(
      count = n(),
      .groups = "drop"
    ) %>%
    mutate(
      percentage = round(100 * count / sum(count), 2),
      prevalence = round(count / sum(count), 4),
      odds = round(count / (sum(count) - count), 4)
    )

  neg_count <- target_tbl$count[target_tbl$fraud_reported == "No"]
  pos_count <- target_tbl$count[target_tbl$fraud_reported == "Yes"]
  imbalance_ratio <- round(neg_count / pos_count, 2)

  attr(target_tbl, "imbalance_ratio") <- imbalance_ratio
  attr(target_tbl, "minority_pct") <- target_tbl$percentage[target_tbl$fraud_reported == "Yes"]

  return(target_tbl)
}

#' Profile Categorical Features vs Fraud Rate
#'
#' Computes category-level sample sizes, fraud frequencies, and fraud incidence percentages.
#'
#' @param df Data frame containing features and `fraud_reported`.
#' @param cat_cols Character vector of categorical column names to analyze.
#' @return A tibble of category-level fraud incidence rates.
#' @export
profile_categorical_fraud <- function(df, cat_cols = NULL) {
  if (is.null(cat_cols)) {
    cat_cols <- c(
      "incident_type", "collision_type", "incident_severity",
      "authorities_contacted", "incident_state", "policy_state",
      "insured_education_level", "insured_occupation", "insured_hobbies",
      "property_damage", "police_report_available", "auto_make"
    )
  }

  cat_cols <- intersect(cat_cols, names(df))

  results <- list()
  for (col in cat_cols) {
    tbl <- df %>%
      mutate(var_val = as.character(.data[[col]])) %>%
      mutate(var_val = ifelse(is.na(var_val), "(Missing / NA)", var_val)) %>%
      group_by(var_val) %>%
      summarise(
        total_claims = n(),
        fraud_claims = sum(fraud_reported == "Yes"),
        fraud_rate_pct = round(100 * fraud_claims / total_claims, 2),
        .groups = "drop"
      ) %>%
      arrange(desc(fraud_rate_pct)) %>%
      mutate(feature_name = col) %>%
      select(feature_name, level = var_val, total_claims, fraud_claims, fraud_rate_pct)

    results[[col]] <- tbl
  }

  bind_rows(results)
}

#' Profile Numeric Features by Fraud Status
#'
#' Calculates median, interquartile range (IQR), mean, and standard deviation
#' for continuous numeric predictors across fraud vs non-fraud outcomes.
#'
#' @param df Data frame containing features and `fraud_reported`.
#' @param num_cols Character vector of numeric column names to analyze.
#' @return A tibble of stratified numeric summaries.
#' @export
profile_numeric_fraud <- function(df, num_cols = NULL) {
  if (is.null(num_cols)) {
    num_cols <- c(
      "months_as_customer", "age", "policy_annual_premium", "policy_deductable",
      "umbrella_limit", "capital_gains", "capital_loss", "incident_hour_of_the_day",
      "number_of_vehicles_involved", "bodily_injuries", "witnesses",
      "total_claim_amount", "injury_claim", "property_claim", "vehicle_claim", "auto_year"
    )
  }

  num_cols <- intersect(num_cols, names(df))

  df %>%
    select(fraud_reported, all_of(num_cols)) %>%
    pivot_longer(cols = -fraud_reported, names_to = "feature", values_to = "value") %>%
    group_by(feature, fraud_reported) %>%
    summarise(
      n = n(),
      median = round(median(value, na.rm = TRUE), 2),
      iqr = round(IQR(value, na.rm = TRUE), 2),
      mean = round(mean(value, na.rm = TRUE), 2),
      sd = round(sd(value, na.rm = TRUE), 2),
      min = round(min(value, na.rm = TRUE), 2),
      max = round(max(value, na.rm = TRUE), 2),
      .groups = "drop"
    ) %>%
    arrange(feature, fraud_reported)
}

#' Execute and Save Full EDA Summary Suite
#'
#' Runs all EDA profiling functions and exports summaries to `outputs/metrics/`.
#'
#' @param df Cleaned data frame from `load_insurance_data`.
#' @param output_dir Directory where metric files should be saved.
#' @return A named list of all calculated EDA summaries.
#' @export
run_eda_pipeline <- function(df, output_dir = "outputs/metrics") {
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

  message(">>> Executing Exploratory Data Analysis profiling...")
  overview <- compute_dataset_overview(df)
  target_summary <- compute_target_analysis(df)
  cat_profile <- profile_categorical_fraud(df)
  num_profile <- profile_numeric_fraud(df)

  # Save metrics as CSV and RDS for downstream scripts and reports
  saveRDS(overview, file.path(output_dir, "eda_overview.rds"))
  readr::write_csv(target_summary, file.path(output_dir, "eda_target_summary.csv"))
  readr::write_csv(cat_profile, file.path(output_dir, "eda_categorical_profiles.csv"))
  readr::write_csv(num_profile, file.path(output_dir, "eda_numeric_profiles.csv"))

  message(">>> EDA profiling completed. Summaries stored in: ", output_dir)

  list(
    overview = overview,
    target_summary = target_summary,
    cat_profile = cat_profile,
    num_profile = num_profile
  )
}
