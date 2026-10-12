# ==============================================================================
# R/01_load_data.R
# Project: Insurance Claim Risk Prediction & Fraud Pattern Analysis
# Purpose: Robust data ingestion, missing value sanitization ('?' to NA),
#          target factor formatting, and schema validation.
# ==============================================================================

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tibble)
})

#' Load and Clean Auto Insurance Claims Dataset
#'
#' Reads the auto insurance claims dataset from CSV, safely converting literal
#' question marks ('?') into native R `NA` values without corrupting numeric columns,
#' drops export artifact columns (e.g. `_c39`), standardizes column names, and
#' converts the target `fraud_reported` into an appropriately ordered classification factor.
#'
#' @param file_path Character. Path to the CSV dataset. Defaults to "data/insurance_claims.csv".
#' @return A cleaned tibble with standardized column names and factor target.
#' @export
load_insurance_data <- function(file_path = "data/insurance_claims.csv") {
  if (!file.exists(file_path)) {
    stop(
      paste0(
        "Dataset not found at '", file_path, "'.\n",
        "Please follow instructions in 'data/README.md' to obtain 'insurance_claims.csv'\n",
        "from Kaggle (brijbhushannanda1979/auto-insurance-fraud-dataset) or Mendeley Data."
      ),
      call. = FALSE
    )
  }

  message(">>> Loading raw data from: ", file_path)

  # Read CSV specifying '?' as an explicit missing value representation
  raw_df <- readr::read_csv(
    file_path,
    na = c("", "NA", "?", "NULL"),
    show_col_types = FALSE,
    trim_ws = TRUE
  )

  # Drop export artifact column `_c39` if present (often produced by trailing commas)
  if ("_c39" %in% names(raw_df)) {
    raw_df <- raw_df %>% dplyr::select(-`_c39`)
  }

  # Standardize column names (replace hyphens with underscores, ensure lower case)
  clean_names <- names(raw_df) %>%
    gsub("-", "_", .) %>%
    gsub(" ", "_", .) %>%
    tolower()
  names(raw_df) <- clean_names

  # Validate presence of target variable
  if (!"fraud_reported" %in% names(raw_df)) {
    stop("Target variable 'fraud_reported' not found in dataset columns.", call. = FALSE)
  }

  # Standardize target variable: ensure Yes/No factor representation
  # Raw data frequently contains 'Y' and 'N' or 'Yes' and 'No'
  raw_df <- raw_df %>%
    dplyr::mutate(
      fraud_reported = case_when(
        fraud_reported %in% c("Y", "Yes", "yes", "1") ~ "Yes",
        fraud_reported %in% c("N", "No", "no", "0")   ~ "No",
        TRUE ~ as.character(fraud_reported)
      ),
      # Set factor levels strictly as c("No", "Yes")
      # In tidymodels, event_level = "second" designates "Yes" as the positive event
      fraud_reported = factor(fraud_reported, levels = c("No", "Yes"))
    )

  # Safety check: Ensure no NA values in target
  if (any(is.na(raw_df$fraud_reported))) {
    warning("Target variable 'fraud_reported' contains missing values. Removing rows with missing target.")
    raw_df <- raw_df %>% dplyr::filter(!is.na(fraud_reported))
  }

  message(sprintf(">>> Ingestion complete: %d rows, %d columns.", nrow(raw_df), ncol(raw_df)))
  message(sprintf(">>> Target distribution: No = %d (%.2f%%), Yes = %d (%.2f%%).",
                  sum(raw_df$fraud_reported == "No"),
                  100 * mean(raw_df$fraud_reported == "No"),
                  sum(raw_df$fraud_reported == "Yes"),
                  100 * mean(raw_df$fraud_reported == "Yes")))

  return(raw_df)
}

#' Validate Schema & Quality of Loaded Dataset
#'
#' Performs structural assertions on the dataset dimensions, target levels,
#' and missing value columns.
#'
#' @param df A data frame or tibble loaded by `load_insurance_data`.
#' @return A named list of validation statistics.
#' @export
validate_insurance_data <- function(df) {
  stopifnot(is.data.frame(df))
  stopifnot("fraud_reported" %in% names(df))
  stopifnot(is.factor(df$fraud_reported))
  stopifnot(identical(levels(df$fraud_reported), c("No", "Yes")))

  missing_counts <- colSums(is.na(df))
  cols_with_na <- missing_counts[missing_counts > 0]

  class_counts <- table(df$fraud_reported)
  imbalance_ratio <- max(class_counts) / min(class_counts)

  summary_info <- list(
    n_rows = nrow(df),
    n_cols = ncol(df),
    target_levels = levels(df$fraud_reported),
    class_counts = class_counts,
    fraud_prevalence = mean(df$fraud_reported == "Yes"),
    imbalance_ratio = imbalance_ratio,
    cols_with_missing = cols_with_na,
    total_missing_cells = sum(missing_counts)
  )

  message("--- Data Validation Summary ---")
  message(sprintf("Total Observations: %d", summary_info$n_rows))
  message(sprintf("Total Predictors: %d", summary_info$n_cols - 1))
  message(sprintf("Fraud Prevalence: %.2f%% (Imbalance Ratio: %.2f:1)",
                  100 * summary_info$fraud_prevalence, summary_info$imbalance_ratio))
  message(sprintf("Columns with Missingness: %d", length(cols_with_na)))

  return(summary_info)
}

# If run directly as a script:
if (sys.nframe() == 0) {
  data_file <- "data/insurance_claims.csv"
  if (file.exists(data_file)) {
    claims <- load_insurance_data(data_file)
    val <- validate_insurance_data(claims)
  } else {
    message("Notice: 'data/insurance_claims.csv' not found. Run data/download_data.sh to download.")
  }
}
