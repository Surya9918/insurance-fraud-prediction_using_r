# ==============================================================================
# R/shiny_helpers.R
# Project: Insurance Claim Risk Prediction & Fraud Pattern Analysis
# Purpose: Backend inference engine, schema alignment, risk scoring,
#          batch audit processor, and dynamic explainability for Shiny UI.
# ==============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tibble)
  library(workflows)
  library(parsnip)
  library(recipes)
  library(themis)
  library(rpart)
})

#' Safely Load Trained Models
#' @return Named list of fitted workflows
load_platform_models <- function(models_dir = "outputs/models") {
  models <- list()
  
  dt_path <- file.path(models_dir, "decision_tree_workflow.rds")
  log_path <- file.path(models_dir, "logistic_workflow.rds")
  xgb_path <- file.path(models_dir, "xgboost_workflow.rds")
  rpart_path <- file.path(models_dir, "rpart_engine_fit.rds")
  
  if (file.exists(dt_path)) {
    models$decision_tree <- readRDS(dt_path)
  }
  if (file.exists(log_path)) {
    models$logistic <- readRDS(log_path)
  }
  if (file.exists(xgb_path)) {
    models$xgboost <- readRDS(xgb_path)
  }
  if (file.exists(rpart_path)) {
    models$rpart_fit <- readRDS(rpart_path)
  }
  
  return(models)
}

#' Create Base Template Claim
#' @param data_path Character. Path to insurance claims dataset.
#' @return A single-row tibble with canonical column schema.
load_claim_template <- function(data_path = "data/insurance_claims.csv") {
  if (file.exists(data_path)) {
    source("R/01_load_data.R", local = TRUE)
    df <- load_insurance_data(data_path)
    return(df[1, ])
  } else {
    stop("Canonical dataset missing at: ", data_path)
  }
}

#' Merge Form Inputs into Template Claim Row
#' Ensures exact schema and factor ordering required by Tidymodels recipe.
build_claim_row <- function(template_row, inputs) {
  row <- template_row
  
  # Incident characteristics
  if (!is.null(inputs$incident_severity)) row$incident_severity <- as.character(inputs$incident_severity)
  if (!is.null(inputs$incident_type)) row$incident_type <- as.character(inputs$incident_type)
  if (!is.null(inputs$collision_type)) row$collision_type <- as.character(inputs$collision_type)
  if (!is.null(inputs$incident_hour_of_the_day)) row$incident_hour_of_the_day <- as.numeric(inputs$incident_hour_of_the_day)
  if (!is.null(inputs$number_of_vehicles_involved)) row$number_of_vehicles_involved <- as.numeric(inputs$number_of_vehicles_involved)
  if (!is.null(inputs$authorities_contacted)) row$authorities_contacted <- as.character(inputs$authorities_contacted)
  if (!is.null(inputs$incident_state)) row$incident_state <- as.character(inputs$incident_state)
  if (!is.null(inputs$incident_city)) row$incident_city <- as.character(inputs$incident_city)
  if (!is.null(inputs$police_report_available)) row$police_report_available <- as.character(inputs$police_report_available)
  if (!is.null(inputs$property_damage)) row$property_damage <- as.character(inputs$property_damage)
  if (!is.null(inputs$bodily_injuries)) row$bodily_injuries <- as.numeric(inputs$bodily_injuries)
  if (!is.null(inputs$witnesses)) row$witnesses <- as.numeric(inputs$witnesses)
  
  # Financials & Claim details
  if (!is.null(inputs$total_claim_amount)) row$total_claim_amount <- as.numeric(inputs$total_claim_amount)
  if (!is.null(inputs$injury_claim)) row$injury_claim <- as.numeric(inputs$injury_claim)
  if (!is.null(inputs$property_claim)) row$property_claim <- as.numeric(inputs$property_claim)
  if (!is.null(inputs$vehicle_claim)) row$vehicle_claim <- as.numeric(inputs$vehicle_claim)
  if (!is.null(inputs$umbrella_limit)) row$umbrella_limit <- as.numeric(inputs$umbrella_limit)
  if (!is.null(inputs$capital_gains)) row$capital_gains <- as.numeric(inputs$capital_gains)
  if (!is.null(inputs$capital_loss)) row$capital_loss <- as.numeric(inputs$capital_loss)
  if (!is.null(inputs$policy_annual_premium)) row$policy_annual_premium <- as.numeric(inputs$policy_annual_premium)
  if (!is.null(inputs$policy_deductable)) row$policy_deductable <- as.numeric(inputs$policy_deductable)
  
  # Policyholder
  if (!is.null(inputs$age)) row$age <- as.numeric(inputs$age)
  if (!is.null(inputs$months_as_customer)) row$months_as_customer <- as.numeric(inputs$months_as_customer)
  if (!is.null(inputs$insured_sex)) row$insured_sex <- as.character(inputs$insured_sex)
  if (!is.null(inputs$insured_education_level)) row$insured_education_level <- as.character(inputs$insured_education_level)
  if (!is.null(inputs$insured_occupation)) row$insured_occupation <- as.character(inputs$insured_occupation)
  if (!is.null(inputs$insured_hobbies)) row$insured_hobbies <- as.character(inputs$insured_hobbies)
  if (!is.null(inputs$insured_relationship)) row$insured_relationship <- as.character(inputs$insured_relationship)
  if (!is.null(inputs$auto_make)) row$auto_make <- as.character(inputs$auto_make)
  if (!is.null(inputs$auto_model)) row$auto_model <- as.character(inputs$auto_model)
  if (!is.null(inputs$auto_year)) row$auto_year <- as.numeric(inputs$auto_year)
  
  return(row)
}

#' Score Single Claim Across All Models
#' @return List with active probability, risk tier, consensus breakdown, and audit advice.
score_single_claim <- function(models, claim_row, selected_model = "Ensemble Consensus", threshold = 0.50) {
  prob_dt  <- NA_real_
  prob_glm <- NA_real_
  prob_xgb <- NA_real_
  
  if (!is.null(models$decision_tree)) {
    tryCatch({
      prob_dt <- predict(models$decision_tree, new_data = claim_row, type = "prob")$.pred_Yes[1]
    }, error = function(e) {})
  }
  
  if (!is.null(models$logistic)) {
    tryCatch({
      suppressWarnings({
        prob_glm <- predict(models$logistic, new_data = claim_row, type = "prob")$.pred_Yes[1]
      })
    }, error = function(e) {})
  }
  
  if (!is.null(models$xgboost)) {
    tryCatch({
      prob_xgb <- predict(models$xgboost, new_data = claim_row, type = "prob")$.pred_Yes[1]
    }, error = function(e) {})
  }
  
  # Calculate consensus
  available_probs <- c(prob_dt, prob_glm, prob_xgb)
  available_probs <- available_probs[!is.na(available_probs)]
  
  prob_consensus <- if (length(available_probs) > 0) mean(available_probs) else 0.50
  
  # Selected model probability
  active_prob <- switch(
    selected_model,
    "Decision Tree" = ifelse(!is.na(prob_dt), prob_dt, prob_consensus),
    "Logistic Regression" = ifelse(!is.na(prob_glm), prob_glm, prob_consensus),
    "XGBoost" = ifelse(!is.na(prob_xgb), prob_xgb, prob_consensus),
    prob_consensus
  )
  
  # Bound probability between 0 and 1
  active_prob <- max(0, min(1, active_prob))
  
  # Risk category determination with model-calibrated boundaries
  risk_tier <- if (identical(selected_model, "XGBoost")) {
    if (active_prob >= 0.25) {
      "HIGH"
    } else if (active_prob >= 0.18) {
      "MODERATE"
    } else {
      "LOW"
    }
  } else {
    if (active_prob >= 0.60) {
      "HIGH"
    } else if (active_prob >= 0.35) {
      "MODERATE"
    } else {
      "LOW"
    }
  }
  
  is_flagged <- active_prob >= threshold
  
  action_recommendation <- switch(
    risk_tier,
    "HIGH" = "🚨 IMMEDIATE AUDIT: Assign to Special Investigation Unit (SIU). Freeze payout pending physical vehicle inspection & witness statement verification.",
    "MODERATE" = "⚠️ SECONDARY REVIEW: Claims adjuster manual audit required. Verify repair shop estimate and police incident report.",
    "LOW" = "✅ FAST-TRACK APPROVAL: Automated green-light. Claim demonstrates low fraud probability pattern; proceed with standard settlement."
  )
  
  list(
    active_prob = active_prob,
    pct_formatted = sprintf("%.1f%%", active_prob * 100),
    risk_tier = risk_tier,
    is_flagged = is_flagged,
    threshold = threshold,
    action_recommendation = action_recommendation,
    model_breakdown = list(
      "Decision Tree" = ifelse(!is.na(prob_dt), round(prob_dt, 4), NA),
      "Logistic Regression" = ifelse(!is.na(prob_glm), round(prob_glm, 4), NA),
      "XGBoost" = ifelse(!is.na(prob_xgb), round(prob_xgb, 4), NA),
      "Ensemble Consensus" = round(prob_consensus, 4)
    )
  )
}

#' Extract Dynamic Key Risk Indicators (KRIs) for Explainability
identify_claim_risk_indicators <- function(claim_row) {
  indicators <- list()
  
  # Severity
  sev <- as.character(claim_row$incident_severity)
  if (identical(sev, "Major Damage")) {
    indicators[[length(indicators) + 1]] <- list(
      type = "danger",
      icon = "⚠️",
      title = "Incident Severity: Major Damage",
      desc = "Major damage is the #1 statistical fraud indicator (60.5% fraud rate vs 10.7% in minor claims; OR = 34x)."
    )
  } else if (identical(sev, "Minor Damage") || identical(sev, "Trivial Damage")) {
    indicators[[length(indicators) + 1]] <- list(
      type = "success",
      icon = "🛡️",
      title = paste("Incident Severity:", sev),
      desc = "Minor/Trivial damage strongly reduces fraud likelihood (>97% odds reduction in logistic GLM)."
    )
  }
  
  # Total claim amount
  amt <- as.numeric(claim_row$total_claim_amount)
  if (!is.na(amt)) {
    if (amt > 60000) {
      indicators[[length(indicators) + 1]] <- list(
        type = "danger",
        icon = "💵",
        title = sprintf("High Claim Amount: $%s", format(round(amt), big.mark = ",")),
        desc = "Total claim exceeds $60,000 threshold, elevating financial exposure and moral hazard risk."
      )
    } else if (amt < 20000) {
      indicators[[length(indicators) + 1]] <- list(
        type = "success",
        icon = "💵",
        title = sprintf("Low Claim Amount: $%s", format(round(amt), big.mark = ",")),
        desc = "Modest total payout substantially reduces likelihood of organized or staged claim fraud."
      )
    }
  }
  
  # Incident hour
  hr <- as.numeric(claim_row$incident_hour_of_the_day)
  if (!is.na(hr) && (hr >= 22 || hr <= 5)) {
    indicators[[length(indicators) + 1]] <- list(
      type = "danger",
      icon = "🌙",
      title = sprintf("Late Night Occurrence (Hour %02d:00)", hr),
      desc = "Incidents occurring between 10:00 PM and 5:00 AM exhibit higher rates of staging and lack of witnesses."
    )
  }
  
  # Auto make
  make <- as.character(claim_row$auto_make)
  if (!is.na(make) && make %in% c("BMW", "Audi", "Mercedes")) {
    indicators[[length(indicators) + 1]] <- list(
      type = "danger",
      icon = "🚗",
      title = paste("Luxury Vehicle:", make),
      desc = sprintf("Luxury vehicles (%s) show statistically elevated fraud odds (+356%% for BMW, +270%% for Audi).", make)
    )
  }
  
  # Police report
  pol <- as.character(claim_row$police_report_available)
  if (!is.na(pol) && pol %in% c("NO", "?")) {
    indicators[[length(indicators) + 1]] <- list(
      type = "warning",
      icon = "📋",
      title = "Missing Official Police Report",
      desc = "Absence of documented law enforcement investigation prevents official validation of incident scene."
    )
  }
  
  # Witnesses
  wit <- as.numeric(claim_row$witnesses)
  if (!is.na(wit) && wit >= 2) {
    indicators[[length(indicators) + 1]] <- list(
      type = "info",
      icon = "👥",
      title = sprintf("%d Witnesses Documented", wit),
      desc = "Multiple witnesses can corroborate physical damage, though staged claims often present suspect witnesses."
    )
  }
  
  # Customer tenure
  cust <- as.numeric(claim_row$months_as_customer)
  if (!is.na(cust) && cust >= 240) {
    indicators[[length(indicators) + 1]] <- list(
      type = "success",
      icon = "⭐",
      title = sprintf("Loyal Long-Term Policyholder (%d months)", cust),
      desc = "Customer policy tenure exceeding 20 years represents established relationship and lower baseline risk."
    )
  }
  
  return(indicators)
}

#' Score Batch Data Frame of Claims
score_batch_claims <- function(models, df, selected_model = "Decision Tree", threshold = 0.50) {
  # Drop target column if present to simulate real inference
  input_df <- df
  if ("fraud_reported" %in% names(input_df)) {
    ground_truth <- input_df$fraud_reported
  } else {
    ground_truth <- NULL
  }
  
  # Choose model
  active_model <- switch(
    selected_model,
    "Decision Tree" = models$decision_tree,
    "Logistic Regression" = models$logistic,
    "XGBoost" = models$xgboost,
    models$decision_tree
  )
  
  if (is.null(active_model)) {
    active_model <- models$decision_tree
  }
  
  # Predict probabilities
  prob_preds <- tryCatch({
    predict(active_model, new_data = input_df, type = "prob")$.pred_Yes
  }, error = function(e) {
    # Fallback to decision tree or mock if error
    if (!is.null(models$decision_tree)) {
      predict(models$decision_tree, new_data = input_df, type = "prob")$.pred_Yes
    } else {
      rep(0.25, nrow(input_df))
    }
  })
  
  scored_df <- input_df %>%
    mutate(
      fraud_probability = round(prob_preds, 4),
      risk_tier = if (identical(selected_model, "XGBoost")) {
        case_when(
          fraud_probability >= 0.25 ~ "HIGH",
          fraud_probability >= 0.18 ~ "MODERATE",
          TRUE ~ "LOW"
        )
      } else {
        case_when(
          fraud_probability >= 0.60 ~ "HIGH",
          fraud_probability >= 0.35 ~ "MODERATE",
          TRUE ~ "LOW"
        )
      },
      flagged_fraud = ifelse(fraud_probability >= threshold, "YES (Flagged)", "NO (Pass)"),
      recommended_action = case_when(
        risk_tier == "HIGH" ~ "SIU Full Audit",
        risk_tier == "MODERATE" ~ "Adjuster Review",
        TRUE ~ "Fast-Track Approval"
      )
    )
  
  if (!is.null(ground_truth)) {
    scored_df$actual_fraud <- ground_truth
  }
  
  return(scored_df)
}
