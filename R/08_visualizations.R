# ==============================================================================
# R/08_visualizations.R
# Project: Insurance Claim Risk Prediction & Fraud Pattern Analysis
# Purpose: High-resolution publication-quality ggplot2 visualization pipeline
#          for exploratory data analysis, evaluation curves, feature importance,
#          odds ratios, and confusion matrix diagnostics.
# ==============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(scales)
  library(yardstick)
})

# Define publication-ready theme
theme_fraud <- function(base_size = 11) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.title = element_text(face = "bold", size = rel(1.15), margin = margin(b = 6)),
      plot.subtitle = element_text(color = "grey30", size = rel(0.95), margin = margin(b = 10)),
      plot.caption = element_text(color = "grey50", size = rel(0.8), hjust = 0, margin = margin(t = 8)),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_line(color = "grey90", linewidth = 0.4),
      panel.grid.major.y = element_line(color = "grey90", linewidth = 0.4),
      legend.position = "bottom",
      legend.title = element_text(face = "bold"),
      strip.text = element_text(face = "bold", size = rel(1.0))
    )
}

#' Plot Class Imbalance and the Accuracy Paradox
#' @export
plot_class_distribution <- function(df, output_path = "outputs/figures/01_class_imbalance.png") {
  p_data <- df %>%
    group_by(fraud_reported) %>%
    summarise(count = n(), .groups = "drop") %>%
    mutate(
      pct = count / sum(count),
      label = sprintf("%d\n(%.1f%%)", count, 100 * pct),
      class_type = ifelse(fraud_reported == "Yes", "Minority Class (Fraud)", "Majority Class (Legitimate)")
    )

  p <- ggplot(p_data, aes(x = fraud_reported, y = count, fill = fraud_reported)) +
    geom_col(width = 0.55, show.legend = FALSE, alpha = 0.9) +
    geom_text(aes(label = label), vjust = 1.3, color = "white", fontface = "bold", size = 4.2) +
    scale_fill_manual(values = c("No" = "#2b5c8f", "Yes" = "#d95f02")) +
    scale_y_continuous(labels = comma, expand = expansion(mult = c(0, 0.08))) +
    labs(
      title = "Class Imbalance: Auto Insurance Fraud vs Legitimate Claims",
      subtitle = "Severe ~3:1 imbalance illustrates the Accuracy Paradox: a naive 'No Fraud' rule yields 75.3% accuracy",
      x = "Fraud Reported Ground Truth",
      y = "Claim Count (N)",
      caption = "Source: Kaggle Auto Insurance Fraud Dataset (N = 1,000)"
    ) +
    theme_fraud()

  ggsave(output_path, plot = p, width = 7, height = 5, dpi = 300)
  return(p)
}

#' Plot Missing Value Frequencies
#' @export
plot_missing_values <- function(df, output_path = "outputs/figures/02_missing_values.png") {
  miss_df <- tibble(
    feature = names(df),
    missing_count = sapply(df, function(x) sum(is.na(x))),
    missing_pct = sapply(df, function(x) round(100 * sum(is.na(x)) / nrow(df), 2))
  ) %>%
    filter(missing_count > 0) %>%
    arrange(desc(missing_count))

  if (nrow(miss_df) == 0) {
    message(">>> No missing values detected in dataset.")
    return(NULL)
  }

  p <- ggplot(miss_df, aes(x = reorder(feature, missing_pct), y = missing_pct)) +
    geom_col(fill = "#e7298a", width = 0.6, alpha = 0.85) +
    geom_text(aes(label = sprintf("%d (%.1f%%)", missing_count, missing_pct)),
              hjust = -0.1, size = 3.8, fontface = "bold") +
    coord_flip() +
    scale_y_continuous(limits = c(0, max(miss_df$missing_pct) * 1.25), labels = function(x) paste0(x, "%")) +
    labs(
      title = "Missingness Profile: Encoded '?' Characters Converted to NA",
      subtitle = "Missing values concentrated in collision dynamics and administrative documentation fields",
      x = "Feature",
      y = "Percentage Missing (%)",
      caption = "Converted from literal '?' strings prior to imputation"
    ) +
    theme_fraud()

  ggsave(output_path, plot = p, width = 7.5, height = 4.5, dpi = 300)
  return(p)
}

#' Plot Incident Severity vs Fraud Rate
#' @export
plot_incident_severity <- function(df, output_path = "outputs/figures/03_incident_severity_vs_fraud.png") {
  p_data <- df %>%
    group_by(incident_severity) %>%
    summarise(
      total = n(),
      fraud = sum(fraud_reported == "Yes"),
      rate = fraud / total,
      .groups = "drop"
    ) %>%
    arrange(desc(rate))

  p <- ggplot(p_data, aes(x = reorder(incident_severity, rate), y = rate, fill = rate)) +
    geom_col(width = 0.6, show.legend = FALSE) +
    geom_text(aes(label = sprintf("%.1f%%\n(%d/%d)", 100 * rate, fraud, total)),
              hjust = -0.15, size = 3.6, fontface = "bold") +
    coord_flip() +
    scale_fill_gradient(low = "#6baed6", high = "#cb181d") +
    scale_y_continuous(labels = percent_format(), limits = c(0, max(p_data$rate) * 1.25)) +
    labs(
      title = "Fraud Rate by Incident Severity",
      subtitle = "Major Damage incidents demonstrate dramatically higher empirical fraud incidence",
      x = "Incident Severity",
      y = "Fraud Proportion (%)",
      caption = "Categories: Major Damage, Total Loss, Minor Damage, Trivial Damage"
    ) +
    theme_fraud()

  ggsave(output_path, plot = p, width = 7.5, height = 4.5, dpi = 300)
  return(p)
}

#' Plot Total Claim Amount Distribution by Fraud Status
#' @export
plot_claim_amounts <- function(df, output_path = "outputs/figures/04_claim_amount_distribution.png") {
  p <- ggplot(df, aes(x = fraud_reported, y = total_claim_amount, fill = fraud_reported)) +
    geom_boxplot(alpha = 0.7, outlier.color = "red", outlier.alpha = 0.6, width = 0.45, show.legend = FALSE) +
    stat_summary(fun = mean, geom = "point", shape = 18, size = 3.5, color = "black") +
    scale_fill_manual(values = c("No" = "#4393c3", "Yes" = "#d6604d")) +
    scale_y_continuous(labels = dollar_format()) +
    labs(
      title = "Total Claim Amount Distribution by Fraud Status",
      subtitle = "Fraudulent claims exhibit significantly elevated median and mean (diamond) claim amounts",
      x = "Fraud Reported",
      y = "Total Claim Amount ($)",
      caption = "Black diamonds indicate group mean; red points denote outliers"
    ) +
    theme_fraud()

  ggsave(output_path, plot = p, width = 7, height = 5, dpi = 300)
  return(p)
}

#' Plot Multi-Model Precision-Recall Curves
#' @export
plot_precision_recall_curves <- function(pred_list, output_path = "outputs/figures/06_precision_recall_curves.png") {
  pr_curves_data <- bind_rows(lapply(names(pred_list), function(m_name) {
    df <- pred_list[[m_name]]
    yardstick::pr_curve(df, truth = truth, .pred_Yes, event_level = "second") %>%
      mutate(model = m_name)
  }))

  # Calculate PR-AUC values for legend annotation
  auc_labels <- sapply(names(pred_list), function(m_name) {
    df <- pred_list[[m_name]]
    auc_val <- yardstick::pr_auc(df, truth = truth, .pred_Yes, event_level = "second")$.estimate
    sprintf("%s (PR-AUC = %.3f)", m_name, auc_val)
  })

  # Set baseline prevalence
  base_prevalence <- mean(pred_list[[1]]$truth == "Yes")

  p <- ggplot(pr_curves_data, aes(x = recall, y = precision, color = model)) +
    geom_path(linewidth = 1.1) +
    geom_hline(yintercept = base_prevalence, linetype = "dashed", color = "grey50", linewidth = 0.7) +
    annotate("text", x = 0.1, y = base_prevalence + 0.03, label = "Baseline (Random Guess)", color = "grey40", size = 3.3) +
    scale_color_manual(values = c(
      "Logistic Regression" = "#1f78b4",
      "XGBoost" = "#e31a1c",
      "Decision Tree" = "#33a02c"
    )) +
    scale_x_continuous(labels = percent_format(), limits = c(0, 1)) +
    scale_y_continuous(labels = percent_format(), limits = c(0, 1)) +
    labs(
      title = "Precision-Recall Curves Across Competing Models",
      subtitle = "Primary evaluation metric for class imbalance; higher curve indicates superior fraud detection trade-off",
      x = "Recall (Sensitivity / Detection Rate)",
      y = "Precision (Positive Predictive Value)",
      color = "Model",
      caption = "Evaluated on independent holdout test set (N = 200, un-smoted original distribution)"
    ) +
    theme_fraud() +
    theme(legend.position = "right")

  ggsave(output_path, plot = p, width = 8, height = 5.5, dpi = 300)
  return(p)
}

#' Plot Model ROC Curves
#' @export
plot_roc_curves <- function(pred_list, output_path = "outputs/figures/07_roc_curves.png") {
  roc_curves_data <- bind_rows(lapply(names(pred_list), function(m_name) {
    df <- pred_list[[m_name]]
    yardstick::roc_curve(df, truth = truth, .pred_Yes, event_level = "second") %>%
      mutate(model = m_name)
  }))

  p <- ggplot(roc_curves_data, aes(x = 1 - specificity, y = sensitivity, color = model)) +
    geom_path(linewidth = 1.1) +
    geom_abline(lty = 2, color = "grey50", linewidth = 0.7) +
    scale_color_manual(values = c(
      "Logistic Regression" = "#1f78b4",
      "XGBoost" = "#e31a1c",
      "Decision Tree" = "#33a02c"
    )) +
    scale_x_continuous(labels = percent_format(), limits = c(0, 1)) +
    scale_y_continuous(labels = percent_format(), limits = c(0, 1)) +
    labs(
      title = "Receiver Operating Characteristic (ROC) Curves",
      subtitle = "Cross-model comparison of sensitivity vs 1 - specificity across all probability thresholds",
      x = "False Positive Rate (1 - Specificity)",
      y = "True Positive Rate (Recall / Sensitivity)",
      color = "Model",
      caption = "Evaluated on independent holdout test set"
    ) +
    theme_fraud() +
    theme(legend.position = "right")

  ggsave(output_path, plot = p, width = 8, height = 5.5, dpi = 300)
  return(p)
}

#' Plot Confusion Matrices Heatmap
#' @export
plot_confusion_matrices <- function(pred_list, output_path = "outputs/figures/08_confusion_matrices.png") {
  cm_combined <- bind_rows(lapply(names(pred_list), function(m_name) {
    df <- pred_list[[m_name]]
    cm <- yardstick::conf_mat(df, truth = truth, estimate = .pred_class)
    as.data.frame(cm$table) %>%
      mutate(
        model = m_name,
        Outcome = case_when(
          Prediction == "Yes" & Truth == "Yes" ~ "True Positive (Fraud Caught)",
          Prediction == "Yes" & Truth == "No"  ~ "False Positive (False Alarm)",
          Prediction == "No"  & Truth == "No"  ~ "True Negative (Legit Approved)",
          Prediction == "No"  & Truth == "Yes" ~ "False Negative (Missed Fraud)"
        )
      )
  }))

  p <- ggplot(cm_combined, aes(x = Truth, y = Prediction, fill = Freq)) +
    geom_tile(color = "white", linewidth = 1) +
    geom_text(aes(label = sprintf("%d\n%s", Freq, gsub(" \\(.*\\)", "", Outcome))),
              size = 3.6, fontface = "bold", color = "black") +
    facet_wrap(~ model) +
    scale_fill_gradient(low = "#f7fbff", high = "#2171b5") +
    labs(
      title = "Confusion Matrices on Unseen Holdout Test Partition",
      subtitle = "False Negatives represent missed fraud liabilities; False Positives represent investigation costs",
      x = "Ground Truth (Actual Status)",
      y = "Model Predicted Class",
      fill = "Count",
      caption = "Holdout Test Set N = 200 (True Fraud = 49, True Legitimate = 151)"
    ) +
    theme_fraud() +
    theme(legend.position = "none")

  ggsave(output_path, plot = p, width = 9.5, height = 4.5, dpi = 300)
  return(p)
}

#' Plot Top XGBoost Feature Importance
#' @export
plot_xgboost_importance <- function(vi_df, top_n = 15, output_path = "outputs/figures/09_feature_importance_xgboost.png") {
  plot_data <- vi_df %>%
    head(top_n) %>%
    arrange(Importance) %>%
    mutate(Variable = factor(Variable, levels = Variable))

  p <- ggplot(plot_data, aes(x = Importance, y = Variable)) +
    geom_col(fill = "#d95f02", width = 0.65, alpha = 0.88) +
    geom_text(aes(label = sprintf("%.2f", Importance)), hjust = -0.15, size = 3.4, fontface = "bold") +
    scale_x_continuous(expand = expansion(mult = c(0, 0.15))) +
    labs(
      title = sprintf("Top %d Predictors of Fraud Risk: XGBoost Variable Importance", top_n),
      subtitle = "Relative contribution to split gain. Note: Reflects predictive utility, not direct causal mechanism.",
      x = "Feature Importance (Split Gain)",
      y = "Predictor Feature",
      caption = "Extracted via vip::vi() from finalized XGBoost model"
    ) +
    theme_fraud()

  ggsave(output_path, plot = p, width = 8, height = 6, dpi = 300)
  return(p)
}

#' Plot Logistic Regression Odds Ratios Forest Plot
#' @export
plot_logistic_odds_ratios <- function(odds_df, top_n = 12, output_path = "outputs/figures/10_logistic_odds_ratios.png") {
  plot_data <- odds_df %>%
    filter(term != "(Intercept)") %>%
    filter(!is.na(odds_ratio) & is.finite(odds_ratio) & is.finite(conf_high)) %>%
    filter(conf_high < 100 & conf_low > 0.01) %>% # Filter extreme numerical blowouts
    head(top_n) %>%
    arrange(odds_ratio) %>%
    mutate(term = factor(term, levels = term))

  p <- ggplot(plot_data, aes(x = odds_ratio, y = term)) +
    geom_vline(xintercept = 1.0, linetype = "dashed", color = "grey40", linewidth = 0.8) +
    geom_errorbarh(aes(xmin = conf_low, xmax = conf_high), height = 0.25, color = "#1f78b4", linewidth = 0.8) +
    geom_point(size = 3.2, color = "#1f78b4") +
    geom_text(aes(label = sprintf("OR=%.2f", odds_ratio)), vjust = -1.1, size = 3.2, fontface = "bold") +
    scale_x_log10(labels = comma) +
    labs(
      title = "Logistic Regression Odds Ratios with 95% Confidence Intervals",
      subtitle = "OR > 1 indicates increased odds of fraud; OR < 1 indicates protective / decreased odds",
      x = "Odds Ratio: exp(Beta) (Log Scale)",
      y = "Model Term / Dummy Feature",
      caption = "Wald 95% Confidence Intervals. Dashed vertical line marks neutral odds (OR = 1.0)"
    ) +
    theme_fraud()

  ggsave(output_path, plot = p, width = 8, height = 5.5, dpi = 300)
  return(p)
}

#' Plot Model Comparison Multi-Metric Bar Chart
#' @export
plot_model_comparison <- function(comp_df, output_path = "outputs/figures/11_model_comparison_bar.png") {
  long_df <- comp_df %>%
    select(model, pr_auc, recall, precision, f1_score, roc_auc) %>%
    pivot_longer(cols = -model, names_to = "metric", values_to = "value") %>%
    mutate(
      metric_label = case_when(
        metric == "pr_auc"    ~ "PR-AUC (Primary)",
        metric == "recall"    ~ "Recall (Sensitivity)",
        metric == "precision" ~ "Precision",
        metric == "f1_score"  ~ "F1-Score",
        metric == "roc_auc"   ~ "ROC-AUC"
      )
    )

  p <- ggplot(long_df, aes(x = model, y = value, fill = model)) +
    geom_col(position = position_dodge(0.8), width = 0.65, alpha = 0.9) +
    geom_text(aes(label = sprintf("%.2f", value)), vjust = -0.4, size = 3.2, fontface = "bold") +
    facet_wrap(~ metric_label, scales = "free_y") +
    scale_fill_manual(values = c(
      "Logistic Regression" = "#1f78b4",
      "XGBoost" = "#e31a1c",
      "Decision Tree" = "#33a02c"
    )) +
    scale_y_continuous(labels = percent_format(), limits = c(0, 1.15)) +
    labs(
      title = "Holistic Performance Comparison Across Competing Architectures",
      subtitle = "Comparing Interpretable Baseline (GLM), Nonlinear Ensemble (XGBoost), and Decision Tree",
      x = "Model Architecture",
      y = "Metric Score",
      caption = "Evaluation conducted on un-smoted test partition"
    ) +
    theme_fraud() +
    theme(legend.position = "none")

  ggsave(output_path, plot = p, width = 9.5, height = 6, dpi = 300)
  return(p)
}
