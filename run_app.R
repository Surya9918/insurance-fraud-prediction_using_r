# ==============================================================================
# run_app.R
# Project: Insurance Claim Risk Prediction & Fraud Pattern Analysis
# Purpose: One-click runner script to launch the interactive R Shiny UI.
# Usage:   Rscript run_app.R
#          or in R console: source("run_app.R")
# ==============================================================================

message("==========================================================================")
message(" Launching ShieldClaims Fraud Risk Intelligence Shiny Application ")
message("==========================================================================")

# Check shiny and bslib
if (!requireNamespace("shiny", quietly = TRUE)) {
  stop("Package 'shiny' is required. Please install via install.packages('shiny')", call. = FALSE)
}
if (!requireNamespace("bslib", quietly = TRUE)) {
  stop("Package 'bslib' is required. Please install via install.packages('bslib')", call. = FALSE)
}

port <- 7860
host <- "127.0.0.1"

message(sprintf(">>> Starting Shiny server at http://%s:%d ...", host, port))
message(">>> Press Ctrl+C in terminal or Esc in RStudio to stop the server.")

shiny::runApp(
  appDir = ".",
  port = port,
  host = host,
  launch.browser = interactive()
)
