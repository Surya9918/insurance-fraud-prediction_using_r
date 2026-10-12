# .Rprofile - Insurance Claim Risk Prediction Project Environment Configuration

options(
  repos = c(CRAN = "https://cloud.r-project.org"),
  stringsAsFactors = FALSE,
  scipen = 999,
  digits = 4,
  readr.show_col_types = FALSE
)

message(">>> Insurance Claim Risk Prediction & Fraud Pattern Analysis Project Loaded <<<")
message(">>> Base working directory: ", getwd())
message(">>> Ensure packages are installed via: source('run_pipeline.R') or devtools::install_deps()")

pandoc_dir <- file.path(Sys.getenv("LOCALAPPDATA"), "Pandoc")
if (dir.exists(pandoc_dir) && Sys.getenv("RSTUDIO_PANDOC") == "") {
  Sys.setenv(RSTUDIO_PANDOC = pandoc_dir)
}

