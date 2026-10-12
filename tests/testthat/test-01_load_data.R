# tests/testthat/test-01_load_data.R

test_that("load_insurance_data properly sanitizes missing values and factor levels", {
  # Create a mock CSV mimicking the raw Kaggle dataset structure
  mock_dir <- tempdir()
  mock_file <- file.path(mock_dir, "mock_insurance.csv")

  mock_df <- data.frame(
    months_as_customer = c(12, 24, 36, 48),
    age = c(25, 35, 45, 55),
    policy_number = c(101, 102, 103, 104),
    `policy_annual_premium` = c(1000.5, 1200.0, 950.2, 1400.0),
    `capital-gains` = c(0, 5000, 0, 10000),
    collision_type = c("Front Collision", "?", "Rear Collision", "?"),
    property_damage = c("YES", "NO", "?", "YES"),
    fraud_reported = c("Y", "N", "Yes", "No"),
    `_c39` = c(NA, NA, NA, NA),
    check.names = FALSE
  )

  readr::write_csv(mock_df, mock_file)

  # Source data loading function
  source("../../R/01_load_data.R", local = TRUE)

  loaded <- load_insurance_data(mock_file)

  # Test assertions
  expect_equal(nrow(loaded), 4)
  expect_false("_c39" %in% names(loaded))
  expect_true("capital_gains" %in% names(loaded))
  expect_false("capital-gains" %in% names(loaded))

  # Test '?' to NA conversion
  expect_equal(sum(is.na(loaded$collision_type)), 2)
  expect_equal(sum(is.na(loaded$property_damage)), 1)
  expect_false(any(loaded$collision_type == "?", na.rm = TRUE))

  # Test factor levels: strictly c("No", "Yes")
  expect_true(is.factor(loaded$fraud_reported))
  expect_equal(levels(loaded$fraud_reported), c("No", "Yes"))
  expect_equal(as.character(loaded$fraud_reported), c("Yes", "No", "Yes", "No"))

  # Clean up mock file
  unlink(mock_file)
})

test_that("load_insurance_data throws descriptive error when file is missing", {
  source("../../R/01_load_data.R", local = TRUE)
  expect_error(
    load_insurance_data("non_existent_file.csv"),
    "Dataset not found"
  )
})
