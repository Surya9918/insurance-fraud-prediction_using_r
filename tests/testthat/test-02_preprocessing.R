# tests/testthat/test-02_preprocessing.R

test_that("preprocessing pipeline preserves stratification and eliminates identifiers", {
  # Mock dataset with 40 rows and realistic attributes
  set.seed(42)
  n <- 50
  mock_df <- tibble(
    policy_number = 1000 + 1:n,
    insured_zip = sample(40000:49999, n, replace = TRUE),
    incident_location = paste("Street", 1:n),
    policy_bind_date = as.Date("2015-01-01") + 1:n,
    incident_date = as.Date("2015-06-01") + 1:n,
    age = sample(20:65, n, replace = TRUE),
    months_as_customer = sample(1:200, n, replace = TRUE),
    incident_severity = factor(sample(c("Minor Damage", "Major Damage", "Total Loss", "Trivial Damage"), n, replace = TRUE)),
    collision_type = sample(c("Front Collision", "Rear Collision", "Side Collision", NA), n, replace = TRUE),
    total_claim_amount = runif(n, 5000, 80000),
    fraud_reported = factor(sample(c("No", "Yes"), n, replace = TRUE, prob = c(0.75, 0.25)), levels = c("No", "Yes"))
  )

  source("../../R/03_preprocessing.R", local = TRUE)

  # Test stratified split
  split_res <- create_train_test_split(mock_df, prop = 0.80, seed = 123)
  tr <- rsample::training(split_res)
  te <- rsample::testing(split_res)

  expect_equal(nrow(tr), 40)
  expect_equal(nrow(te), 10)

  # Check that stratification maintains ratio approximately
  tr_rate <- mean(tr$fraud_reported == "Yes")
  te_rate <- mean(te$fraud_reported == "Yes")
  expect_true(abs(tr_rate - te_rate) < 0.15)

  # Test recipe specification
  rec <- build_preprocessing_recipe(tr, use_smote = FALSE, other_threshold = 0.05)
  prepped <- recipes::prep(rec, training = tr)
  baked_tr <- recipes::bake(prepped, new_data = NULL)
  baked_te <- recipes::bake(prepped, new_data = te)

  # Check identifiers removed
  expect_false("policy_number" %in% names(baked_tr))
  expect_false("insured_zip" %in% names(baked_tr))
  expect_false("incident_location" %in% names(baked_tr))
  expect_false("policy_bind_date" %in% names(baked_tr))
  expect_false("incident_date" %in% names(baked_tr))

  # Check no missing values remain
  expect_equal(sum(is.na(baked_tr)), 0)
  expect_equal(sum(is.na(baked_te)), 0)
})
