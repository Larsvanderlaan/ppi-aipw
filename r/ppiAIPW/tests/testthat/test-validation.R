test_that("one-sided intervals and summaries work", {
  set.seed(2001)
  y <- rnorm(40)
  yhat <- y + rnorm(40, sd = 0.3)
  yhat_u <- rnorm(80)

  ci_large <- mean_ci(y, yhat, yhat_u, method = "linear", alternative = "larger")
  ci_small <- mean_ci(y, yhat, yhat_u, method = "linear", alternative = "smaller")
  summary_large <- paste(capture.output(summary(mean_inference(y, yhat, yhat_u, method = "linear"), alternative = "larger")), collapse = "\n")

  expect_true(is.infinite(ci_large[[2]]))
  expect_true(is.infinite(ci_small[[1]]))
  expect_match(summary_large, "wald_alternative: larger")
})

test_that("mean API validates basic inputs and methods", {
  y <- 1:4
  yhat <- c(0.9, 2.1, 3, 4.2)
  yhat_u <- 1:3

  expect_error(mean_inference(y, yhat[-1], yhat_u), "same shape")
  expect_error(mean_inference(y, yhat, matrix(1:6, ncol = 2)), "column")
  expect_error(mean_inference(y, yhat, yhat_u, method = "unknown"), "Unknown calibration method")
  expect_error(mean_ci(y, yhat, yhat_u, alpha = 1.5), "alpha")
  expect_error(mean_ci(y, yhat, yhat_u, alternative = "bad"), "alternative")
  expect_error(mean_inference(y, yhat, yhat_u, inference = "bad"), "inference")
  expect_error(mean_inference(y, yhat, yhat_u, w = c(1, 1, 1)), "Expected weights")
  expect_error(mean_inference(y, yhat, yhat_u, w = c(-1, 1, 1, 1)), "nonnegative")
})

test_that("fit_calibrator validates special methods and supports data frame inputs", {
  y_df <- data.frame(y = c(0, 1, 0, 1))
  yhat_df <- data.frame(y = c(0.1, 0.3, 0.7, 0.9))
  model <- fit_calibrator(y_df, yhat_df, method = "linear")

  expect_s3_class(model, "ppi_calibration_model")
  expect_error(fit_calibrator(y_df, yhat_df, method = "prognostic_linear"), "requires optional covariates")
})

test_that("calibration models expose return_model and informative print output", {
  y <- c(0, 1, 0, 1, 1)
  yhat <- c(0.1, 0.3, 0.4, 0.8, 0.9)
  yhat_u <- c(0.2, 0.7)

  calibrated <- calibrate_predictions(y, yhat, yhat_u, method = "isotonic", return_model = TRUE)
  print_text <- paste(capture.output(print(calibrated$model)), collapse = "\n")

  expect_named(calibrated, c("pred_labeled", "pred_unlabeled", "model"))
  expect_identical(calibrated$model$metadata$isocal_backend, "weighted_pava")
  expect_match(print_text, "ppi_calibration_model")
})

test_that("prognostic linear models validate prediction shapes", {
  set.seed(2002)
  x <- matrix(rnorm(80), ncol = 2)
  x_u <- matrix(rnorm(160), ncol = 2)
  y <- 1 + 0.4 * x[, 1] - 0.2 * x[, 2] + rnorm(40, sd = 0.2)
  yhat <- y + rnorm(40, sd = 0.1)
  yhat_u <- 1 + 0.4 * x_u[, 1] - 0.2 * x_u[, 2] + rnorm(80, sd = 0.1)

  result <- mean_inference(y, yhat, yhat_u, X = x, X_unlabeled = x_u, method = "prognostic_linear")
  model <- result$calibrator
  print_text <- paste(capture.output(print(model)), collapse = "\n")

  expect_match(print_text, "ppi_prognostic_linear_model")
  expect_error(predict(model, yhat[1:3], X = x[1:2, ]), "same number of rows")
  expect_error(predict(model, yhat[1:3], X = matrix(rnorm(9), ncol = 3)), "Expected X to have")
})

test_that("auto selection diagnostics include candidate scores and efficiency candidate", {
  set.seed(2003)
  yhat <- seq(-1, 1, length.out = 40)
  y <- 0.5 + 1.5 * yhat + rnorm(40, sd = 0.05)
  yhat_u <- seq(-1.2, 1.2, length.out = 120)

  selected <- select_mean_method_cv(
    y, yhat, yhat_u,
    candidate_methods = c("aipw", "linear"),
    num_folds = 4,
    selection_random_state = 0
  )

  expect_true("candidate_scores" %in% names(selected$diagnostics))
  expect_true("aipw_efficiency_maximization" %in% names(selected$diagnostics$candidate_scores))
})

test_that("calibration diagnostics validate num_bins and object types", {
  y <- c(0, 1, 0, 1)
  yhat <- c(0.1, 0.2, 0.8, 0.9)
  model <- fit_calibrator(y, yhat, method = "linear")

  expect_error(calibration_diagnostics(model, y, yhat, num_bins = 0), "at least 1")
  expect_error(calibration_diagnostics(list(), y, yhat), "Expected a ppi_mean_result")
})
