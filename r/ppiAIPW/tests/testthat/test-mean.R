test_that("default mean method matches explicit monotone spline", {
  set.seed(11)
  y <- rnorm(60)
  yhat <- y + rnorm(60, sd = 0.25)
  yhat_u <- rnorm(120)

  default_result <- mean_inference(y, yhat, yhat_u)
  explicit_result <- mean_inference(y, yhat, yhat_u, method = "monotone_spline")
  default_model <- fit_calibrator(y, yhat)

  expect_equal(default_result$pointestimate, explicit_result$pointestimate, tolerance = 1e-10)
  expect_equal(default_result$se, explicit_result$se, tolerance = 1e-10)
  expect_equal(default_result$ci[[1]], explicit_result$ci[[1]], tolerance = 1e-10)
  expect_equal(default_result$ci[[2]], explicit_result$ci[[2]], tolerance = 1e-10)
  expect_identical(default_model$method, "monotone_spline")
})

test_that("aipw matches manual augmented estimator", {
  y <- c(0, 1, 0, 1)
  yhat <- c(0.1, 0.4, 0.6, 0.8)
  yhat_u <- c(0.2, 0.3, 0.7)

  estimate <- mean_pointestimate(y, yhat, yhat_u, method = "aipw")
  rho <- length(y) / (length(y) + length(yhat_u))
  expected <- rho * mean(yhat) + (1 - rho) * mean(yhat_u) + mean(y - yhat)

  expect_equal(estimate, expected, tolerance = 1e-12)
})

test_that("uniform weights reproduce the unweighted result", {
  set.seed(12)
  y <- rnorm(50)
  yhat <- y + rnorm(50, sd = 0.3)
  yhat_u <- rnorm(90)

  unweighted <- mean_inference(y, yhat, yhat_u, method = "linear")
  weighted <- mean_inference(
    y, yhat, yhat_u, method = "linear",
    w = rep(1, length(y)), w_unlabeled = rep(1, length(yhat_u))
  )

  expect_equal(weighted$pointestimate, unweighted$pointestimate, tolerance = 1e-10)
  expect_equal(weighted$se, unweighted$se, tolerance = 1e-10)
  expect_equal(weighted$ci[[1]], unweighted$ci[[1]], tolerance = 1e-10)
  expect_equal(weighted$ci[[2]], unweighted$ci[[2]], tolerance = 1e-10)
})

test_that("uniform weight scaling leaves mean inference unchanged", {
  set.seed(220)
  y <- rnorm(50)
  yhat <- y + rnorm(50, sd = 0.3)
  yhat_u <- rnorm(90)
  w <- runif(50, 0.5, 2)
  w_u <- runif(90, 0.5, 2)

  baseline <- mean_inference(y, yhat, yhat_u, method = "linear", w = w, w_unlabeled = w_u)
  rescaled <- mean_inference(y, yhat, yhat_u, method = "linear", w = 7 * w, w_unlabeled = 7 * w_u)

  expect_equal(rescaled$pointestimate, baseline$pointestimate, tolerance = 1e-8)
  expect_equal(rescaled$se, baseline$se, tolerance = 1e-8)
  expect_equal(rescaled$ci[[1]], baseline$ci[[1]], tolerance = 1e-8)
  expect_equal(rescaled$ci[[2]], baseline$ci[[2]], tolerance = 1e-8)
})

test_that("mean_inference linear wald matches separate calls", {
  set.seed(32)
  y <- rnorm(45)
  yhat <- y + rnorm(45, sd = 0.35)
  yhat_u <- rnorm(90)

  result <- mean_inference(y, yhat, yhat_u, method = "linear", alpha = 0.1, inference = "wald")

  expect_s3_class(result, "ppi_mean_result")
  expect_equal(result$pointestimate, mean_pointestimate(y, yhat, yhat_u, method = "linear"), tolerance = 1e-10)
  expect_equal(result$se, mean_se(y, yhat, yhat_u, method = "linear"), tolerance = 1e-10)
  ci <- mean_ci(y, yhat, yhat_u, method = "linear", alpha = 0.1)
  expect_equal(result$ci[[1]], ci[[1]], tolerance = 1e-10)
  expect_equal(result$ci[[2]], ci[[2]], tolerance = 1e-10)
  expect_identical(result$method, "linear")
  expect_identical(result$selected_candidate, "linear")
  expect_false(result$selected_efficiency_maximization)
  expect_null(result$efficiency_lambda)
  expect_identical(result$calibrator$method, "linear")
})

test_that("auto wald matches separate calls and selection diagnostics", {
  set.seed(33)
  yhat <- seq(-1, 1, length.out = 60)
  y <- 0.75 + 2.1 * yhat + rnorm(60, sd = 0.08)
  yhat_u <- seq(-1.2, 1.2, length.out = 120)

  result <- mean_inference(
    y, yhat, yhat_u,
    method = "auto",
    candidate_methods = c("aipw", "linear", "monotone_spline", "isotonic"),
    num_folds = 5,
    selection_random_state = 0,
    inference = "wald"
  )
  selected <- select_mean_method_cv(
    y, yhat, yhat_u,
    candidate_methods = c("aipw", "linear", "monotone_spline", "isotonic"),
    num_folds = 5,
    selection_random_state = 0
  )

  expect_equal(
    result$pointestimate,
    mean_pointestimate(
      y, yhat, yhat_u,
      method = "auto",
      candidate_methods = c("aipw", "linear", "monotone_spline", "isotonic"),
      num_folds = 5,
      selection_random_state = 0
    ),
    tolerance = 1e-10
  )
  expect_equal(
    result$se,
    mean_se(
      y, yhat, yhat_u,
      method = "auto",
      candidate_methods = c("aipw", "linear", "monotone_spline", "isotonic"),
      num_folds = 5,
      selection_random_state = 0
    ),
    tolerance = 1e-10
  )
  ci <- mean_ci(
    y, yhat, yhat_u,
    method = "auto",
    candidate_methods = c("aipw", "linear", "monotone_spline", "isotonic"),
    num_folds = 5,
    selection_random_state = 0,
    alpha = 0.1
  )
  expect_equal(result$ci[[1]], ci[[1]], tolerance = 1e-10)
  expect_equal(result$ci[[2]], ci[[2]], tolerance = 1e-10)
  expect_identical(result$method, selected$method)
  expect_identical(result$selected_candidate, selected$diagnostics$selected_candidate)
  expect_identical(result$selected_efficiency_maximization, selected$diagnostics$selected_efficiency_maximization)
  expect_true("auto_unlabeled_subsample_size" %in% names(result$diagnostics))
})

test_that("auto bootstrap and jackknife reuse the selected method", {
  set.seed(34)
  yhat <- seq(-1, 1, length.out = 50)
  y <- 0.9 + 2 * yhat + rnorm(50, sd = 0.06)
  yhat_u <- seq(-1.1, 1.1, length.out = 100)

  selected <- select_mean_method_cv(
    y, yhat, yhat_u,
    candidate_methods = c("aipw", "linear", "monotone_spline", "isotonic"),
    num_folds = 5,
    selection_random_state = 0
  )

  boot <- mean_inference(
    y, yhat, yhat_u,
    method = "auto",
    candidate_methods = c("aipw", "linear", "monotone_spline", "isotonic"),
    num_folds = 5,
    selection_random_state = 0,
    inference = "bootstrap",
    n_resamples = 30,
    random_state = 7
  )
  expect_true(isTRUE(boot$diagnostics$bootstrap_selected_once))
  expect_identical(boot$method, selected$method)
  expect_identical(boot$diagnostics$bootstrap_method, selected$method)
  expect_equal(
    boot$pointestimate,
    mean_pointestimate(
      y, yhat, yhat_u,
      method = selected$method,
      efficiency_maximization = selected$diagnostics$selected_efficiency_maximization
    ),
    tolerance = 1e-10
  )

  jack <- mean_inference(
    y, yhat, yhat_u,
    method = "auto",
    candidate_methods = c("aipw", "linear", "monotone_spline", "isotonic"),
    num_folds = 5,
    selection_random_state = 0,
    inference = "jackknife",
    jackknife_folds = 5,
    random_state = 7
  )
  expect_true(isTRUE(jack$diagnostics$jackknife_selected_once))
  expect_identical(jack$method, selected$method)
  expect_identical(jack$diagnostics$jackknife_method, selected$method)
  expect_identical(jack$diagnostics$jackknife_efficiency_maximization, selected$diagnostics$selected_efficiency_maximization)
  expect_identical(jack$diagnostics$jackknife_folds, 5L)
})

test_that("bootstrap and jackknife are reproducible", {
  set.seed(103)
  y <- rnorm(30)
  yhat <- y + rnorm(30, sd = 0.4)
  yhat_u <- rnorm(80)

  ci_boot_1 <- mean_ci(y, yhat, yhat_u, method = "linear", inference = "bootstrap", n_resamples = 60, random_state = 0)
  ci_boot_2 <- mean_ci(y, yhat, yhat_u, method = "linear", inference = "bootstrap", n_resamples = 60, random_state = 0)
  expect_equal(ci_boot_1[[1]], ci_boot_2[[1]], tolerance = 1e-12)
  expect_equal(ci_boot_1[[2]], ci_boot_2[[2]], tolerance = 1e-12)

  ci_jack_1 <- mean_ci(y, yhat, yhat_u, method = "linear", inference = "jackknife", jackknife_folds = 10, random_state = 0)
  ci_jack_2 <- mean_ci(y, yhat, yhat_u, method = "linear", inference = "jackknife", jackknife_folds = 10, random_state = 0)
  expect_equal(ci_jack_1[[1]], ci_jack_2[[1]], tolerance = 1e-12)
  expect_equal(ci_jack_1[[2]], ci_jack_2[[2]], tolerance = 1e-12)
})

test_that("efficiency lambda is reported when applicable", {
  set.seed(35)
  y <- rnorm(40)
  yhat <- y + rnorm(40, sd = 0.5)
  yhat_u <- rnorm(80)

  result <- mean_inference(y, yhat, yhat_u, method = "linear", efficiency_maximization = TRUE)

  expect_true(result$selected_efficiency_maximization)
  expect_false(is.null(result$efficiency_lambda))
  expect_identical(result$diagnostics$efficiency_lambda_source, "full_sample")
  expect_identical(result$calibrator$metadata$efficiency_lambda_source, "full_sample")
})

test_that("vector outputs and aliases agree with the generic API", {
  set.seed(1)
  y <- matrix(rnorm(60), ncol = 2)
  yhat <- y + matrix(rnorm(60, sd = 0.3), ncol = 2)
  yhat_u <- matrix(rnorm(100), ncol = 2)

  estimate <- ppi_aipw_mean_pointestimate(y, yhat, yhat_u, method = "linear")
  se <- ppi_aipw_mean_se(y, yhat, yhat_u, method = "linear")
  ci <- aipw_mean_ci(y, yhat, yhat_u, method = "linear", alpha = 0.1)

  expect_equal(dim(y)[2], length(estimate))
  expect_equal(dim(y)[2], length(se))
  expect_equal(dim(y)[2], length(ci[[1]]))
  expect_equal(dim(y)[2], length(ci[[2]]))
  expect_equal(linear_calibration_mean_pointestimate(y, yhat, yhat_u), mean_pointestimate(y, yhat, yhat_u, method = "linear"))
  expect_equal(linear_calibration_mean_se(y, yhat, yhat_u), mean_se(y, yhat, yhat_u, method = "linear"))
  expect_equal(sigmoid_mean_pointestimate(y[, 1], yhat[, 1], yhat_u[, 1]), mean_pointestimate(y[, 1], yhat[, 1], yhat_u[, 1], method = "sigmoid"))
  expect_equal(isotonic_mean_se(y[, 1], yhat[, 1], yhat_u[, 1]), mean_se(y[, 1], yhat[, 1], yhat_u[, 1], method = "isotonic"))
  expect_equal(pi_aipw_mean_se(y[, 1], yhat[, 1], yhat_u[, 1], method = "linear"), mean_se(y[, 1], yhat[, 1], yhat_u[, 1], method = "linear"))
})

test_that("summary and print methods are informative", {
  set.seed(2024)
  y <- rnorm(60)
  yhat <- y + rnorm(60, sd = 0.35)
  yhat_u <- rnorm(120)

  result <- mean_inference(y, yhat, yhat_u, method = "linear")
  summary_text <- paste(capture.output(summary(result)), collapse = "\n")
  print_text <- paste(capture.output(print(result)), collapse = "\n")

  expect_match(summary_text, "ppi_mean_result summary")
  expect_match(summary_text, "wald_t:")
  expect_match(summary_text, "wald_p_value:")
  expect_match(print_text, "ppi_mean_result\\(")
})
