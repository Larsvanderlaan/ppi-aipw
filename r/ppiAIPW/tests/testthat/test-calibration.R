test_that("linear calibration is identity on perfect labeled predictions", {
  y <- c(0, 1, 0, 1)
  yhat_u <- c(0.2, 0.8, 0.4)

  calibrated <- calibrate_predictions(y, y, yhat_u, method = "linear")

  expect_equal(calibrated$pred_labeled, y, tolerance = 1e-12)
  expect_equal(calibrated$pred_unlabeled, yhat_u, tolerance = 1e-12)
})

test_that("sigmoid and monotone methods stay bounded and monotone", {
  y <- c(0, 0, 0, 1, 1, 1)
  yhat <- c(0.05, 0.15, 0.45, 0.55, 0.85, 0.95)
  yhat_u <- c(0.1, 0.2, 0.8, 0.9)

  sigmoid <- calibrate_predictions(y, yhat, yhat_u, method = "sigmoid")
  iso <- calibrate_predictions(y, yhat, yhat_u, method = "isotonic")
  spline <- calibrate_predictions(y, yhat, yhat_u, method = "monotone_spline")

  expect_true(all(sigmoid$pred_labeled >= 0 & sigmoid$pred_labeled <= 1))
  expect_true(all(sigmoid$pred_unlabeled >= 0 & sigmoid$pred_unlabeled <= 1))
  expect_true(all(diff(sigmoid$pred_unlabeled[order(yhat_u)]) >= -1e-12))

  expect_true(all(iso$pred_labeled >= 0 & iso$pred_labeled <= 1))
  expect_true(all(iso$pred_unlabeled >= 0 & iso$pred_unlabeled <= 1))
  expect_true(all(diff(iso$pred_labeled[order(yhat)]) >= -1e-12))

  expect_true(all(spline$pred_labeled >= 0 & spline$pred_labeled <= 1))
  expect_true(all(spline$pred_unlabeled >= 0 & spline$pred_unlabeled <= 1))
  expect_true(all(diff(spline$pred_labeled[order(yhat)]) >= -1e-8))
})

test_that("fit_calibrator returns a reusable model", {
  y <- c(0, 1, 0, 1, 1)
  yhat <- c(0.1, 0.3, 0.4, 0.8, 0.9)
  model <- fit_calibrator(y, yhat, method = "linear")
  pred <- predict(model, c(0.2, 0.7))

  expect_s3_class(model, "ppi_calibration_model")
  expect_length(pred, 2)
  expect_true(all(pred >= 0 & pred <= 1))
})

test_that("calibration diagnostics return expected structure", {
  y <- c(0, 0.2, 0.7, 1)
  yhat <- c(0.1, 0.3, 0.6, 0.9)
  model <- fit_calibrator(y, yhat, method = "linear")

  diagnostics <- calibration_diagnostics(model, y, yhat, num_bins = 3)

  expect_identical(diagnostics$method, "linear")
  expect_identical(diagnostics$diagnostic_mode, "out_of_fold")
  expect_identical(diagnostics$n_outputs, 1L)
  expect_identical(diagnostics$n_labeled, 4L)
  expect_identical(diagnostics$num_bins, 3L)
  expect_identical(diagnostics$effective_num_folds, 4L)
  expect_length(diagnostics$per_output, 1)
  record <- diagnostics$per_output[[1]]
  expect_length(record$raw_labeled_scores, 4)
  expect_length(record$calibrated_labeled_scores, 4)
  expect_length(record$observed_outcomes, 4)
  expect_equal(sum(record$bin_counts), 4)
  expect_equal(length(record$grid_scores), length(record$fitted_curve))
  expect_true(is.finite(record$blp$slope))
  expect_true(is.finite(record$blp$slope_se))
  expect_identical(record$blp$slope_null, 1.0)
})

test_that("calibration diagnostics respect labeled weights and support nonlinear models", {
  y <- c(0, 1, 0.5, 0.5)
  yhat <- c(0.1, 0.2, 0.8, 0.9)
  w <- c(1, 3, 1, 1)
  model <- fit_calibrator(y, yhat, method = "linear", w = w)

  diagnostics <- calibration_diagnostics(model, y, yhat, w = w, num_bins = 2)
  record <- diagnostics$per_output[[1]]

  expect_equal(record$bin_counts, c(2L, 2L))
  expect_equal(record$bin_mean_outcome[[1]], 0.75, tolerance = 1e-12)

  x <- seq(0, 1, length.out = 40)
  y_nl <- 0.15 + 0.65 * x^2
  yhat_nl <- x + 0.05 * sin(3 * pi * x)
  nonlinear <- fit_calibrator(y_nl, yhat_nl, method = "monotone_spline")
  nonlinear_diag <- calibration_diagnostics(nonlinear, y_nl, yhat_nl, num_bins = 5)

  expect_identical(nonlinear_diag$method, "monotone_spline")
  expect_true(all(diff(nonlinear_diag$per_output[[1]]$fitted_curve) >= -1e-8))
})

test_that("calibration diagnostics BLP uses labeled weights", {
  y <- c(0, 1, 1.4, 2)
  yhat <- c(0.1, 0.3, 0.8, 0.9)
  w <- c(1, 5, 1, 1)
  model <- fit_calibrator(y, yhat, method = "linear")

  diagnostics <- calibration_diagnostics(
    model,
    y,
    yhat,
    w = w,
    diagnostic_mode = "in_sample",
    num_bins = 2
  )
  record <- diagnostics$per_output[[1]]
  weights <- w / sum(w) * length(w)
  coef <- coef(stats::lm.wfit(
    x = cbind(1, record$calibrated_labeled_scores),
    y = y,
    w = weights
  ))

  expect_equal(record$blp$intercept, unname(coef[[1]]), tolerance = 1e-12)
  expect_equal(record$blp$slope, unname(coef[[2]]), tolerance = 1e-12)
})

test_that("in-sample calibration diagnostics match the fitted model exactly", {
  y <- c(0, 0.25, 0.8, 1)
  yhat <- c(0.1, 0.35, 0.55, 0.9)
  model <- fit_calibrator(y, yhat, method = "linear")

  diagnostics <- calibration_diagnostics(
    model,
    y,
    yhat,
    diagnostic_mode = "in_sample",
    num_bins = 4
  )

  expect_identical(diagnostics$diagnostic_mode, "in_sample")
  expect_null(diagnostics$effective_num_folds)
  expect_equal(
    diagnostics$per_output[[1]]$calibrated_labeled_scores,
    as.numeric(predict(model, yhat)),
    tolerance = 1e-12
  )
})

test_that("calibration diagnostics BLP follows the diagnostic calibrated scores", {
  set.seed(778)
  y <- rnorm(40)
  yhat <- y + rnorm(40, sd = 0.4)
  model <- fit_calibrator(y, yhat, method = "linear")

  in_sample <- calibration_diagnostics(model, y, yhat, diagnostic_mode = "in_sample", num_bins = 4)
  out_of_fold <- calibration_diagnostics(model, y, yhat, diagnostic_mode = "out_of_fold", num_bins = 4)

  for (diagnostics in list(in_sample, out_of_fold)) {
    record <- diagnostics$per_output[[1]]
    coef <- coef(stats::lm(
      record$observed_outcomes ~ record$calibrated_labeled_scores
    ))
    expect_equal(record$blp$intercept, unname(coef[[1]]), tolerance = 1e-10)
    expect_equal(record$blp$slope, unname(coef[[2]]), tolerance = 1e-10)
  }

  expect_false(isTRUE(all.equal(
    in_sample$per_output[[1]]$calibrated_labeled_scores,
    out_of_fold$per_output[[1]]$calibrated_labeled_scores,
    tolerance = 1e-12
  )))
})

test_that("calibration diagnostics accept mean results and prognostic models require X", {
  set.seed(99)
  y <- rnorm(50)
  yhat <- y + rnorm(50, sd = 0.25)
  yhat_u <- rnorm(100)
  result <- mean_inference(y, yhat, yhat_u, method = "linear")

  diagnostics <- calibration_diagnostics(result, y, yhat, num_bins = 4)
  expect_identical(diagnostics$method, result$calibrator$method)
  expect_length(diagnostics$per_output[[1]]$raw_labeled_scores, 50)

  set.seed(101)
  x <- matrix(rnorm(120), ncol = 2)
  x_u <- matrix(rnorm(240), ncol = 2)
  y2 <- 1 + 0.8 * x[, 1] - 0.4 * x[, 2] + rnorm(60, sd = 0.2)
  yhat2 <- 0.9 * y2 + 0.3 * x[, 1]
  yhat2_u <- 0.9 * (1 + 0.8 * x_u[, 1] - 0.4 * x_u[, 2]) + 0.3 * x_u[, 1]

  prognostic_result <- mean_inference(
    y2, yhat2, yhat2_u,
    X = x, X_unlabeled = x_u,
    method = "prognostic_linear"
  )

  expect_error(calibration_diagnostics(prognostic_result, y2, yhat2), "X is required")
  prognostic_diag <- calibration_diagnostics(prognostic_result, y2, yhat2, X = x)
  expect_length(prognostic_diag$reference_covariates, 2)
})

test_that("calibration diagnostics summary reports BLP slope", {
  set.seed(124)
  y <- rnorm(35)
  yhat <- y + rnorm(35, sd = 0.25)
  model <- fit_calibrator(y, yhat, method = "linear")

  diagnostics <- calibration_diagnostics(model, y, yhat, num_bins = 4)
  summary_text <- paste(capture.output(summary(diagnostics)), collapse = "\n")

  expect_match(summary_text, "ppi_calibration_diagnostics summary")
  expect_match(summary_text, "blp_slope_null: 1")
  expect_match(summary_text, "calibrated_blp_slope:")
  expect_match(summary_text, "p_value=")
})

test_that("plot calibration runs without error", {
  set.seed(321)
  y <- matrix(rnorm(60), ncol = 2)
  yhat <- y + matrix(rnorm(60, sd = 0.25), ncol = 2)
  diagnostics <- calibration_diagnostics(fit_calibrator(y, yhat, method = "linear"), y, yhat, num_bins = 4)

  plot_path <- tempfile(fileext = ".pdf")
  grDevices::pdf(plot_path)
  on.exit({
    grDevices::dev.off()
    unlink(plot_path)
  }, add = TRUE)
  expect_invisible(plot_calibration(diagnostics, output_index = 1))
})
