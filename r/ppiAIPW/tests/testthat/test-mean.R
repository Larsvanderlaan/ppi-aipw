test_that("aipw matches manual augmented estimator", {
  y <- c(0, 1, 0, 1)
  yhat <- c(0.1, 0.4, 0.6, 0.8)
  yhat_u <- c(0.2, 0.3, 0.7)

  estimate <- mean_pointestimate(y, yhat, yhat_u, method = "aipw")
  rho <- length(y) / (length(y) + length(yhat_u))
  expected <- rho * mean(yhat) + (1 - rho) * mean(yhat_u) + mean(y - yhat)

  expect_equal(estimate, expected, tolerance = 1e-12)
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

test_that("monotone methods stay bounded and monotone on binary data", {
  y <- c(0, 0, 0, 1, 1, 1)
  yhat <- c(0.05, 0.15, 0.45, 0.55, 0.85, 0.95)
  yhat_u <- c(0.1, 0.2, 0.8, 0.9)

  iso <- calibrate_predictions(y, yhat, yhat_u, method = "isotonic")
  ms <- calibrate_predictions(y, yhat, yhat_u, method = "monotone_spline")

  expect_true(all(iso$pred_labeled >= 0 & iso$pred_labeled <= 1))
  expect_true(all(iso$pred_unlabeled >= 0 & iso$pred_unlabeled <= 1))
  expect_true(all(diff(iso$pred_labeled[order(yhat)]) >= -1e-8))

  expect_true(all(ms$pred_labeled >= 0 & ms$pred_labeled <= 1))
  expect_true(all(ms$pred_unlabeled >= 0 & ms$pred_unlabeled <= 1))
  expect_true(all(diff(ms$pred_labeled[order(yhat)]) >= -1e-5))
})

test_that("auto selection returns a valid method and diagnostics", {
  set.seed(21)
  yhat <- seq(-1, 1, length.out = 60)
  y <- 0.75 + 2.5 * yhat + rnorm(60, sd = 0.05)
  yhat_u <- seq(-1.2, 1.2, length.out = 120)

  selected <- select_mean_method_cv(
    y,
    yhat,
    yhat_u,
    candidate_methods = c("aipw", "linear", "monotone_spline", "isotonic"),
    num_folds = 5,
    selection_random_state = 0
  )

  expect_true(selected$method %in% c("aipw", "linear", "monotone_spline", "isotonic"))
  expect_true("candidate_scores" %in% names(selected$diagnostics))
})
