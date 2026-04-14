make_causal_data <- function(seed, n, treatment_levels) {
  set.seed(seed)
  x <- rnorm(n)
  arm_idx <- sample(seq_along(treatment_levels), size = n, replace = TRUE)
  A <- treatment_levels[arm_idx]
  potential_means <- lapply(
    seq_along(treatment_levels),
    function(idx) 0.5 * (idx - 1) + 0.8 * x + 0.15 * (idx - 1) * x
  )
  potential_outcomes <- lapply(
    seq_along(treatment_levels),
    function(idx) potential_means[[idx]] + rnorm(n, sd = 0.25)
  )
  Y <- numeric(n)
  for (idx in seq_along(treatment_levels)) {
    mask <- A == treatment_levels[[idx]]
    Y[mask] <- potential_outcomes[[idx]][mask]
  }
  Yhat_potential <- do.call(cbind, lapply(potential_means, function(mu) mu + rnorm(n, sd = 0.08)))
  list(Y = Y, A = A, Yhat_potential = Yhat_potential, X = cbind(x, x^2))
}

test_that("causal linear path matches armwise direct mean inference", {
  dat <- make_causal_data(seed = 1, n = 120, treatment_levels = c(0, 1))
  result <- causal_inference(dat$Y, dat$A, dat$Yhat_potential, method = "linear", alpha = 0.1)

  for (arm_idx in seq_along(c(0, 1))) {
    arm <- c(0, 1)[[arm_idx]]
    mask <- dat$A == arm
    direct <- mean_inference(
      dat$Y[mask],
      dat$Yhat_potential[mask, arm_idx],
      dat$Yhat_potential[!mask, arm_idx],
      method = "linear",
      alpha = 0.1
    )
    expect_equal(result$arm_means[[as.character(arm)]], direct$pointestimate, tolerance = 1e-8)
    expect_equal(result$arm_ses[[as.character(arm)]], direct$se, tolerance = 1e-8)
  }
})

test_that("causal inference supports auto and prognostic linear paths", {
  dat <- make_causal_data(seed = 2, n = 150, treatment_levels = c(0, 1, 2))

  auto_result <- causal_inference(
    dat$Y, dat$A, dat$Yhat_potential,
    method = "auto",
    num_folds = 5,
    selection_random_state = 0
  )
  expect_s3_class(auto_result, "ppi_causal_result")
  expect_length(auto_result$arm_results, 3)
  expect_true(all(vapply(auto_result$arm_results, inherits, logical(1), "ppi_mean_result")))

  prognostic_result <- causal_inference(
    dat$Y, dat$A, dat$Yhat_potential,
    X = dat$X,
    method = "prognostic_linear"
  )
  expect_true(all(vapply(prognostic_result$arm_results, function(x) inherits(x$calibrator, "ppi_prognostic_linear_model"), logical(1))))
})

test_that("uniform weights and weight scaling leave causal inference unchanged", {
  dat <- make_causal_data(seed = 3, n = 160, treatment_levels = c(0, 1))
  w <- runif(length(dat$Y), 0.5, 2)

  unweighted <- causal_inference(dat$Y, dat$A, dat$Yhat_potential, method = "linear")
  weighted_uniform <- causal_inference(dat$Y, dat$A, dat$Yhat_potential, method = "linear", w = rep(1, length(dat$Y)))
  baseline <- causal_inference(dat$Y, dat$A, dat$Yhat_potential, method = "linear", w = w)
  rescaled <- causal_inference(dat$Y, dat$A, dat$Yhat_potential, method = "linear", w = 11 * w)

  expect_equal(weighted_uniform$ate[["1"]], unweighted$ate[["1"]], tolerance = 1e-10)
  expect_equal(rescaled$arm_means[["0"]], baseline$arm_means[["0"]], tolerance = 1e-8)
  expect_equal(rescaled$arm_means[["1"]], baseline$arm_means[["1"]], tolerance = 1e-8)
  expect_equal(rescaled$ate[["1"]], baseline$ate[["1"]], tolerance = 1e-8)
})

test_that("causal summary and print methods are informative", {
  dat <- make_causal_data(seed = 4, n = 120, treatment_levels = c(0, 1))
  result <- causal_inference(dat$Y, dat$A, dat$Yhat_potential, method = "linear")

  summary_text <- paste(capture.output(summary(result)), collapse = "\n")
  print_text <- paste(capture.output(print(result)), collapse = "\n")

  expect_match(summary_text, "ppi_causal_result summary")
  expect_match(summary_text, "ATEs vs control")
  expect_match(print_text, "ppi_causal_result\\(")
})

test_that("causal inference validates unsupported inputs", {
  dat <- make_causal_data(seed = 5, n = 80, treatment_levels = c(0, 1))

  expect_error(causal_inference(dat$Y, dat$A, dat$Yhat_potential, inference = "bootstrap"), "supports inference='wald' only")
  expect_error(causal_inference(dat$Y, dat$A, dat$Yhat_potential, control_arm = 99), "control_arm")
  expect_error(causal_inference(dat$Y, dat$A, dat$Yhat_potential[, 1, drop = FALSE], treatment_levels = c(0, 1)), "number of treatment_levels")
  expect_error(causal_inference(dat$Y, rep(0, length(dat$A)), dat$Yhat_potential[, 1, drop = FALSE]), "at least two observed treatment arms")
})
