make_causal_data <- function(seed, n, treatment_levels) {
  set.seed(seed)
  x <- rnorm(n)
  arm_idx <- sample(seq_along(treatment_levels), size = n, replace = TRUE)
  A <- treatment_levels[arm_idx]
  potential_means <- lapply(seq_along(treatment_levels), function(idx) 0.5 * (idx - 1) + 0.8 * x + 0.15 * (idx - 1) * x)
  potential_outcomes <- lapply(seq_along(treatment_levels), function(idx) potential_means[[idx]] + rnorm(n, sd = 0.25))
  Y <- numeric(n)
  for (idx in seq_along(treatment_levels)) {
    mask <- A == treatment_levels[[idx]]
    Y[mask] <- potential_outcomes[[idx]][mask]
  }
  Yhat_potential <- do.call(cbind, lapply(potential_means, function(mu) mu + rnorm(n, sd = 0.08)))
  list(Y = Y, A = A, Yhat_potential = Yhat_potential)
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
