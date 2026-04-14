test_that("balancing weights match pooled target moments approximately", {
  x_labeled <- matrix(c(0, 1, 2, 3), ncol = 1)
  x_unlabeled <- matrix(c(1, 2), ncol = 1)

  out <- compute_two_sample_balancing_weights(
    x_labeled,
    x_unlabeled,
    target = "pooled",
    return_diagnostics = TRUE
  )

  rho <- nrow(x_labeled) / (nrow(x_labeled) + nrow(x_unlabeled))
  pooled_mean <- rho * mean(x_labeled[, 1]) + (1 - rho) * mean(x_unlabeled[, 1])

  expect_equal(mean(out$weights), 1, tolerance = 1e-4)
  expect_equal(mean(out$weights * x_labeled[, 1]), pooled_mean, tolerance = 1e-4)
  expect_true(all(out$weights >= 0))
})
