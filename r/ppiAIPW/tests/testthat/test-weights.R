test_that("balancing weights match pooled target moments", {
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

  expect_equal(mean(out$weights), 1, tolerance = 1e-8)
  expect_equal(mean(out$weights * x_labeled[, 1]), pooled_mean, tolerance = 1e-8)
  expect_true(all(out$weights >= 0))
  expect_lt(out$diagnostics$max_abs_balance_error, 1e-8)
})

test_that("balancing weights can target unlabeled moments directly", {
  x_labeled <- matrix(c(0, 2, 4, 6), ncol = 1)
  x_unlabeled <- matrix(c(3, 5), ncol = 1)

  weights <- compute_two_sample_balancing_weights(x_labeled, x_unlabeled, target = "unlabeled")

  expect_equal(mean(weights * x_labeled[, 1]), mean(x_unlabeled[, 1]), tolerance = 1e-8)
})

test_that("balancing weights return ones when already balanced", {
  x_labeled <- matrix(c(1, 2, 3, 4), ncol = 1)
  x_unlabeled <- matrix(c(1, 2, 3, 4), ncol = 1)

  weights <- compute_two_sample_balancing_weights(x_labeled, x_unlabeled)

  expect_equal(weights, rep(1, nrow(x_labeled)), tolerance = 1e-12)
})

test_that("balancing weights fail cleanly when nonnegative balance is infeasible", {
  x_labeled <- matrix(c(0, 0), ncol = 1)
  x_unlabeled <- matrix(c(1, 1), ncol = 1)

  expect_error(
    compute_two_sample_balancing_weights(x_labeled, x_unlabeled, target = "unlabeled"),
    "Could not compute nonnegative balancing weights|Could not achieve the requested balance tolerance"
  )
})

test_that("balancing weights validate basic inputs", {
  expect_error(compute_two_sample_balancing_weights(matrix(numeric(0), ncol = 1), matrix(1, ncol = 1)), "nonempty")
  expect_error(compute_two_sample_balancing_weights(matrix(1:4, ncol = 2), matrix(1:3, ncol = 1)), "same number of columns")
  expect_error(compute_two_sample_balancing_weights(matrix(1:4, ncol = 1), matrix(1:2, ncol = 1), target = "bad"), "target")
  expect_error(
    compute_two_sample_balancing_weights(matrix(c(0, NA, 2), ncol = 1), matrix(c(1, 2), ncol = 1)),
    "X_labeled must contain only finite values"
  )
  expect_error(
    compute_two_sample_balancing_weights(matrix(c(0, 1, 2), ncol = 1), matrix(c(1, Inf), ncol = 1)),
    "X_unlabeled must contain only finite values"
  )
})
