test_that("V-fold jackknife uses paper pseudo-values and t critical values", {
  y <- seq(-2, 3, length.out = 40)^2
  splits <- ppiAIPW:::kfold_splits(40, 10, 7)
  deleted <- vapply(splits, function(s) mean(y[s$train]), numeric(1))
  pseudo <- 10 * mean(y) - 9 * deleted
  expected_se <- sd(pseudo) / sqrt(10)
  for (alt in c("two-sided", "larger", "smaller")) {
    result <- mean_inference(y, rep(0, 40), rep(0, 80), method = "aipw",
                             inference = "jackknife", random_state = 7, alternative = alt)
    expect_equal(result$se, expected_se, tolerance = 1e-12)
    expect_equal(result$pointestimate, mean(y), tolerance = 1e-12)
    expect_identical(result$diagnostics$jackknife_folds, 10L)
    expect_identical(result$diagnostics$jackknife_degrees_of_freedom, 9L)
    critical <- qt(if (alt == "two-sided") .95 else .9, 9)
    expect_equal(result$ci[[1]], if (alt == "smaller") -Inf else mean(y) - critical * expected_se)
    expect_equal(result$ci[[2]], if (alt == "larger") Inf else mean(y) + critical * expected_se)
    stat <- ppiAIPW:::compute_wald_statistics(result$pointestimate, result$se,
                                            alternative = alt, degrees_of_freedom = 9)
    z <- result$pointestimate / result$se
    p_value <- if (alt == "two-sided") 2 * pt(abs(z), 9, lower.tail = FALSE) else
      if (alt == "larger") pt(z, 9, lower.tail = FALSE) else pt(z, 9)
    expect_equal(stat$p_value, p_value)
    expect_match(summary(result)$text, "Student t \\(df=9\\)")
  }
})

test_that("honest calibration pairs score differences for unequal fold sizes", {
  y <- seq_len(23)^2
  for (weighted in c(FALSE, TRUE)) {
    w <- if (weighted) seq(1, 3, length.out = 23) else NULL
    w_u <- if (weighted) seq(1, 4, length.out = 47) else NULL
    result <- mean_inference(y, rep(0, 23), rep(0, 47), method = "linear",
                             cross_fit = TRUE, w = w, w_unlabeled = w_u)
    target <- if (weighted) weighted.mean(y, w) else mean(y)
    expect_equal(result$pointestimate, target, tolerance = 1e-10)
    expect_false(result$diagnostics$pointestimate_uses_full_sample_refit)
    expect_identical(result$diagnostics$calibration_folds, 10L)
  }
})

test_that("honest selection and scaling exclude held-out labels and predictions", {
  y <- seq_len(12)^2 + .3
  scores <- seq(.1, 2, length.out = 12)
  scores_u <- seq(.2, 3, length.out = 20)
  options <- list(method = "auto", w = NULL, w_unlabeled = NULL,
                  efficiency_maximization = TRUE, candidate_methods = c("aipw", "linear"),
                  num_folds = 2, cross_fit = TRUE, calibration_folds = 3, selection_random_state = 4)
  a <- do.call(ppiAIPW:::prepare_inference_inputs, c(list(Y = y, Yhat = scores, Yhat_unlabeled = scores_u), options))
  val <- ppiAIPW:::kfold_splits(12, 3, 4)[[1]]$val
  val_u <- ppiAIPW:::kfold_splits(20, 3, 4)[[1]]$val
  y[val] <- y[val] + 1000
  scores_u[val_u] <- scores_u[val_u] + 500
  b <- do.call(ppiAIPW:::prepare_inference_inputs, c(list(Y = y, Yhat = scores, Yhat_unlabeled = scores_u), options))
  expect_equal(a$pred_labeled_point[val, ], b$pred_labeled_point[val, ])
  expect_identical(a$diagnostics$calibration_fold_methods[[1]], b$diagnostics$calibration_fold_methods[[1]])
  expect_equal(a$diagnostics$calibration_fold_efficiency_lambda[[1]], b$diagnostics$calibration_fold_efficiency_lambda[[1]])
  expect_equal(a$pred_labeled_point, a$pred_labeled_variance)
})

test_that("honest point estimation is shared across inference methods and wrappers", {
  x <- seq(-1, 1, length.out = 23); y <- x + x^2; u <- seq(-2, 2, length.out = 47)
  expected <- mean_pointestimate(y, x, u, method = "linear", cross_fit = TRUE, calibration_folds = 3)
  for (inf in c("wald", "jackknife", "bootstrap")) {
    opts <- list(Y = y, Yhat = x, Yhat_unlabeled = u, method = "linear",
                 cross_fit = TRUE, calibration_folds = 3, inference = inf, random_state = 7, n_resamples = 4)
    r <- do.call(mean_inference, opts)
    expect_equal(r$pointestimate, expected)
    expect_equal(r$se, do.call(mean_se, opts))
    expect_equal(r$ci, do.call(mean_ci, opts))
  }
  a <- mean_inference(y, x, u)
  b <- mean_inference(y, x, u, cross_fit = FALSE)
  expect_equal(a$pointestimate, b$pointestimate)
  expect_equal(a$se, b$se)
  expect_false(a$diagnostics$cross_fit)
})

test_that("honest calibration supports leave-one-out and validates fold settings", {
  x <- 0:5; y <- x^2; u <- 0:11
  r <- mean_inference(y, x, u, cross_fit = TRUE, calibration_folds = length(y))
  expect_identical(r$diagnostics$calibration_folds, 6L)
  for (bad in list(1, 2.5, TRUE, NA_real_)) {
    expect_error(mean_inference(y, x, u, cross_fit = TRUE, calibration_folds = bad), "calibration_folds")
    expect_error(mean_inference(y, x, u, inference = "jackknife", jackknife_folds = bad), "jackknife_folds")
  }
  expect_error(mean_inference(y, x, u, cross_fit = "yes"), "cross_fit")
})

test_that("auto jackknife matches explicit re-selection on deleted samples", {
  set.seed(67)
  scores <- seq(-1, 1, length.out = 20)
  y <- scores^2 + rnorm(20, sd = .2)
  u <- seq(-1.2, 1.2, length.out = 40)
  opts <- list(method = "auto", candidate_methods = c("aipw", "linear"),
               num_folds = 2, selection_random_state = 0)
  r <- do.call(mean_inference, c(list(Y = y, Yhat = scores, Yhat_unlabeled = u,
                                    inference = "jackknife", jackknife_folds = 4, random_state = 7), opts))
  split <- ppiAIPW:::kfold_splits(20, 4, 7)
  split_u <- ppiAIPW:::kfold_splits(40, 4, 7)
  deleted <- vapply(seq_len(4), function(k) {
    do.call(mean_pointestimate, c(list(Y = y[split[[k]]$train], Yhat = scores[split[[k]]$train],
                                     Yhat_unlabeled = u[split_u[[k]]$train]), opts))
  }, numeric(1))
  pseudo <- 4 * r$pointestimate - 3 * deleted
  expect_equal(r$se, sd(pseudo) / 2, tolerance = 1e-12)
})

test_that("honest calibration supports every calibrator and multiple outputs", {
  set.seed(135)
  x <- matrix(rnorm(48), 24, 2); x_u <- matrix(rnorm(96), 48, 2)
  scores <- plogis(x); u <- plogis(x_u)
  y <- scores^2 + matrix(rnorm(48, sd = .1), 24, 2)
  for (method in c("linear", "prognostic_linear", "sigmoid", "monotone_spline", "isotonic")) {
    r <- mean_inference(y, scores, u, method = method, cross_fit = TRUE, calibration_folds = 3,
                        X = x, X_unlabeled = x_u, w = seq(1, 2, length.out = 24),
                        w_unlabeled = seq(1, 3, length.out = 48))
    expect_length(r$pointestimate, 2)
    expect_true(all(is.finite(r$pointestimate)))
    expect_true(all(r$se > 0))
  }
})

test_that("causal Wald inference passes honest calibration to each arm", {
  set.seed(17)
  a <- rep(0:1, each = 20)
  x <- rnorm(40); predictions <- cbind(x, x + 1)
  colnames(predictions) <- c("0", "1")
  y <- x + a + rnorm(40, sd = .2)
  for (weighted in c(FALSE, TRUE)) {
    r <- causal_inference(y, a, predictions, method = "linear", cross_fit = TRUE,
                          calibration_folds = 3, w = if (weighted) seq(1, 2, length.out = 40) else NULL)
    expect_true(r$arm_results[["0"]]$diagnostics$cross_fit)
    expect_true(r$arm_results[["1"]]$diagnostics$cross_fit)
    expect_equal(r$ate[["1"]], r$arm_means[["1"]] - r$arm_means[["0"]])
  }
})
