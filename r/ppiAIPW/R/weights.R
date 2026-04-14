#' Compute nonnegative balancing weights for a two-sample design
#'
#' @param X_labeled Feature matrix for the labeled sample.
#' @param X_unlabeled Feature matrix for the unlabeled sample.
#' @param target One of `"pooled"` or `"unlabeled"`.
#' @param include_intercept Whether to include an intercept balance moment.
#' @param tolerance Maximum allowed balance error.
#' @param maxiter Maximum optimizer iterations.
#' @param return_diagnostics Whether to return a diagnostics list.
#' @return A weight vector, or a list with weights and diagnostics.
project_balancing_weights_active_set <- function(constraint_matrix, constraint_rhs, tolerance, maxiter) {
  n_weights <- ncol(constraint_matrix)
  active_zero <- rep(FALSE, n_weights)
  iterations <- 0L
  while (iterations < maxiter) {
    iterations <- iterations + 1L
    free_idx <- which(!active_zero)
    if (length(free_idx) == 0L) {
      stop(
        "Could not compute nonnegative balancing weights with the requested balance moments. Try a lower-dimensional balance representation or target='pooled'.",
        call. = FALSE
      )
    }
    A_free <- constraint_matrix[, free_idx, drop = FALSE]
    rhs_gap <- A_free %*% rep(1, length(free_idx)) - constraint_rhs
    lambda <- MASS::ginv(A_free %*% t(A_free)) %*% rhs_gap
    weights_free <- as.numeric(rep(1, length(free_idx)) - t(A_free) %*% lambda)
    if (all(weights_free >= -tolerance)) {
      weights <- numeric(n_weights)
      weights[free_idx] <- pmax(weights_free, 0)
      return(list(weights = weights, iterations = iterations))
    }
    active_zero[free_idx[[which.min(weights_free)]]] <- TRUE
  }
  stop(
    "Could not compute nonnegative balancing weights with the requested balance moments. Try a lower-dimensional balance representation or target='pooled'.",
    call. = FALSE
  )
}

compute_two_sample_balancing_weights <- function(X_labeled, X_unlabeled,
                                                 target = "pooled",
                                                 include_intercept = TRUE,
                                                 tolerance = 1e-8,
                                                 maxiter = 1000,
                                                 return_diagnostics = FALSE) {
  X_labeled_2d <- reshape_to_2d(X_labeled, "X_labeled")
  X_unlabeled_2d <- reshape_to_2d(X_unlabeled, "X_unlabeled")
  if (ncol(X_labeled_2d) != ncol(X_unlabeled_2d)) {
    stop(
      sprintf("X_labeled and X_unlabeled must have the same number of columns. Got %d and %d.", ncol(X_labeled_2d), ncol(X_unlabeled_2d)),
      call. = FALSE
    )
  }
  if (nrow(X_labeled_2d) == 0L || nrow(X_unlabeled_2d) == 0L) {
    stop("Both labeled and unlabeled samples must be nonempty.", call. = FALSE)
  }
  resolved_target <- tolower(target[[1]])
  if (!resolved_target %in% c("pooled", "unlabeled")) {
    stop("target must be either 'pooled' or 'unlabeled'.", call. = FALSE)
  }
  if (!(is.numeric(tolerance) && tolerance > 0)) {
    stop("tolerance must be strictly positive.", call. = FALSE)
  }
  if (!(is.numeric(maxiter) && maxiter >= 1)) {
    stop("maxiter must be at least 1.", call. = FALSE)
  }
  Z_labeled <- if (isTRUE(include_intercept)) cbind(1, X_labeled_2d) else X_labeled_2d
  Z_unlabeled <- if (isTRUE(include_intercept)) cbind(1, X_unlabeled_2d) else X_unlabeled_2d
  n_labeled <- nrow(Z_labeled)
  n_unlabeled <- nrow(Z_unlabeled)
  rho <- n_labeled / (n_labeled + n_unlabeled)
  labeled_mean <- colMeans(Z_labeled)
  unlabeled_mean <- colMeans(Z_unlabeled)
  target_mean <- if (resolved_target == "pooled") {
    rho * labeled_mean + (1 - rho) * unlabeled_mean
  } else {
    unlabeled_mean
  }
  if (max(abs(labeled_mean - target_mean)) <= tolerance) {
    weights <- rep(1, n_labeled)
    diagnostics <- list(
      target = resolved_target,
      include_intercept = include_intercept,
      n_labeled = n_labeled,
      n_unlabeled = n_unlabeled,
      n_balance_functions = ncol(Z_labeled),
      target_mean = target_mean,
      weighted_labeled_mean = labeled_mean,
      balance_error = labeled_mean - target_mean,
      max_abs_balance_error = max(abs(labeled_mean - target_mean)),
      min_weight = 1,
      max_weight = 1,
      optimizer_success = TRUE,
      optimizer_status = 0L,
      optimizer_message = "already balanced",
      optimizer_iterations = 0L
    )
    if (isTRUE(return_diagnostics)) {
      return(list(weights = weights, diagnostics = diagnostics))
    }
    return(weights)
  }

  constraint_matrix <- t(Z_labeled)
  constraint_rhs <- n_labeled * target_mean
  fit <- project_balancing_weights_active_set(
    constraint_matrix = constraint_matrix,
    constraint_rhs = constraint_rhs,
    tolerance = tolerance,
    maxiter = maxiter
  )
  weights <- fit$weights
  weighted_mean_labeled <- colMeans(weights * Z_labeled)
  balance_error <- weighted_mean_labeled - target_mean
  max_abs_balance_error <- max(abs(balance_error))
  if (max_abs_balance_error > tolerance) {
    stop(
      sprintf(
        "Could not achieve the requested balance tolerance with nonnegative weights. Max absolute balance error was %.3e.",
        max_abs_balance_error
      ),
      call. = FALSE
    )
  }
  diagnostics <- list(
    target = resolved_target,
    include_intercept = include_intercept,
    n_labeled = n_labeled,
    n_unlabeled = n_unlabeled,
    n_balance_functions = ncol(Z_labeled),
    target_mean = target_mean,
    weighted_labeled_mean = weighted_mean_labeled,
    balance_error = balance_error,
    max_abs_balance_error = max_abs_balance_error,
    min_weight = min(weights),
    max_weight = max(weights),
    optimizer_success = TRUE,
    optimizer_status = 0L,
    optimizer_message = "active_set_projection",
    optimizer_iterations = fit$iterations
  )
  if (isTRUE(return_diagnostics)) {
    return(list(weights = weights, diagnostics = diagnostics))
  }
  weights
}
