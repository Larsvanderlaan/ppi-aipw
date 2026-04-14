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

  objective_factory <- function(penalty) {
    function(weights) {
      weighted_mean <- colMeans(weights * Z_labeled)
      balance_error <- weighted_mean - target_mean
      0.5 * sum((weights - 1)^2) + penalty * sum(balance_error^2)
    }
  }

  best <- NULL
  penalties <- c(1e2, 1e4, 1e6, 1e8)
  for (penalty in penalties) {
    fit <- try(
      stats::optim(
        par = rep(1, n_labeled),
        fn = objective_factory(penalty),
        method = "L-BFGS-B",
        lower = rep(0, n_labeled),
        control = list(maxit = maxiter)
      ),
      silent = TRUE
    )
    if (inherits(fit, "try-error") || is.null(fit$par)) {
      next
    }
    weights <- as.numeric(fit$par)
    weighted_mean_labeled <- colMeans(weights * Z_labeled)
    balance_error <- weighted_mean_labeled - target_mean
    max_abs_balance_error <- max(abs(balance_error))
    best <- list(
      weights = weights,
      diagnostics = list(
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
        optimizer_success = isTRUE(fit$convergence == 0L),
        optimizer_status = fit$convergence,
        optimizer_message = if (!is.null(fit$message)) fit$message else "",
        optimizer_iterations = if (!is.null(fit$counts[["function"]])) fit$counts[["function"]] else NA_integer_
      )
    )
    if (max_abs_balance_error <= max(tolerance, 1e-5)) {
      break
    }
  }

  if (is.null(best) || best$diagnostics$max_abs_balance_error > max(tolerance, 1e-5)) {
    stop(
      "Could not compute nonnegative balancing weights with the requested balance moments. Try a lower-dimensional balance representation or target='pooled'.",
      call. = FALSE
    )
  }
  if (isTRUE(return_diagnostics)) {
    return(best)
  }
  best$weights
}
