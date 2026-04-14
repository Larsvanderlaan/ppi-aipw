resolve_model <- function(obj) {
  if (inherits(obj, "ppi_mean_result")) {
    return(obj$calibrator)
  }
  if (inherits(obj, "ppi_calibration_model") || inherits(obj, "ppi_prognostic_linear_model")) {
    return(obj)
  }
  stop("Expected a ppi_mean_result, ppi_calibration_model, or ppi_prognostic_linear_model.", call. = FALSE)
}

weighted_average_scalar <- function(x, weights) {
  stats::weighted.mean(as.numeric(x), w = as.numeric(weights))
}

bin_diagnostics <- function(raw_scores, calibrated_scores, outcomes, weights, num_bins) {
  order_idx <- order(raw_scores)
  groups <- split(order_idx, cut(seq_along(order_idx), breaks = min(num_bins, length(order_idx)), labels = FALSE))
  groups <- Filter(length, groups)
  bin_centers <- numeric(length(groups))
  bin_mean_raw <- numeric(length(groups))
  bin_mean_calibrated <- numeric(length(groups))
  bin_mean_outcome <- numeric(length(groups))
  bin_counts <- integer(length(groups))
  for (i in seq_along(groups)) {
    idx <- groups[[i]]
    bin_centers[[i]] <- 0.5 * (min(raw_scores[idx]) + max(raw_scores[idx]))
    bin_mean_raw[[i]] <- weighted_average_scalar(raw_scores[idx], weights[idx])
    bin_mean_calibrated[[i]] <- weighted_average_scalar(calibrated_scores[idx], weights[idx])
    bin_mean_outcome[[i]] <- weighted_average_scalar(outcomes[idx], weights[idx])
    bin_counts[[i]] <- length(idx)
  }
  list(
    bin_centers = bin_centers,
    bin_mean_raw = bin_mean_raw,
    bin_mean_calibrated = bin_mean_calibrated,
    bin_mean_outcome = bin_mean_outcome,
    bin_counts = bin_counts
  )
}

new_calibration_diagnostics <- function(method, n_outputs, n_labeled, num_bins, per_output,
                                        reference_covariates = NULL) {
  structure(
    list(
      method = method,
      n_outputs = n_outputs,
      n_labeled = n_labeled,
      num_bins = num_bins,
      per_output = per_output,
      reference_covariates = reference_covariates
    ),
    class = "ppi_calibration_diagnostics"
  )
}

print.ppi_calibration_diagnostics <- function(x, ...) {
  cat(sprintf(
    "ppi_calibration_diagnostics(method='%s', n_outputs=%d, n_labeled=%d, num_bins=%d)\n",
    x$method, x$n_outputs, x$n_labeled, x$num_bins
  ))
  invisible(x)
}

#' Calibration diagnostics on the labeled sample
#'
#' @param obj A fitted result or model.
#' @param Y Observed outcomes.
#' @param Yhat Labeled-sample predictions.
#' @param X Optional covariates for prognostic-linear diagnostics.
#' @param w Optional labeled-sample weights.
#' @param num_bins Number of bins.
#' @return A `ppi_calibration_diagnostics` object.
calibration_diagnostics <- function(obj, Y, Yhat, X = NULL, w = NULL, num_bins = 10) {
  if (num_bins < 1L) {
    stop("num_bins must be at least 1.", call. = FALSE)
  }
  model <- resolve_model(obj)
  pair <- validate_pair_inputs(Y, Yhat)
  weights <- construct_weight_vector(nrow(pair$Y_2d), w, vectorized = FALSE)
  X_2d <- if (is.null(X)) NULL else coerce_covariates(X, nrow(pair$Y_2d), "X")
  if (inherits(model, "ppi_prognostic_linear_model")) {
    if (model$x_dim > 0L && is.null(X_2d)) {
      stop("X is required for calibration diagnostics when the fitted model uses prognostic covariates.", call. = FALSE)
    }
    reference_covariates <- if (model$x_dim > 0L) colMeans(weight_matrix(weights, nrow(X_2d), ncol(X_2d)) * X_2d) else NULL
    calibrated_labeled <- reshape_to_2d(predict(model, pair$Yhat_2d, X = X_2d), "calibrated_labeled")
  } else {
    reference_covariates <- NULL
    calibrated_labeled <- reshape_to_2d(predict(model, pair$Yhat_2d), "calibrated_labeled")
  }
  per_output <- vector("list", ncol(pair$Y_2d))
  for (j in seq_len(ncol(pair$Y_2d))) {
    raw_scores <- pair$Yhat_2d[, j]
    calibrated_scores <- calibrated_labeled[, j]
    outcomes <- pair$Y_2d[, j]
    bins <- bin_diagnostics(raw_scores, calibrated_scores, outcomes, weights, num_bins)
    grid_scores <- seq(min(raw_scores), max(raw_scores), length.out = 200L)
    if (inherits(model, "ppi_prognostic_linear_model")) {
      X_grid <- if (model$x_dim > 0L) matrix(reference_covariates, nrow = length(grid_scores), ncol = model$x_dim, byrow = TRUE) else NULL
      fitted_curve <- as.numeric(reshape_to_2d(predict(model, matrix(grid_scores, ncol = 1L), X = X_grid), "fitted_curve"))
    } else {
      fitted_curve <- predict_coordinate_calibrator(model$calibrators[[j]], grid_scores)
    }
    per_output[[j]] <- list(
      raw_labeled_scores = raw_scores,
      calibrated_labeled_scores = calibrated_scores,
      observed_outcomes = outcomes,
      bin_centers = bins$bin_centers,
      bin_mean_raw_score = bins$bin_mean_raw,
      bin_mean_calibrated_score = bins$bin_mean_calibrated,
      bin_mean_outcome = bins$bin_mean_outcome,
      bin_counts = bins$bin_counts,
      grid_scores = grid_scores,
      fitted_curve = fitted_curve
    )
  }
  new_calibration_diagnostics(
    method = model$method,
    n_outputs = ncol(pair$Y_2d),
    n_labeled = nrow(pair$Y_2d),
    num_bins = as.integer(min(num_bins, nrow(pair$Y_2d))),
    per_output = per_output,
    reference_covariates = reference_covariates
  )
}

plot.ppi_calibration_diagnostics <- function(x, output_index = 1L, show_identity = TRUE,
                                             show_bins = TRUE, ...) {
  idx <- as.integer(output_index[[1]])
  if (idx < 1L || idx > x$n_outputs) {
    stop(sprintf("output_index must lie in [1, %d].", x$n_outputs), call. = FALSE)
  }
  record <- x$per_output[[idx]]
  x_min <- min(c(record$grid_scores, record$raw_labeled_scores))
  x_max <- max(c(record$grid_scores, record$raw_labeled_scores))
  y_min <- min(c(record$observed_outcomes, record$fitted_curve))
  y_max <- max(c(record$observed_outcomes, record$fitted_curve))
  graphics::plot(
    NA_real_,
    xlim = c(x_min, x_max),
    ylim = c(y_min, y_max),
    xlab = "Prediction score",
    ylab = "Observed outcome",
    main = if (x$n_outputs > 1L) sprintf("%s calibration (output %d)", x$method, idx) else sprintf("%s calibration", x$method)
  )
  if (isTRUE(show_identity)) {
    lo <- min(x_min, y_min)
    hi <- max(x_max, y_max)
    graphics::abline(a = 0, b = 1, lty = 2, col = "grey60")
    graphics::segments(lo, lo, hi, hi, lty = 2, col = "grey60")
  }
  graphics::lines(record$grid_scores, record$fitted_curve, lwd = 2, col = "steelblue")
  if (isTRUE(show_bins)) {
    graphics::points(record$bin_mean_raw_score, record$bin_mean_outcome, pch = 16, col = "firebrick")
    graphics::points(record$bin_mean_calibrated_score, record$bin_mean_outcome, pch = 1, col = "darkgreen")
  }
  invisible(x)
}

#' Plot calibration diagnostics
#'
#' @param diagnostics A `ppi_calibration_diagnostics` object.
#' @param output_index Output coordinate to plot.
#' @return Invisibly returns `diagnostics`.
plot_calibration <- function(diagnostics, output_index = 1L, show_identity = TRUE, show_bins = TRUE, ...) {
  plot(diagnostics, output_index = output_index, show_identity = show_identity, show_bins = show_bins, ...)
}
