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
                                        diagnostic_mode,
                                        effective_num_folds = NULL,
                                        reference_covariates = NULL) {
  structure(
    list(
      method = method,
      diagnostic_mode = diagnostic_mode,
      n_outputs = n_outputs,
      n_labeled = n_labeled,
      num_bins = num_bins,
      per_output = per_output,
      effective_num_folds = effective_num_folds,
      reference_covariates = reference_covariates
    ),
    class = "ppi_calibration_diagnostics"
  )
}

print.ppi_calibration_diagnostics <- function(x, ...) {
  cat(sprintf(
    "ppi_calibration_diagnostics(method='%s', diagnostic_mode='%s', n_outputs=%d, n_labeled=%d, num_bins=%d)\n",
    x$method, x$diagnostic_mode, x$n_outputs, x$n_labeled, x$num_bins
  ))
  invisible(x)
}

normalize_diagnostic_mode <- function(diagnostic_mode) {
  mode <- gsub("-", "_", tolower(as.character(diagnostic_mode[[1]])))
  if (mode %in% c("out_of_fold", "oof")) {
    return("out_of_fold")
  }
  if (mode %in% c("in_sample", "insample")) {
    return("in_sample")
  }
  stop("diagnostic_mode must be 'out_of_fold' or 'in_sample'.", call. = FALSE)
}

fit_blp_diagnostics <- function(outcomes, calibrated_scores, weights) {
  outcomes <- as.numeric(outcomes)
  calibrated_scores <- as.numeric(calibrated_scores)
  weights <- as.numeric(weights)
  predictor_centered <- calibrated_scores - stats::weighted.mean(calibrated_scores, w = weights)

  if (all(isTRUE(all.equal(predictor_centered, rep(0, length(predictor_centered)), tolerance = 1e-12)))) {
    intercept <- stats::weighted.mean(outcomes, w = weights)
    residuals <- outcomes - intercept
    intercept_var <- sum((weights * residuals)^2) / (sum(weights)^2)
    intercept_se <- sqrt(max(intercept_var, 0))
    intercept_ci <- z_interval(intercept, intercept_se, alpha = 0.05, alternative = "two-sided")
    return(list(
      intercept = intercept,
      slope = NA_real_,
      intercept_se = intercept_se,
      slope_se = NA_real_,
      intercept_ci = c(intercept_ci$lower[[1]], intercept_ci$upper[[1]]),
      slope_ci = c(NA_real_, NA_real_),
      slope_null = 1.0,
      slope_wald_t = NA_real_,
      slope_p_value = NA_real_
    ))
  }

  design <- cbind(1, calibrated_scores)
  fit <- stats::lm.wfit(x = design, y = outcomes, w = weights)
  coef <- as.numeric(fit$coefficients)
  residuals <- as.numeric(fit$residuals)
  xtwx <- crossprod(design, weights * design)
  xtwx_inv <- solve(xtwx)
  design_weighted_resid <- design * (weights * residuals)
  meat <- crossprod(design_weighted_resid)
  covariance <- xtwx_inv %*% meat %*% xtwx_inv
  se <- sqrt(pmax(diag(covariance), 0))
  ci <- z_interval(coef, se, alpha = 0.05, alternative = "two-sided")
  slope_stats <- compute_wald_statistics(
    pointestimate = coef[[2]],
    standard_error = se[[2]],
    null = 1.0,
    alternative = "two-sided"
  )
  list(
    intercept = coef[[1]],
    slope = coef[[2]],
    intercept_se = se[[1]],
    slope_se = se[[2]],
    intercept_ci = c(ci$lower[[1]], ci$upper[[1]]),
    slope_ci = c(ci$lower[[2]], ci$upper[[2]]),
    slope_null = 1.0,
    slope_wald_t = slope_stats$t_stat[[1]],
    slope_p_value = slope_stats$p_value[[1]]
  )
}

summary.ppi_calibration_diagnostics <- function(object, null = 1, alternative = "two-sided", ...) {
  null_value <- as.numeric(null)
  if (length(null_value) != 1L || !is.finite(null_value)) {
    stop("summary.ppi_calibration_diagnostics expects a single finite null value.", call. = FALSE)
  }
  alt <- normalize_alternative(alternative)
  lines <- c(
    "ppi_calibration_diagnostics summary",
    sprintf("method: %s", object$method),
    sprintf("diagnostic_mode: %s", object$diagnostic_mode),
    sprintf("n_outputs: %d", object$n_outputs),
    sprintf("n_labeled: %d", object$n_labeled),
    sprintf("num_bins: %d", object$num_bins),
    sprintf("blp_slope_null: %s", trimws(summary_value(null_value))),
    sprintf("blp_slope_alternative: %s", alt)
  )
  if (!is.null(object$effective_num_folds)) {
    lines <- c(lines, sprintf("effective_num_folds: %d", object$effective_num_folds))
  }
  for (i in seq_along(object$per_output)) {
    blp <- object$per_output[[i]]$blp
    stats_obj <- compute_wald_statistics(
      pointestimate = blp$slope,
      standard_error = blp$slope_se,
      null = null_value,
      alternative = alt
    )
    ci <- z_interval(blp$slope, blp$slope_se, alpha = 0.05, alternative = alt)
    prefix <- if (object$n_outputs == 1L) {
      "calibrated_blp_slope"
    } else {
      sprintf("output[%d] calibrated_blp_slope", i - 1L)
    }
    lines <- c(lines, sprintf(
      "%s: estimate=%s, se=%s, ci=(%s, %s), wald_t=%s, p_value=%s",
      prefix,
      summary_value(blp$slope),
      summary_value(blp$slope_se),
      summary_value(ci$lower[[1]]),
      summary_value(ci$upper[[1]]),
      summary_value(stats_obj$t_stat[[1]]),
      summary_value(stats_obj$p_value[[1]])
    ))
  }
  structure(
    list(lines = lines, text = paste(lines, collapse = "\n")),
    class = "summary_ppi_calibration_diagnostics"
  )
}

print.summary_ppi_calibration_diagnostics <- function(x, ...) {
  cat(x$text, "\n")
  invisible(x)
}

#' Calibration diagnostics for fitted calibration models
#'
#' @param obj A fitted result or model.
#' @param Y Observed outcomes.
#' @param Yhat Labeled-sample predictions.
#' @param X Optional covariates for prognostic-linear diagnostics.
#' @param w Optional labeled-sample weights.
#' @param diagnostic_mode Either `"out_of_fold"` for honest out-of-fold diagnostics
#'   or `"in_sample"` for descriptive fit-on-fit diagnostics.
#' @param num_folds Number of folds used when `diagnostic_mode = "out_of_fold"`.
#' @param num_bins Number of bins.
#' @return A `ppi_calibration_diagnostics` object.
calibration_diagnostics <- function(obj, Y, Yhat, X = NULL, w = NULL,
                                    diagnostic_mode = "out_of_fold",
                                    num_folds = 10,
                                    num_bins = 10) {
  if (num_bins < 1L) {
    stop("num_bins must be at least 1.", call. = FALSE)
  }
  if (num_folds < 2L) {
    stop("num_folds must be at least 2.", call. = FALSE)
  }
  model <- resolve_model(obj)
  diagnostic_mode <- normalize_diagnostic_mode(diagnostic_mode)
  pair <- validate_pair_inputs(Y, Yhat)
  weights <- construct_weight_vector(nrow(pair$Y_2d), w, vectorized = FALSE)
  X_2d <- if (is.null(X)) NULL else coerce_covariates(X, nrow(pair$Y_2d), "X")

  if (inherits(model, "ppi_prognostic_linear_model") && model$x_dim > 0L && is.null(X_2d)) {
    stop("X is required for calibration diagnostics when the fitted model uses prognostic covariates.", call. = FALSE)
  }

  if (inherits(model, "ppi_prognostic_linear_model") && model$x_dim > 0L) {
    reference_covariates <- colMeans(weight_matrix(weights, nrow(X_2d), ncol(X_2d)) * X_2d)
  } else {
    reference_covariates <- NULL
  }

  if (identical(diagnostic_mode, "in_sample")) {
    if (inherits(model, "ppi_prognostic_linear_model")) {
      calibrated_labeled <- reshape_to_2d(predict(model, pair$Yhat_2d, X = X_2d), "calibrated_labeled")
    } else {
      calibrated_labeled <- reshape_to_2d(predict(model, pair$Yhat_2d), "calibrated_labeled")
    }
    effective_num_folds <- NULL
  } else {
    n_splits <- min(as.integer(num_folds), nrow(pair$Y_2d))
    if (n_splits < 2L) {
      stop("out_of_fold calibration diagnostics require at least two labeled observations.", call. = FALSE)
    }
    splits <- kfold_splits(nrow(pair$Y_2d), n_splits = n_splits, seed = 0L)
    calibrated_labeled <- matrix(NA_real_, nrow(pair$Y_2d), ncol(pair$Y_2d))
    for (split in splits) {
      w_train <- if (is.null(w)) NULL else as.numeric(w)[split$train]
      if (inherits(model, "ppi_prognostic_linear_model")) {
        fitted <- fit_prognostic_linear(
          Y = pair$Y_2d[split$train, , drop = FALSE],
          Yhat = pair$Yhat_2d[split$train, , drop = FALSE],
          Yhat_unlabeled = pair$Yhat_2d[split$val, , drop = FALSE],
          X = if (is.null(X_2d)) NULL else X_2d[split$train, , drop = FALSE],
          X_unlabeled = if (is.null(X_2d)) NULL else X_2d[split$val, , drop = FALSE],
          w = w_train
        )
      } else {
        fitted <- fit_and_calibrate(
          Y = pair$Y_2d[split$train, , drop = FALSE],
          Yhat = pair$Yhat_2d[split$train, , drop = FALSE],
          Yhat_unlabeled = pair$Yhat_2d[split$val, , drop = FALSE],
          method = model$method,
          w = w_train,
          isocal_backend = if (!is.null(model$metadata$isocal_backend)) model$metadata$isocal_backend else "weighted_pava",
          isocal_max_depth = if (!is.null(model$metadata$isocal_max_depth)) model$metadata$isocal_max_depth else 20,
          isocal_min_child_weight = if (!is.null(model$metadata$isocal_min_child_weight)) model$metadata$isocal_min_child_weight else 10
        )
      }
      calibrated_labeled[split$val, ] <- fitted$pred_unlabeled
    }
    effective_num_folds <- n_splits
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
      fitted_curve = fitted_curve,
      blp = fit_blp_diagnostics(outcomes, calibrated_scores, weights)
    )
  }
  new_calibration_diagnostics(
    method = model$method,
    diagnostic_mode = diagnostic_mode,
    n_outputs = ncol(pair$Y_2d),
    n_labeled = nrow(pair$Y_2d),
    num_bins = as.integer(min(num_bins, nrow(pair$Y_2d))),
    per_output = per_output,
    effective_num_folds = effective_num_folds,
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
    xlab = "Score value",
    ylab = "Observed outcome",
    main = if (x$n_outputs > 1L) {
      sprintf(
        "%s calibration%s (output %d)",
        x$method,
        if (identical(x$diagnostic_mode, "out_of_fold")) " (out-of-fold)" else "",
        idx
      )
    } else {
      sprintf(
        "%s calibration%s",
        x$method,
        if (identical(x$diagnostic_mode, "out_of_fold")) " (out-of-fold)" else ""
      )
    }
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
  legend_labels <- c()
  legend_lty <- c()
  legend_lwd <- c()
  legend_pch <- c()
  legend_col <- c()
  if (isTRUE(show_identity)) {
    legend_labels <- c(legend_labels, "Identity")
    legend_lty <- c(legend_lty, 2)
    legend_lwd <- c(legend_lwd, 1)
    legend_pch <- c(legend_pch, NA_integer_)
    legend_col <- c(legend_col, "grey60")
  }
  legend_labels <- c(legend_labels, "Fitted calibration")
  legend_lty <- c(legend_lty, 1)
  legend_lwd <- c(legend_lwd, 2)
  legend_pch <- c(legend_pch, NA_integer_)
  legend_col <- c(legend_col, "steelblue")
  if (isTRUE(show_bins)) {
    legend_labels <- c(
      legend_labels,
      "Bin mean outcome at raw score",
      "Same bin outcome at calibrated score"
    )
    legend_lty <- c(legend_lty, NA_integer_, NA_integer_)
    legend_lwd <- c(legend_lwd, NA_integer_, NA_integer_)
    legend_pch <- c(legend_pch, 16, 1)
    legend_col <- c(legend_col, "firebrick", "darkgreen")
  }
  graphics::legend(
    "topleft",
    legend = legend_labels,
    lty = legend_lty,
    lwd = legend_lwd,
    pch = legend_pch,
    col = legend_col,
    bty = "n",
    cex = 0.85
  )
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
