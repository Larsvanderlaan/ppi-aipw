new_coordinate_calibrator <- function(method, fitted, y_min, y_max, metadata = list()) {
  structure(
    list(
      method = method,
      fitted = fitted,
      y_min = as.numeric(y_min),
      y_max = as.numeric(y_max),
      metadata = metadata
    ),
    class = "ppi_coordinate_calibrator"
  )
}

new_calibration_model <- function(method, calibrators, metadata = list()) {
  structure(
    list(method = method, calibrators = calibrators, metadata = metadata),
    class = "ppi_calibration_model"
  )
}

new_prognostic_linear_model <- function(coefficients, x_dim, metadata = list()) {
  structure(
    list(coefficients = coefficients, x_dim = as.integer(x_dim), metadata = metadata),
    class = "ppi_prognostic_linear_model"
  )
}

print.ppi_calibration_model <- function(x, ...) {
  parts <- c(
    sprintf("method='%s'", x$method),
    sprintf("n_outputs=%d", length(x$calibrators))
  )
  backend <- x$metadata$isocal_backend
  if (!is.null(backend) && !is.na(backend)) {
    parts <- c(parts, sprintf("isocal_backend='%s'", backend))
  }
  cat(sprintf("ppi_calibration_model(%s)\n", paste(parts, collapse = ", ")))
  invisible(x)
}

print.ppi_prognostic_linear_model <- function(x, ...) {
  cat(
    sprintf(
      "ppi_prognostic_linear_model(n_outputs=%d, x_dim=%d)\n",
      length(x$coefficients),
      x$x_dim
    )
  )
  invisible(x)
}

predict.ppi_prognostic_linear_model <- function(object, scores, X = NULL, ...) {
  scores_mat <- reshape_to_2d(scores, "scores")
  if (is.null(X)) {
    X_mat <- matrix(0, nrow(scores_mat), object$x_dim)
  } else {
    X_mat <- reshape_to_2d(X, "X")
    if (nrow(X_mat) != nrow(scores_mat)) {
      stop("X must have the same number of rows as scores.", call. = FALSE)
    }
    if (ncol(X_mat) != object$x_dim) {
      stop(sprintf("Expected X to have %d column(s), got %d.", object$x_dim, ncol(X_mat)), call. = FALSE)
    }
  }
  preds <- matrix(NA_real_, nrow(scores_mat), length(object$coefficients))
  for (j in seq_along(object$coefficients)) {
    coef <- object$coefficients[[j]]
    design <- cbind(1, scores_mat[, j], X_mat)
    preds[, j] <- as.numeric(design %*% coef)
  }
  restore_shape(preds, scores)
}

predict_coordinate_calibrator <- function(calibrator, scores) {
  scores <- as.numeric(scores)
  if (calibrator$method == "aipw") {
    return(scores)
  }
  if (isTRUE(all.equal(calibrator$y_max, calibrator$y_min))) {
    return(rep(calibrator$y_min, length(scores)))
  }
  if (calibrator$method == "linear") {
    pred <- calibrator$fitted$slope * scores + calibrator$fitted$intercept
    return(clip_range(pred, calibrator$y_min, calibrator$y_max))
  }
  if (calibrator$method == "sigmoid") {
    scores_scaled <- clip_unit((scores - calibrator$y_min) / (calibrator$y_max - calibrator$y_min))
    logits <- safe_logit(scores_scaled)
    calibrated <- sigmoid(calibrator$fitted$slope * logits + calibrator$fitted$intercept)
    pred <- calibrator$y_min + (calibrator$y_max - calibrator$y_min) * calibrated
    return(clip_range(pred, calibrator$y_min, calibrator$y_max))
  }
  if (calibrator$method == "isotonic") {
    x_vals <- calibrator$fitted$x
    y_vals <- calibrator$fitted$y
    pred <- stats::approx(
      x = x_vals,
      y = y_vals,
      xout = scores,
      method = "constant",
      f = 0,
      rule = 2,
      ties = "ordered"
    )$y
    return(clip_range(pred, calibrator$y_min, calibrator$y_max))
  }
  if (calibrator$method == "monotone_spline") {
    fitted <- calibrator$fitted
    score_scale <- fitted$score_scale
    if (isTRUE(all.equal(score_scale, 0))) {
      return(rep(fitted$intercept, length(scores)))
    }
    z <- pmax(pmin((scores - fitted$score_min) / score_scale, 1), 0)
    pred <- evaluate_monotone_spline(
      scores_scaled = z,
      knots = fitted$knots,
      coef = fitted$coef,
      degree = fitted$basis_degree,
      intercept = fitted$intercept
    )
    return(clip_range(pred, calibrator$y_min, calibrator$y_max))
  }
  stop(sprintf("Unsupported calibrator method '%s'.", calibrator$method), call. = FALSE)
}

predict.ppi_calibration_model <- function(object, scores, ...) {
  scores_mat <- reshape_to_2d(scores, "scores")
  if (ncol(scores_mat) != length(object$calibrators)) {
    stop(sprintf("Expected %d columns, got %d.", length(object$calibrators), ncol(scores_mat)), call. = FALSE)
  }
  pred <- vapply(
    seq_along(object$calibrators),
    function(j) predict_coordinate_calibrator(object$calibrators[[j]], scores_mat[, j]),
    numeric(nrow(scores_mat))
  )
  if (is.vector(pred)) {
    pred <- matrix(pred, ncol = 1L)
  }
  restore_shape(pred, scores)
}

fit_linear_coordinate <- function(y, scores, sample_weight) {
  design <- cbind(scores, 1)
  fit <- stats::lm.wfit(x = design, y = y, w = sample_weight)
  coef <- fit$coefficients
  coef[is.na(coef)] <- 0
  new_coordinate_calibrator(
    method = "linear",
    fitted = list(slope = unname(coef[[1]]), intercept = unname(coef[[2]])),
    y_min = min(y),
    y_max = max(y)
  )
}

fit_platt_coordinate <- function(y, scores, sample_weight) {
  y_min <- min(y)
  y_max <- max(y)
  if (isTRUE(all.equal(y_max, y_min))) {
    return(new_coordinate_calibrator(
      method = "sigmoid",
      fitted = list(slope = 0, intercept = 0),
      y_min = y_min,
      y_max = y_max
    ))
  }
  y_scaled <- clip_unit((y - y_min) / (y_max - y_min))
  scores_scaled <- clip_unit((scores - y_min) / (y_max - y_min))
  logits <- safe_logit(scores_scaled)
  mean_y <- stats::weighted.mean(y_scaled, sample_weight)
  start <- c(1, safe_logit(mean_y))
  if (length(unique(round(logits, 12))) == 1L) {
    return(new_coordinate_calibrator(
      method = "sigmoid",
      fitted = list(slope = 0, intercept = unname(start[[2]])),
      y_min = y_min,
      y_max = y_max
    ))
  }
  normalized_weight <- sample_weight / sum(sample_weight)
  objective <- function(beta) {
    p <- clip_unit(sigmoid(beta[[1]] * logits + beta[[2]]))
    loss <- -(y_scaled * log(p) + (1 - y_scaled) * log(1 - p))
    sum(normalized_weight * loss)
  }
  fit <- try(stats::optim(start, objective, method = "BFGS"), silent = TRUE)
  beta <- if (inherits(fit, "try-error") || is.null(fit$par) || any(!is.finite(fit$par))) start else fit$par
  new_coordinate_calibrator(
    method = "sigmoid",
    fitted = list(slope = unname(beta[[1]]), intercept = unname(beta[[2]])),
    y_min = y_min,
    y_max = y_max
  )
}

weighted_pava <- function(y, w) {
  y <- as.numeric(y)
  w <- as.numeric(w)
  level <- y
  weight <- w
  length_block <- rep(1L, length(y))
  i <- 1L
  while (i < length(level)) {
    if (level[[i]] <= level[[i + 1L]] + 1e-12) {
      i <- i + 1L
      next
    }
    new_weight <- weight[[i]] + weight[[i + 1L]]
    new_level <- (weight[[i]] * level[[i]] + weight[[i + 1L]] * level[[i + 1L]]) / new_weight
    level[[i]] <- new_level
    weight[[i]] <- new_weight
    length_block[[i]] <- length_block[[i]] + length_block[[i + 1L]]
    level <- level[-(i + 1L)]
    weight <- weight[-(i + 1L)]
    length_block <- length_block[-(i + 1L)]
    while (i > 1L && level[[i - 1L]] > level[[i]] + 1e-12) {
      new_weight <- weight[[i - 1L]] + weight[[i]]
      new_level <- (weight[[i - 1L]] * level[[i - 1L]] + weight[[i]] * level[[i]]) / new_weight
      level[[i - 1L]] <- new_level
      weight[[i - 1L]] <- new_weight
      length_block[[i - 1L]] <- length_block[[i - 1L]] + length_block[[i]]
      level <- level[-i]
      weight <- weight[-i]
      length_block <- length_block[-i]
      i <- i - 1L
    }
  }
  rep(level, length_block)
}

fit_isotonic_coordinate <- function(y, scores, sample_weight) {
  y_min <- min(y)
  y_max <- max(y)
  if (length(unique(round(scores, 12))) <= 1L || isTRUE(all.equal(y_max, y_min))) {
    mean_y <- stats::weighted.mean(y, sample_weight)
    return(new_coordinate_calibrator(
      method = "linear",
      fitted = list(slope = 0, intercept = mean_y),
      y_min = y_min,
      y_max = y_max,
      metadata = list(fallback = 1)
    ))
  }
  ord <- order(scores)
  scores_ord <- scores[ord]
  y_ord <- y[ord]
  w_ord <- sample_weight[ord]
  uniq_scores <- sort(unique(scores_ord))
  y_agg <- numeric(length(uniq_scores))
  w_agg <- numeric(length(uniq_scores))
  for (i in seq_along(uniq_scores)) {
    mask <- scores_ord == uniq_scores[[i]]
    w_agg[[i]] <- sum(w_ord[mask])
    y_agg[[i]] <- sum(w_ord[mask] * y_ord[mask]) / w_agg[[i]]
  }
  y_fit <- weighted_pava(y_agg, w_agg)
  new_coordinate_calibrator(
    method = "isotonic",
    fitted = list(x = uniq_scores, y = clip_range(y_fit, y_min, y_max)),
    y_min = y_min,
    y_max = y_max,
    metadata = list(backend = "weighted_pava")
  )
}

choose_monotone_spline_knots <- function(scores_scaled, max_internal_knots, degree) {
  unique_scores <- sort(unique(scores_scaled))
  max_allowed <- max(0L, length(unique_scores) - degree - 1L)
  n_internal <- min(max_internal_knots, max_allowed)
  if (n_internal <= 0L) {
    internal <- numeric(0)
  } else {
    probs <- seq(0, 1, length.out = n_internal + 2L)[-c(1L, n_internal + 2L)]
    internal <- unique(as.numeric(stats::quantile(scores_scaled, probs = probs, type = 7)))
    internal <- internal[internal > 1e-8 & internal < 1 - 1e-8]
  }
  c(rep(0, degree + 1L), internal, rep(1, degree + 1L))
}

integrated_bspline_design <- function(scores_scaled, knots, degree) {
  x <- pmax(pmin(as.numeric(scores_scaled), 1), 0)
  grid <- sort(unique(c(0, x)))
  basis <- splines::splineDesign(knots = knots, x = grid, ord = degree + 1L, outer.ok = TRUE)
  n_basis <- ncol(basis)
  integ <- matrix(0, nrow = length(grid), ncol = n_basis)
  if (length(grid) > 1L) {
    dx <- diff(grid)
    for (j in seq_len(n_basis)) {
      for (i in 2:length(grid)) {
        integ[i, j] <- integ[i - 1L, j] + 0.5 * dx[i - 1L] * (basis[i - 1L, j] + basis[i, j])
      }
    }
  }
  integ[match(x, grid), , drop = FALSE]
}

evaluate_monotone_spline <- function(scores_scaled, knots, coef, degree, intercept) {
  basis <- integrated_bspline_design(scores_scaled, knots = knots, degree = degree)
  as.numeric(intercept + basis %*% coef)
}

fit_monotone_spline_coordinate <- function(y, scores, sample_weight) {
  y_min <- min(y)
  y_max <- max(y)
  if (length(unique(round(scores, 12))) <= 1L || isTRUE(all.equal(y_max, y_min))) {
    mean_y <- stats::weighted.mean(y, sample_weight)
    return(new_coordinate_calibrator(
      method = "linear",
      fitted = list(slope = 0, intercept = mean_y),
      y_min = y_min,
      y_max = y_max,
      metadata = list(fallback = 1)
    ))
  }
  score_min <- min(scores)
  score_max <- max(scores)
  score_scale <- score_max - score_min
  if (isTRUE(all.equal(score_scale, 0)) || length(unique(scores)) < 4L) {
    return(fit_linear_coordinate(y, scores, sample_weight))
  }
  scores_scaled <- pmax(pmin((scores - score_min) / score_scale, 1), 0)
  degree <- .monotone_spline_derivative_degree
  knots <- choose_monotone_spline_knots(
    scores_scaled,
    max_internal_knots = .monotone_spline_max_internal_knots,
    degree = degree
  )
  basis <- integrated_bspline_design(scores_scaled, knots = knots, degree = degree)
  n_basis <- ncol(basis)
  objective <- function(par) {
    intercept <- par[[1]]
    coef <- par[-1L]
    pred <- as.numeric(intercept + basis %*% coef)
    penalty <- if (length(coef) < 3L) sum(coef^2) else sum(diff(coef, differences = 2L)^2)
    sum(sample_weight * (y - pred)^2) + .monotone_spline_penalty * penalty
  }
  start_intercept <- stats::weighted.mean(y, sample_weight)
  start <- c(start_intercept, rep(0.1, n_basis))
  fit <- try(
    stats::optim(
      par = start,
      fn = objective,
      method = "L-BFGS-B",
      lower = c(-Inf, rep(0, n_basis))
    ),
    silent = TRUE
  )
  if (inherits(fit, "try-error") || is.null(fit$par) || any(!is.finite(fit$par))) {
    return(fit_linear_coordinate(y, scores, sample_weight))
  }
  new_coordinate_calibrator(
    method = "monotone_spline",
    fitted = list(
      intercept = unname(fit$par[[1]]),
      coef = unname(fit$par[-1L]),
      knots = knots,
      basis_degree = degree,
      score_min = score_min,
      score_scale = score_scale
    ),
    y_min = y_min,
    y_max = y_max,
    metadata = list(
      max_internal_knots = .monotone_spline_max_internal_knots,
      basis_degree = degree + 1L,
      penalty = .monotone_spline_penalty
    )
  )
}

#' Fit a calibration model
#'
#' @param Y Observed outcomes on the labeled sample.
#' @param Yhat Predictions on the labeled sample.
#' @param method Calibration method.
#' @param w Optional labeled-sample weights.
#' @return A `ppi_calibration_model`.
fit_calibrator <- function(Y, Yhat, method = "monotone_spline", w = NULL,
                           isocal_backend = "weighted_pava",
                           isocal_max_depth = 20,
                           isocal_min_child_weight = 10) {
  method <- canonical_method(method)
  if (identical(method, "prognostic_linear")) {
    stop(
      "method='prognostic_linear' requires optional covariates X/X_unlabeled and is available through mean_inference(...) and causal_inference(...).",
      call. = FALSE
    )
  }
  pair <- validate_pair_inputs(Y, Yhat)
  weights <- construct_weight_vector(nrow(pair$Y_2d), w, vectorized = FALSE)
  calibrators <- vector("list", ncol(pair$Y_2d))
  for (j in seq_len(ncol(pair$Y_2d))) {
    y_coord <- pair$Y_2d[, j]
    score_coord <- pair$Yhat_2d[, j]
    calibrators[[j]] <- switch(
      method,
      aipw = new_coordinate_calibrator(
        method = "aipw",
        fitted = NULL,
        y_min = min(y_coord),
        y_max = max(y_coord)
      ),
      linear = fit_linear_coordinate(y_coord, score_coord, weights),
      sigmoid = fit_platt_coordinate(y_coord, score_coord, weights),
      isotonic = fit_isotonic_coordinate(y_coord, score_coord, weights),
      monotone_spline = fit_monotone_spline_coordinate(y_coord, score_coord, weights),
      stop(sprintf("Unsupported method '%s'.", method), call. = FALSE)
    )
  }
  new_calibration_model(
    method = method,
    calibrators = calibrators,
    metadata = list(
      n_outputs = ncol(pair$Y_2d),
      isocal_backend = if (method == "isotonic") isocal_backend else NA_character_,
      isocal_max_depth = isocal_max_depth,
      isocal_min_child_weight = isocal_min_child_weight
    )
  )
}

#' Calibrate labeled and unlabeled predictions
#'
#' @param Y Observed outcomes on the labeled sample.
#' @param Yhat Predictions on the labeled sample.
#' @param Yhat_unlabeled Optional unlabeled predictions.
#' @param method Calibration method.
#' @param w Optional labeled-sample weights.
#' @param return_model If `TRUE`, also return the fitted model.
#' @return Calibrated predictions, and optionally the model.
calibrate_predictions <- function(Y, Yhat, Yhat_unlabeled = NULL, method = "monotone_spline",
                                  w = NULL, isocal_backend = "weighted_pava",
                                  isocal_max_depth = 20, isocal_min_child_weight = 10,
                                  return_model = FALSE) {
  model <- fit_calibrator(
    Y = Y,
    Yhat = Yhat,
    method = method,
    w = w,
    isocal_backend = isocal_backend,
    isocal_max_depth = isocal_max_depth,
    isocal_min_child_weight = isocal_min_child_weight
  )
  pred_labeled <- predict(model, Yhat)
  pred_unlabeled <- if (is.null(Yhat_unlabeled)) NULL else predict(model, Yhat_unlabeled)
  if (return_model) {
    return(list(pred_labeled = pred_labeled, pred_unlabeled = pred_unlabeled, model = model))
  }
  list(pred_labeled = pred_labeled, pred_unlabeled = pred_unlabeled)
}
