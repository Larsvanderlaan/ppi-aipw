new_causal_result <- function(arm_means, arm_ses, arm_cis, ate, ate_ses, ate_cis,
                              control_arm, treatment_levels, arm_results, diagnostics) {
  structure(
    list(
      arm_means = arm_means,
      arm_ses = arm_ses,
      arm_cis = arm_cis,
      ate = ate,
      ate_ses = ate_ses,
      ate_cis = ate_cis,
      control_arm = control_arm,
      treatment_levels = treatment_levels,
      arm_results = arm_results,
      diagnostics = diagnostics
    ),
    class = "ppi_causal_result"
  )
}

print.ppi_causal_result <- function(x, ...) {
  ate_preview <- paste(
    vapply(names(x$ate), function(name) sprintf("%s=%s", name, summary_value(x$ate[[name]])), character(1L)),
    collapse = ", "
  )
  cat(sprintf(
    "ppi_causal_result(control_arm=%s, treatment_levels=%s, inference='%s', ate={%s})\n",
    deparse(x$control_arm),
    paste0("(", paste(vapply(x$treatment_levels, deparse, character(1L)), collapse = ", "), ")"),
    x$diagnostics$inference,
    ate_preview
  ))
  invisible(x)
}

summary.ppi_causal_result <- function(object, null = 0, alternative = "two-sided", ...) {
  ordered_arms <- object$treatment_levels
  arm_estimates <- vapply(ordered_arms, function(arm) object$arm_means[[as.character(arm)]], numeric(1L))
  arm_ses <- vapply(ordered_arms, function(arm) object$arm_ses[[as.character(arm)]], numeric(1L))
  arm_ci_lower <- vapply(ordered_arms, function(arm) object$arm_cis[[as.character(arm)]][[1L]], numeric(1L))
  arm_ci_upper <- vapply(ordered_arms, function(arm) object$arm_cis[[as.character(arm)]][[2L]], numeric(1L))
  arm_stats <- compute_wald_statistics(arm_estimates, arm_ses, null = null, alternative = alternative)
  comparison_arms <- ordered_arms[ordered_arms != object$control_arm]
  ate_estimates <- vapply(comparison_arms, function(arm) object$ate[[as.character(arm)]], numeric(1L))
  ate_ses <- vapply(comparison_arms, function(arm) object$ate_ses[[as.character(arm)]], numeric(1L))
  ate_ci_lower <- vapply(comparison_arms, function(arm) object$ate_cis[[as.character(arm)]][[1L]], numeric(1L))
  ate_ci_upper <- vapply(comparison_arms, function(arm) object$ate_cis[[as.character(arm)]][[2L]], numeric(1L))
  ate_stats <- compute_wald_statistics(ate_estimates, ate_ses, null = null, alternative = alternative)
  lines <- c(
    "ppi_causal_result summary",
    sprintf("control_arm: %s", deparse(object$control_arm)),
    sprintf("treatment_levels: %s", paste(vapply(ordered_arms, deparse, character(1L)), collapse = ", ")),
    sprintf("inference: %s", object$diagnostics$inference),
    sprintf("wald_null: %s", preview_value(null, digits = 6L)),
    sprintf("wald_alternative: %s", normalize_alternative(alternative)),
    "",
    "Arm means:"
  )
  for (i in seq_along(ordered_arms)) {
    lines <- c(lines, sprintf(
      "arm=%s: estimate=%s, se=%s, ci=(%s, %s), wald_t=%s, p_value=%s",
      deparse(ordered_arms[[i]]),
      summary_value(arm_estimates[[i]]),
      summary_value(arm_ses[[i]]),
      summary_value(arm_ci_lower[[i]]),
      summary_value(arm_ci_upper[[i]]),
      summary_value(arm_stats$t_stat[[i]]),
      summary_value(arm_stats$p_value[[i]])
    ))
  }
  lines <- c(lines, "", "ATEs vs control:")
  for (i in seq_along(comparison_arms)) {
    lines <- c(lines, sprintf(
      "%s - %s: estimate=%s, se=%s, ci=(%s, %s), wald_t=%s, p_value=%s",
      deparse(comparison_arms[[i]]),
      deparse(object$control_arm),
      summary_value(ate_estimates[[i]]),
      summary_value(ate_ses[[i]]),
      summary_value(ate_ci_lower[[i]]),
      summary_value(ate_ci_upper[[i]]),
      summary_value(ate_stats$t_stat[[i]]),
      summary_value(ate_stats$p_value[[i]])
    ))
  }
  structure(list(lines = lines, text = paste(lines, collapse = "\n")), class = "summary_ppi_causal_result")
}

print.summary_ppi_causal_result <- function(x, ...) {
  cat(x$text, "\n")
  invisible(x)
}

validate_outcome_vector <- function(Y) {
  Y_arr <- as.numeric(Y)
  if (is.matrix(Y) && ncol(Y) != 1L) {
    stop("causal_inference currently supports one-dimensional outcomes only.", call. = FALSE)
  }
  if (length(Y_arr) == 0L) {
    stop("Y must be nonempty.", call. = FALSE)
  }
  validate_finite_numeric(Y_arr, "Y")
}

validate_weight_vector <- function(w, n_obs) {
  if (is.null(w)) {
    return(NULL)
  }
  weights <- as.numeric(w)
  if (length(weights) != n_obs) {
    stop(sprintf("Expected weights with length %d, got %d.", n_obs, length(weights)), call. = FALSE)
  }
  validate_finite_numeric(weights, "Weights")
  if (any(weights < 0)) {
    stop("Weights must be nonnegative.", call. = FALSE)
  }
  if (!any(weights > 0)) {
    stop("At least one weight must be strictly positive.", call. = FALSE)
  }
  weights
}

validate_covariate_matrix <- function(X, n_obs) {
  if (is.null(X)) {
    return(NULL)
  }
  X_arr <- reshape_to_2d(X, "X")
  if (nrow(X_arr) != n_obs) {
    stop(sprintf("X must have %d rows, got %d.", n_obs, nrow(X_arr)), call. = FALSE)
  }
  X_arr
}

resolve_potential_outcome_inputs <- function(A, Yhat_potential, treatment_levels = NULL) {
  A_arr <- as.vector(A)
  if (is.numeric(A_arr)) {
    validate_finite_numeric(A_arr, "A")
  } else if (any(is.na(A_arr))) {
    stop("A must not contain missing values.", call. = FALSE)
  }
  potential_matrix <- as.matrix(Yhat_potential)
  storage.mode(potential_matrix) <- "double"
  validate_finite_numeric(potential_matrix, "Yhat_potential")
  if (nrow(potential_matrix) != length(A_arr)) {
    stop("Yhat_potential must have the same number of rows as Y and A.", call. = FALSE)
  }
  observed_levels <- unique(A_arr)
  if (length(observed_levels) < 2L) {
    stop("causal_inference requires at least two observed treatment arms in A.", call. = FALSE)
  }
  if (is.null(treatment_levels)) {
    resolved_levels <- if (!is.null(colnames(potential_matrix))) colnames(potential_matrix) else sort(observed_levels)
  } else {
    resolved_levels <- treatment_levels
  }
  if (length(resolved_levels) != ncol(potential_matrix)) {
    stop("The number of treatment_levels must match the number of columns in Yhat_potential.", call. = FALSE)
  }
  counts <- setNames(vapply(resolved_levels, function(arm) sum(A_arr == arm), integer(1L)), resolved_levels)
  if (any(counts == 0L)) {
    stop("Every treatment arm must have at least one observed unit.", call. = FALSE)
  }
  list(A = A_arr, potential_matrix = potential_matrix, treatment_levels = resolved_levels, arm_to_column = setNames(seq_along(resolved_levels), resolved_levels))
}

resolve_control_arm <- function(treatment_levels, control_arm = NULL) {
  if (!is.null(control_arm)) {
    if (!control_arm %in% treatment_levels) {
      stop(sprintf("control_arm=%s is not one of the resolved treatment levels.", deparse(control_arm)), call. = FALSE)
    }
    return(control_arm)
  }
  suppressWarnings({
    tryCatch(min(treatment_levels), error = function(e) stop("Could not determine the default control arm from treatment_levels. Pass control_arm explicitly.", call. = FALSE))
  })
}

assemble_full_sample_prediction <- function(labeled_mask, pred_labeled, pred_unlabeled) {
  full_prediction <- numeric(length(labeled_mask))
  full_prediction[labeled_mask] <- as.numeric(pred_labeled)
  full_prediction[!labeled_mask] <- as.numeric(pred_unlabeled)
  full_prediction
}

causal_arm_pointestimate_and_influence <- function(Y, labeled_mask, pred_point, pred_variance, full_sample_weights) {
  arm_probability <- mean(labeled_mask)
  if (arm_probability <= 0) {
    stop("Each treatment arm must contain at least one observation.", call. = FALSE)
  }
  labeled_indicator <- as.numeric(labeled_mask)
  pseudo_outcome_point <- full_sample_weights * (pred_point + (labeled_indicator / arm_probability) * (Y - pred_point))
  pointestimate <- mean(pseudo_outcome_point)
  pseudo_outcome_variance <- full_sample_weights * (pred_variance + (labeled_indicator / arm_probability) * (Y - pred_variance))
  influence <- (pseudo_outcome_variance - mean(pseudo_outcome_variance)) / length(Y)
  list(pointestimate = pointestimate, influence = influence)
}

aligned_wald_influence <- function(labeled_mask, prepared, n_obs) {
  components <- wald_influence_components(prepared)
  influence <- matrix(0, n_obs, ncol(components$labeled))
  influence[labeled_mask, ] <- (components$labeled - matrix(colMeans(components$labeled), nrow(components$labeled), ncol(components$labeled), byrow = TRUE)) / nrow(components$labeled)
  influence[!labeled_mask, ] <- (components$unlabeled - matrix(colMeans(components$unlabeled), nrow(components$unlabeled), ncol(components$unlabeled), byrow = TRUE)) / nrow(components$unlabeled)
  influence
}

#' Causal inference wrapper over the semi-supervised mean engine
#'
#' @param Y Observed outcomes for all units.
#' @param A Treatment assignments.
#' @param Yhat_potential Matrix of predicted potential outcomes.
#' @return A `ppi_causal_result`.
causal_inference <- function(Y, A, Yhat_potential, treatment_levels = NULL, control_arm = NULL,
                             method = "monotone_spline", w = NULL, X = NULL, alpha = 0.1,
                             alternative = "two-sided", efficiency_maximization = FALSE,
                             candidate_methods = c("aipw", "linear", "monotone_spline", "isotonic"),
                             num_folds = 100, auto_unlabeled_subsample_size = NULL,
                             selection_random_state = NULL, isocal_backend = "weighted_pava",
                             isocal_max_depth = 20, isocal_min_child_weight = 10,
                             inference = "wald") {
  if (tolower(inference) != "wald") {
    stop("causal_inference currently supports inference='wald' only.", call. = FALSE)
  }
  Y_arr <- validate_outcome_vector(Y)
  raw_weights <- validate_weight_vector(w, length(Y_arr))
  use_unweighted_path <- is.null(raw_weights) || isTRUE(all.equal(raw_weights, rep(raw_weights[[1]], length(raw_weights))))
  full_sample_weights <- if (use_unweighted_path) NULL else construct_weight_vector(length(Y_arr), raw_weights, vectorized = FALSE)
  X_arr <- validate_covariate_matrix(X, length(Y_arr))
  resolved <- resolve_potential_outcome_inputs(A, Yhat_potential, treatment_levels)
  resolved_control_arm <- resolve_control_arm(resolved$treatment_levels, control_arm)

  arm_results <- list()
  arm_means <- list()
  arm_ses <- list()
  arm_cis <- list()
  arm_influences <- list()

  for (arm in resolved$treatment_levels) {
    labeled_mask <- resolved$A == arm
    unlabeled_mask <- !labeled_mask
    arm_predictions <- resolved$potential_matrix[, resolved$arm_to_column[[as.character(arm)]], drop = FALSE]
    prepared <- prepare_inference_inputs(
      Y = Y_arr[labeled_mask],
      Yhat = arm_predictions[labeled_mask, , drop = FALSE],
      Yhat_unlabeled = arm_predictions[unlabeled_mask, , drop = FALSE],
      method = method,
      w = if (is.null(raw_weights)) NULL else raw_weights[labeled_mask],
      w_unlabeled = if (is.null(raw_weights)) NULL else raw_weights[unlabeled_mask],
      X = if (is.null(X_arr)) NULL else X_arr[labeled_mask, , drop = FALSE],
      X_unlabeled = if (is.null(X_arr)) NULL else X_arr[unlabeled_mask, , drop = FALSE],
      efficiency_maximization = efficiency_maximization,
      candidate_methods = candidate_methods,
      num_folds = num_folds,
      auto_unlabeled_subsample_size = auto_unlabeled_subsample_size,
      selection_random_state = selection_random_state,
      isocal_backend = isocal_backend,
      isocal_max_depth = isocal_max_depth,
      isocal_min_child_weight = isocal_min_child_weight
    )
    pointestimate <- aipw_mean_pointestimate_from_predictions(
      Y = prepared$Y_2d,
      pred_labeled = prepared$pred_labeled_point,
      pred_unlabeled = prepared$pred_unlabeled_point,
      w = prepared$weights,
      w_unlabeled = prepared$weights_unlabeled_point
    )
    se <- wald_standard_error(prepared)
    ci <- z_interval(pointestimate, se, alpha = alpha, alternative = alternative)
    result <- new_mean_result(
      pointestimate = pointestimate[[1]],
      se = se[[1]],
      ci = list(ci$lower[[1]], ci$upper[[1]]),
      method = prepared$method,
      selected_candidate = prepared$selected_candidate,
      selected_efficiency_maximization = prepared$selected_efficiency_maximization,
      efficiency_lambda = prepared$efficiency_lambda,
      inference = "wald",
      diagnostics = prepared$diagnostics,
      calibrator = prepared$calibrator
    )
    if (use_unweighted_path) {
      arm_results[[as.character(arm)]] <- result
      arm_means[[as.character(arm)]] <- result$pointestimate
      arm_ses[[as.character(arm)]] <- result$se
      arm_cis[[as.character(arm)]] <- result$ci
      arm_influences[[as.character(arm)]] <- as.numeric(aligned_wald_influence(labeled_mask, prepared, length(Y_arr)))
    } else {
      pred_point_full <- assemble_full_sample_prediction(labeled_mask, prepared$pred_labeled_point, prepared$pred_unlabeled_point)
      pred_variance_full <- assemble_full_sample_prediction(labeled_mask, prepared$pred_labeled_variance, prepared$pred_unlabeled_variance)
      arm_obj <- causal_arm_pointestimate_and_influence(
        Y = Y_arr,
        labeled_mask = labeled_mask,
        pred_point = pred_point_full,
        pred_variance = pred_variance_full,
        full_sample_weights = full_sample_weights
      )
      arm_se <- sqrt(sum(arm_obj$influence^2))
      arm_interval <- z_interval(arm_obj$pointestimate, arm_se, alpha = alpha, alternative = alternative)
      arm_results[[as.character(arm)]] <- new_mean_result(
        pointestimate = arm_obj$pointestimate,
        se = arm_se,
        ci = list(arm_interval$lower[[1]], arm_interval$upper[[1]]),
        method = result$method,
        selected_candidate = result$selected_candidate,
        selected_efficiency_maximization = result$selected_efficiency_maximization,
        efficiency_lambda = result$efficiency_lambda,
        inference = result$inference,
        diagnostics = c(result$diagnostics, list(causal_weight_normalization = "global_full_sample")),
        calibrator = result$calibrator
      )
      arm_means[[as.character(arm)]] <- arm_obj$pointestimate
      arm_ses[[as.character(arm)]] <- arm_se
      arm_cis[[as.character(arm)]] <- list(arm_interval$lower[[1]], arm_interval$upper[[1]])
      arm_influences[[as.character(arm)]] <- arm_obj$influence
    }
  }

  ordered_arm_names <- as.character(resolved$treatment_levels)
  influence_matrix <- do.call(cbind, arm_influences[ordered_arm_names])
  covariance <- t(influence_matrix) %*% influence_matrix
  control_idx <- match(as.character(resolved_control_arm), ordered_arm_names)

  ate <- list()
  ate_ses <- list()
  ate_cis <- list()
  for (arm in resolved$treatment_levels) {
    if (identical(arm, resolved_control_arm)) next
    arm_idx <- match(as.character(arm), ordered_arm_names)
    estimate <- arm_means[[as.character(arm)]] - arm_means[[as.character(resolved_control_arm)]]
    variance <- covariance[arm_idx, arm_idx] + covariance[control_idx, control_idx] - 2 * covariance[arm_idx, control_idx]
    ate_se <- sqrt(max(variance, 0))
    interval <- z_interval(estimate, ate_se, alpha = alpha, alternative = alternative)
    ate[[as.character(arm)]] <- estimate
    ate_ses[[as.character(arm)]] <- ate_se
    ate_cis[[as.character(arm)]] <- list(interval$lower[[1]], interval$upper[[1]])
  }

  diagnostics <- list(
    inference = "wald",
    treatment_levels = resolved$treatment_levels,
    control_arm = resolved_control_arm,
    arm_counts = setNames(vapply(resolved$treatment_levels, function(arm) sum(resolved$A == arm), integer(1L)), as.character(resolved$treatment_levels)),
    arm_prediction_columns = resolved$arm_to_column,
    per_arm = lapply(arm_results, `[[`, "diagnostics")
  )
  new_causal_result(
    arm_means = arm_means,
    arm_ses = arm_ses,
    arm_cis = arm_cis,
    ate = ate,
    ate_ses = ate_ses,
    ate_cis = ate_cis,
    control_arm = resolved_control_arm,
    treatment_levels = resolved$treatment_levels,
    arm_results = arm_results,
    diagnostics = diagnostics
  )
}
