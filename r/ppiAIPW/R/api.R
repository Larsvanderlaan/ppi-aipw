solve_prognostic_linear_system <- function(score, X, y, sample_weight, alpha) {
  z <- cbind(1, score)
  z_w <- z * sample_weight
  if (ncol(X) == 0L) {
    gram <- crossprod(z, z_w)
    rhs <- crossprod(z_w, y)
    return(as.numeric(MASS::ginv(gram) %*% rhs))
  }
  x_w <- X * sample_weight
  zz <- crossprod(z, z_w)
  zx <- crossprod(z, x_w)
  xx <- crossprod(X, x_w) + alpha * diag(ncol(X))
  gram <- rbind(cbind(zz, zx), cbind(t(zx), xx))
  rhs <- c(crossprod(z_w, y), crossprod(x_w, y))
  as.numeric(MASS::ginv(gram) %*% rhs)
}

predict_prognostic_linear_from_coef <- function(coef, score, X) {
  design <- cbind(1, score, X)
  as.numeric(design %*% coef)
}

weighted_prediction_error <- function(y_true, y_pred, sample_weight) {
  normalized_weight <- sample_weight / sum(sample_weight)
  sum(normalized_weight * (y_true - y_pred)^2)
}

select_prognostic_linear_alpha <- function(y, score, X, sample_weight) {
  if (ncol(X) == 0L) {
    return(0)
  }
  n_splits <- min(5L, length(y))
  if (n_splits < 2L) {
    return(1)
  }
  splits <- kfold_splits(length(y), n_splits = n_splits, seed = 0L)
  best_alpha <- .prognostic_linear_ridge_grid[[1]]
  best_score <- Inf
  for (alpha in .prognostic_linear_ridge_grid) {
    fold_errors <- numeric(length(splits))
    for (i in seq_along(splits)) {
      split <- splits[[i]]
      coef <- solve_prognostic_linear_system(
        score = score[split$train],
        X = X[split$train, , drop = FALSE],
        y = y[split$train],
        sample_weight = sample_weight[split$train],
        alpha = alpha
      )
      pred_val <- predict_prognostic_linear_from_coef(
        coef = coef,
        score = score[split$val],
        X = X[split$val, , drop = FALSE]
      )
      fold_errors[[i]] <- weighted_prediction_error(
        y_true = y[split$val],
        y_pred = pred_val,
        sample_weight = sample_weight[split$val]
      )
    }
    mean_error <- mean(fold_errors)
    if (mean_error < best_score) {
      best_score <- mean_error
      best_alpha <- alpha
    }
  }
  best_alpha
}

fit_prognostic_linear <- function(Y, Yhat, Yhat_unlabeled, X = NULL, X_unlabeled = NULL, w = NULL) {
  validated <- validate_mean_inputs(Y, Yhat, Yhat_unlabeled)
  X_valid <- validate_prognostic_covariates(
    X = X,
    X_unlabeled = X_unlabeled,
    n_labeled = nrow(validated$Y_2d),
    n_unlabeled = nrow(validated$Yhat_unlabeled_2d)
  )
  X_design <- if (is.null(X_valid$X_2d)) matrix(0, nrow(validated$Y_2d), 0L) else X_valid$X_2d
  X_unlabeled_design <- if (is.null(X_valid$X_unlabeled_2d)) {
    matrix(0, nrow(validated$Yhat_unlabeled_2d), 0L)
  } else {
    X_valid$X_unlabeled_2d
  }
  weights <- construct_weight_vector(nrow(validated$Y_2d), w, vectorized = FALSE)
  pred_labeled <- matrix(0, nrow(validated$Y_2d), ncol(validated$Y_2d))
  pred_unlabeled <- matrix(0, nrow(validated$Yhat_unlabeled_2d), ncol(validated$Yhat_unlabeled_2d))
  coefficients <- vector("list", ncol(validated$Y_2d))
  selected_alphas <- numeric(ncol(validated$Y_2d))
  for (j in seq_len(ncol(validated$Y_2d))) {
    alpha <- select_prognostic_linear_alpha(
      y = validated$Y_2d[, j],
      score = validated$Yhat_2d[, j],
      X = X_design,
      sample_weight = weights
    )
    coef <- solve_prognostic_linear_system(
      score = validated$Yhat_2d[, j],
      X = X_design,
      y = validated$Y_2d[, j],
      sample_weight = weights,
      alpha = alpha
    )
    pred_labeled[, j] <- predict_prognostic_linear_from_coef(coef, validated$Yhat_2d[, j], X_design)
    pred_unlabeled[, j] <- predict_prognostic_linear_from_coef(coef, validated$Yhat_unlabeled_2d[, j], X_unlabeled_design)
    coefficients[[j]] <- coef
    selected_alphas[[j]] <- alpha
  }
  model <- new_prognostic_linear_model(
    coefficients = coefficients,
    x_dim = ncol(X_design),
    metadata = list(
      n_outputs = ncol(validated$Y_2d),
      x_dim = ncol(X_design),
      uses_covariates = ncol(X_design) > 0L,
      ridge_alpha = if (length(selected_alphas) == 1L) selected_alphas[[1]] else selected_alphas,
      ridge_penalizes_only_covariates = TRUE
    )
  )
  list(
    Y_2d = validated$Y_2d,
    Yhat_2d = validated$Yhat_2d,
    pred_labeled = pred_labeled,
    pred_unlabeled = pred_unlabeled,
    model = model
  )
}

fit_and_calibrate <- function(Y, Yhat, Yhat_unlabeled, method, w = NULL, X = NULL, X_unlabeled = NULL,
                              isocal_backend = "weighted_pava", isocal_max_depth = 20,
                              isocal_min_child_weight = 10) {
  if (identical(method, "prognostic_linear")) {
    return(fit_prognostic_linear(
      Y = Y,
      Yhat = Yhat,
      Yhat_unlabeled = Yhat_unlabeled,
      X = X,
      X_unlabeled = X_unlabeled,
      w = w
    ))
  }
  validated <- validate_mean_inputs(Y, Yhat, Yhat_unlabeled)
  model <- fit_calibrator(
    Y = validated$Y_2d,
    Yhat = validated$Yhat_2d,
    method = method,
    w = w,
    isocal_backend = isocal_backend,
    isocal_max_depth = isocal_max_depth,
    isocal_min_child_weight = isocal_min_child_weight
  )
  list(
    Y_2d = validated$Y_2d,
    Yhat_2d = validated$Yhat_2d,
    pred_labeled = reshape_to_2d(predict(model, validated$Yhat_2d), "pred_labeled"),
    pred_unlabeled = reshape_to_2d(predict(model, validated$Yhat_unlabeled_2d), "pred_unlabeled"),
    model = model
  )
}

estimate_efficiency_lambda <- function(Y, pred_labeled, pred_unlabeled, w, w_unlabeled) {
  n_labeled <- nrow(Y)
  n_unlabeled <- nrow(pred_unlabeled)
  c_weight <- 1 - labeled_fraction(n_labeled, n_unlabeled)
  labeled_outcome <- weight_matrix(w, n_labeled, ncol(Y)) * Y
  labeled_score <- c_weight * weight_matrix(w, n_labeled, ncol(pred_labeled)) * pred_labeled
  unlabeled_score <- c_weight * weight_matrix(w_unlabeled, n_unlabeled, ncol(pred_unlabeled)) * pred_unlabeled
  numerator <- colMeans(
    (labeled_outcome - matrix(colMeans(labeled_outcome), nrow(labeled_outcome), ncol(labeled_outcome), byrow = TRUE)) *
      (labeled_score - matrix(colMeans(labeled_score), nrow(labeled_score), ncol(labeled_score), byrow = TRUE))
  )
  denominator <- apply(labeled_score, 2L, stats::var) +
    (n_labeled / n_unlabeled) * apply(unlabeled_score, 2L, stats::var)
  ifelse(denominator > 0, numerator / denominator, 0)
}

apply_efficiency_scaling <- function(Y_train, pred_train, pred_unlabeled, predictions_to_scale,
                                     w_train = NULL, w_unlabeled = NULL) {
  weights_train <- construct_weight_vector(nrow(Y_train), w_train, vectorized = TRUE)
  weights_unlabeled <- construct_weight_vector(nrow(pred_unlabeled), w_unlabeled, vectorized = TRUE)
  lambda_hat <- estimate_efficiency_lambda(
    Y = Y_train,
    pred_labeled = pred_train,
    pred_unlabeled = pred_unlabeled,
    w = weights_train,
    w_unlabeled = weights_unlabeled
  )
  scaled_predictions <- lapply(predictions_to_scale, function(pred) pred * matrix(lambda_hat, nrow(pred), ncol(pred), byrow = TRUE))
  list(lambda_hat = lambda_hat, predictions = scaled_predictions)
}

auto_candidate_specs <- function(candidate_methods) {
  canonical_candidates <- unique(vapply(candidate_methods, canonical_method, character(1L)))
  if (length(canonical_candidates) == 0L) {
    stop("candidate_methods must contain at least one valid method.", call. = FALSE)
  }
  specs <- lapply(canonical_candidates, function(method_name) {
    list(method = method_name, label = method_name, efficiency_maximization = FALSE)
  })
  if ("aipw" %in% canonical_candidates) {
    specs[[length(specs) + 1L]] <- list(
      method = "aipw",
      label = .auto_aipw_efficiency_label,
      efficiency_maximization = TRUE
    )
  }
  list(canonical_candidates = canonical_candidates, specs = specs)
}

cv_selection_score <- function(Y, pred_labeled, pred_unlabeled, w = NULL, w_unlabeled = NULL) {
  n_labeled <- nrow(Y)
  n_unlabeled <- nrow(pred_unlabeled)
  c_weight <- 1 - labeled_fraction(n_labeled, n_unlabeled)
  weights <- construct_weight_vector(n_labeled, w, vectorized = TRUE)
  weights_unlabeled <- construct_weight_vector(n_unlabeled, w_unlabeled, vectorized = TRUE)
  labeled_component <- weight_matrix(weights, n_labeled, ncol(Y)) * (Y - c_weight * pred_labeled)
  unlabeled_component <- weight_matrix(weights_unlabeled, n_unlabeled, ncol(pred_unlabeled)) * (c_weight * pred_unlabeled)
  score <- apply(labeled_component, 2L, stats::var) / n_labeled +
    apply(unlabeled_component, 2L, stats::var) / n_unlabeled
  sum(score)
}

candidate_cv_predictions <- function(Y_2d, Yhat_2d, Yhat_unlabeled_2d, method_name, splits,
                                     w = NULL, w_unlabeled = NULL, X_2d = NULL, X_unlabeled_2d = NULL,
                                     efficiency_maximization = FALSE, isocal_backend = "weighted_pava",
                                     isocal_max_depth = 20, isocal_min_child_weight = 10) {
  pred_oof <- matrix(0, nrow(Y_2d), ncol(Y_2d))
  pred_unlabeled_folds <- vector("list", length(splits))
  for (i in seq_along(splits)) {
    split <- splits[[i]]
    y_train <- Y_2d[split$train, , drop = FALSE]
    yhat_train <- Yhat_2d[split$train, , drop = FALSE]
    yhat_val <- Yhat_2d[split$val, , drop = FALSE]
    w_train <- if (is.null(w)) NULL else w[split$train]
    X_train <- if (is.null(X_2d)) NULL else X_2d[split$train, , drop = FALSE]
    X_val <- if (is.null(X_2d)) NULL else X_2d[split$val, , drop = FALSE]
    if (identical(method_name, "prognostic_linear")) {
      fitted <- fit_prognostic_linear(
        Y = y_train,
        Yhat = yhat_train,
        Yhat_unlabeled = Yhat_unlabeled_2d,
        X = X_train,
        X_unlabeled = X_unlabeled_2d,
        w = w_train
      )
      pred_train <- fitted$pred_labeled
      pred_val <- reshape_to_2d(predict(fitted$model, yhat_val, X = X_val), "pred_val")
      pred_unlabeled <- reshape_to_2d(predict(fitted$model, Yhat_unlabeled_2d, X = X_unlabeled_2d), "pred_unlabeled")
    } else {
      model <- fit_calibrator(
        Y = y_train,
        Yhat = yhat_train,
        method = method_name,
        w = w_train,
        isocal_backend = isocal_backend,
        isocal_max_depth = isocal_max_depth,
        isocal_min_child_weight = isocal_min_child_weight
      )
      pred_train <- reshape_to_2d(predict(model, yhat_train), "pred_train")
      pred_val <- reshape_to_2d(predict(model, yhat_val), "pred_val")
      pred_unlabeled <- reshape_to_2d(predict(model, Yhat_unlabeled_2d), "pred_unlabeled")
    }
    if (isTRUE(efficiency_maximization)) {
      scaled <- apply_efficiency_scaling(
        Y_train = y_train,
        pred_train = pred_train,
        pred_unlabeled = pred_unlabeled,
        predictions_to_scale = list(pred_val, pred_unlabeled),
        w_train = w_train,
        w_unlabeled = w_unlabeled
      )
      pred_val <- scaled$predictions[[1L]]
      pred_unlabeled <- scaled$predictions[[2L]]
    }
    pred_oof[split$val, ] <- pred_val
    pred_unlabeled_folds[[i]] <- pred_unlabeled
  }
  mean_pred_unlabeled <- Reduce(`+`, pred_unlabeled_folds) / length(pred_unlabeled_folds)
  list(pred_oof = pred_oof, pred_unlabeled = mean_pred_unlabeled)
}

select_mean_method_cv_internal <- function(Y, Yhat, Yhat_unlabeled,
                                           candidate_methods = c("aipw", "linear", "monotone_spline", "isotonic"),
                                           w = NULL, w_unlabeled = NULL, X = NULL, X_unlabeled = NULL,
                                           efficiency_maximization = FALSE, num_folds = 100,
                                           auto_unlabeled_subsample_size = NULL,
                                           selection_random_state = NULL,
                                           isocal_backend = "weighted_pava",
                                           isocal_max_depth = 20,
                                           isocal_min_child_weight = 10) {
  validated <- validate_mean_inputs(Y, Yhat, Yhat_unlabeled)
  X_valid <- validate_prognostic_covariates(
    X = X,
    X_unlabeled = X_unlabeled,
    n_labeled = nrow(validated$Y_2d),
    n_unlabeled = nrow(validated$Yhat_unlabeled_2d)
  )
  seeds <- resolve_selection_seeds(selection_random_state)
  subset <- subset_unlabeled_for_auto(
    Yhat_unlabeled_2d = validated$Yhat_unlabeled_2d,
    w_unlabeled = if (is.null(w_unlabeled)) NULL else as.numeric(w_unlabeled),
    X_unlabeled = X_valid$X_unlabeled_2d,
    n_labeled = nrow(validated$Y_2d),
    auto_unlabeled_subsample_size = auto_unlabeled_subsample_size,
    subset_seed = seeds$subset_seed
  )
  specs <- auto_candidate_specs(candidate_methods)
  n_splits <- min(resolve_num_folds(num_folds), nrow(validated$Y_2d))
  if (n_splits < 2L) {
    stop("method='auto' requires at least two labeled observations.", call. = FALSE)
  }
  splits <- kfold_splits(nrow(validated$Y_2d), n_splits = n_splits, seed = seeds$cv_seed)
  candidate_scores <- numeric(length(specs$specs))
  candidate_preds <- vector("list", length(specs$specs))
  for (i in seq_along(specs$specs)) {
    spec <- specs$specs[[i]]
    preds <- candidate_cv_predictions(
      Y_2d = validated$Y_2d,
      Yhat_2d = validated$Yhat_2d,
      Yhat_unlabeled_2d = subset$Yhat_unlabeled,
      method_name = spec$method,
      splits = splits,
      w = w,
      w_unlabeled = subset$w_unlabeled,
      X_2d = X_valid$X_2d,
      X_unlabeled_2d = subset$X_unlabeled,
      efficiency_maximization = spec$efficiency_maximization,
      isocal_backend = isocal_backend,
      isocal_max_depth = isocal_max_depth,
      isocal_min_child_weight = isocal_min_child_weight
    )
    candidate_preds[[i]] <- preds
    candidate_scores[[i]] <- cv_selection_score(
      Y = validated$Y_2d,
      pred_labeled = preds$pred_oof,
      pred_unlabeled = preds$pred_unlabeled,
      w = w,
      w_unlabeled = subset$w_unlabeled
    )
  }
  best_idx <- which.min(candidate_scores)
  selected <- specs$specs[[best_idx]]
  diagnostics <- c(
    list(
      selected_method = selected$method,
      selected_candidate = selected$label,
      selected_efficiency_maximization = selected$efficiency_maximization,
      candidate_scores = stats::setNames(as.list(candidate_scores), vapply(specs$specs, `[[`, character(1L), "label")),
      num_folds = n_splits,
      selection_random_state = if (is.null(selection_random_state)) NULL else as.integer(selection_random_state[[1]])
    ),
    subset$diagnostics
  )
  list(
    selected_method = selected$method,
    diagnostics = diagnostics,
    pred_labeled_cf = candidate_preds[[best_idx]]$pred_oof,
    pred_unlabeled_cf = candidate_preds[[best_idx]]$pred_unlabeled
  )
}

#' Select a mean-inference method by cross-validation
#'
#' @param Y Observed outcomes on the labeled sample.
#' @param Yhat Predictions on the labeled sample.
#' @param Yhat_unlabeled Predictions on the unlabeled sample.
#' @param candidate_methods Candidate methods for `method = "auto"`.
#' @return A list with the selected method and diagnostics.
select_mean_method_cv <- function(Y, Yhat, Yhat_unlabeled,
                                  candidate_methods = c("aipw", "linear", "monotone_spline", "isotonic"),
                                  w = NULL, w_unlabeled = NULL, X = NULL, X_unlabeled = NULL,
                                  efficiency_maximization = FALSE, num_folds = NULL,
                                  auto_unlabeled_subsample_size = NULL,
                                  selection_random_state = NULL,
                                  isocal_backend = "weighted_pava",
                                  isocal_max_depth = 20,
                                  isocal_min_child_weight = 10) {
  selected <- select_mean_method_cv_internal(
    Y = Y,
    Yhat = Yhat,
    Yhat_unlabeled = Yhat_unlabeled,
    candidate_methods = candidate_methods,
    w = w,
    w_unlabeled = w_unlabeled,
    X = X,
    X_unlabeled = X_unlabeled,
    efficiency_maximization = efficiency_maximization,
    num_folds = resolve_num_folds(num_folds),
    auto_unlabeled_subsample_size = auto_unlabeled_subsample_size,
    selection_random_state = selection_random_state,
    isocal_backend = isocal_backend,
    isocal_max_depth = isocal_max_depth,
    isocal_min_child_weight = isocal_min_child_weight
  )
  list(method = selected$selected_method, diagnostics = selected$diagnostics)
}

format_coordinate_parameter <- function(x) {
  if (length(x) == 1L) x[[1]] else as.numeric(x)
}

prepare_inference_inputs <- function(Y, Yhat, Yhat_unlabeled, method, w, w_unlabeled,
                                     X = NULL, X_unlabeled = NULL,
                                     efficiency_maximization = FALSE,
                                     candidate_methods = c("aipw", "linear", "monotone_spline", "isotonic"),
                                     num_folds = 100,
                                     auto_unlabeled_subsample_size = NULL,
                                     selection_random_state = NULL,
                                     isocal_backend = "weighted_pava",
                                     isocal_max_depth = 20,
                                     isocal_min_child_weight = 10) {
  if (tolower(method) == "auto") {
    selected <- select_mean_method_cv_internal(
      Y = Y,
      Yhat = Yhat,
      Yhat_unlabeled = Yhat_unlabeled,
      candidate_methods = candidate_methods,
      w = w,
      w_unlabeled = w_unlabeled,
      X = X,
      X_unlabeled = X_unlabeled,
      efficiency_maximization = efficiency_maximization,
      num_folds = num_folds,
      auto_unlabeled_subsample_size = auto_unlabeled_subsample_size,
      selection_random_state = selection_random_state,
      isocal_backend = isocal_backend,
      isocal_max_depth = isocal_max_depth,
      isocal_min_child_weight = isocal_min_child_weight
    )
    fitted <- fit_and_calibrate(
      Y = Y,
      Yhat = Yhat,
      Yhat_unlabeled = Yhat_unlabeled,
      method = selected$selected_method,
      w = w,
      X = X,
      X_unlabeled = X_unlabeled,
      isocal_backend = isocal_backend,
      isocal_max_depth = isocal_max_depth,
      isocal_min_child_weight = isocal_min_child_weight
    )
    weights <- construct_weight_vector(nrow(fitted$Y_2d), w, vectorized = TRUE)
    weights_unlabeled_point <- construct_weight_vector(nrow(fitted$pred_unlabeled), w_unlabeled, vectorized = TRUE)
    validated <- validate_mean_inputs(Y, Yhat, Yhat_unlabeled)
    X_valid <- validate_prognostic_covariates(
      X = X,
      X_unlabeled = X_unlabeled,
      n_labeled = nrow(validated$Y_2d),
      n_unlabeled = nrow(validated$Yhat_unlabeled_2d)
    )
    subset <- subset_unlabeled_for_auto(
      Yhat_unlabeled_2d = validated$Yhat_unlabeled_2d,
      w_unlabeled = if (is.null(w_unlabeled)) NULL else as.numeric(w_unlabeled),
      X_unlabeled = X_valid$X_unlabeled_2d,
      n_labeled = nrow(validated$Y_2d),
      auto_unlabeled_subsample_size = auto_unlabeled_subsample_size,
      subset_seed = if (!is.null(selected$diagnostics$auto_unlabeled_subsample_seed)) selected$diagnostics$auto_unlabeled_subsample_seed else 0L
    )
    weights_unlabeled_variance <- construct_weight_vector(
      nrow(selected$pred_unlabeled_cf),
      subset$w_unlabeled,
      vectorized = TRUE
    )
    pred_labeled_point <- fitted$pred_labeled
    pred_unlabeled_point <- fitted$pred_unlabeled
    pred_labeled_variance <- selected$pred_labeled_cf
    pred_unlabeled_variance <- selected$pred_unlabeled_cf
    final_efficiency_maximization <- isTRUE(efficiency_maximization) || isTRUE(selected$diagnostics$selected_efficiency_maximization)
    diagnostics <- selected$diagnostics
    diagnostics$final_efficiency_maximization <- final_efficiency_maximization
    diagnostics$lambda_from_cross_fitted_estimates <- final_efficiency_maximization
    efficiency_lambda_out <- NULL
    if (isTRUE(final_efficiency_maximization)) {
      lambda_hat <- estimate_efficiency_lambda(
        Y = fitted$Y_2d,
        pred_labeled = pred_labeled_variance,
        pred_unlabeled = pred_unlabeled_variance,
        w = weights,
        w_unlabeled = weights_unlabeled_variance
      )
      scale_point <- matrix(lambda_hat, nrow(pred_labeled_point), ncol(pred_labeled_point), byrow = TRUE)
      scale_unl <- matrix(lambda_hat, nrow(pred_unlabeled_point), ncol(pred_unlabeled_point), byrow = TRUE)
      scale_var_lab <- matrix(lambda_hat, nrow(pred_labeled_variance), ncol(pred_labeled_variance), byrow = TRUE)
      scale_var_unl <- matrix(lambda_hat, nrow(pred_unlabeled_variance), ncol(pred_unlabeled_variance), byrow = TRUE)
      pred_labeled_point <- pred_labeled_point * scale_point
      pred_unlabeled_point <- pred_unlabeled_point * scale_unl
      pred_labeled_variance <- pred_labeled_variance * scale_var_lab
      pred_unlabeled_variance <- pred_unlabeled_variance * scale_var_unl
      efficiency_lambda_out <- format_coordinate_parameter(lambda_hat)
      diagnostics$efficiency_lambda_source <- "cross_fitted"
      fitted$model$metadata$efficiency_lambda <- efficiency_lambda_out
      fitted$model$metadata$efficiency_lambda_source <- "cross_fitted"
    }
    fitted$model$metadata <- c(fitted$model$metadata, diagnostics)
    return(list(
      Y_2d = fitted$Y_2d,
      pred_labeled_point = pred_labeled_point,
      pred_unlabeled_point = pred_unlabeled_point,
      pred_labeled_variance = pred_labeled_variance,
      pred_unlabeled_variance = pred_unlabeled_variance,
      weights = weights,
      weights_unlabeled_point = weights_unlabeled_point,
      weights_unlabeled_variance = weights_unlabeled_variance,
      calibrator = fitted$model,
      method = selected$selected_method,
      selected_candidate = selected$diagnostics$selected_candidate,
      selected_efficiency_maximization = isTRUE(selected$diagnostics$selected_efficiency_maximization),
      final_efficiency_maximization = final_efficiency_maximization,
      diagnostics = diagnostics,
      efficiency_lambda = efficiency_lambda_out
    ))
  }

  canonical <- canonical_method(method)
  fitted <- fit_and_calibrate(
    Y = Y,
    Yhat = Yhat,
    Yhat_unlabeled = Yhat_unlabeled,
    method = canonical,
    w = w,
    X = X,
    X_unlabeled = X_unlabeled,
    isocal_backend = isocal_backend,
    isocal_max_depth = isocal_max_depth,
    isocal_min_child_weight = isocal_min_child_weight
  )
  weights <- construct_weight_vector(nrow(fitted$Y_2d), w, vectorized = TRUE)
  weights_unlabeled <- construct_weight_vector(nrow(fitted$pred_unlabeled), w_unlabeled, vectorized = TRUE)
  diagnostics <- list(
    selected_method = canonical,
    selected_candidate = canonical,
    selected_efficiency_maximization = isTRUE(efficiency_maximization),
    final_efficiency_maximization = isTRUE(efficiency_maximization),
    lambda_from_cross_fitted_estimates = FALSE
  )
  efficiency_lambda_out <- NULL
  pred_labeled <- fitted$pred_labeled
  pred_unlabeled <- fitted$pred_unlabeled
  if (isTRUE(efficiency_maximization)) {
    lambda_hat <- estimate_efficiency_lambda(
      Y = fitted$Y_2d,
      pred_labeled = pred_labeled,
      pred_unlabeled = pred_unlabeled,
      w = weights,
      w_unlabeled = weights_unlabeled
    )
    pred_labeled <- pred_labeled * matrix(lambda_hat, nrow(pred_labeled), ncol(pred_labeled), byrow = TRUE)
    pred_unlabeled <- pred_unlabeled * matrix(lambda_hat, nrow(pred_unlabeled), ncol(pred_unlabeled), byrow = TRUE)
    efficiency_lambda_out <- format_coordinate_parameter(lambda_hat)
    diagnostics$efficiency_lambda_source <- "full_sample"
    fitted$model$metadata$efficiency_lambda <- efficiency_lambda_out
    fitted$model$metadata$efficiency_lambda_source <- "full_sample"
  }
  fitted$model$metadata <- c(fitted$model$metadata, diagnostics)
  list(
    Y_2d = fitted$Y_2d,
    pred_labeled_point = pred_labeled,
    pred_unlabeled_point = pred_unlabeled,
    pred_labeled_variance = pred_labeled,
    pred_unlabeled_variance = pred_unlabeled,
    weights = weights,
    weights_unlabeled_point = weights_unlabeled,
    weights_unlabeled_variance = weights_unlabeled,
    calibrator = fitted$model,
    method = canonical,
    selected_candidate = canonical,
    selected_efficiency_maximization = isTRUE(efficiency_maximization),
    final_efficiency_maximization = isTRUE(efficiency_maximization),
    diagnostics = diagnostics,
    efficiency_lambda = efficiency_lambda_out
  )
}

aipw_mean_pointestimate_from_predictions <- function(Y, pred_labeled, pred_unlabeled, w, w_unlabeled) {
  n_labeled <- nrow(Y)
  n_unlabeled <- nrow(pred_unlabeled)
  c_weight <- 1 - labeled_fraction(n_labeled, n_unlabeled)
  labeled_term <- colMeans(weight_matrix(w, n_labeled, ncol(Y)) * (Y - c_weight * pred_labeled))
  unlabeled_term <- colMeans(weight_matrix(w_unlabeled, n_unlabeled, ncol(pred_unlabeled)) * (c_weight * pred_unlabeled))
  labeled_term + unlabeled_term
}

wald_influence_components <- function(prepared) {
  n_labeled <- nrow(prepared$Y_2d)
  n_unlabeled <- nrow(prepared$pred_unlabeled_variance)
  c_weight <- 1 - labeled_fraction(n_labeled, n_unlabeled)
  labeled_component <- weight_matrix(prepared$weights, n_labeled, ncol(prepared$Y_2d)) *
    (prepared$Y_2d - c_weight * prepared$pred_labeled_variance)
  unlabeled_component <- weight_matrix(prepared$weights_unlabeled_variance, n_unlabeled, ncol(prepared$pred_unlabeled_variance)) *
    (c_weight * prepared$pred_unlabeled_variance)
  list(labeled = labeled_component, unlabeled = unlabeled_component)
}

wald_standard_error <- function(prepared) {
  components <- wald_influence_components(prepared)
  sqrt(apply(components$labeled, 2L, stats::var) / nrow(prepared$Y_2d) +
         apply(components$unlabeled, 2L, stats::var) / nrow(prepared$pred_unlabeled_variance))
}

bootstrap_interval <- function(bootstrap_estimates, alpha, alternative) {
  alternative <- normalize_alternative(alternative)
  if (alternative == "two-sided") {
    lower <- apply(bootstrap_estimates, 2L, stats::quantile, probs = alpha / 2, names = FALSE)
    upper <- apply(bootstrap_estimates, 2L, stats::quantile, probs = 1 - alpha / 2, names = FALSE)
  } else if (alternative == "larger") {
    lower <- apply(bootstrap_estimates, 2L, stats::quantile, probs = alpha, names = FALSE)
    upper <- rep(Inf, ncol(bootstrap_estimates))
  } else {
    lower <- rep(-Inf, ncol(bootstrap_estimates))
    upper <- apply(bootstrap_estimates, 2L, stats::quantile, probs = 1 - alpha, names = FALSE)
  }
  list(lower = lower, upper = upper)
}

resolve_jackknife_effective_folds <- function(jackknife_folds, n_labeled, n_unlabeled) {
  if (jackknife_folds < 2L) {
    stop("jackknife_folds must be at least 2 when inference='jackknife'.", call. = FALSE)
  }
  effective <- min(as.integer(jackknife_folds), n_labeled, n_unlabeled)
  if (effective < 2L) {
    stop("jackknife requires at least two labeled and two unlabeled observations.", call. = FALSE)
  }
  max_labeled_fold_size <- ceiling(n_labeled / effective)
  max_unlabeled_fold_size <- ceiling(n_unlabeled / effective)
  if (n_labeled - max_labeled_fold_size < 2L || n_unlabeled - max_unlabeled_fold_size < 1L) {
    stop("jackknife leaves too little data in at least one refit. Try fewer jackknife_folds or inference='wald'.", call. = FALSE)
  }
  effective
}

bootstrap_pointestimates <- function(Y, Yhat, Yhat_unlabeled, method, w, w_unlabeled,
                                     X = NULL, X_unlabeled = NULL, efficiency_maximization = FALSE,
                                     candidate_methods, num_folds,
                                     auto_unlabeled_subsample_size,
                                     selection_random_state, isocal_backend,
                                     isocal_max_depth, isocal_min_child_weight,
                                     n_resamples, random_state) {
  if (n_resamples < 2L) {
    stop("n_resamples must be at least 2 when inference='bootstrap'.", call. = FALSE)
  }
  old_seed <- .Random.seed
  on.exit({
    if (!is.null(old_seed)) {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    }
  }, add = TRUE)
  if (!is.null(random_state)) {
    set.seed(as.integer(random_state[[1]]))
  }
  n_labeled <- NROW(Y)
  n_unlabeled <- NROW(Yhat_unlabeled)
  estimates <- matrix(NA_real_, n_resamples, NCOL(reshape_to_2d(Y, "Y")))
  for (b in seq_len(n_resamples)) {
    labeled_idx <- sample.int(n_labeled, n_labeled, replace = TRUE)
    unlabeled_idx <- sample.int(n_unlabeled, n_unlabeled, replace = TRUE)
    prepared <- prepare_inference_inputs(
      Y = Y[labeled_idx, , drop = FALSE],
      Yhat = Yhat[labeled_idx, , drop = FALSE],
      Yhat_unlabeled = Yhat_unlabeled[unlabeled_idx, , drop = FALSE],
      method = method,
      w = if (is.null(w)) NULL else w[labeled_idx],
      w_unlabeled = if (is.null(w_unlabeled)) NULL else w_unlabeled[unlabeled_idx],
      X = if (is.null(X)) NULL else X[labeled_idx, , drop = FALSE],
      X_unlabeled = if (is.null(X_unlabeled)) NULL else X_unlabeled[unlabeled_idx, , drop = FALSE],
      efficiency_maximization = efficiency_maximization,
      candidate_methods = candidate_methods,
      num_folds = num_folds,
      auto_unlabeled_subsample_size = auto_unlabeled_subsample_size,
      selection_random_state = selection_random_state,
      isocal_backend = isocal_backend,
      isocal_max_depth = isocal_max_depth,
      isocal_min_child_weight = isocal_min_child_weight
    )
    estimates[b, ] <- aipw_mean_pointestimate_from_predictions(
      Y = prepared$Y_2d,
      pred_labeled = prepared$pred_labeled_point,
      pred_unlabeled = prepared$pred_unlabeled_point,
      w = prepared$weights,
      w_unlabeled = prepared$weights_unlabeled_point
    )
  }
  estimates
}

jackknife_pointestimates <- function(Y, Yhat, Yhat_unlabeled, method, w, w_unlabeled,
                                     X = NULL, X_unlabeled = NULL, efficiency_maximization = FALSE,
                                     candidate_methods, num_folds,
                                     auto_unlabeled_subsample_size, selection_random_state,
                                     isocal_backend, isocal_max_depth,
                                     isocal_min_child_weight, jackknife_folds,
                                     random_state) {
  n_labeled <- NROW(Y)
  n_unlabeled <- NROW(Yhat_unlabeled)
  effective_folds <- resolve_jackknife_effective_folds(jackknife_folds, n_labeled, n_unlabeled)
  seed <- if (is.null(random_state)) 1L else as.integer(random_state[[1]])
  labeled_splits <- kfold_splits(n_labeled, effective_folds, seed)
  unlabeled_splits <- kfold_splits(n_unlabeled, effective_folds, seed)
  estimates <- matrix(NA_real_, effective_folds, NCOL(reshape_to_2d(Y, "Y")))
  for (k in seq_len(effective_folds)) {
    prepared <- prepare_inference_inputs(
      Y = Y[labeled_splits[[k]]$train, , drop = FALSE],
      Yhat = Yhat[labeled_splits[[k]]$train, , drop = FALSE],
      Yhat_unlabeled = Yhat_unlabeled[unlabeled_splits[[k]]$train, , drop = FALSE],
      method = method,
      w = if (is.null(w)) NULL else w[labeled_splits[[k]]$train],
      w_unlabeled = if (is.null(w_unlabeled)) NULL else w_unlabeled[unlabeled_splits[[k]]$train],
      X = if (is.null(X)) NULL else X[labeled_splits[[k]]$train, , drop = FALSE],
      X_unlabeled = if (is.null(X_unlabeled)) NULL else X_unlabeled[unlabeled_splits[[k]]$train, , drop = FALSE],
      efficiency_maximization = efficiency_maximization,
      candidate_methods = candidate_methods,
      num_folds = num_folds,
      auto_unlabeled_subsample_size = auto_unlabeled_subsample_size,
      selection_random_state = selection_random_state,
      isocal_backend = isocal_backend,
      isocal_max_depth = isocal_max_depth,
      isocal_min_child_weight = isocal_min_child_weight
    )
    estimates[k, ] <- aipw_mean_pointestimate_from_predictions(
      Y = prepared$Y_2d,
      pred_labeled = prepared$pred_labeled_point,
      pred_unlabeled = prepared$pred_unlabeled_point,
      w = prepared$weights,
      w_unlabeled = prepared$weights_unlabeled_point
    )
  }
  list(estimates = estimates, effective_folds = effective_folds)
}

jackknife_standard_error <- function(jackknife_estimates) {
  n_estimates <- nrow(jackknife_estimates)
  mean_estimate <- colMeans(jackknife_estimates)
  variance <- ((n_estimates - 1) / n_estimates) * colSums((jackknife_estimates - matrix(mean_estimate, n_estimates, ncol(jackknife_estimates), byrow = TRUE))^2)
  sqrt(pmax(variance, 0))
}

new_mean_result <- function(pointestimate, se, ci, method, selected_candidate,
                            selected_efficiency_maximization, efficiency_lambda,
                            inference, diagnostics, calibrator) {
  structure(
    list(
      pointestimate = pointestimate,
      se = se,
      ci = ci,
      method = method,
      selected_candidate = selected_candidate,
      selected_efficiency_maximization = selected_efficiency_maximization,
      efficiency_lambda = efficiency_lambda,
      inference = inference,
      diagnostics = diagnostics,
      calibrator = calibrator
    ),
    class = "ppi_mean_result"
  )
}

print.ppi_mean_result <- function(x, ...) {
  parts <- c(
    sprintf("method='%s'", x$method),
    sprintf("pointestimate=%s", preview_value(x$pointestimate)),
    sprintf("se=%s", preview_value(x$se)),
    sprintf("ci=%s", preview_ci(x$ci)),
    sprintf("inference='%s'", x$inference)
  )
  if (!identical(x$selected_candidate, x$method)) {
    parts <- c(parts, sprintf("selected_candidate='%s'", x$selected_candidate))
  }
  if (!is.null(x$efficiency_lambda)) {
    parts <- c(parts, sprintf("efficiency_lambda=%s", preview_value(x$efficiency_lambda)))
  }
  cat(sprintf("ppi_mean_result(%s)\n", paste(parts, collapse = ", ")))
  invisible(x)
}

summary.ppi_mean_result <- function(object, null = 0, alternative = "two-sided", ...) {
  stats_obj <- compute_wald_statistics(
    pointestimate = object$pointestimate,
    standard_error = object$se,
    null = null,
    alternative = alternative
  )
  object$diagnostics$wald_null <- format_coordinate_output(stats_obj$null)
  object$diagnostics$wald_alternative <- normalize_alternative(alternative)
  object$diagnostics$wald_t_statistic <- format_coordinate_output(stats_obj$t_stat)
  object$diagnostics$wald_p_value <- format_coordinate_output(stats_obj$p_value)
  estimate_arr <- flatten_parameter(object$pointestimate)
  se_arr <- flatten_parameter(object$se)
  lower_arr <- flatten_parameter(object$ci[[1]])
  upper_arr <- flatten_parameter(object$ci[[2]])
  lines <- c(
    "ppi_mean_result summary",
    sprintf("method: %s", object$method),
    sprintf("inference: %s", object$inference)
  )
  if (!identical(object$selected_candidate, object$method)) {
    lines <- c(lines, sprintf("selected_candidate: %s", object$selected_candidate))
  }
  if (isTRUE(object$selected_efficiency_maximization)) {
    lines <- c(lines, "efficiency_maximization: TRUE")
  }
  if (!is.null(object$efficiency_lambda)) {
    lines <- c(lines, sprintf("efficiency_lambda: %s", preview_value(object$efficiency_lambda, digits = 6L)))
  }
  lines <- c(
    lines,
    sprintf("wald_null: %s", preview_value(stats_obj$null, digits = 6L)),
    sprintf("wald_alternative: %s", normalize_alternative(alternative))
  )
  if (length(estimate_arr) == 1L) {
    lines <- c(
      lines,
      sprintf("estimate: %s", summary_value(estimate_arr[[1]])),
      sprintf("se: %s", summary_value(se_arr[[1]])),
      sprintf("ci: (%s, %s)", summary_value(lower_arr[[1]]), summary_value(upper_arr[[1]])),
      sprintf("wald_t: %s", summary_value(stats_obj$t_stat[[1]])),
      sprintf("wald_p_value: %s", summary_value(stats_obj$p_value[[1]]))
    )
  } else {
    for (i in seq_along(estimate_arr)) {
      lines <- c(lines, sprintf(
        "output[%d]: estimate=%s, se=%s, ci=(%s, %s), wald_t=%s, p_value=%s",
        i - 1L,
        summary_value(estimate_arr[[i]]),
        summary_value(se_arr[[i]]),
        summary_value(lower_arr[[i]]),
        summary_value(upper_arr[[i]]),
        summary_value(stats_obj$t_stat[[i]]),
        summary_value(stats_obj$p_value[[i]])
      ))
    }
  }
  structure(list(lines = lines, text = paste(lines, collapse = "\n")), class = "summary_ppi_mean_result")
}

print.summary_ppi_mean_result <- function(x, ...) {
  cat(x$text, "\n")
  invisible(x)
}

fit_mean_inference <- function(Y, Yhat, Yhat_unlabeled, alpha = 0.1, alternative = "two-sided",
                               method = "monotone_spline", w = NULL, w_unlabeled = NULL,
                               X = NULL, X_unlabeled = NULL, efficiency_maximization = FALSE,
                               candidate_methods = c("aipw", "linear", "monotone_spline", "isotonic"),
                               num_folds = NULL, auto_unlabeled_subsample_size = NULL,
                               selection_random_state = NULL, isocal_backend = "weighted_pava",
                               isocal_max_depth = 20, isocal_min_child_weight = 10,
                               inference = "wald", n_resamples = 1000, jackknife_folds = 20,
                               random_state = NULL, compute_se = TRUE, compute_ci = TRUE) {
  inference <- tolower(inference)
  if (!inference %in% c("wald", "jackknife", "bootstrap")) {
    stop("inference must be 'wald', 'jackknife', or 'bootstrap'.", call. = FALSE)
  }
  if (isTRUE(compute_ci)) {
    compute_se <- TRUE
  }
  prepared <- prepare_inference_inputs(
    Y = Y,
    Yhat = Yhat,
    Yhat_unlabeled = Yhat_unlabeled,
    method = method,
    w = w,
    w_unlabeled = w_unlabeled,
    X = X,
    X_unlabeled = X_unlabeled,
    efficiency_maximization = resolve_efficiency_maximization(efficiency_maximization),
    candidate_methods = candidate_methods,
    num_folds = resolve_num_folds(num_folds),
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
  diagnostics <- prepared$diagnostics
  diagnostics$inference <- inference
  standard_error <- NULL
  ci <- NULL
  if (inference == "wald") {
    if (isTRUE(compute_se)) {
      standard_error <- wald_standard_error(prepared)
    }
    if (isTRUE(compute_ci)) {
      interval <- z_interval(pointestimate, standard_error, alpha, alternative)
      ci <- list(interval$lower, interval$upper)
    }
  } else if (inference == "bootstrap") {
    if (tolower(method) == "auto") {
      diagnostics$bootstrap_selected_once <- TRUE
      diagnostics$bootstrap_method <- prepared$method
      diagnostics$bootstrap_efficiency_maximization <- prepared$final_efficiency_maximization
    }
    bootstrap_estimates <- bootstrap_pointestimates(
      Y = prepared$Y_2d,
      Yhat = reshape_to_2d(Yhat, "Yhat"),
      Yhat_unlabeled = reshape_to_2d(Yhat_unlabeled, "Yhat_unlabeled"),
      method = prepared$method,
      w = w,
      w_unlabeled = w_unlabeled,
      X = if (is.null(X)) NULL else reshape_to_2d(X, "X"),
      X_unlabeled = if (is.null(X_unlabeled)) NULL else reshape_to_2d(X_unlabeled, "X_unlabeled"),
      efficiency_maximization = prepared$final_efficiency_maximization,
      candidate_methods = candidate_methods,
      num_folds = resolve_num_folds(num_folds),
      auto_unlabeled_subsample_size = auto_unlabeled_subsample_size,
      selection_random_state = selection_random_state,
      isocal_backend = isocal_backend,
      isocal_max_depth = isocal_max_depth,
      isocal_min_child_weight = isocal_min_child_weight,
      n_resamples = n_resamples,
      random_state = random_state
    )
    if (isTRUE(compute_se)) {
      standard_error <- apply(bootstrap_estimates, 2L, stats::sd)
    }
    if (isTRUE(compute_ci)) {
      interval <- bootstrap_interval(bootstrap_estimates, alpha = alpha, alternative = alternative)
      ci <- list(interval$lower, interval$upper)
    }
  } else {
    if (tolower(method) == "auto") {
      diagnostics$jackknife_selected_once <- TRUE
      diagnostics$jackknife_method <- prepared$method
      diagnostics$jackknife_efficiency_maximization <- prepared$final_efficiency_maximization
    }
    jk <- jackknife_pointestimates(
      Y = prepared$Y_2d,
      Yhat = reshape_to_2d(Yhat, "Yhat"),
      Yhat_unlabeled = reshape_to_2d(Yhat_unlabeled, "Yhat_unlabeled"),
      method = prepared$method,
      w = w,
      w_unlabeled = w_unlabeled,
      X = if (is.null(X)) NULL else reshape_to_2d(X, "X"),
      X_unlabeled = if (is.null(X_unlabeled)) NULL else reshape_to_2d(X_unlabeled, "X_unlabeled"),
      efficiency_maximization = prepared$final_efficiency_maximization,
      candidate_methods = candidate_methods,
      num_folds = resolve_num_folds(num_folds),
      auto_unlabeled_subsample_size = auto_unlabeled_subsample_size,
      selection_random_state = selection_random_state,
      isocal_backend = isocal_backend,
      isocal_max_depth = isocal_max_depth,
      isocal_min_child_weight = isocal_min_child_weight,
      jackknife_folds = jackknife_folds,
      random_state = random_state
    )
    diagnostics$jackknife_folds <- jk$effective_folds
    if (isTRUE(compute_se)) {
      standard_error <- jackknife_standard_error(jk$estimates)
    }
    if (isTRUE(compute_ci)) {
      interval <- z_interval(pointestimate, standard_error, alpha, alternative)
      ci <- list(interval$lower, interval$upper)
    }
  }
  list(
    pointestimate = pointestimate,
    se = standard_error,
    ci = ci,
    method = prepared$method,
    selected_candidate = prepared$selected_candidate,
    selected_efficiency_maximization = prepared$selected_efficiency_maximization,
    final_efficiency_maximization = prepared$final_efficiency_maximization,
    efficiency_lambda = prepared$efficiency_lambda,
    inference = inference,
    diagnostics = diagnostics,
    calibrator = prepared$calibrator
  )
}

#' Mean inference in one call
#'
#' @param Y Observed outcomes on the labeled sample.
#' @param Yhat Predictions on the labeled sample.
#' @param Yhat_unlabeled Predictions on the unlabeled sample.
#' @return A `ppi_mean_result`.
mean_inference <- function(Y, Yhat, Yhat_unlabeled, alpha = 0.1, alternative = "two-sided",
                           method = "monotone_spline", w = NULL, w_unlabeled = NULL,
                           X = NULL, X_unlabeled = NULL, efficiency_maximization = FALSE,
                           candidate_methods = c("aipw", "linear", "monotone_spline", "isotonic"),
                           num_folds = NULL, auto_unlabeled_subsample_size = NULL,
                           selection_random_state = NULL, isocal_backend = "weighted_pava",
                           isocal_max_depth = 20, isocal_min_child_weight = 10,
                           inference = "wald", n_resamples = 1000, jackknife_folds = 20,
                           random_state = NULL) {
  state <- fit_mean_inference(
    Y = Y,
    Yhat = Yhat,
    Yhat_unlabeled = Yhat_unlabeled,
    alpha = alpha,
    alternative = alternative,
    method = method,
    w = w,
    w_unlabeled = w_unlabeled,
    X = X,
    X_unlabeled = X_unlabeled,
    efficiency_maximization = efficiency_maximization,
    candidate_methods = candidate_methods,
    num_folds = num_folds,
    auto_unlabeled_subsample_size = auto_unlabeled_subsample_size,
    selection_random_state = selection_random_state,
    isocal_backend = isocal_backend,
    isocal_max_depth = isocal_max_depth,
    isocal_min_child_weight = isocal_min_child_weight,
    inference = inference,
    n_resamples = n_resamples,
    jackknife_folds = jackknife_folds,
    random_state = random_state,
    compute_se = TRUE,
    compute_ci = TRUE
  )
  new_mean_result(
    pointestimate = restore_shape(state$pointestimate, Y),
    se = restore_shape(state$se, Y),
    ci = list(restore_shape(state$ci[[1]], Y), restore_shape(state$ci[[2]], Y)),
    method = state$method,
    selected_candidate = state$selected_candidate,
    selected_efficiency_maximization = state$selected_efficiency_maximization,
    efficiency_lambda = state$efficiency_lambda,
    inference = state$inference,
    diagnostics = state$diagnostics,
    calibrator = state$calibrator
  )
}

mean_pointestimate <- function(Y, Yhat, Yhat_unlabeled, method = "monotone_spline",
                               w = NULL, w_unlabeled = NULL, X = NULL, X_unlabeled = NULL,
                               efficiency_maximization = FALSE,
                               candidate_methods = c("aipw", "linear", "monotone_spline", "isotonic"),
                               num_folds = NULL, auto_unlabeled_subsample_size = NULL,
                               selection_random_state = NULL, isocal_backend = "weighted_pava",
                               isocal_max_depth = 20, isocal_min_child_weight = 10,
                               return_calibrator = FALSE) {
  prepared <- prepare_inference_inputs(
    Y = Y,
    Yhat = Yhat,
    Yhat_unlabeled = Yhat_unlabeled,
    method = method,
    w = w,
    w_unlabeled = w_unlabeled,
    X = X,
    X_unlabeled = X_unlabeled,
    efficiency_maximization = efficiency_maximization,
    candidate_methods = candidate_methods,
    num_folds = resolve_num_folds(num_folds),
    auto_unlabeled_subsample_size = auto_unlabeled_subsample_size,
    selection_random_state = selection_random_state,
    isocal_backend = isocal_backend,
    isocal_max_depth = isocal_max_depth,
    isocal_min_child_weight = isocal_min_child_weight
  )
  estimate <- aipw_mean_pointestimate_from_predictions(
    Y = prepared$Y_2d,
    pred_labeled = prepared$pred_labeled_point,
    pred_unlabeled = prepared$pred_unlabeled_point,
    w = prepared$weights,
    w_unlabeled = prepared$weights_unlabeled_point
  )
  estimate <- restore_shape(estimate, Y)
  if (isTRUE(return_calibrator)) {
    return(list(pointestimate = estimate, calibrator = prepared$calibrator))
  }
  estimate
}

mean_se <- function(Y, Yhat, Yhat_unlabeled, method = "monotone_spline",
                    w = NULL, w_unlabeled = NULL, X = NULL, X_unlabeled = NULL,
                    efficiency_maximization = FALSE,
                    candidate_methods = c("aipw", "linear", "monotone_spline", "isotonic"),
                    num_folds = NULL, auto_unlabeled_subsample_size = NULL,
                    selection_random_state = NULL, isocal_backend = "weighted_pava",
                    isocal_max_depth = 20, isocal_min_child_weight = 10,
                    inference = "wald", n_resamples = 1000, jackknife_folds = 20,
                    random_state = NULL) {
  state <- fit_mean_inference(
    Y = Y,
    Yhat = Yhat,
    Yhat_unlabeled = Yhat_unlabeled,
    method = method,
    w = w,
    w_unlabeled = w_unlabeled,
    X = X,
    X_unlabeled = X_unlabeled,
    efficiency_maximization = efficiency_maximization,
    candidate_methods = candidate_methods,
    num_folds = num_folds,
    auto_unlabeled_subsample_size = auto_unlabeled_subsample_size,
    selection_random_state = selection_random_state,
    isocal_backend = isocal_backend,
    isocal_max_depth = isocal_max_depth,
    isocal_min_child_weight = isocal_min_child_weight,
    inference = inference,
    n_resamples = n_resamples,
    jackknife_folds = jackknife_folds,
    random_state = random_state,
    compute_se = TRUE,
    compute_ci = FALSE
  )
  restore_shape(state$se, Y)
}

mean_ci <- function(Y, Yhat, Yhat_unlabeled, alpha = 0.1, alternative = "two-sided",
                    method = "monotone_spline", w = NULL, w_unlabeled = NULL,
                    X = NULL, X_unlabeled = NULL, efficiency_maximization = FALSE,
                    candidate_methods = c("aipw", "linear", "monotone_spline", "isotonic"),
                    num_folds = NULL, auto_unlabeled_subsample_size = NULL,
                    selection_random_state = NULL, isocal_backend = "weighted_pava",
                    isocal_max_depth = 20, isocal_min_child_weight = 10,
                    inference = "wald", n_resamples = 1000, jackknife_folds = 20,
                    random_state = NULL) {
  state <- fit_mean_inference(
    Y = Y,
    Yhat = Yhat,
    Yhat_unlabeled = Yhat_unlabeled,
    alpha = alpha,
    alternative = alternative,
    method = method,
    w = w,
    w_unlabeled = w_unlabeled,
    X = X,
    X_unlabeled = X_unlabeled,
    efficiency_maximization = efficiency_maximization,
    candidate_methods = candidate_methods,
    num_folds = num_folds,
    auto_unlabeled_subsample_size = auto_unlabeled_subsample_size,
    selection_random_state = selection_random_state,
    isocal_backend = isocal_backend,
    isocal_max_depth = isocal_max_depth,
    isocal_min_child_weight = isocal_min_child_weight,
    inference = inference,
    n_resamples = n_resamples,
    jackknife_folds = jackknife_folds,
    random_state = random_state,
    compute_se = TRUE,
    compute_ci = TRUE
  )
  list(restore_shape(state$ci[[1]], Y), restore_shape(state$ci[[2]], Y))
}

aipw_mean_inference <- mean_inference
aipw_mean_pointestimate <- mean_pointestimate
aipw_mean_se <- mean_se
aipw_mean_ci <- mean_ci
ppi_aipw_mean_inference <- mean_inference
ppi_aipw_mean_pointestimate <- mean_pointestimate
ppi_aipw_mean_se <- mean_se
ppi_aipw_mean_ci <- mean_ci
pi_aipw_mean_inference <- mean_inference
pi_aipw_mean_pointestimate <- mean_pointestimate
pi_aipw_mean_se <- mean_se
pi_aipw_mean_ci <- mean_ci

linear_calibration_mean_pointestimate <- function(...) mean_pointestimate(..., method = "linear")
linear_calibration_mean_se <- function(...) mean_se(..., method = "linear")
linear_calibration_mean_ci <- function(...) mean_ci(..., method = "linear")
sigmoid_mean_pointestimate <- function(...) mean_pointestimate(..., method = "sigmoid")
sigmoid_mean_se <- function(...) mean_se(..., method = "sigmoid")
sigmoid_mean_ci <- function(...) mean_ci(..., method = "sigmoid")
isotonic_mean_pointestimate <- function(...) mean_pointestimate(..., method = "isotonic")
isotonic_mean_se <- function(...) mean_se(..., method = "isotonic")
isotonic_mean_ci <- function(...) mean_ci(..., method = "isotonic")
