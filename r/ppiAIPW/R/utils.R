method_aliases <- c(
  aipw = "aipw",
  identity = "aipw",
  none = "aipw",
  linear = "linear",
  linear_calibration = "linear",
  prognostic_linear = "prognostic_linear",
  linear_adjustment = "prognostic_linear",
  prognostic = "prognostic_linear",
  sigmoid = "sigmoid",
  platt = "sigmoid",
  platt_scaling = "sigmoid",
  isotonic = "isotonic",
  isocal = "isotonic",
  isotonic_calibration = "isotonic",
  monotone_spline = "monotone_spline",
  isotonic_spline = "monotone_spline",
  smooth_spline = "monotone_spline",
  mspline = "monotone_spline"
)

.auto_aipw_efficiency_label <- "aipw_efficiency_maximization"
.prognostic_linear_ridge_grid <- c(1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0)
.monotone_spline_max_internal_knots <- 6L
.monotone_spline_derivative_degree <- 2L
.monotone_spline_penalty <- 1e-3

canonical_method <- function(method) {
  key <- tolower(as.character(method[[1]]))
  if (!key %in% names(method_aliases)) {
    stop(
      sprintf(
        "Unknown calibration method '%s'. Expected one of: %s.",
        key,
        paste(sort(names(method_aliases)), collapse = ", ")
      ),
      call. = FALSE
    )
  }
  unname(method_aliases[[key]])
}

normalize_alternative <- function(alternative) {
  alt <- tolower(as.character(alternative[[1]]))
  if (alt %in% c("two.sided", "two-sided")) {
    return("two-sided")
  }
  if (alt %in% c("larger", "greater")) {
    return("larger")
  }
  if (alt %in% c("smaller", "less")) {
    return("smaller")
  }
  stop("alternative must be 'two-sided', 'larger', or 'smaller'.", call. = FALSE)
}

reshape_to_2d <- function(x, name = deparse(substitute(x))) {
  if (is.data.frame(x)) {
    x <- as.matrix(x)
  }
  if (is.null(dim(x))) {
    x <- as.numeric(x)
    if (length(x) == 0L) {
      stop(sprintf("%s must be nonempty.", name), call. = FALSE)
    }
    return(matrix(x, ncol = 1L))
  }
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  if (nrow(x) == 0L) {
    stop(sprintf("%s must be nonempty.", name), call. = FALSE)
  }
  x
}

restore_shape <- function(x, reference) {
  x <- as.matrix(x)
  if (is.null(dim(reference))) {
    flat <- as.numeric(x)
    if (length(flat) == 1L) {
      return(flat[[1]])
    }
    return(flat)
  }
  if (is.data.frame(reference)) {
    reference <- as.matrix(reference)
  }
  reference <- as.matrix(reference)
  if (ncol(reference) == 1L) {
    flat <- as.numeric(x)
    if (length(flat) == 1L) {
      return(flat[[1]])
    }
    return(flat)
  }
  x
}

construct_weight_vector <- function(n_obs, existing_weight = NULL, vectorized = FALSE) {
  if (is.null(existing_weight)) {
    weights <- rep(1, n_obs)
  } else {
    weights <- as.numeric(existing_weight)
    if (length(weights) != n_obs) {
      stop(
        sprintf("Expected weights with length %d, got %d.", n_obs, length(weights)),
        call. = FALSE
      )
    }
    if (any(weights < 0)) {
      stop("Weights must be nonnegative.", call. = FALSE)
    }
    if (!any(weights > 0)) {
      stop("At least one weight must be strictly positive.", call. = FALSE)
    }
    weights <- weights / sum(weights) * n_obs
  }
  if (vectorized) {
    weights <- matrix(weights, ncol = 1L)
  }
  weights
}

validate_pair_inputs <- function(Y, Yhat) {
  Y_2d <- reshape_to_2d(Y, "Y")
  Yhat_2d <- reshape_to_2d(Yhat, "Yhat")
  if (!identical(dim(Y_2d), dim(Yhat_2d))) {
    stop(
      sprintf(
        "Y and Yhat must have the same shape, got (%d, %d) and (%d, %d).",
        nrow(Y_2d), ncol(Y_2d), nrow(Yhat_2d), ncol(Yhat_2d)
      ),
      call. = FALSE
    )
  }
  list(Y_2d = Y_2d, Yhat_2d = Yhat_2d)
}

validate_mean_inputs <- function(Y, Yhat, Yhat_unlabeled) {
  pair <- validate_pair_inputs(Y, Yhat)
  Yhat_unlabeled_2d <- reshape_to_2d(Yhat_unlabeled, "Yhat_unlabeled")
  if (ncol(Yhat_unlabeled_2d) != ncol(pair$Y_2d)) {
    stop(
      sprintf(
        "Yhat_unlabeled must have %d column(s), got %d.",
        ncol(pair$Y_2d), ncol(Yhat_unlabeled_2d)
      ),
      call. = FALSE
    )
  }
  list(Y_2d = pair$Y_2d, Yhat_2d = pair$Yhat_2d, Yhat_unlabeled_2d = Yhat_unlabeled_2d)
}

clip_unit <- function(x, eps = 1e-6) {
  pmin(pmax(as.numeric(x), eps), 1 - eps)
}

clip_range <- function(x, lower, upper) {
  pmin(pmax(as.numeric(x), lower), upper)
}

safe_logit <- function(p) {
  p <- clip_unit(p)
  log(p / (1 - p))
}

sigmoid <- function(z) {
  1 / (1 + exp(-pmax(pmin(as.numeric(z), 40), -40)))
}

weighted_mean <- function(x, w) {
  colMeans(w * x)
}

weighted_var <- function(x, w) {
  centered <- sweep(x, 2L, weighted_mean(x, w), "-", check.margin = FALSE)
  colMeans(w * centered^2)
}

weighted_cov <- function(x, y, w) {
  x_centered <- sweep(x, 2L, weighted_mean(x, w), "-", check.margin = FALSE)
  y_centered <- sweep(y, 2L, weighted_mean(y, w), "-", check.margin = FALSE)
  colMeans(w * x_centered * y_centered)
}

z_interval <- function(pointestimate, standard_error, alpha, alternative = "two-sided") {
  alternative <- normalize_alternative(alternative)
  if (!(is.numeric(alpha) && length(alpha) == 1L && is.finite(alpha) && alpha > 0 && alpha < 1)) {
    stop("alpha must lie in (0, 1).", call. = FALSE)
  }
  pointestimate <- as.numeric(pointestimate)
  standard_error <- as.numeric(standard_error)
  if (alternative == "two-sided") {
    z_value <- stats::qnorm(1 - alpha / 2)
    lower <- pointestimate - z_value * standard_error
    upper <- pointestimate + z_value * standard_error
  } else if (alternative == "larger") {
    z_value <- stats::qnorm(1 - alpha)
    lower <- pointestimate - z_value * standard_error
    upper <- rep(Inf, length(lower))
  } else {
    z_value <- stats::qnorm(1 - alpha)
    lower <- rep(-Inf, length(pointestimate))
    upper <- pointestimate + z_value * standard_error
  }
  list(lower = lower, upper = upper)
}

flatten_parameter <- function(x) {
  as.numeric(x)
}

coerce_null_array <- function(null, size) {
  null_arr <- as.numeric(null)
  if (length(null_arr) == 1L) {
    return(rep(null_arr[[1]], size))
  }
  if (length(null_arr) != size) {
    stop(sprintf("Expected null to have %d value(s), got %d.", size, length(null_arr)), call. = FALSE)
  }
  null_arr
}

format_coordinate_output <- function(x) {
  x <- flatten_parameter(x)
  if (length(x) == 1L) x[[1]] else x
}

compute_wald_statistics <- function(pointestimate, standard_error, null = 0, alternative = "two-sided") {
  alternative <- normalize_alternative(alternative)
  estimate_arr <- flatten_parameter(pointestimate)
  se_arr <- flatten_parameter(standard_error)
  if (length(estimate_arr) != length(se_arr)) {
    stop("pointestimate and standard_error must have the same number of coordinates.", call. = FALSE)
  }
  null_arr <- coerce_null_array(null, length(estimate_arr))
  valid <- is.finite(estimate_arr) & is.finite(se_arr) & (se_arr > 0)
  t_stat <- rep(NaN, length(estimate_arr))
  t_stat[valid] <- (estimate_arr[valid] - null_arr[valid]) / se_arr[valid]
  p_value <- rep(NaN, length(estimate_arr))
  if (alternative == "two-sided") {
    p_value[valid] <- 2 * stats::pnorm(abs(t_stat[valid]), lower.tail = FALSE)
  } else if (alternative == "larger") {
    p_value[valid] <- stats::pnorm(t_stat[valid], lower.tail = FALSE)
  } else {
    p_value[valid] <- stats::pnorm(t_stat[valid], lower.tail = TRUE)
  }
  list(null = null_arr, t_stat = t_stat, p_value = p_value)
}

preview_value <- function(x, digits = 4L) {
  arr <- flatten_parameter(x)
  fmt <- function(v) if (!is.finite(v)) "NA" else formatC(v, digits = digits, format = "fg")
  if (length(arr) == 1L) {
    return(fmt(arr[[1]]))
  }
  preview <- paste(vapply(utils::head(arr, 3L), fmt, character(1L)), collapse = ", ")
  if (length(arr) > 3L) preview <- paste0(preview, ", ...")
  paste0("[", preview, "]")
}

preview_ci <- function(ci) {
  lower <- flatten_parameter(ci[[1]])
  upper <- flatten_parameter(ci[[2]])
  if (length(lower) == 1L && length(upper) == 1L) {
    return(sprintf("(%s, %s)", preview_value(lower), preview_value(upper)))
  }
  sprintf("(%s, %s)", preview_value(lower), preview_value(upper))
}

summary_value <- function(x) {
  if (!is.finite(x)) {
    return("NA")
  }
  formatC(x, digits = 6, format = "fg")
}

labeled_fraction <- function(n_labeled, n_unlabeled) {
  n_labeled / (n_labeled + n_unlabeled)
}

coerce_covariates <- function(X, n_obs, name) {
  if (is.null(X)) {
    return(NULL)
  }
  X_2d <- reshape_to_2d(X, name)
  if (nrow(X_2d) != n_obs) {
    stop(sprintf("%s must have %d rows, got %d.", name, n_obs, nrow(X_2d)), call. = FALSE)
  }
  X_2d
}

validate_prognostic_covariates <- function(X, X_unlabeled, n_labeled, n_unlabeled) {
  X_2d <- coerce_covariates(X, n_labeled, "X")
  X_unlabeled_2d <- coerce_covariates(X_unlabeled, n_unlabeled, "X_unlabeled")
  if (is.null(X_2d) != is.null(X_unlabeled_2d)) {
    stop("Provide both X and X_unlabeled together, or neither.", call. = FALSE)
  }
  if (!is.null(X_2d) && ncol(X_2d) != ncol(X_unlabeled_2d)) {
    stop(
      sprintf("X and X_unlabeled must have the same number of columns. Got %d and %d.", ncol(X_2d), ncol(X_unlabeled_2d)),
      call. = FALSE
    )
  }
  list(X_2d = X_2d, X_unlabeled_2d = X_unlabeled_2d)
}

resolve_efficiency_maximization <- function(efficiency_maximization) {
  isTRUE(efficiency_maximization)
}

resolve_num_folds <- function(num_folds) {
  if (is.null(num_folds)) 100L else as.integer(num_folds[[1]])
}

resolve_auto_unlabeled_subsample_size <- function(n_labeled, n_unlabeled, auto_unlabeled_subsample_size = NULL) {
  if (is.null(auto_unlabeled_subsample_size)) {
    requested_size <- 10L * n_labeled
  } else {
    requested_size <- as.integer(auto_unlabeled_subsample_size[[1]])
    if (requested_size < 1L) {
      stop("auto_unlabeled_subsample_size must be at least 1 when provided.", call. = FALSE)
    }
  }
  min(n_unlabeled, requested_size)
}

resolve_selection_seeds <- function(selection_random_state = NULL) {
  if (is.null(selection_random_state)) {
    seed <- as.integer(sample.int(.Machine$integer.max, 1L))
  } else {
    seed <- as.integer(selection_random_state[[1]])
  }
  list(subset_seed = seed, cv_seed = seed)
}

subset_unlabeled_for_auto <- function(Yhat_unlabeled_2d, w_unlabeled = NULL, X_unlabeled = NULL,
                                      n_labeled, auto_unlabeled_subsample_size = NULL,
                                      subset_seed) {
  n_unlabeled <- nrow(Yhat_unlabeled_2d)
  subset_size <- resolve_auto_unlabeled_subsample_size(
    n_labeled = n_labeled,
    n_unlabeled = n_unlabeled,
    auto_unlabeled_subsample_size = auto_unlabeled_subsample_size
  )
  if (subset_size >= n_unlabeled) {
    return(list(
      Yhat_unlabeled = Yhat_unlabeled_2d,
      w_unlabeled = w_unlabeled,
      X_unlabeled = X_unlabeled,
      diagnostics = list(
        auto_unlabeled_subsample_size = n_unlabeled,
        auto_unlabeled_subsample_default = is.null(auto_unlabeled_subsample_size),
        unlabeled_strategy = "all_unlabeled_rows_in_each_fold"
      )
    ))
  }
  old_seed <- if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) get(".Random.seed", envir = .GlobalEnv) else NULL
  on.exit({
    if (!is.null(old_seed)) {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    }
  }, add = TRUE)
  set.seed(as.integer(subset_seed))
  subset_idx <- sort(sample.int(n_unlabeled, subset_size, replace = FALSE))
  list(
    Yhat_unlabeled = Yhat_unlabeled_2d[subset_idx, , drop = FALSE],
    w_unlabeled = if (is.null(w_unlabeled)) NULL else w_unlabeled[subset_idx],
    X_unlabeled = if (is.null(X_unlabeled)) NULL else X_unlabeled[subset_idx, , drop = FALSE],
    diagnostics = list(
      auto_unlabeled_subsample_size = subset_size,
      auto_unlabeled_subsample_default = is.null(auto_unlabeled_subsample_size),
      auto_unlabeled_subsample_seed = as.integer(subset_seed),
      unlabeled_strategy = "subsampled_unlabeled_rows_in_each_fold"
    )
  )
}

kfold_splits <- function(n, n_splits, seed = 0L) {
  n_splits <- min(as.integer(n_splits), n)
  if (n_splits < 2L) {
    stop("Need at least 2 folds.", call. = FALSE)
  }
  old_seed <- if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) get(".Random.seed", envir = .GlobalEnv) else NULL
  on.exit({
    if (!is.null(old_seed)) {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    }
  }, add = TRUE)
  set.seed(as.integer(seed))
  idx <- sample.int(n)
  fold_id <- rep(seq_len(n_splits), length.out = n)
  splits <- vector("list", n_splits)
  for (k in seq_len(n_splits)) {
    val_idx <- sort(idx[fold_id == k])
    train_idx <- setdiff(seq_len(n), val_idx)
    splits[[k]] <- list(train = train_idx, val = val_idx)
  }
  splits
}
