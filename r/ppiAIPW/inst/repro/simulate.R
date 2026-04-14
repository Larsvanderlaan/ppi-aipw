run_synthetic_study <- function(n_labeled = 100, n_unlabeled = 500, n_rep = 200,
                                methods = c("aipw", "linear", "monotone_spline", "isotonic"),
                                seed = 20260413) {
  old_seed <- .Random.seed
  on.exit({
    if (!is.null(old_seed)) {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    }
  }, add = TRUE)
  set.seed(seed)
  results <- vector("list", n_rep * length(methods))
  row_id <- 1L
  for (rep_id in seq_len(n_rep)) {
    x_l <- rnorm(n_labeled)
    x_u <- rnorm(n_unlabeled)
    mu_l <- 0.5 + x_l
    mu_u <- 0.5 + x_u
    y_l <- mu_l + rnorm(n_labeled, sd = 1)
    yhat_l <- mu_l + rnorm(n_labeled, sd = 0.6)
    yhat_u <- mu_u + rnorm(n_unlabeled, sd = 0.6)
    true_mean <- mean(c(mu_l, mu_u))
    for (method in methods) {
      fit <- mean_inference(y_l, yhat_l, yhat_u, method = method)
      results[[row_id]] <- data.frame(
        replication = rep_id,
        method = method,
        truth = true_mean,
        estimate = fit$pointestimate,
        se = fit$se,
        ci_lower = fit$ci[[1]],
        ci_upper = fit$ci[[2]],
        covered = fit$ci[[1]] <= true_mean && true_mean <= fit$ci[[2]]
      )
      row_id <- row_id + 1L
    }
  }
  do.call(rbind, results)
}

summarize_synthetic_study <- function(results) {
  split_results <- split(results, results$method)
  do.call(rbind, lapply(names(split_results), function(method) {
    dat <- split_results[[method]]
    err <- dat$estimate - dat$truth
    data.frame(
      method = method,
      bias = mean(err),
      mse = mean(err^2),
      coverage = mean(dat$covered),
      mean_se = mean(dat$se)
    )
  }))
}
