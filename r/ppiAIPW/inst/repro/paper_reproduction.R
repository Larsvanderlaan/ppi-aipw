run_paper_style_comparison <- function(Y, Yhat, Yhat_unlabeled,
                                       methods = c("aipw", "linear", "monotone_spline", "isotonic", "auto"),
                                       alpha = 0.1) {
  out <- lapply(methods, function(method) {
    fit <- if (identical(method, "auto")) {
      mean_inference(
        Y,
        Yhat,
        Yhat_unlabeled,
        method = "auto",
        candidate_methods = c("aipw", "linear", "monotone_spline", "isotonic"),
        num_folds = 20,
        selection_random_state = 0,
        alpha = alpha
      )
    } else {
      mean_inference(Y, Yhat, Yhat_unlabeled, method = method, alpha = alpha)
    }
    data.frame(
      method = method,
      estimate = fit$pointestimate,
      se = fit$se,
      ci_lower = fit$ci[[1]],
      ci_upper = fit$ci[[2]],
      selected_candidate = if (!is.null(fit$selected_candidate)) fit$selected_candidate else method
    )
  })
  do.call(rbind, out)
}
