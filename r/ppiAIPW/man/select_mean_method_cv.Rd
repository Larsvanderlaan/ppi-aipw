\name{select_mean_method_cv}
\alias{select_mean_method_cv}
\title{Select a mean-inference method by cross-validation}
\description{
Cross-validate the candidate methods used by \code{method = "auto"} and return
the selected method together with the recorded diagnostics.
}
\usage{
select_mean_method_cv(
  Y, Yhat, Yhat_unlabeled,
  candidate_methods = c("aipw", "linear", "monotone_spline", "isotonic"),
  w = NULL, w_unlabeled = NULL, X = NULL, X_unlabeled = NULL,
  efficiency_maximization = FALSE, num_folds = NULL,
  auto_unlabeled_subsample_size = NULL,
  selection_random_state = NULL,
  isocal_backend = "weighted_pava",
  isocal_max_depth = 20,
  isocal_min_child_weight = 10
)
}
\arguments{
  \item{Y, Yhat, Yhat_unlabeled}{Labeled outcomes, labeled predictions, and unlabeled predictions.}
  \item{candidate_methods}{Candidate methods considered during selection.}
  \item{w, w_unlabeled}{Optional labeled and unlabeled observation weights.}
  \item{X, X_unlabeled}{Optional covariates for prognostic-linear candidates.}
  \item{efficiency_maximization}{Whether to allow efficiency-maximized candidates.}
  \item{num_folds}{Number of labeled folds used for selection.}
  \item{auto_unlabeled_subsample_size}{Optional unlabeled subset size used during selection.}
  \item{selection_random_state}{Seed used for fold construction and unlabeled subsampling.}
  \item{isocal_backend, isocal_max_depth, isocal_min_child_weight}{Isotonic backend controls.}
}
\value{
  A list with entries \code{method} and \code{diagnostics}.
}
