\name{mean_inference}
\alias{mean_inference}
\alias{mean_pointestimate}
\alias{mean_se}
\alias{mean_ci}
\alias{aipw_mean_inference}
\alias{aipw_mean_pointestimate}
\alias{aipw_mean_se}
\alias{aipw_mean_ci}
\alias{ppi_aipw_mean_inference}
\alias{ppi_aipw_mean_pointestimate}
\alias{ppi_aipw_mean_se}
\alias{ppi_aipw_mean_ci}
\alias{pi_aipw_mean_inference}
\alias{pi_aipw_mean_pointestimate}
\alias{pi_aipw_mean_se}
\alias{pi_aipw_mean_ci}
\alias{linear_calibration_mean_pointestimate}
\alias{linear_calibration_mean_se}
\alias{linear_calibration_mean_ci}
\alias{sigmoid_mean_pointestimate}
\alias{sigmoid_mean_se}
\alias{sigmoid_mean_ci}
\alias{isotonic_mean_pointestimate}
\alias{isotonic_mean_se}
\alias{isotonic_mean_ci}
\title{Semi-supervised mean inference}
\description{
Fit the package's semi-supervised mean estimators, compute standard errors and
confidence intervals, or call the convenience aliases that pin a specific
calibration method.
}
\usage{
mean_inference(
  Y, Yhat, Yhat_unlabeled, alpha = 0.1, alternative = "two-sided",
  method = "monotone_spline", w = NULL, w_unlabeled = NULL,
  X = NULL, X_unlabeled = NULL, efficiency_maximization = FALSE,
  candidate_methods = c("aipw", "linear", "monotone_spline", "isotonic"),
  num_folds = NULL, auto_unlabeled_subsample_size = NULL,
  selection_random_state = NULL, isocal_backend = "weighted_pava",
  isocal_max_depth = 20, isocal_min_child_weight = 10,
  inference = "wald", n_resamples = 1000, jackknife_folds = 20,
  random_state = NULL
)

mean_pointestimate(
  Y, Yhat, Yhat_unlabeled, method = "monotone_spline",
  w = NULL, w_unlabeled = NULL, X = NULL, X_unlabeled = NULL,
  efficiency_maximization = FALSE,
  candidate_methods = c("aipw", "linear", "monotone_spline", "isotonic"),
  num_folds = NULL, auto_unlabeled_subsample_size = NULL,
  selection_random_state = NULL, isocal_backend = "weighted_pava",
  isocal_max_depth = 20, isocal_min_child_weight = 10,
  return_calibrator = FALSE
)

mean_se(
  Y, Yhat, Yhat_unlabeled, method = "monotone_spline",
  w = NULL, w_unlabeled = NULL, X = NULL, X_unlabeled = NULL,
  efficiency_maximization = FALSE,
  candidate_methods = c("aipw", "linear", "monotone_spline", "isotonic"),
  num_folds = NULL, auto_unlabeled_subsample_size = NULL,
  selection_random_state = NULL, isocal_backend = "weighted_pava",
  isocal_max_depth = 20, isocal_min_child_weight = 10,
  inference = "wald", n_resamples = 1000, jackknife_folds = 20,
  random_state = NULL
)

mean_ci(
  Y, Yhat, Yhat_unlabeled, alpha = 0.1, alternative = "two-sided",
  method = "monotone_spline", w = NULL, w_unlabeled = NULL,
  X = NULL, X_unlabeled = NULL, efficiency_maximization = FALSE,
  candidate_methods = c("aipw", "linear", "monotone_spline", "isotonic"),
  num_folds = NULL, auto_unlabeled_subsample_size = NULL,
  selection_random_state = NULL, isocal_backend = "weighted_pava",
  isocal_max_depth = 20, isocal_min_child_weight = 10,
  inference = "wald", n_resamples = 1000, jackknife_folds = 20,
  random_state = NULL
)

linear_calibration_mean_pointestimate(...)
linear_calibration_mean_se(...)
linear_calibration_mean_ci(...)
sigmoid_mean_pointestimate(...)
sigmoid_mean_se(...)
sigmoid_mean_ci(...)
isotonic_mean_pointestimate(...)
isotonic_mean_se(...)
isotonic_mean_ci(...)
}
\arguments{
  \item{Y}{Observed outcomes on the labeled sample.}
  \item{Yhat}{Predictions on the same labeled rows.}
  \item{Yhat_unlabeled}{Predictions on the unlabeled rows.}
  \item{alpha}{Confidence level parameter used by \code{mean_inference()} and \code{mean_ci()}.}
  \item{alternative}{One of \code{"two-sided"}, \code{"larger"}, or \code{"smaller"}.}
  \item{method}{One of \code{"aipw"}, \code{"linear"}, \code{"prognostic_linear"}, \code{"sigmoid"}, \code{"isotonic"}, \code{"monotone_spline"}, or \code{"auto"}.}
  \item{w, w_unlabeled}{Optional labeled and unlabeled observation weights.}
  \item{X, X_unlabeled}{Optional extra covariates used by \code{method = "prognostic_linear"}.}
  \item{efficiency_maximization}{Whether to replace the fitted prediction by the estimated efficient rescaling \eqn{\lambda m(X)}.}
  \item{candidate_methods}{Candidate methods considered when \code{method = "auto"}.}
  \item{num_folds}{Number of labeled folds used for automatic method selection.}
  \item{auto_unlabeled_subsample_size}{Optional unlabeled subset size used during \code{method = "auto"} selection.}
  \item{selection_random_state}{Seed used for fold construction and unlabeled subsampling in \code{method = "auto"}.}
  \item{isocal_backend, isocal_max_depth, isocal_min_child_weight}{Controls for the isotonic backend metadata.}
  \item{inference}{One of \code{"wald"}, \code{"jackknife"}, or \code{"bootstrap"}.}
  \item{n_resamples}{Number of bootstrap resamples when \code{inference = "bootstrap"}.}
  \item{jackknife_folds}{Number of delete-a-group folds when \code{inference = "jackknife"}.}
  \item{random_state}{Seed used by bootstrap and jackknife paths.}
  \item{return_calibrator}{If \code{TRUE}, return both the point estimate and fitted model.}
  \item{...}{Arguments forwarded to the generic mean API with the method pinned by the wrapper.}
}
\value{
  \code{mean_inference()} returns a \code{ppi_mean_result}. \code{mean_pointestimate()}
  returns the point estimate, \code{mean_se()} returns a standard error, and
  \code{mean_ci()} returns \code{list(lower, upper)}. The convenience wrappers
  call the corresponding generic function with a fixed method.
}
\details{
  The default method is \code{"monotone_spline"}. When \code{method = "auto"},
  the package cross-validates the candidate methods on the labeled sample and
  refits the selected method on the full data before returning the final
  estimate.
}
