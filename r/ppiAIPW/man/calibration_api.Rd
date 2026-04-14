\name{fit_calibrator}
\alias{fit_calibrator}
\alias{calibrate_predictions}
\alias{calibration_diagnostics}
\alias{plot_calibration}
\title{Calibration helpers and diagnostics}
\description{
Fit reusable calibration models, apply them to labeled and unlabeled scores,
and inspect honest out-of-fold calibration diagnostics by default.
}
\usage{
fit_calibrator(
  Y, Yhat, method = "monotone_spline", w = NULL,
  isocal_backend = "weighted_pava", isocal_max_depth = 20,
  isocal_min_child_weight = 10
)

calibrate_predictions(
  Y, Yhat, Yhat_unlabeled = NULL, method = "monotone_spline",
  w = NULL, isocal_backend = "weighted_pava",
  isocal_max_depth = 20, isocal_min_child_weight = 10,
  return_model = FALSE
)

calibration_diagnostics(
  obj, Y, Yhat, X = NULL, w = NULL,
  diagnostic_mode = "out_of_fold", num_folds = 10, num_bins = 10
)

plot_calibration(diagnostics, output_index = 1L, show_identity = TRUE, show_bins = TRUE, ...)
}
\arguments{
  \item{Y}{Observed outcomes on the labeled sample.}
  \item{Yhat}{Predictions on the labeled sample.}
  \item{Yhat_unlabeled}{Optional predictions on the unlabeled sample.}
  \item{method}{Calibration method.}
  \item{w}{Optional labeled-sample weights.}
  \item{isocal_backend, isocal_max_depth, isocal_min_child_weight}{Options recorded on isotonic models and diagnostics metadata.}
  \item{return_model}{If \code{TRUE}, also return the fitted calibration model.}
  \item{obj}{A \code{ppi_mean_result}, \code{ppi_calibration_model}, or \code{ppi_prognostic_linear_model}.}
  \item{X}{Optional labeled-sample covariates for prognostic-linear diagnostics.}
  \item{diagnostic_mode}{Either \code{"out_of_fold"} for an honest held-out diagnostic
  view or \code{"in_sample"} for a descriptive fit-on-fit view.}
  \item{num_folds}{Number of folds used when \code{diagnostic_mode = "out_of_fold"}.}
  \item{num_bins}{Number of bins used by \code{calibration_diagnostics()}.}
  \item{diagnostics}{A \code{ppi_calibration_diagnostics} object.}
  \item{output_index}{Output coordinate to plot for multi-output models.}
  \item{show_identity, show_bins}{Whether to add the identity line and bin summaries to the plot.}
  \item{...}{Passed through to the underlying plotting method.}
}
\value{
  \code{fit_calibrator()} returns a reusable fitted model. \code{calibrate_predictions()}
  returns calibrated labeled and unlabeled predictions, optionally with the
  model. \code{calibration_diagnostics()} returns a
  \code{ppi_calibration_diagnostics} object with out-of-fold labeled diagnostics
  by default, and \code{plot_calibration()} draws a base R diagnostic plot.
}
