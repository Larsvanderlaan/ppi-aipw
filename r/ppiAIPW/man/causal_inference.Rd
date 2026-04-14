\name{causal_inference}
\alias{causal_inference}
\title{Armwise causal wrapper over the mean engine}
\description{
Estimate arm-specific potential outcome means and treatment-versus-control
average treatment effects using the semi-supervised mean engine arm by arm.
}
\usage{
causal_inference(
  Y, A, Yhat_potential, treatment_levels = NULL, control_arm = NULL,
  method = "monotone_spline", w = NULL, X = NULL, alpha = 0.1,
  alternative = "two-sided", efficiency_maximization = FALSE,
  candidate_methods = c("aipw", "linear", "monotone_spline", "isotonic"),
  num_folds = 100, auto_unlabeled_subsample_size = NULL,
  selection_random_state = NULL, isocal_backend = "weighted_pava",
  isocal_max_depth = 20, isocal_min_child_weight = 10,
  inference = "wald"
)
}
\arguments{
  \item{Y}{Observed outcomes for all units.}
  \item{A}{Observed treatment assignments.}
  \item{Yhat_potential}{Matrix of predicted potential outcomes with one column per treatment arm.}
  \item{treatment_levels}{Optional explicit ordering of treatment arms.}
  \item{control_arm}{Optional reference arm used for ATE contrasts.}
  \item{method}{Mean-estimation method used within each arm.}
  \item{w}{Optional full-sample observation weights.}
  \item{X}{Optional full-sample covariates for \code{method = "prognostic_linear"}.}
  \item{alpha, alternative}{Confidence interval parameters passed to each armwise fit.}
  \item{efficiency_maximization}{Whether to use efficient rescaling within each arm.}
  \item{candidate_methods, num_folds, auto_unlabeled_subsample_size, selection_random_state}{Automatic-selection controls passed arm by arm.}
  \item{isocal_backend, isocal_max_depth, isocal_min_child_weight}{Isotonic backend controls.}
  \item{inference}{Currently only \code{"wald"} is supported.}
}
\value{
  A \code{ppi_causal_result} containing arm-specific means, standard errors,
  confidence intervals, ATEs relative to the control arm, and per-arm fitted
  mean results.
}
