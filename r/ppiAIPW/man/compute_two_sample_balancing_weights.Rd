\name{compute_two_sample_balancing_weights}
\alias{compute_two_sample_balancing_weights}
\title{Compute two-sample balancing weights}
\description{
Construct nonnegative labeled-sample weights that balance supplied features
between labeled rows and a pooled or unlabeled target sample.
}
\usage{
compute_two_sample_balancing_weights(
  X_labeled, X_unlabeled, target = "pooled",
  include_intercept = TRUE, tolerance = 1e-8, maxiter = 1000,
  return_diagnostics = FALSE
)
}
\arguments{
  \item{X_labeled}{Feature matrix for the labeled sample.}
  \item{X_unlabeled}{Feature matrix for the unlabeled sample.}
  \item{target}{One of \code{"pooled"} or \code{"unlabeled"}.}
  \item{include_intercept}{Whether to also balance the intercept, so the returned weights average to one.}
  \item{tolerance}{Maximum allowed balance error.}
  \item{maxiter}{Maximum optimizer iterations.}
  \item{return_diagnostics}{If \code{TRUE}, also return a diagnostics list.}
}
\value{
  A weight vector, or a list with entries \code{weights} and \code{diagnostics}
  when \code{return_diagnostics = TRUE}.
}
