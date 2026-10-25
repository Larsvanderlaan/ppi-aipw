"""Reference implementations for calibrated semisupervised means."""

import numpy as np
import xgboost as xgb

def isoreg_with_xgboost(x, y, max_depth=15, min_child_weight=20, weights=None):
    """
    Fit a monotone calibrator f so that f(x) is nondecreasing in x.

    Parameters
    ----------
    x : array-like, shape (n,) or (n, 1)
        Predictor used for calibration. In the PPI application this is usually
        the score m(X), so x is typically one-dimensional.
    y : array-like, shape (n,)
        Labeled outcomes.
    max_depth : int, default=15
        Maximum depth for the one-round XGBoost fit.
    min_child_weight : float, default=20
        Minimum total weight in a terminal node.
    weights : array-like, optional
        Optional observation weights.

    Returns
    -------
    predict_fn : callable
        Function mapping new x values to calibrated predictions f(x).
    """
    x = np.asarray(x)
    y = np.asarray(y).reshape(-1)
    x = x.reshape(len(y), -1)

    data = xgb.DMatrix(data=x, label=y, weight=weights)
    params = {
        "max_depth": max_depth,
        "min_child_weight": min_child_weight,
        "monotone_constraints": "(" + ",".join(["1"] * x.shape[1]) + ")",
        "eta": 1.0,
        "gamma": 0.0,
        "lambda": 0.0,
        "objective": "reg:squarederror",
        "verbosity": 0,
    }
    iso_fit = xgb.train(params=params, dtrain=data, num_boost_round=1)

    def predict_fn(x_new):
        x_new = np.asarray(x_new)
        if x_new.ndim == 1:
            x_new = x_new.reshape(-1, 1)
        data_pred = xgb.DMatrix(data=x_new)
        return iso_fit.predict(data_pred)

    return predict_fn


import numpy as np
from statistics import NormalDist


def _resolve_scores(
    Y_labeled,
    X_labeled=None,
    X_unlabeled=None,
    score_fn=None,
    score_labeled=None,
    score_unlabeled=None,
):
    Y_labeled = np.asarray(Y_labeled).reshape(-1)

    if score_fn is not None:
        if X_labeled is None or X_unlabeled is None:
            raise ValueError(
                "If score_fn is provided, then X_labeled and X_unlabeled "
                "must also be provided."
            )
        score_labeled = score_fn(X_labeled)
        score_unlabeled = score_fn(X_unlabeled)
    elif score_labeled is None or score_unlabeled is None:
        raise ValueError(
            "Provide either score_fn together with X_labeled/X_unlabeled, "
            "or provide score_labeled and score_unlabeled."
        )

    score_labeled = np.asarray(score_labeled).reshape(-1)
    score_unlabeled = np.asarray(score_unlabeled).reshape(-1)

    if len(score_labeled) != len(Y_labeled):
        raise ValueError("Y_labeled and score_labeled must have the same length.")

    return Y_labeled, score_labeled, score_unlabeled


def isotonic_calibrated_plugin(
    Y_labeled,
    X_labeled=None,
    X_unlabeled=None,
    score_fn=None,
    score_labeled=None,
    score_unlabeled=None,
    alpha=0.05,
    min_obs=20,
):
    Y_labeled, score_labeled, score_unlabeled = _resolve_scores(
        Y_labeled=Y_labeled,
        X_labeled=X_labeled,
        X_unlabeled=X_unlabeled,
        score_fn=score_fn,
        score_labeled=score_labeled,
        score_unlabeled=score_unlabeled,
    )

    n = len(Y_labeled)
    N = len(score_unlabeled)
    M = n + N
    rho = n / M

    calibrator = isoreg_with_xgboost(
        x=score_labeled,
        y=Y_labeled,
        min_child_weight=min_obs,
    )

    m_labeled = calibrator(score_labeled)
    m_unlabeled = calibrator(score_unlabeled)

    psi_hat = (n * m_labeled.mean() + N * m_unlabeled.mean()) / M

    if_labeled = m_labeled - psi_hat + (Y_labeled - m_labeled) / rho
    if_unlabeled = m_unlabeled - psi_hat
    var_hat = (
        rho * np.var(if_labeled, ddof=1)
        + (1 - rho) * np.var(if_unlabeled, ddof=1)
    )
    se_hat = np.sqrt(var_hat / M)

    z = NormalDist().inv_cdf(1 - alpha / 2)
    ci = (psi_hat - z * se_hat, psi_hat + z * se_hat)

    return {
        "estimate": psi_hat,
        "standard_error": se_hat,
        "confidence_interval": ci,
    }


# Example:
# fit = isotonic_calibrated_plugin(
#     Y_labeled=Y,
#     X_labeled=X,
#     X_unlabeled=X_tilde,
#     score_fn=m,
# )


def linear_calibrated_plugin(
    Y_labeled,
    X_labeled=None,
    X_unlabeled=None,
    score_fn=None,
    score_labeled=None,
    score_unlabeled=None,
    alpha=0.05,
):
    Y_labeled, score_labeled, score_unlabeled = _resolve_scores(
        Y_labeled=Y_labeled,
        X_labeled=X_labeled,
        X_unlabeled=X_unlabeled,
        score_fn=score_fn,
        score_labeled=score_labeled,
        score_unlabeled=score_unlabeled,
    )

    n = len(Y_labeled)
    N = len(score_unlabeled)
    M = n + N
    rho = n / M

    X_design = np.column_stack([np.ones(n), score_labeled])
    beta_hat, _, _, _ = np.linalg.lstsq(X_design, Y_labeled, rcond=None)

    def calibrator(score):
        score = np.asarray(score).reshape(-1)
        return beta_hat[0] + beta_hat[1] * score

    m_labeled = calibrator(score_labeled)
    m_unlabeled = calibrator(score_unlabeled)

    psi_hat = (n * m_labeled.mean() + N * m_unlabeled.mean()) / M

    if_labeled = m_labeled - psi_hat + (Y_labeled - m_labeled) / rho
    if_unlabeled = m_unlabeled - psi_hat
    var_hat = (
        rho * np.var(if_labeled, ddof=1)
        + (1 - rho) * np.var(if_unlabeled, ddof=1)
    )
    se_hat = np.sqrt(var_hat / M)

    z = NormalDist().inv_cdf(1 - alpha / 2)
    ci = (psi_hat - z * se_hat, psi_hat + z * se_hat)

    return {
        "estimate": psi_hat,
        "standard_error": se_hat,
        "confidence_interval": ci,
    }


# Example:
# fit = linear_calibrated_plugin(
#     Y_labeled=Y,
#     score_labeled=mX,
#     score_unlabeled=mX_tilde,
# )
