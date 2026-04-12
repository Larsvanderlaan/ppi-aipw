from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize

from ._utils import reshape_to_2d


def compute_two_sample_balancing_weights(
    X_labeled: np.ndarray,
    X_unlabeled: np.ndarray,
    *,
    target: str = "pooled",
    include_intercept: bool = True,
    tolerance: float = 1e-8,
    maxiter: int = 1000,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Compute nonnegative labeled-sample balancing weights for a two-sample design.

    The function finds labeled-sample weights that are as close as possible to
    uniform weights while exactly matching empirical moments of the supplied
    balance features between the labeled sample and a target sample.

    Parameters
    ----------
    X_labeled
        Feature matrix for the labeled sample. This can be raw covariates, a
        score, or any low-dimensional balance representation.
    X_unlabeled
        Feature matrix for the unlabeled sample, with the same number of
        columns as ``X_labeled``.
    target
        One of ``"pooled"`` or ``"unlabeled"``. ``"pooled"`` balances the
        weighted labeled moments to the pooled empirical target from the paper,
        namely ``rho * mean_labeled + (1-rho) * mean_unlabeled``.
        ``"unlabeled"`` balances directly to the unlabeled empirical mean.
    include_intercept
        If ``True``, also balances the intercept, so the returned weights have
        empirical mean 1 on the labeled sample.
    tolerance
        Maximum allowed absolute balance error in the solved moments.
    maxiter
        Maximum number of optimizer iterations.
    return_diagnostics
        If ``True``, also returns a diagnostics dictionary.

    Returns
    -------
    weights
        A one-dimensional nonnegative weight vector for the labeled sample.

    Notes
    -----
    The returned weights are designed for the semisupervised two-sample mean
    problem and are intentionally mean-side only. They can be supplied as
    ``w=...`` to the package's mean APIs when you want a weighted labeled
    sample fit. The utility does not construct a causal-arm-specific weighting
    scheme.
    """

    X_labeled_2d = reshape_to_2d(np.asarray(X_labeled, dtype=float))
    X_unlabeled_2d = reshape_to_2d(np.asarray(X_unlabeled, dtype=float))
    if X_labeled_2d.shape[1] != X_unlabeled_2d.shape[1]:
        raise ValueError(
            "X_labeled and X_unlabeled must have the same number of columns. "
            f"Got {X_labeled_2d.shape[1]} and {X_unlabeled_2d.shape[1]}."
        )
    if X_labeled_2d.shape[0] == 0 or X_unlabeled_2d.shape[0] == 0:
        raise ValueError("Both labeled and unlabeled samples must be nonempty.")

    resolved_target = target.lower()
    if resolved_target not in {"pooled", "unlabeled"}:
        raise ValueError("target must be either 'pooled' or 'unlabeled'.")
    if tolerance <= 0:
        raise ValueError("tolerance must be strictly positive.")
    if maxiter < 1:
        raise ValueError("maxiter must be at least 1.")

    if include_intercept:
        Z_labeled = np.column_stack([np.ones(X_labeled_2d.shape[0], dtype=float), X_labeled_2d])
        Z_unlabeled = np.column_stack([np.ones(X_unlabeled_2d.shape[0], dtype=float), X_unlabeled_2d])
    else:
        Z_labeled = X_labeled_2d
        Z_unlabeled = X_unlabeled_2d

    n_labeled = Z_labeled.shape[0]
    n_unlabeled = Z_unlabeled.shape[0]
    rho = n_labeled / float(n_labeled + n_unlabeled)
    labeled_mean = np.mean(Z_labeled, axis=0)
    unlabeled_mean = np.mean(Z_unlabeled, axis=0)
    if resolved_target == "pooled":
        target_mean = rho * labeled_mean + (1.0 - rho) * unlabeled_mean
    else:
        target_mean = unlabeled_mean

    constraint_matrix = Z_labeled.T
    constraint_rhs = n_labeled * target_mean

    def objective(weights: np.ndarray) -> float:
        return 0.5 * float(np.sum((weights - 1.0) ** 2))

    def gradient(weights: np.ndarray) -> np.ndarray:
        return weights - 1.0

    constraints = [
        {
            "type": "eq",
            "fun": lambda weights: constraint_matrix @ weights - constraint_rhs,
            "jac": lambda _weights: constraint_matrix,
        }
    ]
    bounds = [(0.0, None)] * n_labeled
    result = minimize(
        objective,
        x0=np.ones(n_labeled, dtype=float),
        jac=gradient,
        constraints=constraints,
        bounds=bounds,
        method="SLSQP",
        options={"ftol": tolerance**2, "maxiter": int(maxiter)},
    )
    if not result.success:
        raise ValueError(
            "Could not compute nonnegative balancing weights with the requested balance moments. "
            "Try a lower-dimensional balance representation or target='pooled'."
        )

    weights = np.asarray(result.x, dtype=float)
    weighted_mean = np.mean(weights[:, None] * Z_labeled, axis=0)
    balance_error = weighted_mean - target_mean
    max_abs_balance_error = float(np.max(np.abs(balance_error)))
    if max_abs_balance_error > tolerance:
        raise ValueError(
            "Could not achieve the requested balance tolerance with nonnegative weights. "
            f"Max absolute balance error was {max_abs_balance_error:.3e}."
        )

    diagnostics = {
        "target": resolved_target,
        "include_intercept": include_intercept,
        "n_labeled": n_labeled,
        "n_unlabeled": n_unlabeled,
        "n_balance_functions": Z_labeled.shape[1],
        "target_mean": target_mean,
        "weighted_labeled_mean": weighted_mean,
        "balance_error": balance_error,
        "max_abs_balance_error": max_abs_balance_error,
        "min_weight": float(np.min(weights)),
        "max_weight": float(np.max(weights)),
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "optimizer_iterations": int(getattr(result, "nit", 0)),
    }
    if return_diagnostics:
        return weights, diagnostics
    return weights
