from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.stats import norm


def reshape_to_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 0:
        raise ValueError("Expected a one- or two-dimensional array, got a scalar.")
    return x.reshape(-1, 1) if x.ndim == 1 else x.copy()


def restore_shape(x: np.ndarray, reference: np.ndarray) -> float | np.ndarray:
    x = np.asarray(x, dtype=float)
    reference = np.asarray(reference)
    if reference.ndim == 1:
        flat = x.reshape(-1)
        return float(flat[0]) if flat.size == 1 else flat
    return x


def construct_weight_vector(
    n_obs: int,
    existing_weight: Optional[np.ndarray],
    *,
    vectorized: bool = False,
) -> np.ndarray:
    if existing_weight is None:
        weights = np.ones(n_obs, dtype=float)
    else:
        weights = np.asarray(existing_weight, dtype=float).reshape(-1)
        if weights.shape != (n_obs,):
            raise ValueError(f"Expected weights with shape {(n_obs,)}, got {weights.shape}.")
        if np.any(weights < 0):
            raise ValueError("Weights must be nonnegative.")
        if not np.any(weights > 0):
            raise ValueError("At least one weight must be strictly positive.")
        weights = weights / weights.sum() * n_obs

    if vectorized:
        weights = weights[:, None]
    return weights


def validate_mean_inputs(
    Y: np.ndarray,
    Yhat: np.ndarray,
    Yhat_unlabeled: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Y_2d = reshape_to_2d(np.asarray(Y, dtype=float))
    Yhat_2d = reshape_to_2d(np.asarray(Yhat, dtype=float))
    Yhat_unlabeled_2d = reshape_to_2d(np.asarray(Yhat_unlabeled, dtype=float))

    if Y_2d.shape != Yhat_2d.shape:
        raise ValueError(
            f"Y and Yhat must have the same shape, got {Y_2d.shape} and {Yhat_2d.shape}."
        )
    if Yhat_unlabeled_2d.shape[1] != Y_2d.shape[1]:
        raise ValueError(
            "Yhat_unlabeled must have the same number of columns as Y and Yhat. "
            f"Got {Yhat_unlabeled_2d.shape[1]} and {Y_2d.shape[1]}."
        )
    if Y_2d.shape[0] == 0 or Yhat_unlabeled_2d.shape[0] == 0:
        raise ValueError("Both the labeled and unlabeled samples must be nonempty.")

    return Y_2d, Yhat_2d, Yhat_unlabeled_2d


def validate_pair_inputs(
    Y: np.ndarray,
    Yhat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    Y_2d = reshape_to_2d(np.asarray(Y, dtype=float))
    Yhat_2d = reshape_to_2d(np.asarray(Yhat, dtype=float))
    if Y_2d.shape != Yhat_2d.shape:
        raise ValueError(f"Y and Yhat must have the same shape, got {Y_2d.shape} and {Yhat_2d.shape}.")
    if Y_2d.shape[0] == 0:
        raise ValueError("The labeled sample must be nonempty.")
    return Y_2d, Yhat_2d


def weighted_mean(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return (w * x).mean(axis=0)


def weighted_var(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    centered = x - weighted_mean(x, w)
    return (w * centered**2).mean(axis=0)


def weighted_cov(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    centered_x = x - weighted_mean(x, w)
    centered_y = y - weighted_mean(y, w)
    return (w * centered_x * centered_y).mean(axis=0)


def z_interval(
    pointestimate: np.ndarray,
    standard_error: np.ndarray,
    alpha: float,
    alternative: str,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1).")

    pointestimate = np.asarray(pointestimate, dtype=float)
    standard_error = np.asarray(standard_error, dtype=float)

    if alternative == "two-sided":
        z_value = norm.ppf(1.0 - alpha / 2.0)
        lower = pointestimate - z_value * standard_error
        upper = pointestimate + z_value * standard_error
    elif alternative == "larger":
        z_value = norm.ppf(1.0 - alpha)
        lower = pointestimate - z_value * standard_error
        upper = np.full_like(lower, np.inf, dtype=float)
    elif alternative == "smaller":
        z_value = norm.ppf(1.0 - alpha)
        lower = np.full_like(pointestimate, -np.inf, dtype=float)
        upper = pointestimate + z_value * standard_error
    else:
        raise ValueError("alternative must be 'two-sided', 'larger', or 'smaller'.")

    return lower, upper
