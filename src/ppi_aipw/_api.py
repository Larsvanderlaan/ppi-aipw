from __future__ import annotations

from typing import Any

import numpy as np

from ._calibration import CalibrationModel, calibrate_predictions, canonical_method, fit_calibrator
from ._utils import (
    construct_weight_vector,
    restore_shape,
    validate_mean_inputs,
    z_interval,
)

def _labeled_fraction(n_labeled: int, n_unlabeled: int) -> float:
    return n_labeled / float(n_labeled + n_unlabeled)


def _fit_and_calibrate(
    Y: np.ndarray,
    Yhat: np.ndarray,
    Yhat_unlabeled: np.ndarray,
    *,
    method: str,
    w: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, CalibrationModel]:
    Y_2d, Yhat_2d, Yhat_unlabeled_2d = validate_mean_inputs(Y, Yhat, Yhat_unlabeled)
    model = fit_calibrator(Y_2d, Yhat_2d, method=method, w=w)
    pred_labeled = np.asarray(model.predict(Yhat_2d), dtype=float)
    pred_unlabeled = np.asarray(model.predict(Yhat_unlabeled_2d), dtype=float)
    return Y_2d, Yhat_2d, pred_labeled, pred_unlabeled, model


def _coerce_generator(random_state: int | np.random.Generator | None) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _bootstrap_pointestimates(
    Y: np.ndarray,
    Yhat: np.ndarray,
    Yhat_unlabeled: np.ndarray,
    *,
    method: str,
    w: np.ndarray | None,
    w_unlabeled: np.ndarray | None,
    n_resamples: int,
    random_state: int | np.random.Generator | None,
) -> np.ndarray:
    if n_resamples < 2:
        raise ValueError("n_resamples must be at least 2 when inference='bootstrap'.")

    Y_arr = np.asarray(Y)
    Yhat_arr = np.asarray(Yhat)
    Yhat_unlabeled_arr = np.asarray(Yhat_unlabeled)
    if w is not None:
        w = np.asarray(w, dtype=float)
    if w_unlabeled is not None:
        w_unlabeled = np.asarray(w_unlabeled, dtype=float)

    n_labeled = Y_arr.shape[0]
    n_unlabeled = Yhat_unlabeled_arr.shape[0]
    rng = _coerce_generator(random_state)

    bootstrap_estimates = []
    for _ in range(n_resamples):
        labeled_idx = rng.integers(0, n_labeled, size=n_labeled)
        unlabeled_idx = rng.integers(0, n_unlabeled, size=n_unlabeled)
        estimate = np.asarray(
            aipw_mean_pointestimate(
                Y_arr[labeled_idx],
                Yhat_arr[labeled_idx],
                Yhat_unlabeled_arr[unlabeled_idx],
                method=method,
                w=None if w is None else w[labeled_idx],
                w_unlabeled=None if w_unlabeled is None else w_unlabeled[unlabeled_idx],
            ),
            dtype=float,
        ).reshape(1, -1)
        bootstrap_estimates.append(estimate)

    return np.concatenate(bootstrap_estimates, axis=0)


def _aipw_mean_pointestimate_from_predictions(
    Y: np.ndarray,
    pred_labeled: np.ndarray,
    pred_unlabeled: np.ndarray,
    *,
    w: np.ndarray,
    w_unlabeled: np.ndarray,
) -> np.ndarray:
    n_labeled = Y.shape[0]
    n_unlabeled = pred_unlabeled.shape[0]
    c = 1.0 - _labeled_fraction(n_labeled, n_unlabeled)

    labeled_term = (w * (Y - c * pred_labeled)).mean(axis=0)
    unlabeled_term = (w_unlabeled * (c * pred_unlabeled)).mean(axis=0)
    return labeled_term + unlabeled_term


def aipw_mean_pointestimate(
    Y: np.ndarray,
    Yhat: np.ndarray,
    Yhat_unlabeled: np.ndarray,
    *,
    method: str = "aipw",
    w: np.ndarray | None = None,
    w_unlabeled: np.ndarray | None = None,
    return_calibrator: bool = False,
) -> float | np.ndarray | tuple[float | np.ndarray, CalibrationModel]:
    """Computes an AIPW-style point estimate for the population mean.

    The calling convention mirrors :mod:`ppi_py`: pass labeled outcomes ``Y``,
    labeled predictions ``Yhat``, and unlabeled predictions ``Yhat_unlabeled``.
    Set ``method`` to ``"aipw"``, ``"linear"``, ``"platt"``, or ``"isocal"``.

    Args:
        Y:
            Gold-standard outcomes for the labeled sample. Shape ``(n_labeled,)``
            or ``(n_labeled, d)``.
        Yhat:
            Model predictions for the same labeled rows as ``Y``. Must have the
            same shape as ``Y``.
        Yhat_unlabeled:
            Model predictions for the unlabeled sample. Shape
            ``(n_unlabeled,)`` or ``(n_unlabeled, d)`` with the same output
            dimension as ``Y``.
        method:
            Calibration method applied before the AIPW augmentation step.
            Available choices are:

            - ``"aipw"``: use the raw predictions directly, with no calibration.
            - ``"linear"``: fit an affine map from predictions to outcomes.
            - ``"platt"``: fit a sigmoid calibration map after rescaling outcomes
              into the observed labeled range.
            - ``"isocal"``: fit a monotone isotonic calibration map.
        w:
            Optional sample weights for labeled data. If provided, must have
            length ``n_labeled``.
        w_unlabeled:
            Optional sample weights for unlabeled data. If provided, must have
            length ``n_unlabeled``.
        return_calibrator:
            If ``True``, also returns the fitted calibration model so it can be
            inspected or reused.

    Returns:
        The estimated population mean, with scalar or vector shape matching ``Y``.

    Examples:
        Basic usage::

            estimate = aipw_mean_pointestimate(Y, Yhat, Yhat_unlabeled, method="linear")

        Use isotonic calibration::

            estimate = aipw_mean_pointestimate(Y, Yhat, Yhat_unlabeled, method="isocal")
    """

    method = canonical_method(method)
    Y_2d, _, pred_labeled, pred_unlabeled, calibrator = _fit_and_calibrate(
        Y,
        Yhat,
        Yhat_unlabeled,
        method=method,
        w=w,
    )

    weights = construct_weight_vector(Y_2d.shape[0], w, vectorized=True)
    weights_unlabeled = construct_weight_vector(pred_unlabeled.shape[0], w_unlabeled, vectorized=True)

    estimate = _aipw_mean_pointestimate_from_predictions(
        Y_2d,
        pred_labeled,
        pred_unlabeled,
        w=weights,
        w_unlabeled=weights_unlabeled,
    )
    estimate_out = restore_shape(estimate, np.asarray(Y))
    if return_calibrator:
        return estimate_out, calibrator
    return estimate_out


def aipw_mean_se(
    Y: np.ndarray,
    Yhat: np.ndarray,
    Yhat_unlabeled: np.ndarray,
    *,
    method: str = "aipw",
    w: np.ndarray | None = None,
    w_unlabeled: np.ndarray | None = None,
    inference: str = "wald",
    n_resamples: int = 1000,
    random_state: int | np.random.Generator | None = None,
) -> float | np.ndarray:
    """Computes a standard error for :func:`aipw_mean_pointestimate`.

    This uses the same arguments as :func:`aipw_mean_pointestimate` and returns a
    scalar or vector standard error matching the shape of ``Y``.

    Use ``inference="wald"`` for the analytic standard error or
    ``inference="bootstrap"`` to estimate the standard error from bootstrap
    resamples.
    """

    inference = inference.lower()
    if inference not in {"wald", "bootstrap"}:
        raise ValueError("inference must be either 'wald' or 'bootstrap'.")

    if inference == "bootstrap":
        bootstrap_estimates = _bootstrap_pointestimates(
            Y,
            Yhat,
            Yhat_unlabeled,
            method=method,
            w=w,
            w_unlabeled=w_unlabeled,
            n_resamples=n_resamples,
            random_state=random_state,
        )
        return restore_shape(np.std(bootstrap_estimates, axis=0, ddof=1), np.asarray(Y))

    method = canonical_method(method)
    Y_2d, _, pred_labeled, pred_unlabeled, _ = _fit_and_calibrate(
        Y,
        Yhat,
        Yhat_unlabeled,
        method=method,
        w=w,
    )

    n_labeled = Y_2d.shape[0]
    n_unlabeled = pred_unlabeled.shape[0]
    weights = construct_weight_vector(n_labeled, w, vectorized=True)
    weights_unlabeled = construct_weight_vector(n_unlabeled, w_unlabeled, vectorized=True)

    c = 1.0 - _labeled_fraction(n_labeled, n_unlabeled)
    labeled_component = weights * (Y_2d - c * pred_labeled)
    unlabeled_component = weights_unlabeled * (c * pred_unlabeled)
    standard_error = np.sqrt(
        np.var(labeled_component, axis=0) / n_labeled + np.var(unlabeled_component, axis=0) / n_unlabeled
    )
    return restore_shape(standard_error, np.asarray(Y))


def aipw_mean_ci(
    Y: np.ndarray,
    Yhat: np.ndarray,
    Yhat_unlabeled: np.ndarray,
    *,
    alpha: float = 0.1,
    alternative: str = "two-sided",
    method: str = "aipw",
    w: np.ndarray | None = None,
    w_unlabeled: np.ndarray | None = None,
    inference: str = "wald",
    n_resamples: int = 1000,
    random_state: int | np.random.Generator | None = None,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Computes a confidence interval for the mean.

    Args:
        Y:
            Observed outcomes on the labeled sample.
        Yhat:
            Predictions for the labeled sample.
        Yhat_unlabeled:
            Predictions for the unlabeled sample.
        alpha:
            Miscoverage level. ``alpha=0.1`` gives a 90% interval and
            ``alpha=0.05`` gives a 95% interval.
        alternative:
            One of ``"two-sided"``, ``"larger"``, or ``"smaller"``.
        method:
            One of ``"aipw"``, ``"linear"``, ``"platt"``, or ``"isocal"``.
        w:
            Optional labeled-sample weights.
        w_unlabeled:
            Optional unlabeled-sample weights.
        inference:
            ``"wald"`` for the analytic normal-approximation interval or
            ``"bootstrap"`` for a percentile bootstrap interval that resamples
            labeled and unlabeled rows and refits the calibration step in each
            resample while holding the prediction model fixed.
        n_resamples:
            Number of bootstrap resamples when ``inference="bootstrap"``.
        random_state:
            Optional random seed or ``numpy.random.Generator`` for bootstrap
            reproducibility.

    Returns:
        A pair ``(lower, upper)`` with the same scalar or vector shape as ``Y``.

    Notes:
        ``inference="wald"`` returns the analytic normal-approximation interval.
        ``inference="bootstrap"`` returns a percentile bootstrap interval.
    """

    inference = inference.lower()
    if inference not in {"wald", "bootstrap"}:
        raise ValueError("inference must be either 'wald' or 'bootstrap'.")

    if inference == "bootstrap":
        bootstrap_estimates = _bootstrap_pointestimates(
            Y,
            Yhat,
            Yhat_unlabeled,
            method=method,
            w=w,
            w_unlabeled=w_unlabeled,
            n_resamples=n_resamples,
            random_state=random_state,
        )
        if alternative == "two-sided":
            lower = np.quantile(bootstrap_estimates, alpha / 2.0, axis=0)
            upper = np.quantile(bootstrap_estimates, 1.0 - alpha / 2.0, axis=0)
        elif alternative == "larger":
            lower = np.quantile(bootstrap_estimates, alpha, axis=0)
            upper = np.full(lower.shape, np.inf, dtype=float)
        elif alternative == "smaller":
            lower = np.full(bootstrap_estimates.shape[1], -np.inf, dtype=float)
            upper = np.quantile(bootstrap_estimates, 1.0 - alpha, axis=0)
        else:
            raise ValueError("alternative must be 'two-sided', 'larger', or 'smaller'.")
        return restore_shape(lower, np.asarray(Y)), restore_shape(upper, np.asarray(Y))

    Y_2d, _, pred_labeled, pred_unlabeled, _ = _fit_and_calibrate(
        Y,
        Yhat,
        Yhat_unlabeled,
        method=method,
        w=w,
    )

    weights = construct_weight_vector(Y_2d.shape[0], w, vectorized=True)
    weights_unlabeled = construct_weight_vector(pred_unlabeled.shape[0], w_unlabeled, vectorized=True)

    pointestimate = _aipw_mean_pointestimate_from_predictions(
        Y_2d,
        pred_labeled,
        pred_unlabeled,
        w=weights,
        w_unlabeled=weights_unlabeled,
    )

    c = 1.0 - _labeled_fraction(Y_2d.shape[0], pred_unlabeled.shape[0])
    labeled_component = weights * (Y_2d - c * pred_labeled)
    unlabeled_component = weights_unlabeled * (c * pred_unlabeled)
    standard_error = np.sqrt(
        np.var(labeled_component, axis=0) / Y_2d.shape[0]
        + np.var(unlabeled_component, axis=0) / pred_unlabeled.shape[0]
    )
    lower, upper = z_interval(pointestimate, standard_error, alpha, alternative)
    return restore_shape(lower, np.asarray(Y)), restore_shape(upper, np.asarray(Y))


def linear_calibration_mean_pointestimate(*args: Any, **kwargs: Any) -> float | np.ndarray:
    """Shortcut for :func:`aipw_mean_pointestimate` with ``method="linear"``."""

    kwargs.setdefault("method", "linear")
    return aipw_mean_pointestimate(*args, **kwargs)


def linear_calibration_mean_ci(*args: Any, **kwargs: Any) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Shortcut for :func:`aipw_mean_ci` with ``method="linear"``."""

    kwargs.setdefault("method", "linear")
    return aipw_mean_ci(*args, **kwargs)


def platt_scaling_mean_pointestimate(*args: Any, **kwargs: Any) -> float | np.ndarray:
    """Shortcut for :func:`aipw_mean_pointestimate` with ``method="platt"``."""

    kwargs.setdefault("method", "platt")
    return aipw_mean_pointestimate(*args, **kwargs)


def platt_scaling_mean_ci(*args: Any, **kwargs: Any) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Shortcut for :func:`aipw_mean_ci` with ``method="platt"``."""

    kwargs.setdefault("method", "platt")
    return aipw_mean_ci(*args, **kwargs)


def isocal_mean_pointestimate(*args: Any, **kwargs: Any) -> float | np.ndarray:
    """Shortcut for :func:`aipw_mean_pointestimate` with ``method="isocal"``."""

    kwargs.setdefault("method", "isocal")
    return aipw_mean_pointestimate(*args, **kwargs)


def isocal_mean_ci(*args: Any, **kwargs: Any) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Shortcut for :func:`aipw_mean_ci` with ``method="isocal"``."""

    kwargs.setdefault("method", "isocal")
    return aipw_mean_ci(*args, **kwargs)


mean_pointestimate = aipw_mean_pointestimate
mean_ci = aipw_mean_ci
mean_se = aipw_mean_se
ppi_aipw_mean_pointestimate = aipw_mean_pointestimate
ppi_aipw_mean_ci = aipw_mean_ci
pi_aipw_mean_pointestimate = aipw_mean_pointestimate
pi_aipw_mean_ci = aipw_mean_ci
