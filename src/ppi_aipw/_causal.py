from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._api import (
    MeanInferenceResult,
    _compute_wald_statistics,
    _mean_inference_result_from_prepared,
    _prepare_inference_inputs,
    _preview_value,
    _summary_value,
    _wald_influence_components,
)
from ._utils import construct_weight_vector, reshape_to_2d, validate_finite_array, z_interval


@dataclass(frozen=True)
class CausalInferenceResult:
    arm_means: dict[Any, float]
    arm_ses: dict[Any, float]
    arm_cis: dict[Any, tuple[float, float]]
    ate: dict[Any, float]
    ate_ses: dict[Any, float]
    ate_cis: dict[Any, tuple[float, float]]
    control_arm: Any
    treatment_levels: tuple[Any, ...]
    arm_results: dict[Any, MeanInferenceResult]
    diagnostics: dict[str, Any]

    def __repr__(self) -> str:
        ate_preview = {arm: _summary_value(float(self.ate[arm])) for arm in self.ate}
        parts = [
            f"control_arm={self.control_arm!r}",
            f"treatment_levels={self.treatment_levels!r}",
            f"inference={self.diagnostics.get('inference', 'wald')!r}",
            f"ate={ate_preview!r}",
        ]
        return f"CausalInferenceResult({', '.join(parts)})"

    def summary(
        self,
        *,
        null: float = 0.0,
        alternative: str = "two-sided",
    ) -> str:
        null_arr = np.asarray(null, dtype=float)
        null_preview: float | np.ndarray
        if null_arr.ndim == 0:
            null_preview = float(null_arr)
        else:
            null_preview = null_arr.reshape(-1).copy()

        ordered_arms = list(self.treatment_levels)
        arm_estimates = np.array([self.arm_means[arm] for arm in ordered_arms], dtype=float)
        arm_ses = np.array([self.arm_ses[arm] for arm in ordered_arms], dtype=float)
        arm_ci_lower = np.array([self.arm_cis[arm][0] for arm in ordered_arms], dtype=float)
        arm_ci_upper = np.array([self.arm_cis[arm][1] for arm in ordered_arms], dtype=float)
        arm_null, arm_t_stat, arm_p_value = _compute_wald_statistics(
            arm_estimates,
            arm_ses,
            null=null,
            alternative=alternative,
        )

        comparison_arms = [arm for arm in ordered_arms if arm != self.control_arm]
        ate_estimates = np.array([self.ate[arm] for arm in comparison_arms], dtype=float)
        ate_ses = np.array([self.ate_ses[arm] for arm in comparison_arms], dtype=float)
        ate_ci_lower = np.array([self.ate_cis[arm][0] for arm in comparison_arms], dtype=float)
        ate_ci_upper = np.array([self.ate_cis[arm][1] for arm in comparison_arms], dtype=float)
        ate_null, ate_t_stat, ate_p_value = _compute_wald_statistics(
            ate_estimates,
            ate_ses,
            null=null,
            alternative=alternative,
        )

        self.diagnostics["arm_wald_null"] = null_preview
        self.diagnostics["arm_wald_alternative"] = alternative
        self.diagnostics["arm_wald_t_statistic"] = {
            arm: float(stat) for arm, stat in zip(ordered_arms, arm_t_stat)
        }
        self.diagnostics["arm_wald_p_value"] = {
            arm: float(value) for arm, value in zip(ordered_arms, arm_p_value)
        }
        self.diagnostics["ate_wald_null"] = null_preview
        self.diagnostics["ate_wald_alternative"] = alternative
        self.diagnostics["ate_wald_t_statistic"] = {
            arm: float(stat) for arm, stat in zip(comparison_arms, ate_t_stat)
        }
        self.diagnostics["ate_wald_p_value"] = {
            arm: float(value) for arm, value in zip(comparison_arms, ate_p_value)
        }

        arm_methods = {arm: result.method for arm, result in self.arm_results.items()}
        unique_methods = tuple(dict.fromkeys(arm_methods[arm] for arm in ordered_arms))
        lines = [
            "CausalInferenceResult summary",
            f"control_arm: {self.control_arm!r}",
            f"treatment_levels: {self.treatment_levels!r}",
            f"inference: {self.diagnostics.get('inference', 'wald')}",
            f"wald_null: {_preview_value(null_preview, digits=6)}",
            f"wald_alternative: {alternative}",
        ]
        if len(unique_methods) == 1:
            lines.append(f"arm_method: {unique_methods[0]}")
        else:
            lines.append(f"arm_methods: {arm_methods}")

        lines.append("")
        lines.append("Arm means:")
        for arm, estimate, se, lower, upper, t_stat, p_value in zip(
            ordered_arms,
            arm_estimates,
            arm_ses,
            arm_ci_lower,
            arm_ci_upper,
            arm_t_stat,
            arm_p_value,
        ):
            lines.append(
                "arm={arm!r}: estimate={estimate}, se={se}, ci=({lower}, {upper}), "
                "wald_t={t_stat}, p_value={p_value}".format(
                    arm=arm,
                    estimate=_summary_value(float(estimate)),
                    se=_summary_value(float(se)),
                    lower=_summary_value(float(lower)),
                    upper=_summary_value(float(upper)),
                    t_stat=_summary_value(float(t_stat)),
                    p_value=_summary_value(float(p_value)),
                )
            )

        lines.append("")
        lines.append("ATEs vs control:")
        for arm, estimate, se, lower, upper, t_stat, p_value in zip(
            comparison_arms,
            ate_estimates,
            ate_ses,
            ate_ci_lower,
            ate_ci_upper,
            ate_t_stat,
            ate_p_value,
        ):
            lines.append(
                "{arm!r} - {control!r}: estimate={estimate}, se={se}, ci=({lower}, {upper}), "
                "wald_t={t_stat}, p_value={p_value}".format(
                    arm=arm,
                    control=self.control_arm,
                    estimate=_summary_value(float(estimate)),
                    se=_summary_value(float(se)),
                    lower=_summary_value(float(lower)),
                    upper=_summary_value(float(upper)),
                    t_stat=_summary_value(float(t_stat)),
                    p_value=_summary_value(float(p_value)),
                )
            )
        return "\n".join(lines)


def _validate_outcome_vector(Y: np.ndarray) -> np.ndarray:
    Y_arr = np.asarray(Y, dtype=float)
    if Y_arr.ndim != 1:
        raise ValueError("causal_inference currently supports one-dimensional outcomes only.")
    if Y_arr.shape[0] == 0:
        raise ValueError("Y must be nonempty.")
    return validate_finite_array(Y_arr, name="Y")


def _validate_weight_vector(w: np.ndarray | None, n_obs: int) -> np.ndarray | None:
    if w is None:
        return None
    weights = np.asarray(w, dtype=float).reshape(-1)
    if weights.shape != (n_obs,):
        raise ValueError(f"Expected weights with shape {(n_obs,)}, got {weights.shape}.")
    if not np.all(np.isfinite(weights)):
        raise ValueError("Weights must contain only finite values.")
    if np.any(weights < 0):
        raise ValueError("Weights must be nonnegative.")
    if not np.any(weights > 0):
        raise ValueError("At least one weight must be strictly positive.")
    return weights


def _validate_covariate_matrix(X: np.ndarray | None, n_obs: int) -> np.ndarray | None:
    if X is None:
        return None
    X_arr = reshape_to_2d(np.asarray(X, dtype=float), name="X")
    if X_arr.shape[0] != n_obs:
        raise ValueError(f"X must have {n_obs} rows, got {X_arr.shape[0]}.")
    return X_arr


def _resolve_potential_outcome_inputs(
    A: np.ndarray,
    Yhat_potential: Any,
    treatment_levels: tuple[Any, ...] | list[Any] | None,
) -> tuple[np.ndarray, np.ndarray, tuple[Any, ...], dict[Any, int]]:
    A_arr = np.asarray(A)
    if A_arr.ndim != 1:
        raise ValueError("A must be a one-dimensional treatment vector.")
    if np.issubdtype(A_arr.dtype, np.number):
        validate_finite_array(A_arr, name="A")
    elif any(item is None for item in A_arr.tolist()):
        raise ValueError("A must not contain missing values.")

    potential_matrix = np.asarray(Yhat_potential, dtype=float)
    if potential_matrix.ndim != 2:
        raise ValueError("Yhat_potential must be a two-dimensional matrix with one column per arm.")
    if potential_matrix.shape[0] != A_arr.shape[0]:
        raise ValueError(
            "Yhat_potential must have the same number of rows as Y and A. "
            f"Got {potential_matrix.shape[0]} and {A_arr.shape[0]}."
        )
    validate_finite_array(potential_matrix, name="Yhat_potential")

    observed_levels = tuple(dict.fromkeys(A_arr.tolist()))
    if len(observed_levels) < 2:
        raise ValueError(
            "causal_inference requires at least two observed treatment arms in A."
        )

    if treatment_levels is None:
        if hasattr(Yhat_potential, "columns"):
            resolved_levels = tuple(list(Yhat_potential.columns))
        else:
            try:
                resolved_levels = tuple(sorted(observed_levels))
            except TypeError as exc:
                raise ValueError(
                    "Could not infer a sortable treatment order from A. "
                    "Pass treatment_levels explicitly."
                ) from exc
    else:
        resolved_levels = tuple(treatment_levels)

    if len(resolved_levels) != potential_matrix.shape[1]:
        raise ValueError(
            "The number of treatment_levels must match the number of columns in Yhat_potential. "
            f"Got {len(resolved_levels)} and {potential_matrix.shape[1]}."
        )
    if len(set(resolved_levels)) != len(resolved_levels):
        raise ValueError("treatment_levels must contain unique arm labels.")

    counts = {arm: int(np.sum(A_arr == arm)) for arm in resolved_levels}
    missing_arms = [arm for arm, count in counts.items() if count == 0]
    if missing_arms:
        raise ValueError(
            "Every treatment arm must have at least one observed unit. "
            f"Missing observed data for: {missing_arms}."
        )
    observed_but_unmapped = [arm for arm in observed_levels if arm not in counts]
    if observed_but_unmapped:
        raise ValueError(
            "Observed treatment values are missing from treatment_levels: "
            f"{observed_but_unmapped}."
        )
    if len(resolved_levels) < 2:
        raise ValueError("Need at least two observed treatment arms for causal_inference.")

    arm_to_column = {arm: idx for idx, arm in enumerate(resolved_levels)}
    return A_arr, potential_matrix, resolved_levels, arm_to_column


def _resolve_control_arm(
    treatment_levels: tuple[Any, ...],
    control_arm: Any | None,
) -> Any:
    if control_arm is not None:
        if control_arm not in treatment_levels:
            raise ValueError(f"control_arm={control_arm!r} is not one of the resolved treatment levels.")
        return control_arm
    try:
        return min(treatment_levels)
    except TypeError as exc:
        raise ValueError(
            "Could not determine the default control arm from treatment_levels. "
            "Pass control_arm explicitly."
        ) from exc


def _coerce_prediction_vector(
    prediction: np.ndarray,
    *,
    expected_length: int,
    name: str,
) -> np.ndarray:
    prediction_arr = np.asarray(prediction, dtype=float).reshape(-1)
    if prediction_arr.shape != (expected_length,):
        raise ValueError(f"Expected {name} with shape {(expected_length,)}, got {prediction_arr.shape}.")
    return prediction_arr


def _assemble_full_sample_prediction(
    labeled_mask: np.ndarray,
    pred_labeled: np.ndarray,
    pred_unlabeled: np.ndarray,
    *,
    name: str,
    fallback_unlabeled: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    n_obs = labeled_mask.shape[0]
    n_labeled = int(np.sum(labeled_mask))
    n_unlabeled = int(n_obs - n_labeled)

    pred_labeled_vec = _coerce_prediction_vector(
        pred_labeled,
        expected_length=n_labeled,
        name=f"{name} labeled predictions",
    )

    pred_unlabeled_vec = np.asarray(pred_unlabeled, dtype=float).reshape(-1)
    used_fallback = False
    if pred_unlabeled_vec.shape != (n_unlabeled,):
        if fallback_unlabeled is None:
            raise ValueError(
                f"Expected {name} unlabeled predictions with shape {(n_unlabeled,)}, "
                f"got {pred_unlabeled_vec.shape}."
            )
        pred_unlabeled_vec = _coerce_prediction_vector(
            fallback_unlabeled,
            expected_length=n_unlabeled,
            name=f"{name} fallback unlabeled predictions",
        )
        used_fallback = True

    full_prediction = np.empty(n_obs, dtype=float)
    full_prediction[labeled_mask] = pred_labeled_vec
    full_prediction[~labeled_mask] = pred_unlabeled_vec
    return full_prediction, used_fallback


def _causal_arm_pointestimate_and_influence(
    Y: np.ndarray,
    *,
    labeled_mask: np.ndarray,
    pred_point: np.ndarray,
    pred_variance: np.ndarray,
    full_sample_weights: np.ndarray,
) -> tuple[float, np.ndarray]:
    arm_probability = float(np.mean(labeled_mask))
    if arm_probability <= 0.0:
        raise ValueError("Each treatment arm must contain at least one observation.")

    labeled_indicator = labeled_mask.astype(float)
    pseudo_outcome_point = full_sample_weights * (
        pred_point + (labeled_indicator / arm_probability) * (Y - pred_point)
    )
    pointestimate = float(np.mean(pseudo_outcome_point))

    pseudo_outcome_variance = full_sample_weights * (
        pred_variance + (labeled_indicator / arm_probability) * (Y - pred_variance)
    )
    influence = (pseudo_outcome_variance - np.mean(pseudo_outcome_variance)) / Y.shape[0]
    return pointestimate, influence


def _aligned_wald_influence(
    labeled_mask: np.ndarray,
    prepared: Any,
    n_obs: int,
) -> np.ndarray:
    labeled_component, unlabeled_component = _wald_influence_components(prepared)
    influence = np.zeros((n_obs, labeled_component.shape[1]), dtype=float)
    influence[labeled_mask] = (
        labeled_component - labeled_component.mean(axis=0)
    ) / labeled_component.shape[0]
    influence[~labeled_mask] = (
        unlabeled_component - unlabeled_component.mean(axis=0)
    ) / unlabeled_component.shape[0]
    return influence


def causal_inference(
    Y: np.ndarray,
    A: np.ndarray,
    Yhat_potential: Any,
    *,
    treatment_levels: tuple[Any, ...] | list[Any] | None = None,
    control_arm: Any | None = None,
    method: str = "monotone_spline",
    w: np.ndarray | None = None,
    X: np.ndarray | None = None,
    alpha: float = 0.1,
    alternative: str = "two-sided",
    efficiency_maximization: bool = False,
    candidate_methods: tuple[str, ...] = ("aipw", "linear", "monotone_spline", "isotonic"),
    num_folds: int = 100,
    auto_unlabeled_subsample_size: int | None = None,
    selection_random_state: int | np.random.Generator | None = None,
    isocal_backend: str = "xgboost",
    isocal_max_depth: int = 20,
    isocal_min_child_weight: float = 10.0,
    inference: str = "wald",
) -> CausalInferenceResult:
    """Estimate arm-specific potential outcome means and control-vs-treatment ATEs.

    The wrapper reuses the package's semisupervised mean engine arm by arm. For
    a target arm ``a``, units with ``A == a`` are treated as the labeled sample
    and units with ``A != a`` are treated as the unlabeled sample, using the
    arm-``a`` prediction column from ``Yhat_potential`` for both groups.

    Parameters
    ----------
    Y
        Observed outcomes for all units. This first version supports one-dimensional outcomes.
    A
        Observed treatment assignments. May be binary, discrete, or categorical.
    Yhat_potential
        Matrix or DataFrame of predicted potential outcomes with one column per treatment arm.
    treatment_levels
        Optional arm labels corresponding to the columns of ``Yhat_potential``.
        When omitted and ``Yhat_potential`` has named columns, those names are used.
        Otherwise the function infers levels from the sorted unique values in ``A``.
    control_arm
        Optional control arm used for ATE comparisons. Defaults to the minimum resolved treatment level.
    method
        Calibration method passed to the underlying semisupervised mean engine.
    w
        Optional observation weights for the full sample. These may also be
        balancing weights if you want to reweight the arm-specific mean targets
        toward a covariate-adjusted population. Uniform weights reproduce the
        unweighted behavior.
    X
        Optional extra covariates for the full sample. These are passed
        arm-by-arm to the underlying semisupervised mean engine and are
        especially useful with ``method="prognostic_linear"``.
    alpha
        Miscoverage level for Wald intervals.
    alternative
        Interval alternative passed through to the Wald interval builder.
    efficiency_maximization
        Whether to rescale the chosen predictor by empirical efficiency maximization.
    candidate_methods
        Candidate methods used when ``method="auto"``.
    num_folds
        Number of folds used when ``method="auto"``.
    auto_unlabeled_subsample_size
        Unlabeled subsample size used by ``method="auto"``.
    selection_random_state
        Random seed for automatic method selection.
    isocal_backend, isocal_max_depth, isocal_min_child_weight
        Controls for the isotonic calibration backend.
    inference
        Must be ``"wald"`` in this version.
    """

    if inference.lower() != "wald":
        raise ValueError("causal_inference currently supports inference='wald' only.")

    Y_arr = _validate_outcome_vector(Y)
    raw_weights = _validate_weight_vector(w, Y_arr.shape[0])
    use_unweighted_causal_path = raw_weights is None or bool(np.allclose(raw_weights, raw_weights[0]))
    full_sample_weights = (
        None
        if use_unweighted_causal_path
        else construct_weight_vector(Y_arr.shape[0], raw_weights, vectorized=False)
    )
    X_arr = _validate_covariate_matrix(X, Y_arr.shape[0])
    A_arr, potential_matrix, resolved_levels, arm_to_column = _resolve_potential_outcome_inputs(
        A,
        Yhat_potential,
        treatment_levels,
    )
    resolved_control_arm = _resolve_control_arm(resolved_levels, control_arm)

    arm_results: dict[Any, MeanInferenceResult] = {}
    arm_means: dict[Any, float] = {}
    arm_ses: dict[Any, float] = {}
    arm_cis: dict[Any, tuple[float, float]] = {}
    arm_influences: dict[Any, np.ndarray] = {}

    for arm in resolved_levels:
        labeled_mask = np.asarray(A_arr == arm, dtype=bool)
        unlabeled_mask = ~labeled_mask
        arm_predictions = potential_matrix[:, arm_to_column[arm]]

        prepared = _prepare_inference_inputs(
            Y_arr[labeled_mask],
            arm_predictions[labeled_mask],
            arm_predictions[unlabeled_mask],
            method=method,
            w=None if raw_weights is None else raw_weights[labeled_mask],
            w_unlabeled=None if raw_weights is None else raw_weights[unlabeled_mask],
            X=None if X_arr is None else X_arr[labeled_mask],
            X_unlabeled=None if X_arr is None else X_arr[unlabeled_mask],
            efficiency_maximization=efficiency_maximization,
            candidate_methods=candidate_methods,
            num_folds=num_folds,
            auto_unlabeled_subsample_size=auto_unlabeled_subsample_size,
            selection_random_state=selection_random_state,
            isocal_backend=isocal_backend,
            isocal_max_depth=isocal_max_depth,
            isocal_min_child_weight=isocal_min_child_weight,
        )
        result = _mean_inference_result_from_prepared(
            prepared,
            alpha=alpha,
            alternative=alternative,
            reference=Y_arr,
        )
        if use_unweighted_causal_path:
            arm_results[arm] = result
            arm_means[arm] = float(result.pointestimate)
            arm_ses[arm] = float(result.se)
            arm_cis[arm] = (float(result.ci[0]), float(result.ci[1]))
            arm_influences[arm] = _aligned_wald_influence(labeled_mask, prepared, Y_arr.shape[0]).reshape(-1)
        else:
            pred_point_full, _ = _assemble_full_sample_prediction(
                labeled_mask,
                prepared.pred_labeled_point,
                prepared.pred_unlabeled_point,
                name=f"arm {arm!r} point",
            )
            pred_variance_full, used_unlabeled_point_fallback = _assemble_full_sample_prediction(
                labeled_mask,
                prepared.pred_labeled_variance,
                prepared.pred_unlabeled_variance,
                name=f"arm {arm!r} variance",
                fallback_unlabeled=prepared.pred_unlabeled_point,
            )
            arm_pointestimate, arm_influence = _causal_arm_pointestimate_and_influence(
                Y_arr,
                labeled_mask=labeled_mask,
                pred_point=pred_point_full,
                pred_variance=pred_variance_full,
                full_sample_weights=full_sample_weights,
            )
            arm_standard_error = float(np.sqrt(np.sum(arm_influence**2)))
            arm_lower, arm_upper = z_interval(
                np.array([arm_pointestimate], dtype=float),
                np.array([arm_standard_error], dtype=float),
                alpha=alpha,
                alternative=alternative,
            )

            arm_diagnostics = dict(result.diagnostics)
            arm_diagnostics["causal_weight_normalization"] = "global_full_sample"
            arm_diagnostics["causal_unlabeled_variance_prediction_source"] = (
                "point_predictions_fallback" if used_unlabeled_point_fallback else "variance_predictions"
            )
            arm_results[arm] = MeanInferenceResult(
                pointestimate=arm_pointestimate,
                se=arm_standard_error,
                ci=(float(arm_lower[0]), float(arm_upper[0])),
                method=result.method,
                selected_candidate=result.selected_candidate,
                selected_efficiency_maximization=result.selected_efficiency_maximization,
                efficiency_lambda=result.efficiency_lambda,
                inference=result.inference,
                diagnostics=arm_diagnostics,
                calibrator=result.calibrator,
            )
            arm_means[arm] = arm_pointestimate
            arm_ses[arm] = arm_standard_error
            arm_cis[arm] = (float(arm_lower[0]), float(arm_upper[0]))
            arm_influences[arm] = arm_influence.reshape(-1)

    ordered_arms = list(resolved_levels)
    influence_matrix = np.column_stack([arm_influences[arm] for arm in ordered_arms])
    covariance = influence_matrix.T @ influence_matrix
    control_idx = ordered_arms.index(resolved_control_arm)

    ate: dict[Any, float] = {}
    ate_ses: dict[Any, float] = {}
    ate_cis: dict[Any, tuple[float, float]] = {}
    for arm in ordered_arms:
        if arm == resolved_control_arm:
            continue
        arm_idx = ordered_arms.index(arm)
        estimate = arm_means[arm] - arm_means[resolved_control_arm]
        variance = covariance[arm_idx, arm_idx] + covariance[control_idx, control_idx] - 2.0 * covariance[
            arm_idx,
            control_idx,
        ]
        standard_error = float(np.sqrt(max(variance, 0.0)))
        lower, upper = z_interval(
            np.array([estimate], dtype=float),
            np.array([standard_error], dtype=float),
            alpha=alpha,
            alternative=alternative,
        )
        ate[arm] = float(estimate)
        ate_ses[arm] = standard_error
        ate_cis[arm] = (float(lower[0]), float(upper[0]))

    arm_wald_null, arm_wald_t_stat, arm_wald_p_value = _compute_wald_statistics(
        np.array([arm_means[arm] for arm in ordered_arms], dtype=float),
        np.array([arm_ses[arm] for arm in ordered_arms], dtype=float),
        null=0.0,
        alternative=alternative,
    )
    comparison_arms = [arm for arm in ordered_arms if arm != resolved_control_arm]
    ate_wald_null, ate_wald_t_stat, ate_wald_p_value = _compute_wald_statistics(
        np.array([ate[arm] for arm in comparison_arms], dtype=float),
        np.array([ate_ses[arm] for arm in comparison_arms], dtype=float),
        null=0.0,
        alternative=alternative,
    )

    diagnostics = {
        "inference": "wald",
        "treatment_levels": resolved_levels,
        "control_arm": resolved_control_arm,
        "arm_counts": {arm: int(np.sum(A_arr == arm)) for arm in resolved_levels},
        "arm_prediction_columns": arm_to_column,
        "per_arm": {arm: result.diagnostics for arm, result in arm_results.items()},
        "arm_wald_null": 0.0,
        "arm_wald_alternative": alternative,
        "arm_wald_t_statistic": {
            arm: float(stat) for arm, stat in zip(ordered_arms, arm_wald_t_stat)
        },
        "arm_wald_p_value": {
            arm: float(value) for arm, value in zip(ordered_arms, arm_wald_p_value)
        },
        "ate_wald_null": 0.0 if ate_wald_null.size > 0 else 0.0,
        "ate_wald_alternative": alternative,
        "ate_wald_t_statistic": {
            arm: float(stat) for arm, stat in zip(comparison_arms, ate_wald_t_stat)
        },
        "ate_wald_p_value": {
            arm: float(value) for arm, value in zip(comparison_arms, ate_wald_p_value)
        },
    }
    if not use_unweighted_causal_path:
        diagnostics["causal_weight_normalization"] = "global_full_sample"
    return CausalInferenceResult(
        arm_means=arm_means,
        arm_ses=arm_ses,
        arm_cis=arm_cis,
        ate=ate,
        ate_ses=ate_ses,
        ate_cis=ate_cis,
        control_arm=resolved_control_arm,
        treatment_levels=resolved_levels,
        arm_results=arm_results,
        diagnostics=diagnostics,
    )
