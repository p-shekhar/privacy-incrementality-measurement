from __future__ import annotations

from math import log, sqrt
from statistics import NormalDist

import numpy as np


def match_loss_band(
    degraded_treat: float,
    degraded_control: float,
    q_treat_low: float,
    q_treat_high: float,
    q_control_low: float,
    q_control_high: float,
    bound: float,
) -> tuple[float, float]:
    lower = degraded_treat / q_treat_high - min(bound, degraded_control / q_control_low)
    upper = min(bound, degraded_treat / q_treat_low) - degraded_control / q_control_high
    return float(lower), float(upper)


def confidence_radius(
    n: int,
    bound: float,
    alpha: float,
    sigma_noise: float = 0.0,
) -> float:
    if n <= 0:
        return float("inf")
    sampling = bound * sqrt(log(2 / alpha) / (2 * n))
    privacy = (sigma_noise / n) * sqrt(2 * log(2 / alpha)) if sigma_noise else 0.0
    return float(sampling + privacy)


def finite_sample_band(
    degraded_treat: float,
    degraded_control: float,
    n_treat: int,
    n_control: int,
    q_low: float,
    q_high: float,
    bound: float,
    alpha: float,
    n_segments: int = 1,
    sigma_noise: float = 0.0,
) -> tuple[float, float]:
    alpha_arm = alpha / max(2 * n_segments, 1)
    rt = confidence_radius(n_treat, bound, alpha_arm, sigma_noise)
    rc = confidence_radius(n_control, bound, alpha_arm, sigma_noise)
    treat_low = max(0.0, degraded_treat - rt)
    treat_high = min(bound, degraded_treat + rt)
    control_low = max(0.0, degraded_control - rc)
    control_high = min(bound, degraded_control + rc)
    lower = treat_low / q_high - min(bound, control_high / q_low)
    upper = min(bound, treat_high / q_low) - control_low / q_high
    return float(lower), float(upper)


def decision(lower: float, upper: float, threshold: float) -> str:
    if lower > threshold:
        return "certify"
    if upper <= threshold:
        return "reject"
    return "unresolved"


def sample_complexity_radius(
    n: int,
    bound: float,
    q_min: float,
    n_segments: int,
    alpha: float,
    sigma_noise: float = 0.0,
) -> float:
    sampling = (2 * bound / q_min) * sqrt(log(4 * n_segments / alpha) / (2 * n))
    privacy = (2 * sigma_noise / (q_min * n)) * sqrt(2 * log(4 * n_segments / alpha))
    return float(sampling + privacy)


def sample_complexity_sufficient_n(
    epsilon: float,
    bound: float,
    q_min: float,
    n_segments: int,
    alpha: float,
    sigma_noise: float = 0.0,
) -> int:
    term_sampling = (8 * bound**2 / (q_min**2 * epsilon**2)) * log(4 * n_segments / alpha)
    term_privacy = 0.0
    if sigma_noise:
        term_privacy = (4 * sigma_noise / (q_min * epsilon)) * sqrt(2 * log(4 * n_segments / alpha))
    return int(np.ceil(max(term_sampling, term_privacy, 1)))


def minimax_sampling_lower_bound(n: int, q: float, bound: float, constant: float = 0.08) -> float:
    return float(constant * bound / sqrt(max(q * n, 1e-12)))


def retention_identification_floor(q_low: float, q_high: float, bound: float) -> float:
    return float((bound / 4.0) * (1.0 - q_low / q_high))


def privacy_variance(
    randomization_variance: float,
    n_treat: int,
    n_control: int,
    sigma_noise: float,
) -> float:
    return float(randomization_variance + sigma_noise**2 * (1 / n_treat**2 + 1 / n_control**2))


def privacy_mde(
    sigma: float,
    n_eff: int,
    n_treat: int,
    n_control: int,
    alpha: float = 0.05,
    beta: float = 0.2,
    sigma_noise: float = 0.0,
) -> float:
    z_alpha = NormalDist().inv_cdf(1 - alpha / 2)
    z_beta = NormalDist().inv_cdf(1 - beta)
    se = sqrt(2 * sigma**2 / max(n_eff, 1) + sigma_noise**2 * (1 / n_treat**2 + 1 / n_control**2))
    return float((z_alpha + z_beta) * se)


def granularity_segment_lower_bound(cell_lower_bound: float, heterogeneity_lipschitz: float, diameter: float) -> float:
    return float(cell_lower_bound - heterogeneity_lipschitz * diameter)


def unresolved_error_lower_bound(total_variation: float) -> float:
    return float((1 - total_variation) / 2)


def fiber_projection(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    return float(values.min()), float(values.max())

