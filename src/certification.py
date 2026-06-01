from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from theory import decision, finite_sample_band, match_loss_band


def arm_means(df: pd.DataFrame, treatment_col: str, value_col: str) -> dict[str, float | int]:
    treat = df[treatment_col].astype(int) == 1
    control = ~treat
    mu_t = float(df.loc[treat, value_col].mean())
    mu_c = float(df.loc[control, value_col].mean())
    return {
        "n_treat": int(treat.sum()),
        "n_control": int(control.sum()),
        "mu_treat": mu_t,
        "mu_control": mu_c,
        "effect": mu_t - mu_c,
    }


def certification_for_frame(
    df: pd.DataFrame,
    treatment_col: str,
    value_col: str,
    q_true: float,
    q_low: float,
    q_high: float,
    threshold: float,
    alpha: float,
    bound: float,
    n_segments: int = 1,
    sigma_noise: float = 0.0,
    suppressed: bool = False,
) -> dict[str, float | int | str]:
    stats = arm_means(df, treatment_col, value_col)
    if suppressed:
        return {
            **stats,
            "q_true": q_true,
            "q_low": q_low,
            "q_high": q_high,
            "threshold": threshold,
            "degraded_treat": np.nan,
            "degraded_control": np.nan,
            "pop_lower": np.nan,
            "pop_upper": np.nan,
            "finite_lower": np.nan,
            "finite_upper": np.nan,
            "pop_decision": "unresolved",
            "finite_decision": "unresolved",
            "pop_width": np.nan,
            "finite_width": np.nan,
            "suppressed": True,
        }
    degraded_treat = q_true * float(stats["mu_treat"])
    degraded_control = q_true * float(stats["mu_control"])
    pop_l, pop_u = match_loss_band(
        degraded_treat,
        degraded_control,
        q_low,
        q_high,
        q_low,
        q_high,
        bound,
    )
    fs_l, fs_u = finite_sample_band(
        degraded_treat,
        degraded_control,
        int(stats["n_treat"]),
        int(stats["n_control"]),
        q_low,
        q_high,
        bound,
        alpha,
        n_segments=n_segments,
        sigma_noise=sigma_noise,
    )
    return {
        **stats,
        "q_true": q_true,
        "q_low": q_low,
        "q_high": q_high,
        "threshold": threshold,
        "degraded_treat": degraded_treat,
        "degraded_control": degraded_control,
        "pop_lower": pop_l,
        "pop_upper": pop_u,
        "finite_lower": fs_l,
        "finite_upper": fs_u,
        "pop_decision": decision(pop_l, pop_u, threshold),
        "finite_decision": decision(fs_l, fs_u, threshold),
        "pop_width": pop_u - pop_l,
        "finite_width": fs_u - fs_l,
        "suppressed": False,
    }


def certification_frontier(
    df: pd.DataFrame,
    treatment_col: str,
    value_col: str,
    dataset: str,
    q_values: list[float],
    q_half_width: float,
    threshold: float,
    alpha: float,
    bound: float,
    sigma_noise: float = 0.0,
    aggregation_threshold: int | None = None,
) -> pd.DataFrame:
    rows = []
    suppressed = bool(aggregation_threshold is not None and len(df) < aggregation_threshold)
    for q in q_values:
        q_low = max(0.02, q - q_half_width)
        q_high = min(1.0, q + q_half_width)
        row = certification_for_frame(
            df,
            treatment_col,
            value_col,
            q_true=q,
            q_low=q_low,
            q_high=q_high,
            threshold=threshold,
            alpha=alpha,
            bound=bound,
            sigma_noise=sigma_noise,
            suppressed=suppressed,
        )
        row["dataset"] = dataset
        if aggregation_threshold is not None:
            row["aggregation_threshold"] = int(aggregation_threshold)
        rows.append(row)
    return pd.DataFrame(rows)


def segment_certification_table(
    df: pd.DataFrame,
    treatment_col: str,
    value_col: str,
    segment_col: str,
    dataset: str,
    q_true: float,
    q_low: float,
    q_high: float,
    threshold: float,
    alpha: float,
    bound: float,
    min_count: int = 200,
    sigma_noise: float = 0.0,
    stress_fn: Callable[[str, int], dict[str, float]] | None = None,
) -> pd.DataFrame:
    counts = df.groupby(segment_col, observed=True).size()
    reported = counts.index
    n_segments = len(reported)
    rows = []
    for segment, sdf in df.groupby(segment_col, observed=True):
        segment_key = str(segment)
        stress = stress_fn(segment_key, int(len(sdf))) if stress_fn else {}
        local_q_true = float(stress.get("q_true", q_true))
        local_q_low = float(stress.get("q_low", q_low))
        local_q_high = float(stress.get("q_high", q_high))
        local_sigma_noise = float(stress.get("sigma_noise", sigma_noise))
        local_suppressed = int(len(sdf)) < min_count
        row = certification_for_frame(
            sdf,
            treatment_col,
            value_col,
            q_true=local_q_true,
            q_low=local_q_low,
            q_high=local_q_high,
            threshold=threshold,
            alpha=alpha,
            bound=bound,
            n_segments=n_segments,
            sigma_noise=local_sigma_noise,
            suppressed=local_suppressed,
        )
        row["dataset"] = dataset
        row["segment_col"] = segment_col
        row["segment"] = segment_key
        row["rows"] = int(len(sdf))
        row["aggregation_threshold"] = int(min_count)
        row["sigma_noise"] = local_sigma_noise
        rows.append(row)
    return pd.DataFrame(rows)


def simulate_finite_sample_coverage(
    mu_treat: float,
    mu_control: float,
    n_treat: int,
    n_control: int,
    q_true: float,
    q_low: float,
    q_high: float,
    bound: float,
    alpha: float,
    reps: int,
    seed: int = 11,
    sigma_noise: float = 0.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    true_effect = mu_treat - mu_control
    for rep in range(reps):
        yt = rng.binomial(1, min(max(mu_treat, 0), 1), n_treat) * bound
        yc = rng.binomial(1, min(max(mu_control, 0), 1), n_control) * bound
        rt = rng.binomial(1, q_true, n_treat)
        rc = rng.binomial(1, q_true, n_control)
        degraded_t = float((yt * rt).mean())
        degraded_c = float((yc * rc).mean())
        if sigma_noise:
            degraded_t += float(rng.normal(0, sigma_noise) / n_treat)
            degraded_c += float(rng.normal(0, sigma_noise) / n_control)
            degraded_t = max(0, degraded_t)
            degraded_c = max(0, degraded_c)
        lb, ub = finite_sample_band(
            degraded_t,
            degraded_c,
            n_treat,
            n_control,
            q_low,
            q_high,
            bound,
            alpha,
            n_segments=1,
            sigma_noise=sigma_noise,
        )
        rows.append({"rep": rep, "lower": lb, "upper": ub, "covered": lb <= true_effect <= ub, "width": ub - lb})
    return pd.DataFrame(rows)
