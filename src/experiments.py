from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from certification import certification_frontier, segment_certification_table
from data import MarketingDataset
from theory import (
    decision,
    fiber_projection,
    match_loss_band,
    minimax_sampling_lower_bound,
    privacy_mde,
    sample_complexity_radius,
    sample_complexity_sufficient_n,
    unresolved_error_lower_bound,
)


@dataclass(frozen=True)
class SignalLossStress:
    label: str
    match_rate: float
    attribution_retention: float
    identity_linkage: float
    q_half_width: float
    aggregation_threshold: int
    sigma_noise: float

    @property
    def q_true(self) -> float:
        return float(self.match_rate * self.attribution_retention * self.identity_linkage)

    @property
    def q_low(self) -> float:
        return float(max(0.02, self.q_true - self.q_half_width))

    @property
    def q_high(self) -> float:
        return float(min(1.0, self.q_true + self.q_half_width))


def signal_loss_stress_grid(dataset_key: str) -> list[SignalLossStress]:
    base_noise = 2.0 if dataset_key == "criteo" else 0.8
    rows = []
    for label, q_target, width, threshold, noise_multiplier in [
        ("near_clean", 0.95, 0.04, 50, 0.5),
        ("mild_loss", 0.80, 0.06, 100, 0.8),
        ("moderate_loss", 0.65, 0.08, 150, 1.0),
        ("severe_loss", 0.50, 0.10, 250, 1.25),
        ("very_severe_loss", 0.35, 0.12, 400, 1.5),
        ("extreme_loss", 0.25, 0.14, 600, 2.0),
    ]:
        component = q_target ** (1 / 3)
        rows.append(
            SignalLossStress(
                label=label,
                match_rate=component,
                attribution_retention=component,
                identity_linkage=component,
                q_half_width=width,
                aggregation_threshold=threshold,
                sigma_noise=base_noise * noise_multiplier,
            )
        )
    return rows


def _stable_jitter(key: str, scale: float) -> float:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    raw = int(digest[:8], 16) / 0xFFFFFFFF
    return scale * (2 * raw - 1)


def main_frontier_results(datasets: dict[str, MarketingDataset]) -> pd.DataFrame:
    rows = []
    for key, ds in datasets.items():
        # Binary conversion keeps the bounded-outcome theory exactly aligned.
        bound = 1.0
        threshold = 0.0005 if key == "criteo" else 0.001
        for stress in signal_loss_stress_grid(key):
            out = certification_frontier(
                ds.frame,
                ds.treatment_col,
                ds.conversion_col,
                ds.name,
                q_values=[stress.q_true],
                q_half_width=stress.q_half_width,
                threshold=threshold,
                alpha=0.05,
                bound=bound,
                sigma_noise=stress.sigma_noise,
                aggregation_threshold=stress.aggregation_threshold,
            )
            out["stress"] = stress.label
            out["stress_label"] = stress.label
            out["match_rate"] = stress.match_rate
            out["attribution_retention"] = stress.attribution_retention
            out["identity_linkage"] = stress.identity_linkage
            out["aggregation_threshold"] = stress.aggregation_threshold
            out["sigma_noise"] = stress.sigma_noise
            rows.append(out)
    return pd.concat(rows, ignore_index=True)


def sample_complexity_grid() -> pd.DataFrame:
    rows = []
    ns = np.unique(np.round(np.geomspace(500, 2_000_000, 80)).astype(int))
    for q in [0.9, 0.7, 0.5, 0.3, 0.15]:
        for sigma in [0.0, 1.0, 3.0]:
            for n in ns:
                rows.append(
                    {
                        "n": int(n),
                        "q_min": q,
                        "sigma_noise": sigma,
                        "radius": sample_complexity_radius(
                            n=n,
                            bound=1.0,
                            q_min=q,
                            n_segments=24,
                            alpha=0.05,
                            sigma_noise=sigma,
                        ),
                        "minimax_lb": minimax_sampling_lower_bound(n=n, q=q, bound=1.0),
                    }
                )
    return pd.DataFrame(rows)


def required_sample_table(epsilons: list[float] | None = None) -> pd.DataFrame:
    epsilons = epsilons or [0.02, 0.01, 0.005]
    rows = []
    for eps in epsilons:
        for q in [0.9, 0.7, 0.5, 0.3, 0.15]:
            rows.append(
                {
                    "epsilon": eps,
                    "q_min": q,
                    "required_n_no_noise": sample_complexity_sufficient_n(
                        epsilon=eps,
                        bound=1.0,
                        q_min=q,
                        n_segments=24,
                        alpha=0.05,
                        sigma_noise=0.0,
                    ),
                    "required_n_sigma_3": sample_complexity_sufficient_n(
                        epsilon=eps,
                        bound=1.0,
                        q_min=q,
                        n_segments=24,
                        alpha=0.05,
                        sigma_noise=3.0,
                    ),
                }
            )
    return pd.DataFrame(rows)


def _cell_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df[cols].astype(str).agg("|".join, axis=1)


def _nested_fine_cell_effects(
    df: pd.DataFrame,
    treatment_col: str,
    value_col: str,
    fine_cols: list[str],
    coarse_cols: list[str],
    min_rows: int = 80,
) -> pd.DataFrame:
    keyed = df.assign(_fine_cell=_cell_key(df, fine_cols), _cell=_cell_key(df, coarse_cols))
    rows = []
    for fine_cell, sdf in keyed.groupby("_fine_cell", observed=True):
        if len(sdf) < min_rows:
            continue
        treated = sdf[treatment_col].astype(int) == 1
        if treated.sum() == 0 or (~treated).sum() == 0:
            continue
        rows.append(
            {
                "_fine_cell": str(fine_cell),
                "segment": str(sdf["_cell"].iloc[0]),
                "fine_rows": int(len(sdf)),
                "fine_effect": float(sdf.loc[treated, value_col].mean() - sdf.loc[~treated, value_col].mean()),
            }
        )
    return pd.DataFrame(rows)


def _empirical_heterogeneity_penalties(
    df: pd.DataFrame,
    treatment_col: str,
    value_col: str,
    fine_cols: list[str],
    coarse_cols: list[str],
) -> pd.DataFrame:
    fine = _nested_fine_cell_effects(df, treatment_col, value_col, fine_cols, coarse_cols)
    if fine.empty:
        return pd.DataFrame(
            columns=[
                "segment",
                "diameter",
                "heterogeneity_lipschitz",
                "heterogeneity_penalty",
                "fine_cells_in_report_cell",
            ]
        )

    global_range = float(fine["fine_effect"].max() - fine["fine_effect"].min())
    heterogeneity_lipschitz = max(global_range, 1e-6)
    rows = []
    for segment, sdf in fine.groupby("segment", observed=True):
        weights = sdf["fine_rows"].to_numpy(dtype=float)
        effects = sdf["fine_effect"].to_numpy(dtype=float)
        avg_effect = float(np.average(effects, weights=weights))
        penalty = float(np.max(np.abs(effects - avg_effect))) if len(effects) else 0.0
        rows.append(
            {
                "segment": str(segment),
                "diameter": penalty / heterogeneity_lipschitz,
                "heterogeneity_lipschitz": heterogeneity_lipschitz,
                "heterogeneity_penalty": penalty,
                "fine_cells_in_report_cell": int(len(sdf)),
            }
        )
    return pd.DataFrame(rows)


def _expected_decision(lower: float, upper: float, threshold: float) -> str:
    if pd.isna(lower) or pd.isna(upper):
        return "unresolved"
    return decision(float(lower), float(upper), float(threshold))


def granularity_results(dataset: MarketingDataset) -> pd.DataFrame:
    df = dataset.frame.copy()
    if dataset.name.startswith("Hillstrom"):
        partitions = [
            ("coarse_channel", ["channel_bucket"]),
            ("medium_channel_zip", ["channel_bucket", "zip_bucket"]),
            ("fine_channel_zip_history", ["channel_bucket", "zip_bucket", "history_bucket"]),
            ("very_fine_plus_recency", ["channel_bucket", "zip_bucket", "history_bucket", "recency_bucket"]),
        ]
        threshold = 0.001
        value_col = dataset.conversion_col
    else:
        partitions = [
            ("coarse_f0", ["f0_bucket"]),
            ("medium_f0_f1", ["f0_bucket", "f1_bucket"]),
            ("fine_f0_f1_f2", ["f0_bucket", "f1_bucket", "f2_bucket"]),
            ("very_fine_plus_exposure", ["f0_bucket", "f1_bucket", "f2_bucket", "exposed_logged"]),
        ]
        threshold = 0.0005
        value_col = dataset.conversion_col

    rows = []
    fine_cols = partitions[-1][1]

    def stress_for_cell(segment_key: str, rows_in_cell: int) -> dict[str, float]:
        size_adjustment = float(np.clip(np.log(max(rows_in_cell, 1) / 500.0) / 20.0, -0.10, 0.10))
        q_true = float(np.clip(0.70 + size_adjustment + _stable_jitter(segment_key, 0.05), 0.35, 0.92))
        width = float(0.08 + max(0.0, 250 - rows_in_cell) / 2500.0)
        return {
            "q_true": q_true,
            "q_low": max(0.02, q_true - width),
            "q_high": min(1.0, q_true + width),
            "sigma_noise": 1.0 + max(0.0, 200 - rows_in_cell) / 200.0,
        }

    for level, cols in partitions:
        cell_key = _cell_key(df, cols)
        tmp = df.assign(_cell=cell_key)
        cert = segment_certification_table(
            tmp,
            dataset.treatment_col,
            value_col,
            "_cell",
            dataset.name,
            q_true=0.7,
            q_low=0.6,
            q_high=0.8,
            threshold=threshold,
            alpha=0.05,
            bound=1.0,
            min_count=100,
            sigma_noise=1.0,
            stress_fn=stress_for_cell,
        )
        if cert.empty:
            continue
        penalties = _empirical_heterogeneity_penalties(
            df,
            dataset.treatment_col,
            value_col,
            fine_cols,
            cols,
        )
        cert = cert.merge(penalties, on="segment", how="left")
        cert["diameter"] = cert["diameter"].fillna(0.0)
        cert["heterogeneity_lipschitz"] = cert["heterogeneity_lipschitz"].fillna(0.0)
        cert["heterogeneity_penalty"] = cert["heterogeneity_penalty"].fillna(0.0)
        cert["fine_cells_in_report_cell"] = cert["fine_cells_in_report_cell"].fillna(0).astype(int)
        cert["segment_safe_lower"] = cert["finite_lower"] - cert["heterogeneity_penalty"]
        cert["segment_safe_upper"] = cert["finite_upper"] + cert["heterogeneity_penalty"]
        cert["segment_safe_decision"] = cert.apply(
            lambda row: _expected_decision(row["segment_safe_lower"], row["segment_safe_upper"], threshold),
            axis=1,
        )
        cert["partition"] = level
        cert["n_cells"] = cert["segment"].nunique()
        cert["median_cell_rows"] = cert["rows"].median()
        cert["share_suppressed"] = cert["suppressed"].mean()
        cert["share_population_certified_cells"] = (cert["pop_decision"] == "certify").mean()
        cert["share_population_rejected_cells"] = (cert["pop_decision"] == "reject").mean()
        cert["share_population_unresolved_cells"] = (cert["pop_decision"] == "unresolved").mean()
        cert["share_certified_cells"] = (cert["finite_decision"] == "certify").mean()
        cert["share_segment_safe"] = (cert["segment_safe_decision"] == "certify").mean()
        cert["share_segment_safe_rejected"] = (cert["segment_safe_decision"] == "reject").mean()
        cert["share_segment_safe_unresolved"] = (cert["segment_safe_decision"] == "unresolved").mean()
        rows.append(cert)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def algorithm_validity_checks(frontier: pd.DataFrame) -> pd.DataFrame:
    checked = frontier.copy()
    checked["expected_pop_decision"] = checked.apply(
        lambda row: _expected_decision(row["pop_lower"], row["pop_upper"], row["threshold"]),
        axis=1,
    )
    checked["expected_finite_decision"] = checked.apply(
        lambda row: _expected_decision(row["finite_lower"], row["finite_upper"], row["threshold"]),
        axis=1,
    )
    checked["pop_rule_consistent"] = checked["pop_decision"].eq(checked["expected_pop_decision"])
    checked["finite_rule_consistent"] = checked["finite_decision"].eq(checked["expected_finite_decision"])
    checked["suppressed_unresolved"] = np.where(
        checked["suppressed"],
        checked["pop_decision"].eq("unresolved") & checked["finite_decision"].eq("unresolved"),
        True,
    )
    not_suppressed = ~checked["suppressed"]
    checked["finite_contains_population_band"] = True
    checked.loc[not_suppressed, "finite_contains_population_band"] = (
        checked.loc[not_suppressed, "finite_lower"].le(checked.loc[not_suppressed, "pop_lower"] + 1e-12)
        & checked.loc[not_suppressed, "finite_upper"].ge(checked.loc[not_suppressed, "pop_upper"] - 1e-12)
    )
    valid_states = {"certify", "reject", "unresolved"}
    return (
        checked.groupby("dataset")
        .agg(
            rows=("dataset", "size"),
            valid_pop_states=("pop_decision", lambda s: set(s).issubset(valid_states)),
            valid_finite_states=("finite_decision", lambda s: set(s).issubset(valid_states)),
            pop_rule_consistent=("pop_rule_consistent", "all"),
            finite_rule_consistent=("finite_rule_consistent", "all"),
            suppressed_unresolved=("suppressed_unresolved", "all"),
            finite_contains_population_band=("finite_contains_population_band", "all"),
            n_certify=("finite_decision", lambda s: (s == "certify").sum()),
            n_reject=("finite_decision", lambda s: (s == "reject").sum()),
            n_unresolved=("finite_decision", lambda s: (s == "unresolved").sum()),
        )
        .reset_index()
    )


def fiber_unresolved_example() -> pd.DataFrame:
    degraded_treat = 0.12
    degraded_control = 0.08
    q_low, q_high = 0.5, 1.0
    values = []
    for qt in np.linspace(q_low, q_high, 101):
        for qc in np.linspace(q_low, q_high, 101):
            mu_t = degraded_treat / qt
            mu_c = degraded_control / qc
            if 0 <= mu_t <= 1 and 0 <= mu_c <= 1:
                values.append(mu_t - mu_c)
    lower, upper = fiber_projection(np.array(values))
    threshold_gap = 0.15 * (upper - lower)
    thresholds = [lower - threshold_gap, (lower + upper) / 2, upper + threshold_gap]
    rows = []
    for threshold in thresholds:
        rows.append(
            {
                "degraded_treat": degraded_treat,
                "degraded_control": degraded_control,
                "q_low": q_low,
                "q_high": q_high,
                "fiber_lower": lower,
                "fiber_upper": upper,
                "threshold": threshold,
                "threshold_position": "inside" if lower <= threshold <= upper else "outside",
                "unresolved_tv0_error_lb": unresolved_error_lower_bound(0.0),
            }
        )
    return pd.DataFrame(rows)


def privacy_mde_grid() -> pd.DataFrame:
    rows = []
    for n in np.unique(np.round(np.geomspace(100, 300_000, 80)).astype(int)):
        for sigma_noise in [0.0, 1.0, 3.0, 10.0]:
            rows.append(
                {
                    "n_eff": int(n),
                    "sigma_noise": sigma_noise,
                    "mde": privacy_mde(
                        sigma=0.5,
                        n_eff=int(n),
                        n_treat=int(n),
                        n_control=int(n),
                        alpha=0.05,
                        beta=0.2,
                        sigma_noise=sigma_noise,
                    ),
                }
            )
    return pd.DataFrame(rows)


def heterogeneous_signal_loss_reversal() -> pd.DataFrame:
    threshold = 0.25
    q_low, q_high = 0.5, 1.0
    worlds = [
        {
            "world": "u",
            "segment_1_clean_treat": 1.0,
            "segment_2_clean_treat": 0.0,
            "segment_1_retention": 0.5,
            "segment_2_retention": 1.0,
        },
        {
            "world": "u_prime",
            "segment_1_clean_treat": 0.0,
            "segment_2_clean_treat": 1.0,
            "segment_1_retention": 1.0,
            "segment_2_retention": 0.5,
        },
    ]
    rows = []
    for row in worlds:
        row = dict(row)
        row["segment_1_degraded_treat"] = row["segment_1_clean_treat"] * row["segment_1_retention"]
        row["segment_2_degraded_treat"] = row["segment_2_clean_treat"] * row["segment_2_retention"]
        row["aggregate_clean_treat"] = 0.5 * (row["segment_1_clean_treat"] + row["segment_2_clean_treat"])
        row["aggregate_degraded_treat"] = 0.5 * (
            row["segment_1_degraded_treat"] + row["segment_2_degraded_treat"]
        )
        for segment in ["segment_1", "segment_2"]:
            lower, upper = match_loss_band(
                degraded_treat=row[f"{segment}_degraded_treat"],
                degraded_control=0.0,
                q_treat_low=q_low,
                q_treat_high=q_high,
                q_control_low=q_low,
                q_control_high=q_high,
                bound=1.0,
            )
            row[f"{segment}_lower"] = lower
            row[f"{segment}_upper"] = upper
            row[f"{segment}_decision"] = decision(lower, upper, threshold)
        certified = [segment for segment in ["segment_1", "segment_2"] if row[f"{segment}_decision"] == "certify"]
        row["threshold"] = threshold
        row["certified_segment"] = certified[0] if len(certified) == 1 else "none"
        rows.append(row)
    return pd.DataFrame(rows)


def theory_artifact_manifest() -> pd.DataFrame:
    rows = [
        ("Theorem 5.1", "Sharp privacy-loss certification frontier", "01_main_certification_frontier.ipynb", "main", "privacy_frontier_main.png"),
        ("Section 5.2", "Finite-sample simultaneous certification logic", "01_main_certification_frontier.ipynb", "main", "finite_sample_coverage_summary.csv"),
        ("Proposition 5.2", "Sample complexity under privacy signal loss", "02_main_sample_complexity_minimax.ipynb", "main", "sample_complexity_main.png"),
        ("Theorem 5.3", "Minimax lower bound for signal-loss measurement", "02_main_sample_complexity_minimax.ipynb", "main", "sample_complexity_main.png"),
        ("Section 5.5", "Reporting granularity and segment safety", "03_main_granularity_segment_safety.ipynb", "main", "granularity_tradeoff_main.png"),
        ("Lemma B.1", "Fiber projection is the sharp identified set", "04_appendix_fiber_unresolved_diagnostics.ipynb", "appendix", "fiber_projection_appendix.png"),
        ("Theorem 5.1", "Unresolved-region lower-bound visualization", "04_appendix_fiber_unresolved_diagnostics.ipynb", "appendix", "fiber_projection_appendix.png"),
        ("Proposition B.2", "Privacy-aware MDE derivation", "05_appendix_privacy_mde_noise.ipynb", "appendix", "privacy_mde_appendix.png"),
        ("Appendix B.3", "Heterogeneous signal-loss reversal example", "06_appendix_heterogeneous_signal_loss.ipynb", "appendix", "heterogeneous_reversal_appendix.png"),
        ("Algorithm 1", "Decision-state consistency check", "07_appendix_algorithm_manifest.ipynb", "appendix", "algorithm_validity_check.csv"),
    ]
    return pd.DataFrame(rows, columns=["result", "claim", "experiment", "paper_location", "primary_artifact"])
