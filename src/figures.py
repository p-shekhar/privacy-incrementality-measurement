from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 240,
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.show()


def plot_frontier(df: pd.DataFrame, path: Path) -> None:
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=False)
    for ax, (dataset, sdf) in zip(axes, df.groupby("dataset")):
        sdf = sdf.sort_values("q_true")
        ax.plot(sdf["q_true"], sdf["effect"], marker="o", label="clean point lift", color="#1f77b4")
        ax.plot(
            sdf["q_true"],
            sdf["pop_lower"],
            marker="o",
            label="population lower bound (signal loss only)",
            color="#9467bd",
        )
        ax.plot(
            sdf["q_true"],
            sdf["finite_lower"],
            marker="o",
            label="finite-sample lower bound (signal loss + noise)",
            color="#2ca02c",
        )
        ax.axhline(sdf["threshold"].iloc[0], color="black", linestyle="--", linewidth=1.2, label="business threshold")
        ax.set_title(dataset)
        ax.set_xlabel("true retention in stress layer")
        ax.set_ylabel("incremental conversion")
        ax.invert_xaxis()
        ax.grid(alpha=0.25)
    axes[0].legend(loc="best", frameon=False)
    savefig(path)


def plot_sample_complexity(df: pd.DataFrame, path: Path) -> None:
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for q, sdf in df[df["sigma_noise"] == 0.0].groupby("q_min"):
        axes[0].plot(sdf["n"], sdf["radius"], label=f"q={q:.2f}")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_title("Finite-sample radius")
    axes[0].set_xlabel("per-arm sample size")
    axes[0].set_ylabel("radius")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    for q, sdf in df[df["sigma_noise"] == 0.0].groupby("q_min"):
        axes[1].plot(sdf["n"], sdf["minimax_lb"], label=f"q={q:.2f}")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_title("Minimax lower-bound scale")
    axes[1].set_xlabel("per-arm sample size")
    axes[1].set_ylabel("lower-bound scale")
    axes[1].grid(alpha=0.25)
    savefig(path)


def plot_granularity(df: pd.DataFrame, path: Path) -> None:
    set_style()
    summary = (
        df.groupby(["dataset", "partition"], observed=True)
        .agg(
            n_cells=("n_cells", "max"),
            median_cell_rows=("median_cell_rows", "median"),
            share_suppressed=("share_suppressed", "mean"),
            share_population_certified=("share_population_certified_cells", "mean"),
            share_certified=("share_certified_cells", "mean"),
            share_segment_safe=("share_segment_safe", "mean"),
            penalty=("heterogeneity_penalty", "mean"),
        )
        .reset_index()
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for dataset, sdf in summary.groupby("dataset"):
        sdf = sdf.sort_values("n_cells")
        axes[0].plot(
            sdf["n_cells"],
            sdf["share_population_certified"],
            marker="o",
            label=f"{dataset}: population",
        )
        axes[0].plot(
            sdf["n_cells"],
            sdf["share_certified"],
            marker="s",
            linestyle="--",
            label=f"{dataset}: finite",
        )
        axes[1].plot(sdf["n_cells"], sdf["share_suppressed"], marker="o", label=dataset)
        axes[2].plot(sdf["n_cells"], sdf["penalty"], marker="o", label=dataset)
    axes[0].set_title("Certification falls after uncertainty")
    axes[0].set_ylabel("share of cells certified")
    axes[0].set_xlabel("reported cells")
    axes[0].set_xscale("log")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_title("Suppression rises with granularity")
    axes[1].set_ylabel("suppressed-cell share")
    axes[1].set_xlabel("reported cells")
    axes[1].set_xscale("log")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    axes[2].set_title("Hidden heterogeneity falls")
    axes[2].set_ylabel("H · diameter")
    axes[2].set_xlabel("reported cells")
    axes[2].set_xscale("log")
    axes[2].grid(alpha=0.25)
    axes[2].legend(frameon=False)
    savefig(path)


def plot_mde(df: pd.DataFrame, path: Path) -> None:
    set_style()
    plt.figure(figsize=(7, 4.5))
    for sigma, sdf in df.groupby("sigma_noise"):
        plt.plot(sdf["n_eff"], sdf["mde"], label=f"noise σ={sigma:g}")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("effective per-arm cell size")
    plt.ylabel("privacy-aware MDE")
    plt.title("Privacy noise lifts the detectable-effect floor")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    savefig(path)
