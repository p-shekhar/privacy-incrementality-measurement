from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketingDataset:
    name: str
    frame: pd.DataFrame
    treatment_col: str
    conversion_col: str
    value_col: str
    segment_cols: tuple[str, ...]
    notes: str


def _qcut_labels(series: pd.Series, q: int, prefix: str) -> pd.Series:
    ranks = series.rank(method="first")
    bins = pd.qcut(ranks, q=q, labels=False, duplicates="drop")
    return (prefix + bins.astype(str)).astype("category")


def load_criteo(data_dir: Path, sample_n: int = 2_000_000, seed: int = 7) -> MarketingDataset:
    path = data_dir / "criteo" / "criteo-research-uplift-v2.1.csv.gz"
    usecols = ["f0", "f1", "f2", "treatment", "conversion", "visit", "exposure"]
    rng = np.random.default_rng(seed)
    pieces: list[pd.DataFrame] = []
    chunksize = 250_000
    per_chunk = max(2_000, sample_n // 50)
    for chunk_id, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=chunksize)):
        draw = min(len(chunk), per_chunk)
        pieces.append(chunk.sample(draw, random_state=int(rng.integers(0, 2**31 - 1))))
    df = pd.concat(pieces, ignore_index=True)
    if len(df) > sample_n:
        df = df.sample(sample_n, random_state=seed).reset_index(drop=True)
    df["treatment"] = df["treatment"].astype(int)
    df["conversion"] = df["conversion"].astype(float)
    df["value"] = df["conversion"]
    df["visit"] = df["visit"].astype(float)
    df["f0_bucket"] = _qcut_labels(df["f0"], 4, "f0_q")
    df["f1_bucket"] = _qcut_labels(df["f1"], 4, "f1_q")
    df["f2_bucket"] = _qcut_labels(df["f2"], 4, "f2_q")
    df["exposed_logged"] = np.where(df["exposure"].astype(int) == 1, "logged_exposed", "not_logged_exposed")
    return MarketingDataset(
        name="Criteo Uplift",
        frame=df,
        treatment_col="treatment",
        conversion_col="conversion",
        value_col="value",
        segment_cols=("f0_bucket", "f1_bucket", "f2_bucket", "exposed_logged"),
        notes=f"Loaded {len(df):,} rows from Criteo uplift with binary conversion as value.",
    )


def load_hillstrom(data_dir: Path) -> MarketingDataset:
    archive = data_dir / "hillstrom" / "archive.zip"
    with ZipFile(archive) as zf:
        [name] = zf.namelist()
        with zf.open(name) as fh:
            df = pd.read_csv(fh)
    df["treatment"] = (df["segment"] != "No E-Mail").astype(int)
    df["conversion"] = df["conversion"].astype(float)
    df["value"] = df["spend"].astype(float)
    df["history_bucket"] = df["history_segment"].astype("category")
    df["zip_bucket"] = df["zip_code"].astype("category")
    df["channel_bucket"] = df["channel"].astype("category")
    df["recency_bucket"] = _qcut_labels(df["recency"].astype(float), 4, "rec_q")
    return MarketingDataset(
        name="Hillstrom Email",
        frame=df,
        treatment_col="treatment",
        conversion_col="conversion",
        value_col="value",
        segment_cols=("history_bucket", "zip_bucket", "channel_bucket", "recency_bucket", "newbie"),
        notes=f"Loaded {len(df):,} rows from Hillstrom with email vs no-email treatment.",
    )


def load_all(data_dir: Path) -> dict[str, MarketingDataset]:
    criteo = load_criteo(data_dir)
    hillstrom = load_hillstrom(data_dir)
    return {"criteo": criteo, "hillstrom": hillstrom}


def dataset_summary(dataset: MarketingDataset) -> dict[str, float | str | int]:
    df = dataset.frame
    treated = df[dataset.treatment_col].astype(int)
    y = df[dataset.conversion_col].astype(float)
    value = df[dataset.value_col].astype(float)
    return {
        "dataset": dataset.name,
        "rows": int(len(df)),
        "treated_share": float(treated.mean()),
        "conversion_rate": float(y.mean()),
        "mean_value": float(value.mean()),
        "segment_columns": ", ".join(map(str, dataset.segment_cols)),
        "notes": dataset.notes,
    }
