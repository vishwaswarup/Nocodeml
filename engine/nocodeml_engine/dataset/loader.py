"""Dataset ingestion (Section 0). V1 supports CSV only, per MVP scope."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class DatasetOverview:
    filename: str
    n_rows: int
    n_columns: int
    size_bytes: int
    memory_usage_bytes: int
    n_numerical: int
    n_categorical: int
    n_datetime: int
    missing_value_pct: float
    n_duplicate_rows: int


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"load_csv only supports .csv files, got: {path.suffix}")
    return infer_datetimes(pd.read_csv(path))


def infer_datetimes(df: pd.DataFrame, min_parse_rate: float = 0.95) -> pd.DataFrame:
    """Convert text columns that overwhelmingly parse as dates into datetime64."""
    df = df.copy()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        values = df[col].dropna()
        if values.empty:
            continue
        sample = values.astype(str).head(500)
        # Cheap guard: dates contain separators and digits.
        if not sample.str.contains(r"\d").all() or not sample.str.contains(r"[-/:]").all():
            continue
        parsed = pd.to_datetime(values.astype(str), errors="coerce", format="mixed")
        if parsed.notna().mean() >= min_parse_rate:
            df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")
    return df


def dataset_overview(df: pd.DataFrame, filename: str, size_bytes: int) -> DatasetOverview:
    numerical = df.select_dtypes(include="number").columns
    datetime_cols = df.select_dtypes(include="datetime").columns
    categorical = [
        c for c in df.columns if c not in numerical and c not in datetime_cols
    ]

    total_cells = df.shape[0] * df.shape[1] if df.shape[0] and df.shape[1] else 1

    return DatasetOverview(
        filename=filename,
        n_rows=df.shape[0],
        n_columns=df.shape[1],
        size_bytes=size_bytes,
        memory_usage_bytes=int(df.memory_usage(deep=True).sum()),
        n_numerical=len(numerical),
        n_categorical=len(categorical),
        n_datetime=len(datetime_cols),
        missing_value_pct=round(100 * df.isna().sum().sum() / total_cells, 2),
        n_duplicate_rows=int(df.duplicated().sum()),
    )
