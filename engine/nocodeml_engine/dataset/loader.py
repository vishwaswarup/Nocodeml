"""Dataset ingestion (Section 0): CSV, Excel (.xlsx, first sheet) and Parquet."""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# format -> content type stored with the file. Old binary .xls is deliberately not supported (legacy parser, no
# safe way to bound it); "save as .xlsx or .csv" is the right advice.
SUPPORTED_FORMATS = {
    ".csv": "text/csv",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".parquet": "application/vnd.apache.parquet",
}
MAX_ROWS = 5_000_000                       # memory guard: a 100 MB Parquet file can hold far more rows than a CSV
MAX_XLSX_UNCOMPRESSED = 500 * 2**20        # a .xlsx is a zip: refuse "zip bombs" that expand enormously


class DatasetFormatError(ValueError):
    """The file is not a supported dataset format, or could not be read."""


def dataset_format(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise DatasetFormatError("Only .csv, .xlsx and .parquet files are supported."
                                 + (" Save old .xls workbooks as .xlsx or .csv first." if ext == ".xls" else ""))
    return ext


def parse_dataset(data: bytes, filename: str) -> pd.DataFrame:
    """Bytes of an uploaded file -> DataFrame (dates detected). Same input always gives the same frame, which the
    stored fingerprint relies on. Excel: the first sheet, first row as the header."""
    ext = dataset_format(filename)
    try:
        if ext == ".csv":
            df = pd.read_csv(io.BytesIO(data))
        elif ext == ".xlsx":
            df = _read_xlsx(data)
        else:
            df = _read_parquet(data)
    except DatasetFormatError:
        raise
    except ImportError as e:  # pragma: no cover - optional dependency missing on the server
        raise DatasetFormatError(f"This server can't read {ext} files ({e}).") from e
    except Exception as e:  # noqa: BLE001 - malformed upload: one clean message
        raise DatasetFormatError(f"Could not read the {ext[1:].upper()} file: {e}") from e
    if len(df) > MAX_ROWS:
        raise DatasetFormatError(f"The file has {len(df):,} rows; the limit is {MAX_ROWS:,}.")
    df = df.reset_index(drop=True)
    df.columns = [str(c) for c in df.columns]
    if df.columns.duplicated().any():
        dupes = sorted(set(df.columns[df.columns.duplicated()]))
        raise DatasetFormatError(f"Column names must be unique. Repeated: {', '.join(dupes[:5])}.")
    return infer_datetimes(df)


def _read_xlsx(data: bytes) -> pd.DataFrame:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            if sum(i.file_size for i in z.infolist()) > MAX_XLSX_UNCOMPRESSED:
                raise DatasetFormatError("This workbook expands to more than 500 MB, which is too large.")
    except zipfile.BadZipFile:
        raise DatasetFormatError("This isn't a valid .xlsx workbook (it may be an old .xls renamed).") from None
    return pd.read_excel(io.BytesIO(data), sheet_name=0, engine="openpyxl")


def _read_parquet(data: bytes) -> pd.DataFrame:
    import pyarrow.parquet as pq
    rows = pq.ParquetFile(io.BytesIO(data)).metadata.num_rows
    if rows > MAX_ROWS:
        raise DatasetFormatError(f"The file has {rows:,} rows; the limit is {MAX_ROWS:,}.")
    return pd.read_parquet(io.BytesIO(data), engine="pyarrow")


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
