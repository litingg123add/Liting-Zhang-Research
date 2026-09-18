"""Utilities for chronological SCADA preprocessing.

The functions in this module deliberately avoid random train/test splitting.
For condition monitoring, preserving temporal order is important because random
splits can leak information from future operating states into training.
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd


def load_scada_data(path: str, timestamp_col: Optional[str] = None) -> pd.DataFrame:
    """Load SCADA data from CSV and optionally parse a timestamp column."""
    df = pd.read_csv(path)
    if timestamp_col and timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    return df


def basic_cleaning(
    df: pd.DataFrame,
    timestamp_col: Optional[str] = None,
    required_columns: Optional[Iterable[str]] = None,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """Apply conservative cleaning while keeping the original row semantics.

    Parameters
    ----------
    df:
        Input dataframe.
    timestamp_col:
        Optional timestamp column used for chronological sorting.
    required_columns:
        Only rows missing one of these columns are removed. If omitted, rows are
        not globally dropped just because an unrelated sensor value is missing.
    drop_duplicates:
        Remove duplicate rows when True.
    """
    data = df.copy()

    if timestamp_col and timestamp_col in data.columns:
        data[timestamp_col] = pd.to_datetime(data[timestamp_col], errors="coerce")
        data = data.dropna(subset=[timestamp_col])

    if required_columns:
        existing = [c for c in required_columns if c in data.columns]
        if existing:
            data = data.dropna(subset=existing)

    if drop_duplicates:
        data = data.drop_duplicates()

    if timestamp_col and timestamp_col in data.columns:
        data = data.sort_values(timestamp_col)

    return data.reset_index(drop=True)


def filter_numeric_range(
    df: pd.DataFrame,
    column: str,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
) -> pd.DataFrame:
    """Filter a numeric variable to a physically meaningful range."""
    if column not in df.columns:
        return df.copy()

    mask = pd.Series(True, index=df.index)
    if lower is not None:
        mask &= df[column] >= lower
    if upper is not None:
        mask &= df[column] <= upper
    return df.loc[mask].copy()


def chronological_split(df: pd.DataFrame, train_fraction: float = 0.7):
    """Split a dataframe in time order, not randomly."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    cut = int(len(df) * train_fraction)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()
