"""Residual-based monitoring utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ewma(series: pd.Series, span: int = 24) -> pd.Series:
    """Exponentially weighted moving average."""
    if span <= 1:
        raise ValueError("span must be > 1")
    return series.ewm(span=span, adjust=False).mean()


def robust_zscore(
    series: pd.Series,
    window: int = 144,
    min_periods: int | None = None,
) -> pd.Series:
    """Rolling robust z-score using median and MAD.

    A 144-point window corresponds to one day for 10-minute SCADA data.
    """
    min_periods = min_periods or max(12, window // 4)

    median = series.rolling(window, min_periods=min_periods).median()
    mad = (series - median).abs().rolling(
        window, min_periods=min_periods
    ).median()

    scale = 1.4826 * mad.replace(0, np.nan)
    return (series - median) / scale


def build_monitoring_signal(
    residual: pd.Series,
    ewma_span: int = 24,
    z_window: int = 144,
) -> pd.DataFrame:
    """Create interpretable monitoring signals from model residuals."""
    out = pd.DataFrame(index=residual.index)
    out["residual"] = residual
    out["abs_residual"] = residual.abs()
    out["ewma_abs_residual"] = ewma(out["abs_residual"], span=ewma_span)
    out["robust_z"] = robust_zscore(out["ewma_abs_residual"], window=z_window)
    out["anomaly_score"] = out["robust_z"].clip(lower=0)
    return out


def persistent_warning(
    score: pd.Series,
    threshold: float = 3.0,
    min_consecutive: int = 6,
) -> pd.Series:
    """Flag warnings only after a threshold persists for several samples."""
    exceed = (score >= threshold).astype(int)
    run = exceed.groupby((exceed == 0).cumsum()).cumsum()
    return run >= min_consecutive
