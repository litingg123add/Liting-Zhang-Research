"""Maintenance-event-oriented evaluation helpers."""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd


def first_warning_before_event(
    timestamps: pd.Series,
    warnings: pd.Series,
    event_time,
    lookback_days: float = 30.0,
):
    """Return the earliest warning within a pre-event window."""
    event_time = pd.Timestamp(event_time)
    start = event_time - pd.Timedelta(days=lookback_days)

    mask = (
        (timestamps >= start)
        & (timestamps < event_time)
        & warnings.fillna(False)
    )
    if not mask.any():
        return None

    return pd.Timestamp(timestamps[mask].iloc[0])


def lead_time_days(first_warning, event_time) -> Optional[float]:
    """Compute lead time in days."""
    if first_warning is None:
        return None
    delta = pd.Timestamp(event_time) - pd.Timestamp(first_warning)
    return delta.total_seconds() / 86400.0


def event_window_labels(
    timestamps: pd.Series,
    event_times: Iterable,
    lookback_days: float = 30.0,
) -> pd.Series:
    """Label samples that fall inside any pre-maintenance event window."""
    ts = pd.to_datetime(timestamps)
    labels = pd.Series(False, index=timestamps.index)

    for event in event_times:
        event = pd.Timestamp(event)
        start = event - pd.Timedelta(days=lookback_days)
        labels |= (ts >= start) & (ts < event)

    return labels
