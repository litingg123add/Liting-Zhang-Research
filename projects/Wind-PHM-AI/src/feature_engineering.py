"""Operating-condition feature engineering for wind-turbine SCADA data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_cyclic_direction_features(
    df: pd.DataFrame, wind_direction_col: str = "Wind Direction"
) -> pd.DataFrame:
    """Encode a direction in degrees using sine and cosine components."""
    data = df.copy()
    if wind_direction_col in data.columns:
        radians = np.deg2rad(data[wind_direction_col] % 360.0)
        data[f"{wind_direction_col}_sin"] = np.sin(radians)
        data[f"{wind_direction_col}_cos"] = np.cos(radians)
    return data


def add_temperature_difference(
    df: pd.DataFrame,
    component_temp_col: str,
    ambient_temp_col: str,
    output_col: str | None = None,
) -> pd.DataFrame:
    """Create a component-minus-ambient temperature feature."""
    data = df.copy()
    if component_temp_col in data.columns and ambient_temp_col in data.columns:
        name = output_col or f"{component_temp_col}_minus_{ambient_temp_col}"
        data[name] = data[component_temp_col] - data[ambient_temp_col]
    return data


def add_speed_ratio(
    df: pd.DataFrame,
    generator_speed_col: str = "Generator RPM",
    rotor_speed_col: str = "Rotor RPM",
    output_col: str = "generator_rotor_speed_ratio",
) -> pd.DataFrame:
    """Create a generator-to-rotor speed ratio with safe division."""
    data = df.copy()
    if generator_speed_col in data.columns and rotor_speed_col in data.columns:
        denominator = data[rotor_speed_col].replace(0, np.nan)
        data[output_col] = data[generator_speed_col] / denominator
    return data


def create_health_features(
    df: pd.DataFrame,
    wind_speed_col: str = "Wind Speed",
    active_power_col: str = "Active Power",
    ambient_temp_col: str = "Ambient Temperature",
    wind_direction_col: str = "Wind Direction",
) -> pd.DataFrame:
    """Add generic operating-condition features when source columns exist."""
    data = df.copy()

    if wind_speed_col in data.columns:
        data[f"{wind_speed_col}_sq"] = data[wind_speed_col] ** 2

    if active_power_col in data.columns:
        data[f"{active_power_col}_abs"] = data[active_power_col].abs()

    data = add_cyclic_direction_features(data, wind_direction_col)

    # Ambient temperature is intentionally retained as an explicit operating
    # condition rather than removed globally from the data.
    if ambient_temp_col in data.columns:
        data[f"{ambient_temp_col}_centered"] = (
            data[ambient_temp_col] - data[ambient_temp_col].median()
        )

    return data
