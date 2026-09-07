"""
SCADA data preprocessing module.

This module will contain reusable functions for:
- loading turbine operational data
- handling missing values
- filtering abnormal records
- preparing time-series inputs
"""

import pandas as pd


def load_scada_data(path: str) -> pd.DataFrame:
    """Load SCADA dataset from a CSV file."""
    return pd.read_csv(path)


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Basic data cleaning pipeline."""
    data = df.copy()
    data = data.dropna()
    return data


if __name__ == "__main__":
    print("SCADA preprocessing module initialized")
