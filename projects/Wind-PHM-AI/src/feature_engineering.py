"""
Feature engineering for wind turbine health monitoring.

Future extensions:
- operating condition normalization
- thermal health indicators
- physics-informed features
- degradation indicators
"""

import pandas as pd


def create_health_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate health monitoring features."""
    features = df.copy()
    return features
