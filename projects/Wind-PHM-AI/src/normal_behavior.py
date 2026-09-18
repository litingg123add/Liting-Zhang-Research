"""Normal-behaviour regression for operating-condition-aware monitoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - optional dependency
    XGBRegressor = None


@dataclass
class ModelReport:
    mae: float
    rmse: float


class NormalBehaviorModel:
    """Estimate the expected healthy value of a monitored SCADA variable."""

    def __init__(
        self,
        feature_columns: List[str],
        target_column: str,
        prefer_xgboost: bool = True,
        random_state: int = 42,
    ):
        self.feature_columns = feature_columns
        self.target_column = target_column

        if prefer_xgboost and XGBRegressor is not None:
            self.model = XGBRegressor(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=random_state,
                n_jobs=4,
            )
        else:
            self.model = HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=300,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                random_state=random_state,
            )

    def _xy(self, df: pd.DataFrame):
        cols = self.feature_columns + [self.target_column]
        clean = df[cols].dropna()
        return clean[self.feature_columns], clean[self.target_column], clean.index

    def fit(self, df: pd.DataFrame):
        X, y, _ = self._xy(df)
        self.model.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        X = df[self.feature_columns]
        valid = X.dropna()
        pred = pd.Series(np.nan, index=df.index, name="expected")
        if len(valid):
            pred.loc[valid.index] = self.model.predict(valid)
        return pred

    def evaluate(self, df: pd.DataFrame) -> ModelReport:
        X, y, _ = self._xy(df)
        pred = self.model.predict(X)
        return ModelReport(
            mae=float(mean_absolute_error(y, pred)),
            rmse=float(mean_squared_error(y, pred) ** 0.5),
        )

    def residuals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["expected"] = self.predict(out)
        out["residual"] = out[self.target_column] - out["expected"]
        out["abs_residual"] = out["residual"].abs()
        return out
