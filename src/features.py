"""Deterministic calendar designs; learned scaling is fitted inside model pipelines."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CalendarFeatures(TransformerMixin, BaseEstimator):
    """Explicit allowlist prevents target-derived columns entering model inputs.

    Category ranges are defined by calendar semantics, not learned from test data.
    Indicator coding omits the first level. Interaction coding has redundant
    columns by design; Ridge handles the resulting non-unique unpenalized design.
    """
    def __init__(self, encoding='onehot', weather=True):
        self.encoding = encoding
        self.weather = weather

    def fit(self, X, y=None):
        self.feature_names_out_ = np.array(self._frame(X).columns, dtype=object)
        return self

    def _frame(self, X):
        if self.encoding not in {'ordinal', 'cyclic', 'onehot', 'interaction'}:
            raise ValueError('Unknown time encoding')
        fields = {k: X[k].to_numpy(dtype=float) for k in ['yr', 'holiday', 'workingday']}
        for column, period, first in [('hr', 24, 0), ('weekday', 7, 0), ('mnth', 12, 1)]:
            values = X[column].to_numpy()
            if self.encoding == 'ordinal':
                fields[column] = values.astype(float)
            elif self.encoding == 'cyclic':
                angle = 2 * np.pi * (values - first) / period
                fields[column + '_sin'] = np.sin(angle)
                fields[column + '_cos'] = np.cos(angle)
            else:
                for level in range(first + 1, first + period):
                    fields[f'{column}_{level}'] = (values == level).astype(float)
        if self.encoding == 'interaction':
            pair = X.hr.to_numpy() + 24 * X.workingday.to_numpy()
            for level in range(1, 48):
                fields[f'hour_workday_{level}'] = (pair == level).astype(float)
        if self.weather:
            for column in ['temp', 'atemp', 'hum', 'windspeed']:
                fields[column] = X[column].to_numpy(dtype=float)
            for level in [2, 3, 4]:
                fields[f'weather_{level}'] = (X.weathersit.to_numpy() == level).astype(float)
        return pd.DataFrame(fields, index=X.index)

    def transform(self, X):
        values = self._frame(X).to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError('Features must be finite')
        return values

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_
