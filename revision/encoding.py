"""Training-only reversible missingness representation for non-private generators."""
import numpy as np
import pandas as pd
from .protocol import CONTINUOUS_FEATURES, FEATURES


class MissingnessCodec:
    def fit(self, train):
        self.missing_columns = [c for c in FEATURES if train[c].isna().any()]
        self.fills = {}
        for c in FEATURES:
            if c in CONTINUOUS_FEATURES:
                value = train[c].median()
                self.fills[c] = float(value) if pd.notna(value) else 0.0
            else:
                mode = train[c].mode()
                self.fills[c] = mode.iloc[0] if len(mode) else '__MISSING__'
        return self

    def transform(self, frame):
        out = frame[FEATURES].copy()
        for c in self.missing_columns:
            out[c + '__missing'] = np.where(out[c].isna(), 'missing', 'observed')
        for c in FEATURES:
            out[c] = out[c].fillna(self.fills[c])
        return out

    def inverse_transform(self, encoded):
        out = encoded[FEATURES].copy()
        for c in self.missing_columns:
            out.loc[encoded[c + '__missing'].eq('missing'), c] = np.nan
        return out
