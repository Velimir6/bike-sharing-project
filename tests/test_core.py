"""Independent numerical checks and safeguards against data leakage."""
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from src.data import load_data, audit_data, split_data, development_folds
from src.features import CalendarFeatures
from src.models import NumpyLinearGD, mse_loss_gradient, HourlyMean

ROOT = Path(__file__).resolve().parents[1]


class CoreChecks(unittest.TestCase):
    def test_gradient_against_finite_difference(self):
        rng = np.random.default_rng(16)
        X, y, w = rng.normal(size=(31, 6)), rng.normal(size=31), rng.normal(size=6)
        _, analytic = mse_loss_gradient(X, y, w)
        step = 1e-5
        numeric = np.array([(mse_loss_gradient(X, y, w + step * v)[0] -
                             mse_loss_gradient(X, y, w - step * v)[0]) / (2 * step)
                            for v in np.eye(6)])
        np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-7)

    def test_gd_against_exact_linear_solution(self):
        rng = np.random.default_rng(23)
        X = rng.normal(size=(300, 6))
        y = 14 + X @ np.array([2, -3, 0, 4, 1, -1]) + rng.normal(size=300) * 0.2
        model, reference = NumpyLinearGD().fit(X, y), LinearRegression().fit(X, y)
        np.testing.assert_allclose(model.predict(X), reference.predict(X), atol=1e-6)
        self.assertTrue(model.converged_)
        self.assertTrue(np.all(np.diff(np.array(model.history_)[:, 1]) <= 1e-8))

    def test_source_integrity_and_time_splits(self):
        df = load_data(ROOT)
        audit, _ = audit_data(df, ROOT)
        self.assertEqual(audit['rows'], 17379)
        dev, test = split_data(df)
        self.assertFalse(set(dev.index) & set(test.index))
        for _, train, valid in development_folds(dev):
            self.assertLess(train.timestamp.max(), valid.timestamp.min())

    def test_target_columns_cannot_change_features(self):
        df = load_data(ROOT).iloc[:100].copy()
        feature = CalendarFeatures('interaction', True).fit(df)
        original = feature.transform(df)
        for column in ['cnt', 'casual', 'registered', 'instant']:
            df[column] = -999999
        np.testing.assert_array_equal(original, feature.transform(df))

    def test_training_scaler_is_not_changed_by_validation(self):
        df = load_data(ROOT)
        dev, _ = split_data(df)
        _, train, valid = next(development_folds(dev))
        features = CalendarFeatures('ordinal', True).fit(train)
        scaler = StandardScaler().fit(features.transform(train))
        original_mean = scaler.mean_.copy()
        changed = valid.copy()
        changed['temp'] = 10000
        scaler.transform(features.transform(changed))
        np.testing.assert_array_equal(original_mean, scaler.mean_)

    def test_baseline_unseen_group_fallback(self):
        train = pd.DataFrame({'hr': [0, 0], 'workingday': [0, 0]})
        model = HourlyMean().fit(train, np.array([2.0, 6.0]))
        unknown = pd.DataFrame({'hr': [23], 'workingday': [1]})
        self.assertEqual(model.predict(unknown)[0], 4.0)

    def test_gd_on_real_training_features(self):
        df = load_data(ROOT)
        train = df.loc[df.timestamp < '2012-01-01']
        X = StandardScaler().fit_transform(CalendarFeatures('ordinal', True).fit_transform(train))
        gd = NumpyLinearGD().fit(X, train.cnt.to_numpy())
        ref = LinearRegression().fit(X, train.cnt.to_numpy())
        np.testing.assert_allclose(gd.predict(X), ref.predict(X), atol=0.001)


if __name__ == '__main__':
    unittest.main()
