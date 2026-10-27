"""Historical baseline, NumPy gradient descent, and reproducible candidate factories."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .features import CalendarFeatures


class HourlyMean(RegressorMixin, BaseEstimator):
    """Historical means by hour and workday status, with global fallback."""
    def fit(self, X, y):
        table = X[['hr', 'workingday']].copy()
        table['target'] = np.asarray(y)
        self.means_ = table.groupby(['hr', 'workingday']).target.mean().to_dict()
        self.global_mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.array([self.means_.get((h, w), self.global_mean_)
                         for h, w in zip(X.hr, X.workingday)])


def mse_loss_gradient(X, y, weights):
    """Return J(w)=||Xw-y||²/(2n) and its exact gradient, for verification."""
    residual = np.asarray(X) @ weights - np.asarray(y)
    return float(residual @ residual / (2 * len(y))), X.T @ residual / len(y)


class NumpyLinearGD(RegressorMixin, BaseEstimator):
    """Batch gradient descent with an analytically safe constant learning rate.

    A Gram matrix avoids repeating the n-by-p multiplication each iteration.
    This is practical for this small feature count, not a large-p solver.
    An intercept is fitted; preprocessing must be fitted outside on training data.
    """
    def __init__(self, max_iter=50000, tol=1e-9):
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or not len(y):
            raise ValueError('Expected nonempty 2D X and matching 1D y')
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError('Inputs must be finite')
        design = np.column_stack([np.ones(len(X)), X])
        gram = design.T @ design / len(y)
        rhs = design.T @ y / len(y)
        self.learning_rate_ = 0.95 / np.linalg.eigvalsh(gram)[-1]
        weights = np.zeros(design.shape[1])
        scale = max(1.0, np.linalg.norm(rhs))
        self.history_ = []
        y_squared = y @ y / len(y)
        for iteration in range(self.max_iter):
            gradient = gram @ weights - rhs
            if iteration % 25 == 0:
                loss = 0.5 * (weights @ gram @ weights - 2 * weights @ rhs + y_squared)
                self.history_.append((iteration, float(max(0, loss))))
            if np.linalg.norm(gradient) <= self.tol * scale:
                break
            weights -= self.learning_rate_ * gradient
        self.n_iter_ = iteration + 1
        self.gradient_norm_ = float(np.linalg.norm(gram @ weights - rhs))
        self.converged_ = self.gradient_norm_ <= self.tol * scale
        if not self.converged_:
            raise RuntimeError('Gradient descent did not converge at the specified tolerance')
        self.intercept_, self.coef_ = float(weights[0]), weights[1:]
        return self

    def predict(self, X):
        return np.asarray(X) @ self.coef_ + self.intercept_


def candidates():
    """Prespecified grid. No settings are selected using final-test outcomes."""
    options = [{'name': 'Historical mean', 'kind': 'baseline'},
               {'name': 'NumPy OLS ordinal', 'kind': 'gd', 'encoding': 'ordinal', 'weather': True},
               {'name': 'OLS onehot', 'kind': 'ols', 'encoding': 'onehot', 'weather': True}]
    for encoding in ['cyclic', 'onehot', 'interaction']:
        for alpha in [0.1, 10.0, 1000.0]:
            options.append({'name': f'Ridge {encoding} a={alpha:g}', 'kind': 'ridge',
                            'encoding': encoding, 'weather': True, 'alpha': alpha})
    for alpha in [0.1, 10.0, 1000.0]:
        options.append({'name': f'Ridge calendar a={alpha:g}', 'kind': 'ridge',
                        'encoding': 'interaction', 'weather': False, 'alpha': alpha})
    for weather in [True, False]:
        for leaf in [1, 5, 15]:
            options.append({'name': f'Forest {"weather" if weather else "calendar"} leaf={leaf}',
                            'kind': 'forest', 'encoding': 'ordinal', 'weather': weather, 'leaf': leaf})
    return options


def make_model(spec):
    if spec['kind'] == 'baseline':
        return HourlyMean()
    features = CalendarFeatures(spec['encoding'], spec['weather'])
    if spec['kind'] == 'forest':
        model = RandomForestRegressor(n_estimators=120, min_samples_leaf=spec['leaf'],
                                      max_features=1.0, random_state=42, n_jobs=2)
        return make_pipeline(features, model)
    estimators = {'gd': lambda: NumpyLinearGD(), 'ols': lambda: LinearRegression(),
                  'ridge': lambda: Ridge(alpha=spec['alpha'], solver='svd')}
    return make_pipeline(features, StandardScaler(), estimators[spec['kind']]())
