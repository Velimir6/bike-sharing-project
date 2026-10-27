"""Build the English report notebook from versioned analysis components."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md(r'''
# Modeling Hourly Bike Rental Demand
## Linear algebra, optimization and statistical evaluation with real data

This project estimates hourly bicycle rentals in Washington, D.C. from calendar and observed weather conditions. It compares a historical baseline, a NumPy implementation of linear regression, Ridge regression and Random Forest. The central question is whether mathematical feature design and nonlinear modeling improve performance on later observations.

The completed experiment selects a 120-tree Random Forest using chronological validation. On the reserved October–December 2012 period, it achieves an MAE of **48.04 rentals per hour**, compared with **79.95** for the historical baseline. The result is conditional on observed weather and is not a claim of deployable advance forecasting accuracy. Peak-hour errors remain substantially larger than overall errors.

The analysis combines an English technical report, executable Python and mathematical derivations. Original source data, implementation files and the experimental protocol are included in the repository. Assistance and attribution are described in `ASSISTANCE.md`.
''')
md(r'''
## 1. Problem and scope

An operator can use estimates of system-wide rental activity to understand likely busy periods and plan aggregate resources. This dataset does not include station-level inventories, rejected rental attempts or a complete operational cost model. Consequently, the project estimates **observed completed rentals**, rather than unmet demand or an optimal station allocation.

For recorded hour $i$, let $\mathbf{x}_i$ contain the permitted calendar and weather inputs and let $y_i$ be the total number of rentals. We estimate $\hat y_i=f(\mathbf{x}_i)$. Under squared loss, the ideal population predictor is the conditional mean $E[Y\mid X=\mathbf{x}]$. Predictions can therefore be noninteger expected counts even though each observed count is an integer.

**Research question:** How accurately can hourly rental counts be estimated from calendar and weather conditions, and when do nonlinear models improve on interpretable linear models?

The main setting assumes weather for the target hour is known. An actual forecast made in advance would need weather forecasts available at its issue time. A calendar-only comparison isolates information that is known in advance, but does not establish performance for every operational forecast horizon.

**Prespecified hypotheses:**

1. Weather information reduces error compared with an otherwise matched calendar-only model.
2. Representing the daily cycle and its interaction with working-day status improves a linear model.
3. A nonlinear ensemble improves mean chronological validation performance over the linear alternatives.

These are predictive hypotheses, not causal claims. The protocol was written before fitting models, and the selected settings were committed before final evaluation.
''')
md('## 2. Reproducible environment\n\nRun all cells from the project folder with the dependencies in `requirements.txt`. The notebook recomputes validation and final evaluation with the fixed protocol, saves tables and regenerates figures. No network access is needed because the original data are included. Repeated runs reproduce the same experiment; they do not authorize tuning against the final test period.')
code('''
from pathlib import Path
import sys
import json
import platform
import importlib.metadata
import numpy as np
import pandas as pd
from IPython.display import display, Image, Markdown

ROOT = Path.cwd()
assert (ROOT / 'src' / 'data.py').exists(), 'Open this notebook from the project folder.'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_data, audit_data, split_data, development_folds
from src.features import CalendarFeatures
from src.models import NumpyLinearGD, mse_loss_gradient, make_model
from src.evaluation import prepare, run_development, run_final
from src.plots import build_figures

packages = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'nbformat', 'nbclient', 'ipykernel']
environment = {name: importlib.metadata.version(name) for name in packages}
environment['Python'] = platform.python_version()
display(pd.Series(environment, name='version').to_frame())
''')
md(r'''
## 3. Data provenance and quality

The **UCI Bike Sharing** dataset describes Capital Bikeshare activity during 2011–2012 [1]. The downloaded archive includes hourly records, daily totals and the original data dictionary. SHA-256 checksums in `data/raw/manifest.json` detect accidental changes. Attribution is retained under the dataset page's CC BY 4.0 license; the accompanying research paper is also credited [2].

The data file and its original README contain **17,379 hourly records**. The current UCI landing page displays 17,389 instances, so this report uses the verified file count rather than copying the landing-page total. There are no null cells in the hourly table, but **165 expected hourly timestamps are absent**. A missing record is not automatically a zero-rental observation. The model is evaluated on recorded hours only, and this coverage limitation is retained.

The current online metadata and bundled README differ in season labels and temperature normalization descriptions. We avoid season labels, use calendar month directly and keep temperature in its supplied normalized units. We do not report uncertain Celsius conversions.
''')
code('''
df, development, test, audit = prepare(ROOT)
display(pd.Series(audit, name='value').to_frame())
display(df.head())
assert (df['cnt'] == df['casual'] + df['registered']).all()
''')
md(r'''
### Variable roles and leakage control

| Variables | Role and decision |
| --- | --- |
| `cnt` | Target: total completed rentals per recorded hour |
| `hr`, `weekday`, `mnth`, `yr` | Calendar information known in advance |
| `workingday`, `holiday` | Work/holiday context known in advance |
| `temp`, `atemp`, `hum`, `windspeed`, `weathersit` | Observed weather; available only under the conditional-estimation assumption |
| `casual`, `registered` | Excluded: their sum is exactly the target |
| `instant` | Excluded arbitrary row identifier |
| `dteday`, derived `timestamp` | Used to split and label observations, not passed directly to estimators |
| `season` | Excluded because month captures calendar seasonality and source labels are inconsistent |

All supplied counts are retained after structural checks. There is no evidence justifying removal of large counts as data errors. Hourly totals agree with the independent daily file. The feature builder uses an explicit allowlist, so passing a DataFrame containing the target cannot silently introduce it into the model.
''')
md(r'''
## 4. Evaluation design

We reserve **October–December 2012** for the final evaluation: 2,168 recorded hours. The 15,211 earlier observations form the development data. Model selection uses three expanding training windows, with validation in the first, second and third quarters of 2012. Calendar boundaries preserve chronology even when hourly records are absent.

This design tests generalization to later observations. A random split would answer a different question by mixing nearby hours and seasons between training and evaluation. Each scaler is fitted only on the corresponding training fold. Fixed indicator categories come from calendar definitions, not from held-out data [4].

Selection minimizes the **unweighted mean MAE across the three validation folds**, with mean RMSE as the tie-breaker. We report fold variation descriptively; the folds are not independent replicates. The first fold contains no 2012 training examples, which limits all models' ability to learn year-to-year growth. This is an intentionally challenging temporal shift.
''')
code('''
display(pd.read_csv(ROOT / 'results' / 'splits.csv'))
for fold, train_part, valid_part in development_folds(development):
    assert train_part.timestamp.max() < valid_part.timestamp.min()
    assert valid_part.timestamp.max() < test.timestamp.min()
print('All chronological non-overlap checks passed.')
''')
md(r'''
## 5. Exploring development data

Exploratory plots and correlations use only development observations. Working days have a pronounced commuting pattern, while nonworking days have a broader daytime peak. A single linear coefficient for the hour cannot represent both patterns. Monthly changes combine seasonality with possible growth; this observational dataset does not identify the causal contribution of either.

Temperature and apparent temperature are strongly correlated. Correlation describes association and can indicate redundant inputs, but does not prove a weather variable causes additional rentals. We retain both for the prespecified experiment and use regularization as one way to investigate stability.
''')
code('''
display(development[['cnt', 'temp', 'atemp', 'hum', 'windspeed']].describe().round(3))
display(development.groupby('workingday')['cnt'].agg(['count', 'mean', 'median']).round(2))
print('Temperature / apparent-temperature correlation:', round(development.temp.corr(development.atemp), 4))
''')
md(r'''
## 6. Mathematical foundations

### 6.1 Linear regression and matrix notation

For $n$ observations and $p$ features, collect features into $X\in\mathbb{R}^{n\times p}$ and coefficients into $w\in\mathbb{R}^p$. With intercept $b$, the model is

$$\hat y=Xw+b\mathbf{1}.$$

Adding a column of ones gives $\widetilde X=[\mathbf{1},X]$ and $\theta=(b,w^T)^T$. The training objective is half the mean squared error:

$$J(\theta)=\frac{1}{2n}\|\widetilde X\theta-y\|_2^2.$$

This is a mathematical statement of a practical choice: large residuals receive disproportionate weight. Squared-error fitting does not require assuming normally distributed counts; such an assumption would be needed for some classical inferential statements that we do not make here. A linear model can predict negative values, an issue addressed explicitly below.

### 6.2 Deriving the gradient

Writing $r_i=\widetilde x_i^T\theta-y_i$, differentiation with respect to coordinate $\theta_j$ gives

$$\frac{\partial J}{\partial\theta_j}=\frac{1}{n}\sum_{i=1}^{n}r_i\widetilde x_{ij}.$$

Collecting coordinates produces

$$\nabla J(\theta)=\frac{1}{n}\widetilde X^T(\widetilde X\theta-y),\qquad
\nabla^2J(\theta)=\frac{1}{n}\widetilde X^T\widetilde X.$$

For any vector $v$, $v^T\nabla^2Jv=\|\widetilde Xv\|^2/n\ge0$. Therefore the objective is convex. The coefficients need not be unique when columns are dependent, but least-squares fitted values are still well defined. We avoid explicitly inverting a potentially singular matrix [3].

### 6.3 Gradient descent

The iterative update is $\theta_{t+1}=\theta_t-\eta\nabla J(\theta_t)$. For this quadratic objective, let $L$ be the largest eigenvalue of the Hessian. Choosing $\eta=0.95/L$ is below $1/L$ and yields a conservative descent step. We stop when the gradient norm is at most $10^{-9}$ times the larger of one and the right-hand-side norm, or raise an error at the iteration limit.

We precompute $G=\widetilde X^T\widetilde X/n$ and $q=\widetilde X^Ty/n$, so each update uses $G\theta-q$. This costs roughly $O(np^2)$ initially and $O(p^2)$ per update, plus $O(p^2)$ storage and the spectral calculation. It is suitable here because the feature count is small. Forming a Gram matrix can worsen numerical conditioning and is not recommended as a universal high-dimensional least-squares solver. Independent checks against a library implementation are consequently important.
''')
code('''
# A small example makes the loss and gradient concrete.
X_example = np.array([[1., 0.], [1., 1.], [1., 2.]])
y_example = np.array([1., 3., 5.])
theta = np.array([0., 0.])
loss, gradient = mse_loss_gradient(X_example, y_example, theta)
print('At the initial coefficients:', {'loss': loss, 'gradient': gradient.tolist()})
print('At the exact coefficients:', mse_loss_gradient(X_example, y_example, np.array([1., 2.]))[0])
''')
md('### Actual NumPy implementation\n\nThe following cell displays the implementation imported by the experiments. It computes the coefficients itself; it does not call a library regression estimator.')
code('''
import inspect
print(inspect.getsource(NumpyLinearGD))
''')
md(r'''
### 6.4 Standardization and Ridge regularization

For each training feature, standardization computes $z_{ij}=(x_{ij}-\mu_j)/s_j$ from training-only statistics. Constant training columns are handled by the library scaler without division by zero. All linear-model design columns, including indicators, are standardized. This choice affects the meaning of the penalty, so the same transformation is applied consistently within each comparison.

Ridge adds a squared coefficient penalty:

$$J_\lambda(w,b)=\frac{1}{2n}\|Xw+b\mathbf{1}-y\|^2+\frac{\lambda}{2}\|w\|^2.$$

Its coefficient gradient adds $\lambda w$; the intercept is not penalized. The scikit-learn convention minimizes the unnormalized squared error plus $\alpha\|w\|^2$, so **$\alpha=n\lambda$** for the objective above [3]. Our experiment tunes the library's $\alpha$ directly. It does not treat a fixed library alpha as a fixed per-observation penalty across differently sized training folds.

Regularization shrinks coefficients and can reduce sensitivity to correlated features, at the cost of bias. It cannot by itself create a missing nonlinear relationship. We compare $\alpha\in\{0.1,10,1000\}$ under each prespecified design.

### 6.5 Calendar representations and interactions

An ordinal hour assumes a constant change in prediction from hour 2 to 3 and from hour 17 to 18. A sine/cosine pair uses

$$h_{\sin}=\sin(2\pi h/24),\qquad h_{\cos}=\cos(2\pi h/24),$$

so midnight is adjacent to the preceding hour in feature space. One pair expresses a smooth single harmonic, which may still miss two sharp commuting peaks. One-hot indicators allow separate hourly effects. An hour-by-working-day indicator allows the hourly pattern itself to differ between working and nonworking days. These are changes to the representation; a linear model remains linear in its learned coefficients [5].

The enriched design deliberately contains overlapping indicators. Ridge's penalty makes the penalized coefficient problem stable; individual redundant coefficients should not be interpreted as unique causal effects.

### 6.6 Random Forest

A regression tree partitions feature space by threshold tests. Under squared-error splitting, a node chooses a split that reduces within-child squared deviations from their means. A leaf predicts its training-target average. A Random Forest averages predictions from bootstrap-trained trees:

$$\hat f(x)=\frac{1}{B}\sum_{b=1}^{B}T_b(x).$$

This captures nonlinear thresholds and interactions without specifying each interaction manually. We use $B=120$, all features available at each split, and minimum leaf sizes 1, 5 and 15. With `max_features=1.0`, tree diversity primarily comes from bootstrap samples rather than feature subsampling. A larger minimum leaf size smooths local estimates. Forests are less interpretable than linear coefficients and cannot reliably extrapolate growth beyond the target patterns in their training data [6].
''')
md(r'''
### 6.7 Error measures and nonnegative predictions

$$\operatorname{MAE}=\frac1n\sum_i|y_i-\hat y_i|,\qquad
\operatorname{RMSE}=\sqrt{\frac1n\sum_i(y_i-\hat y_i)^2}.$$

Both are in rentals per hour. MAE is the primary selection measure; RMSE emphasizes larger misses. Models fitted under squared loss optimize a different objective from the MAE selection criterion, which is a deliberate practical choice. We additionally report $R^2=1-\sum_i(y_i-\hat y_i)^2/\sum_i(y_i-\bar y)^2$ and signed bias, defined as estimated minus observed rentals. $R^2$ is not a percentage of correct predictions.

All scored predictions are projected to $\max(0,\hat y)$. Since observed counts are nonnegative, replacing a negative prediction by zero cannot increase its absolute or squared error. We apply this fixed rule to every candidate and separately report how often raw predictions were negative. Expected counts are not rounded to integers.
''')
md('## 7. Independent correctness checks\n\nThe checks compare analytical derivatives with finite differences, NumPy predictions with independent least-squares fits on synthetic and real training data, verify time separation and data checksums, and confirm that excluded target columns cannot change model features. They also check baseline fallback behavior and that validation transformations do not change the fitted scaler.')
code('''
import unittest
suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
test_result = unittest.TextTestRunner(verbosity=2).run(suite)
assert test_result.wasSuccessful(), 'Resolve failed correctness checks before proceeding.'
''')
md(r'''
## 8. Development experiments and model selection

There are 21 prespecified candidates and three chronological folds, giving 63 validation fits. The baseline predicts a training-only mean for each hour/working-day combination and falls back to the global training mean for an unseen combination. It is intentionally simple and does not adjust for recent growth or weather.

The table reports each candidate's mean validation errors, the sample standard deviation of its three MAEs, fit time and raw negative-prediction rate. The standard deviation summarizes variation across periods; it is not a confidence interval. Runtime values depend on the machine and include preprocessing.
''')
code('''
validation_summary, validation_folds, selection = run_development(ROOT)
display(validation_summary.round(3))
display(pd.Series(selection['selected'], name='selected setting').to_frame())
''')
md(r'''
### Controlled comparisons

Matched settings distinguish the effect of a feature change from the effect of retuning a parameter. At $\alpha=10$, the Ridge comparison changes the calendar representation while keeping the estimator and weather inputs fixed. The weather ablation keeps the enriched Ridge design and alpha fixed, or the forest leaf size fixed at 5.

The lowest average error does not mean a model wins every period. Reporting the folds individually is especially important because the 2011-to-2012 transition shifts the rental distribution.
''')
code('''
matched = ['Ridge cyclic a=10', 'Ridge onehot a=10', 'Ridge interaction a=10',
           'Ridge calendar a=10', 'Forest calendar leaf=5', 'Forest weather leaf=5']
display(validation_summary.set_index('name').loc[matched].round(3))
display(validation_folds[validation_folds.name.isin(
    ['Historical mean', 'Ridge interaction a=10', selection['selected']['name']]
)][['name', 'fold', 'MAE', 'RMSE', 'bias_pred_minus_actual']].round(3))
''')
md(r'''
**Development findings.** The ordinal NumPy linear model has MAE 132.88, worse than the baseline's 101.36. Correct optimization alone therefore does not ensure an adequate model. With Ridge alpha 10, the mean MAE is 116.62 for the cyclic representation, 100.88 for one-hot time and 84.15 with the working-day interaction. The interaction is more useful here than a single smooth harmonic.

At matched settings, weather inputs reduce Ridge MAE from 92.33 to 84.15 and forest MAE from 69.73 to 58.85. These results support a predictive benefit from observed weather, not a causal interpretation.

The selected forest has the lowest mean validation MAE, 58.00, but loses to the historical baseline in the first fold: 82.70 versus 55.84. The forest cannot learn a 2012 year effect from 2011-only training. Its later-fold gains should not conceal this limitation. Strong regularization does not uniformly improve the Ridge models; choosing an alpha remains an empirical decision within development data.
''')
md(r'''
## 9. Final held-out evaluation

The selected settings are frozen in `results/selection.json`. The protocol hash is checked before evaluation. We refit that candidate and the prespecified historical baseline on all development observations, then evaluate October–December 2012. We do not rank alternative models using this final period or change the settings after seeing it.
''')
code('''
test_metrics, test_predictions, uncertainty = run_final(ROOT)
display(test_metrics.round(3))
baseline_mae = test_metrics.iloc[0]['MAE']
model_mae = test_metrics.iloc[1]['MAE']
print(f'Absolute MAE reduction: {baseline_mae - model_mae:.2f} rentals per hour')
print(f'Relative MAE reduction: {100 * (1 - model_mae / baseline_mae):.2f}%')
display(pd.read_csv(ROOT / 'results' / 'test_subgroups.csv').round(3))
''')
md(r'''
### Uncertainty and dependence

Hourly errors are dependent, so treating every hour as an independent sample would be misleading. Following the block-resampling principle for dependent observations [8], we estimate a descriptive 95% percentile interval for the **paired MAE improvement over the baseline** using 1,000 seven-day moving-block bootstrap samples, with seed 42. Daily absolute-error sums and observation counts are resampled together for both models; consecutive seven-day blocks preserve local dependence, and missing hourly records are not invented.

For each resample, improvement is the baseline's total absolute error divided by its observation count minus the selected model's corresponding ratio. The 2.5th and 97.5th percentiles give the interval. This approximates sampling variability within the observed test period under a local-stationarity assumption. It is not a prediction interval for a future hour or a guarantee for another city or year. Block length is prespecified, and interval sensitivity to other block lengths remains future work.
''')
code('''
display(pd.Series(uncertainty, name='paired bootstrap').to_frame())
assert uncertainty['lower'] < uncertainty['upper']
''')
md(r'''
**Final findings.** The selected forest achieves MAE 48.04, RMSE 73.85 and $R^2=0.866$. Relative to baseline MAE 79.95, the reduction is about 39.9%. The paired bootstrap interval for the absolute MAE improvement is approximately **14.00 to 48.08 rentals per hour**.

The model's mean signed error is -11.76 rentals per hour, indicating overall underestimation. In prespecified working-day rush hours (07–09 and 16–18), MAE rises to 91.19 across 363 observations. Weather category 3 has MAE 68.62 across 160 observations. These are descriptive subgroup results with different sample sizes; they do not establish equal reliability across conditions. There are no category-4 observations in the final test, so no final-test claim about that rare category is justified.
''')
md('## 10. Visual evidence\n\nThe first three figures use development data only. Model-comparison figures use chronological validation. The last two figures show the reserved final test period. The example week is the first seven calendar days of that period, selected by date rather than by favorable predictions.')
code('''
figure_paths = build_figures(ROOT)
for path in figure_paths:
    display(Image(filename=str(path)))
''')
md(r'''
## 11. Limitations and practical interpretation

- **Observed weather is not a weather forecast.** This experiment evaluates a conditional model. Operational deployment needs evaluation with information available at the intended forecast time.
- **Completed rentals are not unmet demand.** Supply constraints may suppress observed rentals. System-wide data cannot determine station rebalancing actions.
- **Historical and geographical scope is narrow.** Two years in one city do not establish present-day or cross-city performance. Growth between years particularly affects the first validation fold.
- **Missing timestamps remain missing.** Performance applies to recorded hours; absent records are not assumed to be zero demand.
- **The baseline is deliberately limited.** Its historical means pool earlier and later training observations, and it cannot adjust to growth. Beating it does not establish superiority to a recent-history seasonal baseline or every forecasting approach.
- **Model selection has uncertainty.** Twenty-one candidates and three folds provide a bounded comparison, not an exhaustive search. Averaging across seasons may hide a weak individual period.
- **Peak errors remain meaningful.** The overall MAE does not represent rush-hour reliability. The model tends to underestimate the busiest conditions.
- **Interpretation is associative.** Correlated weather variables and observational data do not support causal statements. Forest predictions offer less coefficient-level transparency.
- **Uncertainty estimates are approximate.** The block bootstrap captures some temporal dependence but can be affected by trends, block length and the short final interval.

Future work could compare a recent-history seasonal baseline, use archived weather forecasts, add a Poisson count model, study asymmetric underestimation costs and evaluate a later year. These are proposed extensions, not completed experiments.
''')
md(r'''
## 12. Conclusion

The results support all three prespecified hypotheses within their stated predictive scope. Weather information improves matched models, the hour-by-working-day interaction strengthens linear regression, and a Random Forest achieves the best average chronological validation MAE and improves on the historical baseline on the reserved final period.

The mathematical representation is as important as the optimizer: a verified linear solver with ordinal time performs poorly, while a more suitable design substantially reduces its error. Regularization helps modestly in some settings but cannot replace the missing interaction. Nonlinear modeling provides an additional improvement, with limitations in temporal extrapolation and peak-hour error.

The practical outcome is a reproducible, auditable model comparison for system-wide rental activity under known conditions. It is useful evidence for further operational research, not a production allocation system or a guarantee of future performance.
''')
md(r'''
## 13. References and attribution

1. Fanaee-T, H. (2013). *Bike Sharing* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5W894. Original archive and README included under `data/raw`; dataset page: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset.
2. Fanaee-T, H., and Gama, J. *Event labeling combining ensemble detectors and background knowledge*. Progress in Artificial Intelligence. https://doi.org/10.1007/s13748-013-0040-3. Cited as requested by the dataset authors; the present project does not reproduce their anomaly-detection study.
3. Scikit-learn developers. *Linear Models*. https://scikit-learn.org/stable/modules/linear_model.html. Reference for OLS and the library's Ridge penalty convention.
4. Scikit-learn developers. *Common pitfalls and recommended practices*. https://scikit-learn.org/stable/common_pitfalls.html. Reference for training-only preprocessing and leakage prevention.
5. Scikit-learn developers. *Time-related feature engineering*. https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html. Related work motivating time representations; this project uses its own explicit feature builder, protocol, NumPy solver, controlled grid and analysis.
6. Scikit-learn developers. *RandomForestRegressor*. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html. Reference for ensemble behavior and parameters.
7. NumPy developers. *numpy.linalg.eigvalsh*. https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigvalsh.html. Used to obtain the Hessian spectral bound for the learning rate.
8. Künsch, H. R. (1989). *The Jackknife and the Bootstrap for General Stationary Observations*. The Annals of Statistics, 17(3), 1217–1241. https://doi.org/10.1214/aos/1176347265. Background for resampling consecutive blocks under dependence; our short historical test period does not establish all asymptotic assumptions.

External documentation accessed 27 October 2026. The NumPy solver, feature builder, validation driver and report were authored for this project with Codex assistance. No external tutorial notebook was copied. See `ASSISTANCE.md` and `SELF_ASSESSMENT.md` for the contribution statement and rubric evidence.
''')

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
               'language_info': {'name': 'python', 'version': '3.12'}}
nbf.validate(nb)
nbf.write(nb, ROOT / 'bike_demand.ipynb')
print(f'Built {len(cells)} cells ({sum(c.cell_type == "code" for c in cells)} code cells).')
