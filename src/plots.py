"""Readable, reproducible figures with units and explicitly labeled data partitions."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from .data import load_data, split_data
from .features import CalendarFeatures
from .models import NumpyLinearGD

COLORS = ['#146C94', '#E38B2C', '#27866B', '#745FA8']


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 12, 'axes.labelsize': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'axes.facecolor': 'white',
                         'axes.prop_cycle': plt.cycler(color=COLORS),
                         'savefig.dpi': 160, 'axes.titlepad': 12})


def save(fig, folder, name):
    fig.savefig(folder / name, bbox_inches='tight')
    plt.close(fig)


def build_figures(root):
    root = Path(root)
    figures = root / 'figures'
    figures.mkdir(exist_ok=True)
    style()
    df = load_data(root)
    dev, _ = split_data(df)
    result_dir = root / 'results'

    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5), layout='constrained')
    dev.cnt.hist(bins=45, ax=ax[0, 0], color=COLORS[0], edgecolor='white', linewidth=0.4)
    ax[0, 0].set(title='Hourly rental distribution', xlabel='Rentals per recorded hour', ylabel='Recorded hours')
    hourly = dev.groupby(['hr', 'workingday']).cnt.mean().unstack()
    for value, label in [(0, 'Nonworking day'), (1, 'Working day')]:
        ax[0, 1].plot(hourly.index, hourly[value], label=label, linewidth=2)
    ax[0, 1].set(title='Commuting patterns', xlabel='Hour of day', ylabel='Mean rentals per hour', xticks=[0, 6, 12, 18, 23])
    ax[0, 1].legend(frameon=False)
    monthly = dev.set_index('timestamp').cnt.resample('MS').mean()
    ax[1, 0].plot(monthly.index, monthly.values, marker='o', markersize=4)
    ax[1, 0].set(title='Seasonality and growth', ylabel='Mean rentals per recorded hour', xlabel='Month')
    ax[1, 0].tick_params(axis='x', rotation=35)
    bins = pd.cut(dev.temp, bins=np.linspace(0, 1, 11))
    means = dev.groupby(bins, observed=True).cnt.mean()
    ax[1, 1].plot([v.mid for v in means.index], means.values, marker='o')
    ax[1, 1].set(title='Temperature association', xlabel='Source-normalized temperature', ylabel='Mean rentals per hour')
    fig.suptitle('Development data only · January 2011–September 2012', fontsize=14)
    save(fig, figures, '01_development_patterns.png')

    fig, ax = plt.subplots(figsize=(10, 3.8), layout='constrained')
    matrix = dev.groupby(['weekday', 'hr']).cnt.mean().unstack().reindex(range(7))
    im = ax.imshow(matrix, aspect='auto', cmap='YlGnBu')
    ax.set(yticks=range(7), yticklabels=['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'],
           xticks=range(0, 24, 2), xlabel='Hour of day', title='Mean rentals by weekday and hour · development data')
    fig.colorbar(im, ax=ax, label='Mean rentals per hour')
    save(fig, figures, '02_weekly_heatmap.png')

    numeric = ['cnt', 'temp', 'atemp', 'hum', 'windspeed']
    correlation = dev[numeric].corr()
    fig, ax = plt.subplots(figsize=(6.7, 5), layout='constrained')
    im = ax.imshow(correlation, vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set(xticks=range(5), xticklabels=numeric, yticks=range(5), yticklabels=numeric,
           title='Pearson correlations · development data')
    for i in range(5):
        for j in range(5):
            value = correlation.iloc[i, j]
            ax.text(j, i, f'{value:.2f}', ha='center', va='center', color='white' if abs(value) > .65 else 'black')
    fig.colorbar(im, ax=ax, label='Correlation coefficient')
    save(fig, figures, '03_correlations.png')

    summary = pd.read_csv(result_dir / 'validation_summary.csv')
    names = summary.head(10).name.tolist()
    if 'Historical mean' not in names:
        names.append('Historical mean')
    selected = summary.set_index('name').loc[names].sort_values('mean_MAE', ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
    colors = [COLORS[1] if n == 'Historical mean' else COLORS[0] for n in selected.index]
    ax.barh(selected.index, selected.mean_MAE, color=colors)
    for i, value in enumerate(selected.mean_MAE):
        ax.text(value + 1, i, f'{value:.1f}', va='center', fontsize=9)
    ax.set(xlabel='Mean validation MAE (rentals per hour; lower is better)',
           title='Chronological validation · best candidates and baseline', xlim=(0, selected.mean_MAE.max() * 1.17))
    save(fig, figures, '04_validation_comparison.png')

    groups = [(['Ridge cyclic a=10', 'Ridge onehot a=10', 'Ridge interaction a=10'],
               ['Sine/cosine', 'One-hot', 'One-hot + interaction'], 'Time representation · Ridge α = 10'),
              (['Ridge calendar a=10', 'Ridge interaction a=10', 'Forest calendar leaf=5', 'Forest weather leaf=5'],
               ['Ridge calendar', 'Ridge + weather', 'Forest calendar', 'Forest + weather'], 'Weather information · matched settings')]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
    for ax, (models, labels, title) in zip(axes, groups):
        vals = summary.set_index('name').loc[models, 'mean_MAE']
        ax.bar(range(len(vals)), vals, color=COLORS[:len(vals)])
        for i, value in enumerate(vals):
            ax.text(i, value + 1, f'{value:.1f}', ha='center', fontsize=9)
        ax.set(xticks=range(len(vals)), xticklabels=labels, title=title,
               ylabel='Mean validation MAE', ylim=(0, vals.max() * 1.18))
        ax.tick_params(axis='x', rotation=18)
    save(fig, figures, '05_controlled_comparisons.png')

    train = dev.loc[dev.timestamp < '2012-01-01']
    X = StandardScaler().fit_transform(CalendarFeatures('ordinal', True).fit_transform(train))
    gd = NumpyLinearGD().fit(X, train.cnt.to_numpy())
    history = np.array(gd.history_)
    fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(history[:, 0], history[:, 1], linewidth=2)
    ax.set(xlabel='Gradient descent iteration', ylabel='Training loss J = MSE / 2',
           title=f'NumPy optimization · 2011 training data · {gd.n_iter_:,} iterations')
    save(fig, figures, '06_gradient_descent.png')

    predictions = pd.read_csv(result_dir / 'test_predictions.csv', parse_dates=['timestamp'])
    first_week = predictions.loc[predictions.timestamp < predictions.timestamp.min() + pd.Timedelta(days=7)]
    fig, ax = plt.subplots(figsize=(12, 4.5), layout='constrained')
    ax.plot(first_week.timestamp, first_week.actual, label='Observed rentals', color='#263648', linewidth=1.8)
    ax.plot(first_week.timestamp, first_week.selected, label='Selected model', linewidth=1.5)
    ax.plot(first_week.timestamp, first_week.baseline, label='Historical mean', alpha=.65, linewidth=1.1)
    ax.set(title='First seven calendar days of the final test period', xlabel='October 2012', ylabel='Rentals per hour')
    ax.legend(frameon=False, ncol=3)
    save(fig, figures, '07_test_week.png')

    residual = predictions.selected - predictions.actual
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout='constrained')
    axes[0].scatter(predictions.actual, predictions.selected, s=6, alpha=.25)
    maximum = max(predictions.actual.max(), predictions.selected.max())
    axes[0].plot([0, maximum], [0, maximum], color=COLORS[1], linestyle='--')
    axes[0].set(xlabel='Observed rentals', ylabel='Estimated rentals', title='Agreement on held-out hours')
    axes[1].hist(residual, bins=40, color=COLORS[0], edgecolor='white', linewidth=.3)
    axes[1].axvline(0, color=COLORS[1], linestyle='--')
    axes[1].set(xlabel='Estimated minus observed rentals', ylabel='Recorded hours', title='Error distribution')
    hourly_error = pd.DataFrame({'hour': predictions.hr, 'absolute_error': abs(residual)}).groupby('hour').absolute_error.mean()
    axes[2].plot(hourly_error.index, hourly_error.values, marker='o', markersize=3)
    axes[2].set(xlabel='Hour of day', ylabel='MAE (rentals per hour)', title='Error by hour', xticks=[0, 6, 12, 18, 23])
    fig.suptitle('Final test diagnostics · October–December 2012', fontsize=14)
    save(fig, figures, '08_test_diagnostics.png')
    return sorted(figures.glob('*.png'))
