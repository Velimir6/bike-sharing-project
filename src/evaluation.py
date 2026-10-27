"""Chronological model comparison and paired block-bootstrap uncertainty."""
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from threadpoolctl import threadpool_limits
from .data import load_data, audit_data, split_data, development_folds
from .models import candidates, make_model


def metrics(y, prediction):
    """Evaluate continuous expected counts, without integer rounding."""
    y, prediction = np.asarray(y), np.asarray(prediction)
    if not np.isfinite(prediction).all():
        raise ValueError('Non-finite prediction')
    return {'MAE': float(mean_absolute_error(y, prediction)),
            'RMSE': float(np.sqrt(mean_squared_error(y, prediction))),
            'R2': float(r2_score(y, prediction)),
            'bias_pred_minus_actual': float(np.mean(prediction - y))}


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def prepare(root):
    root = Path(root)
    result_dir = root / 'results'
    result_dir.mkdir(exist_ok=True)
    df = load_data(root)
    audit, missing = audit_data(df, root)
    write_json(result_dir / 'data_audit.json', audit)
    pd.DataFrame({'missing_timestamp': missing}).to_csv(result_dir / 'missing_hours.csv', index=False)
    dev, test = split_data(df)
    rows = []
    for fold, train, valid in development_folds(dev):
        rows.append({'fold': fold, 'train_start': train.timestamp.min(),
                     'train_end': train.timestamp.max(), 'train_rows': len(train),
                     'validation_start': valid.timestamp.min(),
                     'validation_end': valid.timestamp.max(), 'validation_rows': len(valid)})
    pd.DataFrame(rows).to_csv(result_dir / 'splits.csv', index=False)
    return df, dev, test, audit


def run_development(root, verbose=True):
    """Fit prespecified candidates; save selection before accessing test outcomes."""
    root = Path(root)
    _, dev, _, audit = prepare(root)
    rows = []
    with threadpool_limits(limits=2):
        for spec in candidates():
            for fold, train, valid in development_folds(dev):
                model = make_model(spec)
                started = perf_counter()
                model.fit(train, train.cnt.to_numpy())
                fit_seconds = perf_counter() - started
                raw = model.predict(valid)
                pred = np.maximum(raw, 0)
                rows.append({'name': spec['name'], 'fold': fold,
                             **metrics(valid.cnt, pred), 'fit_seconds': fit_seconds,
                             'negative_raw_fraction': float(np.mean(raw < 0))})
            if verbose:
                mean_mae = np.mean([r['MAE'] for r in rows if r['name'] == spec['name']])
                print(f"{spec['name']}: validation MAE {mean_mae:.3f}", flush=True)
    detailed = pd.DataFrame(rows)
    detailed.to_csv(root / 'results' / 'validation_folds.csv', index=False)
    summary = detailed.groupby('name', sort=False).agg(
        mean_MAE=('MAE', 'mean'), fold_sd_MAE=('MAE', 'std'),
        mean_RMSE=('RMSE', 'mean'), mean_fit_seconds=('fit_seconds', 'mean'),
        negative_raw_fraction=('negative_raw_fraction', 'mean')).reset_index()
    summary = summary.sort_values(['mean_MAE', 'mean_RMSE']).reset_index(drop=True)
    summary.to_csv(root / 'results' / 'validation_summary.csv', index=False)
    selected = next(s for s in candidates() if s['name'] == summary.iloc[0]['name'])
    selection = {'selected': selected, 'selection_rule': 'minimum mean validation MAE; RMSE tie-break',
                 'protocol_sha256': hashlib.sha256((root / 'EXPERIMENT_PROTOCOL.md').read_bytes()).hexdigest(),
                 'candidates': len(candidates()), 'folds': 3}
    write_json(root / 'results' / 'selection.json', selection)
    return summary, detailed, selection


def paired_block_interval(predictions, repetitions=1000, block_days=7):
    """95% percentile interval for baseline MAE minus selected-model MAE.

    Sample overlapping contiguous day blocks with replacement. Error sums and
    counts remain paired, including days with no recorded observations.
    """
    table = predictions.copy()
    table['day'] = pd.to_datetime(table.timestamp).dt.floor('D')
    table['baseline_abs'] = abs(table.actual - table.baseline)
    table['selected_abs'] = abs(table.actual - table.selected)
    daily = table.groupby('day').agg(baseline_sum=('baseline_abs', 'sum'),
                                      selected_sum=('selected_abs', 'sum'), n=('actual', 'size'))
    calendar = pd.date_range(daily.index.min(), daily.index.max(), freq='D')
    daily = daily.reindex(calendar, fill_value=0)
    values = daily.to_numpy()
    rng = np.random.default_rng(42)
    estimates = []
    for _ in range(repetitions):
        starts = rng.integers(0, len(values) - block_days + 1,
                              size=int(np.ceil(len(values) / block_days)))
        indices = np.concatenate([np.arange(s, s + block_days) for s in starts])[:len(values)]
        baseline_sum, selected_sum, count = values[indices].sum(axis=0)
        estimates.append((baseline_sum - selected_sum) / count)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {'lower': float(low), 'upper': float(high), 'block_days': block_days,
            'repetitions': repetitions, 'seed': 42,
            'estimand': 'baseline MAE minus selected MAE on paired hourly observations'}


def run_final(root):
    """Evaluate frozen selected settings and the baseline on the held-out quarter."""
    root = Path(root)
    selection = json.loads((root / 'results' / 'selection.json').read_text())
    protocol_hash = hashlib.sha256((root / 'EXPERIMENT_PROTOCOL.md').read_bytes()).hexdigest()
    if selection['protocol_sha256'] != protocol_hash:
        raise ValueError('Protocol changed after selection; investigate before evaluation')
    df = load_data(root)
    dev, test = split_data(df)
    table = test[['timestamp', 'hr', 'workingday', 'weathersit']].copy()
    table['actual'] = test.cnt.to_numpy()
    results = []
    with threadpool_limits(limits=2):
        for key, spec in [('baseline', {'kind': 'baseline', 'name': 'Historical mean'}),
                          ('selected', selection['selected'])]:
            started = perf_counter()
            model = make_model(spec).fit(dev, dev.cnt.to_numpy())
            elapsed = perf_counter() - started
            raw = model.predict(test)
            table[key] = np.maximum(raw, 0)
            results.append({'model': spec['name'], **metrics(table.actual, table[key]),
                            'fit_seconds': elapsed, 'negative_raw_fraction': float(np.mean(raw < 0))})
    table.to_csv(root / 'results' / 'test_predictions.csv', index=False)
    final_metrics = pd.DataFrame(results)
    final_metrics.to_csv(root / 'results' / 'test_metrics.csv', index=False)
    interval = paired_block_interval(table)
    write_json(root / 'results' / 'paired_bootstrap.json', interval)
    rows = []
    slices = {'all': np.ones(len(table), dtype=bool),
              'working_day': table.workingday.eq(1), 'nonworking_day': table.workingday.eq(0),
              'rush_hour': table.workingday.eq(1) & table.hr.isin([7, 8, 9, 16, 17, 18])}
    for level in sorted(table.weathersit.unique()):
        slices[f'weather_{level}'] = table.weathersit.eq(level)
    for label, mask in slices.items():
        subset = table.loc[mask]
        rows.append({'slice': label, 'n': len(subset), 'MAE': float(np.mean(abs(subset.actual - subset.selected))),
                     'RMSE': float(np.sqrt(np.mean((subset.actual - subset.selected) ** 2))),
                     'bias_pred_minus_actual': float(np.mean(subset.selected - subset.actual))})
    pd.DataFrame(rows).to_csv(root / 'results' / 'test_subgroups.csv', index=False)
    return final_metrics, table, interval
