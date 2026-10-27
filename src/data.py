"""Source verification, structural audit, and calendar-based evaluation splits."""
from pathlib import Path
import hashlib
import json
import pandas as pd

TEST_START = pd.Timestamp('2012-10-01')
FOLDS = [('2012-01-01', '2012-04-01'),
         ('2012-04-01', '2012-07-01'),
         ('2012-07-01', '2012-10-01')]


def load_data(root):
    """Read unchanged source data, verify its checksum, and construct timestamps."""
    raw = Path(root) / 'data' / 'raw'
    manifest = json.loads((raw / 'manifest.json').read_text())
    for filename, record in manifest['files'].items():
        content = (raw / filename).read_bytes()
        if hashlib.sha256(content).hexdigest() != record['sha256']:
            raise ValueError(f'{filename} does not match its source checksum')
    df = pd.read_csv(raw / 'hour.csv')
    df['timestamp'] = pd.to_datetime(df['dteday']) + pd.to_timedelta(df['hr'], unit='h')
    return df.sort_values('timestamp').reset_index(drop=True)


def audit_data(df, root):
    """Audit integrity without using held-out target distributions for modeling."""
    expected = pd.date_range(df.timestamp.min(), df.timestamp.max(), freq='h')
    missing_hours = expected.difference(pd.DatetimeIndex(df.timestamp))
    daily = pd.read_csv(Path(root) / 'data' / 'raw' / 'day.csv')
    observed_daily = df.groupby('dteday')['cnt'].sum()
    daily_match = (observed_daily.reindex(daily.dteday).to_numpy() == daily.cnt.to_numpy()).all()
    result = {
        'rows': len(df), 'source_columns': len(df.columns) - 1,
        'start': str(df.timestamp.min()), 'end': str(df.timestamp.max()),
        'null_cells': int(df.isna().sum().sum()),
        'duplicate_timestamps': int(df.timestamp.duplicated().sum()),
        'missing_hour_records': len(missing_hours),
        'target_equals_components': bool((df.cnt == df.casual + df.registered).all()),
        'daily_totals_match': bool(daily_match),
        'development_rows': int((df.timestamp < TEST_START).sum()),
        'test_rows': int((df.timestamp >= TEST_START).sum()),
    }
    assert result['null_cells'] == 0
    assert result['duplicate_timestamps'] == 0
    assert result['target_equals_components'] and result['daily_totals_match']
    assert df.hr.between(0, 23).all() and df.cnt.ge(0).all()
    return result, missing_hours


def split_data(df):
    """Return disjoint development and final-test observations."""
    dev = df.loc[df.timestamp < TEST_START].copy()
    test = df.loc[df.timestamp >= TEST_START].copy()
    assert dev.timestamp.max() < test.timestamp.min()
    return dev, test


def development_folds(dev):
    for fold, (start, end) in enumerate(FOLDS, 1):
        train = dev.loc[dev.timestamp < start]
        valid = dev.loc[(dev.timestamp >= start) & (dev.timestamp < end)]
        assert len(train) and len(valid)
        assert train.timestamp.max() < valid.timestamp.min() < TEST_START
        yield fold, train, valid
