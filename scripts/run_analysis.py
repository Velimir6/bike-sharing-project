"""Run development, final evaluation, or both from the repository root."""
from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation import run_development, run_final


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['development', 'final', 'all'], default='all')
    args = parser.parse_args()
    if args.stage in ['development', 'all']:
        summary, _, selection = run_development(ROOT)
        print(summary.to_string(index=False))
        print('Frozen selection:', selection['selected'])
    if args.stage in ['final', 'all']:
        results, _, interval = run_final(ROOT)
        print(results.to_string(index=False))
        print(interval)
