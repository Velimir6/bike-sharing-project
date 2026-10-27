# Modeling Hourly Bike Rental Demand

A mathematics and Python course project using the UCI Bike Sharing dataset.

## Research question

How accurately can hourly rental counts be estimated from calendar and weather conditions, and when do nonlinear models improve on interpretable linear models?

The target is the total number of completed rentals per hour across the system. This is observed rental activity, not a direct measurement of unmet demand or station-level bicycle availability.

Weather observations for the same hour are treated as known inputs. Results under this assumption must not be presented as a deployable advance weather-based forecast.

## Status

Initial project specification and notebook scaffold. Data have not yet been downloaded, models have not been trained, and no experimental results are claimed.

## Work in VS Code

1. Open this folder in VS Code.
2. Use the Microsoft Python and Jupyter extensions.
3. Create a project virtual environment with `python -m venv .venv`.
4. In PowerShell, install dependencies using `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.
5. Open `bike_demand.ipynb` and select the `.venv` Python kernel.

Dependency versions will be recorded after the complete project is tested.

## Planned comparison

- Historical mean baseline grouped by hour and working-day status, with a training-set fallback for unseen groups.
- Linear regression implemented with NumPy gradient descent and checked against a library solution.
- Ridge regression to investigate regularization.
- Random Forest as a nonlinear comparison.

Use chronological development splits and a final untouched test interval. Select the exact boundaries after checking date coverage, before model evaluation. Fit all learned preprocessing on training data only. Report MAE and RMSE, residual patterns, and performance during high-demand periods defined without consulting test targets.

## Data and sources

- Fanaee-T, H. (2013). Bike Sharing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5W894
- Dataset page: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
- Dataset license: CC BY 4.0. Preserve attribution when redistributing data.
- Related methodological reference: https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html

Document any adapted code and all external sources where they are used. Follow the course policy for assistance and attribution.

## Completion checklist

See `PROJECT_CHECKLIST.md` for assessment requirements, experimental checks, and genuine development milestones.
