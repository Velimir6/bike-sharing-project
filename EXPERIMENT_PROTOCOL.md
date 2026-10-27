# Experimental protocol

Specified before model fitting and final test evaluation.

## Scope

Predict the system-wide hourly total `cnt`, conditional on known calendar and contemporaneous weather. This is not a station allocation model or a causal analysis. Include a calendar-only ablation to separate weather information from inputs known in advance.

## Time boundaries

Use all available records from 2011-01-01 through 2012-12-31. Reserve October through December 2012 as the final test interval. Development uses only records before 2012-10-01.

Expanding-window validation:

1. Train before 2012-01-01; validate January through March 2012.
2. Train before 2012-04-01; validate April through June 2012.
3. Train before 2012-07-01; validate July through September 2012.

Use exact calendar boundaries, not a fixed count of rows, because missing timestamps may exist. These folds span different seasons and different training sample sizes; their scores are descriptive, not independent replications.

## Features

Allow `hr`, `weekday`, `mnth`, `yr`, `holiday`, `workingday`, `temp`, `atemp`, `hum`, `windspeed`, `weathersit`. Exclude `instant`, `casual`, `registered` and `cnt` from inputs. Use the date only for splits and presentation. Exclude redundant season labels to avoid ambiguity in source documentation.

Use fixed domain-based category ranges. Compare raw numeric time, sine/cosine time encoding and one-hot time encoding. Include a prespecified one-hot hour-by-working-day interaction in the enriched linear design. Use training-only standardization for numeric columns. Keep source-normalized weather values; no uncertain conversion to physical units.

## Candidates and primary selection rule

- Baseline: mean target by hour and working-day status, training-global-mean fallback.
- Linear regression: ordinal and one-hot time encodings, plus a NumPy gradient-descent reference check.
- Ridge: sine/cosine, one-hot and enriched interaction designs; alpha in {0.1, 10, 1000}. Include enriched calendar-only Ridge.
- Random Forest: 120 trees, minimum leaf size in {1, 5, 15}, max_features=1.0, random_state=42; include full-weather and calendar-only versions.

Select the candidate with the lowest mean validation MAE, with mean RMSE as a tie-breaker. All main reported errors use predictions projected to nonnegative values. Report negative raw prediction rates as a diagnostic for linear models. Do not round predicted means to integers.

## Final evaluation

After selection, refit the selected candidate and the prespecified historical baseline on all development data. Evaluate both on the final test interval once, without changing the model in response. Report MAE, RMSE, R-squared, mean error and baseline improvement. Save predictions and protocol-selected settings.

Report hourly, working-day and weather subgroups with sample counts. Define rush hours a priori as 07:00–09:59 and 16:00–18:59 on working days. These are descriptive slices, not extra tuning sets.

Estimate uncertainty in the paired MAE difference using a seven-day moving-block bootstrap (1000 repetitions, seed 42), resampling paired daily error sums and observation counts. This preserves within-block dependence but assumes local stationarity and is not a guarantee of performance in other cities or years.

## Checks and reproducibility

Verify source hashes, target decomposition, unique timestamps, chronological non-overlap, finite predictions, training-only preprocessing and a numerical gradient check. Compare NumPy gradient descent against least squares on controlled and real training data. Fix random seeds. Record package versions and runtime measurements. Dataset distribution and test-period diagnostics may be reported after evaluation, but must not drive model selection.
