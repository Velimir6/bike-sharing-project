# Project completion and assessment checklist

This is a working checklist, not a completed self-assessment. Assessment points are indicative according to the supplied course guidelines.

## Mandatory requirements

- [ ] At least one Jupyter notebook with English explanations, mathematics, and Python code.
- [ ] Working code verified by restarting the kernel and running all cells in order.
- [ ] GitHub repository with at least 10 meaningful commits reflecting real progress.
- [ ] Repository link submitted before 16:00 on 3 November 2026; deadline timezone must match course instructions.
- [ ] Sources, dataset license, and any adapted code credited.
- [ ] Academic integrity and course assistance policy followed.
- [ ] Safe execution and compliance with applicable legal requirements.
- [ ] Self-assessment completed; obtain a course template if one exists.

## Assessment evidence

| Criterion | Maximum | Evidence to produce |
| --- | ---: | --- |
| Problem understanding, formulation and significance | 15 | Motivation, target definition, prediction setting, objectives, assumptions and scope |
| Writing layout | 20 | Clear English sections, documented methods and experiments, readable equations and labeled figures |
| Mathematical understanding and research | 15 | Regression formulation, loss and gradient derivation, regularization explanation, comparisons and references |
| Code quality | 15 | Clear functions, documentation, suitable algorithms and libraries, correctness checks and efficiency discussion |
| Methods and data handling | 20 | Data audit, correct transformations, chronological validation, controlled experiments and documented limitations |
| Communication | 15 | Conclusions answering the question, limitations, practical interpretation and coherent supporting files |

## Scientific workflow

- [ ] Obtain the source dataset and retain its documentation and attribution.
- [ ] Audit schema, dates, missing values, duplicate records, time gaps and distributions.
- [ ] Distinguish absent time records from confirmed zero-rental hours.
- [ ] Verify target definition and exclude target-derived columns and unavailable inputs.
- [ ] Reserve the final chronological test interval before exploratory model comparison.
- [ ] Perform target-informed exploration and feature design on development data only.
- [ ] Fit encoders, scalers and other learned preprocessing within each training split.
- [ ] Evaluate baseline, NumPy linear regression, Ridge and Random Forest under comparable conditions.
- [ ] Verify gradient numerically and check predictions against a reference implementation.
- [ ] Compare calendar-only and calendar-plus-weather inputs.
- [ ] Compare time encodings and document regularization effects.
- [ ] Report MAE, RMSE, residuals, subgroup performance and training time.
- [ ] Discuss negative predictions where applicable, temporal dependence, historical coverage, observed-weather assumptions and association versus causation.
- [ ] Select the final model without using final test results, then evaluate once.
- [ ] State conclusions supported by results, including unsuccessful hypotheses.
- [ ] Record the tested environment and verify a complete clean execution.

## Development milestones

Each milestone should correspond to substantive work; do not manufacture empty commits or backdate history.

1. Define the research question and scaffold the notebook.
2. Add source data, attribution and data loading.
3. Audit data quality and document the schema.
4. Establish chronological splits and the historical baseline.
5. Explore development data and document initial hypotheses.
6. Derive the regression objective and gradient.
7. Implement and validate NumPy gradient descent.
8. Add Ridge and compare regularization settings.
9. Add the nonlinear comparison and controlled tuning.
10. Compare feature sets and time representations.
11. Analyze errors and evaluate the selected model on the held-out period.
12. Complete conclusions, references, self-assessment and reproducibility checks.
