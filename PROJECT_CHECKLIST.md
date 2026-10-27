# Project completion and assessment checklist

Technical work is complete and verified locally. Assessment points are indicative according to the supplied course guidelines. Evidence is mapped in `SELF_ASSESSMENT.md`; publication and submission are still pending.

## Mandatory requirements

- [x] Jupyter notebook with English explanations, mathematics, and Python code.
- [x] Working code verified by a fresh kernel executing all 12 code cells in order.
- [x] Local Git history with more than 10 meaningful commits reflecting real progress.
- [ ] Publish the existing history to a GitHub repository in the user's account.
- [ ] Repository link submitted before 16:00 on 3 November 2026; deadline timezone must match course instructions.
- [x] Sources, dataset license, related methods and Codex assistance credited.
- [ ] User review and confirmation of any separate course AI-assistance policy.
- [x] Public attributed data and local analytical code; no destructive or external messaging actions.
- [x] Self-assessment completed against all six criteria; adapt to a course template if separately supplied.

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

- [x] Obtain the source dataset and retain its documentation and attribution.
- [x] Audit schema, dates, missing values, duplicate records, time gaps and development distributions.
- [x] Distinguish absent time records from confirmed zero-rental hours.
- [x] Verify target definition, exclude target-derived columns and state the observed-weather input assumption.
- [x] Reserve the final chronological test interval before exploratory model comparison.
- [x] Perform target-informed exploration and feature design on development data only.
- [x] Fit learned scaling within each training split; use fixed semantic category ranges.
- [x] Evaluate baseline, NumPy linear regression, Ridge and Random Forest under comparable conditions.
- [x] Verify gradient numerically and check predictions against a reference implementation.
- [x] Compare calendar-only and calendar-plus-weather inputs.
- [x] Compare time encodings and document regularization effects.
- [x] Report MAE, RMSE, residuals, subgroup performance and training time.
- [x] Discuss negative predictions, temporal dependence, historical coverage, observed-weather assumptions and association versus causation.
- [x] Freeze model selection before final evaluation; subsequent execution repeats the same protocol without test-driven tuning.
- [x] State conclusions supported by results, including an unfavorable first validation fold.
- [x] Record the tested environment and verify a complete clean execution.

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
