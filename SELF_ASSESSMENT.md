# Self-assessment against the course criteria

This assessment maps completed evidence to the published rubric. It does not assign a guaranteed grade or claim that the examiner's judgment is predetermined.

| Criterion | Maximum | Evidence | Limitations |
| --- | ---: | --- | --- |
| Problem understanding, formulation and significance | 15 | Notebook sections 1 and 3 define the hourly target, input availability, operator motivation and scope. The distinction between observed rentals, unmet demand and advance forecasting is explicit. | System-wide historical data cannot support station-level allocation. |
| Writing layout | 20 | English report with ordered sections, mathematical notation, documented experiments, tables, eight labeled figures and references. HTML export offers an additional reading format. | The user should review the English terminology before presenting it. |
| Mathematical understanding and research | 15 | Matrix regression, coordinate gradient derivation, positive-semidefinite Hessian argument, safe gradient step, Ridge penalty convention, cyclical features, tree averaging, error metrics and block bootstrap. The NumPy implementation is exposed in the notebook. | The derivations must be understood by the person presenting the project; no unaided understanding is claimed. |
| Code quality | 15 | Separate modules for data, features, models, evaluation and figures; docstrings; explicit input allowlist; fixed seeds; seven independent checks; measured training times and a fresh-kernel execution script. | The Gram-matrix implementation is appropriate for a small design, not a universal numerical solver. |
| Methods and data handling | 20 | Checksummed source, null/duplicate/time-gap audit, daily-total cross-check, chronological folds, training-only scaling, 21 candidates, controlled ablations, held-out predictions and paired uncertainty analysis. | Only three validation periods, one city and two years; bootstrap stationarity and block-length assumptions are stated. |
| Communication | 15 | Conclusion answers the research question; negative first-fold results, peak-hour underestimation, source metadata discrepancies and known-weather limitations are reported. Bulgarian study guide supports preparation. | The GitHub repository link and submission remain separate user-account actions. |

## Mandatory deliverables and integrity

- Main English Jupyter notebook, Python code and mathematical explanations: provided.
- Working local execution and meaningful correctness checks: see `results/execution_check.json` and notebook test output.
- Sources, original data dictionary, attribution and assistance statement: provided.
- At least ten meaningful Git commits: inspect the included local Git history; preserve it when publishing.
- GitHub hosting: published at https://github.com/Velimir6/bike-sharing-project with the original development history and the user's pre-existing commits preserved.
- Submission by the course deadline: not yet claimed; the repository link must be submitted before 16:00 on 3 November 2026, in the course's intended timezone.
- Course-specific self-assessment template or AI rules: none were included in the supplied document; follow any separately provided instructions.

## Results supported by the experiment

The selected forest achieves final-test MAE 48.04 compared with baseline MAE 79.95, a 39.9% reduction. Its test RMSE is 73.85 and R-squared is 0.866. Rush-hour MAE is 91.19. These figures support a conditional-estimation result for recorded hours in the held-out quarter, not a claim of 86.6% prediction accuracy or universal superiority.
