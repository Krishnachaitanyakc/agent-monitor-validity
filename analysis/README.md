# Analysis

`sap.py` is the study's statistical analysis. It compares instability-first review with disagreement-first and random review at equal review budgets, and reports how many additional reference-discordant first verdicts instability-first review captures. The method follows the analysis plan, draft v3.2, and is frozen only at the study freeze.

Every function is pure. Nothing here reads data from disk, and the tests use synthetic inputs only.

## What it computes

- Slot resolution from the attempt log. REFUSAL, TRUNCATED, FILTERED, MALFORMED and ABSTAIN are INVALID, and verdicts match case-insensitively. A slot that ends in HALT is unfinished and is refused, never counted as INVALID.
- The label-free design, fixed before the label join, and the exact tie-averaged captures of the three policies. Delta is the difference between instability-first and disagreement-first review.
- The existence question, by Fisher's exact test or an exact permutation test, and the size question, by the frozen exact score interval or an approximate bootstrap, depending on the regime.
- The labelled sensitivity analyses, the template checks and a text report.
- `select_panel`, which chooses the default panel of one trajectory per task and the one-task-per-template sensitivity panel without reading labels.

## Primary and secondary results

A task whose released primary label is Unsure or missing stays in the panel, in the selections and in the review count.

- **Primary, per 100 panel tasks N.** Such a task counts as not reference-contradicted, so the threshold of 2 per 100 keeps its meaning. In the coverage regime G_hat = (100/N) L (m1/n1 - m0/n0), with Fisher's test on m1 of n1 and m0 of n0.
- **Secondary, per 100 tasks with a usable label N_Y.** Such tasks leave the rates. In the coverage regime G_hat_Y = (100/N_Y) L_Y (m1/n1Y - m0/n0Y) with L_Y = n1Y (n_D0 - s) / n_D0, Fisher's test on m1 of n1Y and m0 of n0Y, and the same interval method for p1Y - p0Y. The report also prints 100 Delta / N_Y and the part of it due to unequal usable shares between unstable and stable agreeing tasks.

With every label usable, both results equal plan v3.1 exactly. The docstrings of `analyze` and `secondary_analysis` give every formula.

## Coverage

`reference_coverage` counts Unsure and missing labels overall, by benchmark and by review group: invalid first verdict, disagreement, unstable agreeing and stable agreeing. For each policy it gives the exact expected number of these tasks selected for review. The report calls them unknown, never correct.

## Running the tests

From the repository root, with a virtual environment outside the repository:

```
python3 -m venv ~/venvs/analysis
~/venvs/analysis/bin/pip install -r analysis/requirements.txt
~/venvs/analysis/bin/python -m unittest discover -s analysis -v
~/venvs/analysis/bin/python analysis/mutations.py
```

`test_sap.py` holds the tests carried over from v3.1 and `test_amendments.py` those of the v3.2 amendments. `mutations.py` makes eleven deliberate breaks, such as counting an Unsure label as contradicted or treating HALT as INVALID, and expects the tests to catch each one. CI runs both on Python 3.12 and 3.13.

## Not yet covered

The default panel keeps tasks that share a template, while the test and the interval treat tasks as independent. The plan requires both to pass synthetic validation under that retained dependence before the freeze. That validation is not in this folder.
