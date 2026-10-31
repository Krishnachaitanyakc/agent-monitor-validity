# Analysis

`sap.py` is the study's statistical analysis. It compares instability-first review with disagreement-first and random review at equal review budgets, and reports how many additional reference-discordant first verdicts instability-first review captures. The method follows draft v3.2 of the study's statistical analysis plan, which is not yet in this repository and is fixed at the study freeze; v3.1 was the previous draft.

sap.py reads and writes no files, and its results depend only on its inputs and seeds; the tests use synthetic inputs only.

## What it computes

- Slot resolution from the attempt log. REFUSAL, TRUNCATED, FILTERED, MALFORMED and ABSTAIN are INVALID, and verdicts match case-insensitively. A slot with no complete response after three retries is INVALID (TRANSPORT_EXHAUSTED). A slot that ends in HALT, or in a transport failure with retries left, is unfinished and is refused, never counted as INVALID.
- The label-free design, fixed before the label join, and the exact tie-averaged captures of the three policies. Delta is the difference between instability-first and disagreement-first review.
- The existence question, by Fisher's exact test or an exact permutation test, and the size question, by the frozen exact score interval or an approximate bootstrap, depending on the regime. The fragility count flips usable reference labels only.
- The labeled sensitivity analyses, the template checks and a text report.
- `select_panel`, which chooses the default panel of one trajectory per task and the one-task-per-template sensitivity panel without reading labels.

## Primary and secondary results

A released primary label of Successful or Unsuccessful, in any letter case, is usable. Unsure is Unsure, a blank or absent label is missing, and any other value stops the analysis. A task whose label is Unsure or missing stays in the panel, in the selections and in the review count.

- **Primary, per 100 panel tasks N.** Such a task counts as not reference-contradicted, so the threshold of 2 per 100 keeps its meaning. In the coverage regime G_hat = (100/N) L (m1/n1 - m0/n0), with Fisher's test on m1 of n1 and m0 of n0.
- **Secondary, per 100 tasks with a usable label N_Y.** First the plan-literal 100 Delta / N_Y: the primary gain rescaled by N/N_Y, with the primary's test and its interval rescaled the same way. Then a usable-only comparison, printed as a proposed analysis pending confirmation. In the coverage regime it is G_hat_Y = (100/N_Y) L_Y (m1/n1Y - m0/n0Y) with L_Y = n1Y (n_D0 - s) / n_D0, with Fisher's test on m1 of n1Y and m0 of n0Y and the same interval method for p1Y - p0Y. It is not reported when its implied comparator would review a negative number of stable agreeing tasks with a usable label, or more than there are. The threshold of 2 per 100 refers to the primary result only, so neither secondary analysis makes a threshold claim.

When reference statuses are not supplied, every label is treated as usable and the primary result equals plan v3.1 exactly. The report then says that Unsure and missing labels were not assessed, and prints no secondary result and no coverage table. The docstrings of `analyze` and `secondary_analysis` give every formula.

## Reference label coverage

`reference_coverage` counts Unsure and missing labels overall, by benchmark and by review group: invalid first verdict, disagreement, unstable agreeing and stable agreeing. For each policy it gives the exact expected number of these tasks selected for review. The report calls them unknown, never correct.

## Running the tests

The pinned numpy and scipy releases need Python 3.12 or later. From the repository root, with a virtual environment outside the repository:

```bash
python3.12 -m venv ~/venvs/analysis
~/venvs/analysis/bin/pip install --only-binary=:all: -r analysis/requirements.txt
~/venvs/analysis/bin/python -m unittest discover -s analysis -v
~/venvs/analysis/bin/python analysis/mutations.py
```

`test_sap.py` holds the tests of the core method and `test_amendments.py` those of the v3.2 amendments. `mutations.py` makes eighteen deliberate breaks, such as counting an Unsure label as contradicted, treating HALT as INVALID or printing a threshold class for the secondary result, and expects the tests to catch each one. CI runs both on Python 3.12 and 3.13.

## Not yet covered

The default panel keeps tasks that share a template, while the primary test and interval are still those of plan v3.1, which treat tasks as independent. The inference that handles this retained template dependence and its synthetic validation are both still to come, and the plan requires both before the freeze. Neither is in this folder, so the analysis does not yet fully implement the plan's panel amendment.
