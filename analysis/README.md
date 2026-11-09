# Analysis

`sap.py` is the study's analysis. It compares instability-first review with disagreement-first and random review at equal review budgets, and reports how many additional reference-contradicted first verdicts instability-first review captures.

It has two paths. The primary path is descriptive: a completed audit of the frozen panel, whose conclusions apply to these tasks and these realized judge calls, with no population confidence or significance claim. `PLAN_VERSION` names this amendment of Nov 8, 2026 to version 3.2 of the study's statistical analysis plan. The population inference of version 3.2, which is not yet in this repository and is fixed at the study freeze, is outside the primary result and separately scoped. It stays in the module and runs only when asked for by name.

sap.py reads and writes no files, and its results depend only on its inputs and seeds; the tests use synthetic inputs only.

## Common to both paths

- Slot resolution from the attempt log. REFUSAL, TRUNCATED, FILTERED, MALFORMED and ABSTAIN are INVALID, and verdicts match case-insensitively. A slot with no complete response after three retries is INVALID (TRANSPORT_EXHAUSTED). A slot that ends in HALT, or in a transport failure with retries left, is unfinished and is refused, never counted as INVALID.
- The label-free design, fixed before the label join, and the exact tie-averaged captures of the three policies.
- `select_panel`, which chooses the default panel of one trajectory per task and the one-task-per-template sensitivity panel without reading labels. The wider-exclusion sensitivity panel is the default panel with every task whose template appears in the development split dropped. No new selection is made, so each kept task keeps its judged trajectory, and it is analyzed as a separate panel with its own N, s and K.
- `reference_coverage`, which counts Unsure and missing labels overall, by benchmark and by review group: invalid first verdict, disagreement, unstable agreeing and stable agreeing. For each policy it gives the exact expected number of these tasks selected for review. The report calls them unknown, never correct.

A released primary label of Successful or Unsuccessful, in any letter case, is usable. Unsure is Unsure, a blank or absent label is missing, and any other value stops the analysis. A task whose label is Unsure or missing stays in the panel, in the selections and in the review count.

## The primary path

`analyze_panel` runs this path unless it is given `path="population_inference"`. It calls `describe_panel`, which takes the resolved slots and either the released labels `Y` or `E` with its reference statuses, and, when given, the benchmark names, the costs from the spending ledger and the attempt log's provenance. `describe_report`, or `panel_report` on its result, prints the report. The report refuses to run without a prior exposure statement passed as an argument, prints the fixed scope sentence and names the panel as a retrospective evaluation panel with documented prior exposure.

In every regime, degenerate ones included, it gives:

- N, n_inv, n_D1, n_D0, s, K and that K is equal across the three policies; U2, U1, S and R_unobserved counts; invalid outputs by slot and category.
- Usable, Unsure and missing counts overall, by review group and by benchmark.
- M, m1 and m0; each policy's exact expected capture over its own random choices, as a fraction and a decimal, and the share of M it captures.
- The gain of instability-first review over disagreement-first and over random review, in verdicts and per 100 panel tasks. The secondary result is the same gain per 100 tasks with a usable label, N_Y. Nothing else is computed on usable labels only.
- The observed gain against the practical reference of 2 per 100 panel tasks, stated only as observed at or above the reference, or below it.
- A1 and B1 validity against usable references with failure as the positive class: confusion matrix, balanced accuracy, failure precision, failure recall, valid-prediction coverage and prevalence, each with its denominator, on each judge's own valid set and on the shared subset where both are valid. These need the released labels, since `E` alone cannot give B1's reference where A1 is invalid.
- Actual inference costs: total, by slot, the extra cost of A2 and A3, and that cost per extra expected verdict captured when the gain is positive.

A rate with an empty denominator is reported as not defined, never as zero. The report carries no p-value, interval, significance wording, size class, fragility count or bootstrap.

The `costs` argument has `total` and `by_slot`, with one amount for each of A1, A2, A3 and B1, in USD, as decimal strings the way the ledger writes them or as numbers. Whatever the ledger's total holds beyond the four slots, such as a reconciliation adjustment, is printed as not attributed to a slot.

## Population inference, outside the primary result

`analyze`, `analyze_with_sensitivities` and `analyze_panel` run this path only with `path="population_inference"`, and its report opens with a line saying it is outside the primary result and separately scoped. "Primary" in this section is the plan's result per 100 panel tasks.

- The existence question, by Fisher's exact test or an exact permutation test, and the size question, by the exact score interval or an approximate bootstrap, depending on the regime. The fragility count is the fewest flips of usable reference labels among agreeing tasks that reverse the existence decision; it is reported as not defined when no set of such flips reverses it.
- The labeled sensitivity analyses, the template checks and a text report.
- **Primary, per 100 panel tasks N.** A task without a usable label counts as not reference-contradicted, so the threshold of 2 per 100 keeps its meaning. In the coverage regime G_hat = (100/N) L (m1/n1 - m0/n0), with Fisher's test on m1 of n1 and m0 of n0.
- **Secondary, per 100 tasks with a usable label N_Y.** The rescaled gain, 100 Delta / N_Y: the primary gain times N/N_Y, with the primary's test and its interval rescaled the same way. The threshold of 2 per 100 refers to the primary result only, so the secondary result makes no threshold claim.

When reference statuses are not supplied, every label is treated as usable. The report then says that Unsure and missing labels were not assessed, and prints no secondary result and no coverage table. The docstrings of `analyze` and `secondary_analysis` give every formula.

## Running the tests

The pinned numpy and scipy releases need Python 3.12 or later. From the repository root, with a virtual environment outside the repository:

```bash
python3.12 -m venv ~/venvs/analysis
~/venvs/analysis/bin/pip install --only-binary=:all: -r analysis/requirements.txt
~/venvs/analysis/bin/python -m unittest discover -s analysis -v
~/venvs/analysis/bin/python analysis/mutations.py
```

`test_descriptive.py` holds the tests of the primary path, `test_sap.py` those of the core method and `test_plan_rules.py` those of reference handling, label coverage, outcome classes and the panel unit. `mutations.py` makes sixty-six deliberate breaks, such as counting an Unsure label as contradicted, treating HALT as INVALID, reporting an empty denominator as zero, printing a p-value in the descriptive report or making population inference the default, and expects the tests to catch each one. CI runs both on Python 3.12 and 3.13.

## Not yet covered

On the population-inference path, the default panel keeps tasks that share a template, while that path's test and interval treat tasks as independent. The inference that handles this retained template dependence and its synthetic validation are both still to come, and belong to that separately scoped work. Neither is in this folder. The descriptive path makes no population claim and does not depend on them.

The wider-exclusion panel has no labeled report line yet. Its numbers come from running an analysis on the filtered panel.
