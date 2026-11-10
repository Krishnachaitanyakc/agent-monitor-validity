# Analysis

`sap.py` is the study's analysis. It compares instability-first review with disagreement-first and random review at equal review budgets, and reports how many additional reference-contradicted first verdicts instability-first review captures.

It has two paths. The primary path is descriptive: a completed audit of the frozen panel, whose conclusions apply to these tasks and these realized judge calls, with no population confidence or significance claim. `PLAN_VERSION` names version 4.2 of the study's analysis plan, dated Nov 9, 2026, which the primary path implements. The population inference, which follows version 3.2 of that plan (`POPULATION_PLAN_VERSION`), is outside the primary result and separately scoped. It stays in the module and runs only when asked for by name.

sap.py reads and writes no files, and its results depend only on its inputs and seeds; the tests use synthetic inputs only.

## Common to both paths

- Slot resolution from the attempt log. REFUSAL, TRUNCATED, FILTERED, MALFORMED and ABSTAIN are INVALID, and verdicts match case-insensitively. A slot with no complete response after three retries is INVALID (TRANSPORT_EXHAUSTED). A slot that ends in HALT, or in a transport failure with retries left, is unfinished and is refused, never counted as INVALID.
- The label-free design, fixed before the label join, and the exact tie-averaged captures of the three policies.
- `select_panel`, which chooses the default panel of one trajectory per task and the one-task-per-template sensitivity panel without reading labels. The wider-exclusion sensitivity panel is the default panel with every task whose template appears in the development split dropped. No new selection is made, so each kept task keeps its judged trajectory, and it is analyzed as a separate panel with its own N, s and K.
- `reference_coverage`, which counts Unsure and missing labels overall, by benchmark and by review group: invalid first verdict, disagreement, U2, U1 and stable agreeing. For each policy it gives the exact expected number of these tasks selected for review. The report calls them unknown, never correct.

A released primary label of Successful or Unsuccessful, in any letter case, is usable. Unsure is Unsure, a blank or absent label is missing, and any other value stops the analysis. A task whose label is Unsure or missing stays in the panel, in the selections and in the review count.

## The primary path

`analyze_panel` runs this path unless it is given `path="population_inference"`. It calls `describe_panel`, which takes the resolved slots and either the released labels `Y` or `E` with its reference statuses, and, when given, the benchmark names, the costs from the spending ledger, the attempt log's provenance, the task ids, the template ids, the section 2 cluster ids, the judges' rationales and any second annotator's labels. `case_seed`, the seed of the cases in the freeze list, is a required keyword, so a forgotten seed stops the analysis. `describe_report`, or `panel_report` on its result, prints the report. The report refuses to run without a prior exposure statement passed as an argument, prints the fixed scope sentence, names the panel as a retrospective evaluation panel with documented prior exposure, and opens with a RESULT block: the plan's sentence on where the reference-contradicted first verdicts sit, counts only, then the sentences of plan section 8 for this result, word for word, from `result_sentences`. The limitations the plan says are stated with the result follow it, word for word.

In every regime, degenerate ones included, it gives:

- The label-free quantities committed at the prediction lock, from `lock_quantities`: the largest gain per 100 panel tasks the panel can reach, whether 2 per 100 is reachable, and x, the change in the gain per 100 from one unstable task, which is 100/N times the largest absolute weight of an unstable group that has tasks and in PARTIAL is the largest such change. When the section 2 clusters are given, `unstable_clusters` adds how many of them the unstable agreeing tasks span and the largest one's count. Otherwise the report says they were not supplied.
- N with its template multiplicities, n_inv, n_D1, n_D0, s, K and K/N, and that K is equal across the three policies; U2, U1 and S counts; R_unobserved, the valid first verdicts with both repeats invalid, disagreeing tasks included; invalid outputs by slot and category.
- Usable, Unsure and missing counts overall, by benchmark and by review group, one row each for I, G1, U2, U1 and S, with the reference-contradicted first verdicts of each group.
- How often each judge contradicts the released references: the count and the rate over its own valid verdicts with a usable reference, with the denominator and the number of its invalid outputs among usable-reference tasks beside it. A1's needs only `E`; B1's needs the released labels and is otherwise reported as not assessed.
- Two fixed lines on how to read a contradicted U2 and U1 verdict and, under the gain, the plan's readings of a gain and what the table cannot separate, all in the plan's words and with no numeric benchmark.
- M, m1 and m0; each policy's exact expected capture over its own random choices, as a fraction and a decimal, its K and K/N, and the share of M it captures.
- The gain of instability-first review over disagreement-first and over random review, in verdicts and per 100 panel tasks, and disagreement-first minus random on the same two scales as a check. The secondary result is the same gain per 100 tasks with a usable label, N_Y. Nothing else is computed on usable labels only.
- The observed gain against the practical reference of 2 per 100 panel tasks, stated only as observed at or above the reference, or below it, and only in COVERAGE and PARTIAL; a degenerate regime prints that no comparison is defined. A gain just under the reference, or very near zero, keeps as many digits as stop its decimal from reading as the reference or as zero.
- The invalid-outcome variants of plan section 9, from `invalid_outcome_variants`: for each rule the difference in expected capture only, with s held at its primary value, capped at the variant's n_D0, and N and N_Y kept as the per-100 denominators.
- At most 8 cases chosen after the join by the rule of plan section 11, which the report prints word for word with its seed. `select_cases` takes them on valid A1 only, from six cells: A, unstable agreeing with E = 1, two cases, one U2 and one U1 when both exist; B, unstable agreeing with a usable reference and E = 0, one case; C, a disagreement with a valid B1 and E = 1, one case; D, stable agreeing with E = 1, two cases; E, stable agreeing with a usable reference and E = 0, one control matched to the first D case on benchmark and A1 verdict, falling back to A1 verdict alone; F, an Unsure or missing reference, one case, or, if none exists, a second control matched to the second D case. A D case with no control of the same A1 verdict gets no control. Within a cell the tasks are sorted by the task ids and permuted by a generator from `SeedSequence(seed, spawn_key=(6, cell index))`, and a short cell takes all it has. A row of cell A says so when instability-first has no chance of reviewing the task. Each row gives the cell, the task, its group, its slot outcomes, its reference status, its released label, any second annotator, E, each policy's exact review probability and what the cell shows, in the plan's words, with the judges' rationales on the line under it. A row says when the released labels, rationales or second annotators were not supplied. `describe_panel` also returns every task's review probabilities as `review_probability`.
- A1 and B1 validity against usable references with failure as the positive class, one row per judge on all tasks with a usable reference, N_Y, and one on the shared subset where both are valid, with the number of usable tasks excluded from it: the confusion matrix of valid verdicts, invalid outputs split by reference class, failure recall and success recall with every reference of the class in the denominator, balanced accuracy as their mean, so an invalid output counts as not correct, failure precision on valid verdicts, valid coverage over N_Y, prevalence, and each denominator. These need the released labels, since `E` alone cannot give B1's reference where A1 is invalid.
- Actual inference costs: total, by slot, the extra cost of instability, meaning A2, A3 and A1's cache-write premium, and that cost per extra expected verdict captured when the gain is positive.

A rate with an empty denominator is reported as not defined, never as zero. The report carries no p-value, interval, size class, fragility count or bootstrap, and none of the words plan section 8 leaves out for this result; the only use of "significance" is in its two fixed sentences saying that no such claim is made.

The `costs` argument has `total`, `by_slot`, with one amount for each of A1, A2, A3 and B1, and `a1_cache_premium`, the part of A1's cost that the cache write adds, in USD, as decimal strings the way the ledger writes them or as numbers. The premium sits inside A1's cost and cannot be more than it. Whatever the ledger's total holds beyond the four slots, such as a reconciliation adjustment, is printed as not attributed to a slot.

## Population inference, outside the primary result

`analyze`, `analyze_with_sensitivities`, `analyze_panel` and `template_sensitivity` run this path only with `path="population_inference"`, and its report opens with a line saying it is outside the primary result and separately scoped. That line is the report's only use of "primary", which it keeps for the descriptive audit. It calls the plan's own result per 100 panel tasks the per-100-panel-task result.

- The existence question, by Fisher's exact test or an exact permutation test, and the size question, by the exact score interval or an approximate bootstrap, depending on the regime. The fragility count is the fewest flips of usable reference labels among agreeing tasks that reverse the existence decision; it is reported as not defined when no set of such flips reverses it.
- The labeled sensitivity analyses, the template checks and a text report.
- **Per 100 panel tasks N.** A task without a usable label counts as not reference-contradicted, so the threshold of 2 per 100 keeps its meaning. In the coverage regime G_hat = (100/N) L (m1/n1 - m0/n0), with Fisher's test on m1 of n1 and m0 of n0.
- **Secondary, per 100 tasks with a usable label N_Y.** The rescaled gain, 100 Delta / N_Y: the gain per 100 panel tasks times N/N_Y, with that result's test and its interval rescaled the same way. The threshold of 2 per 100 refers to the per-100-panel-task result only, so the secondary result makes no threshold claim.

When reference statuses are not supplied, every label is treated as usable. The report then says that Unsure and missing labels were not assessed, and prints no secondary result and no coverage table. The docstrings of `analyze` and `secondary_analysis` give every formula.

## Running the tests

The pinned numpy and scipy releases need Python 3.12 or later. From the repository root, with a virtual environment outside the repository:

```bash
python3.12 -m venv ~/venvs/analysis
~/venvs/analysis/bin/pip install --only-binary=:all: -r analysis/requirements.txt
~/venvs/analysis/bin/python -m unittest discover -s analysis -v
~/venvs/analysis/bin/python analysis/mutations.py
```

`test_descriptive.py` holds the tests of the primary path, `test_sap.py` those of the core method and `test_plan_rules.py` those of reference handling, label coverage, outcome classes and the panel unit. `mutations.py` makes 384 deliberate breaks, such as counting an Unsure label as contradicted, treating HALT as INVALID, reporting an empty denominator as zero, printing a p-value in the descriptive report or making population inference the default, and expects the tests to catch each one. CI runs both on Python 3.12 and 3.13.

## Not yet covered

On the population-inference path, the default panel keeps tasks that share a template, while that path's test and interval treat tasks as independent. The inference that handles this retained template dependence and its synthetic validation are both still to come, and belong to that separately scoped work. Neither is in this folder. The descriptive path makes no population claim and does not depend on them.

The wider-exclusion and one-task-per-template panels wait, as the analysis plan records.
