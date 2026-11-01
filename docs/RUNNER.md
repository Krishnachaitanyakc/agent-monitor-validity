# Stage runner

`scripts/run_stage.py` runs every stage one way: the request layer builds both bodies, [spend.py](../judge/spend.py) reserves each call on a ledger, and the existing dispatcher sends A1, A2, A3, then B1. Standard library only; the docstring shows the commands.

## Stages

| Stage | Cases | Slots | Guard | Budget |
|---|---|---|---|---|
| P1 | up to 5 development cases | A1 to A3, B1, cached | `dev_only` | pilot, $95 for P1 to P3 |
| P2 | one prompt pass, up to 20 traces | A1, B1, uncached | `dev_only` | pilot |
| P3 | frozen pass, up to 20 tasks | A1 to A3, B1, cached | `dev_only` | pilot |
| EVAL | the frozen panel | A1 to A3, B1, cached | `pre_join` | evaluation, $285 |

A stage starts only after the previous stage's complete `report.json`, in the same mode, and a passing reconciliation after that stage's last call. P3 and EVAL also refuse a rubric whose status is not `frozen`. EVAL needs `--panel-manifest`, its recorded `--panel-sha256` and `--freeze-tag`: the manifest must match the hash, name the tag and config hash, and list the given packs in order, with the code at the tag and a clean tree. `--b-model` picks judge B's model, `gpt-6-sol` or its first fallback `gpt-6.1-sol`; the model is part of the configuration hash, so a fallback is a configuration change, not a code edit. `run.json` fixes the run's mode, study workspace and OpenAI project at its first stage, and every later stage must give the same ones. A resumed stage also keeps its packs, configuration and ledger; a finished one never reruns.

The dispatcher takes all four slots or, for prompt development, A1 and B1 only; any other set is refused. `run_stage` runs only the fake transport in fake mode and only the HTTPS transport in paid mode, on a ledger of the same mode.

## Gates

`progress.json` after every trajectory, and `report.json` once every case has ended, show each slot's invalid share by outcome class, the cache hit share of A2 and A3, spend and projected cost against the cap, calls that reached the 32,000-token output cap, halts with their correction notes, and B's capability check. They are for review only: none selects a judge, and a refusal never triggers a fallback. EVAL's report is complete only after a passing reconciliation that follows its last call: reconcile, then run the same command again to write it.

## Spending rules

- **Reserve.** r = (I × 1.02 + 256) × p_in × 1.25 + 32,000 × p_out: $0.47 at 60,000 input tokens, $0.58 at 100,000. I is the provider's own pre-send count. Paid mode is refused until a provider token-count client exists; twice the request layer's estimate is used in fake mode only, when no count is given. The 1.25 prices all input as a cache write. The ledger header keeps the SHA-256 of the price table, the modifiers, these reservation constants, the request layer's output cap and the source of the functions that compute a reservation, so a code change to any of them refuses a resumed ledger.
- **Admit.** A call goes out only if settled spend, open and held reservations and r stay at or below the cap; otherwise a `spend_cap` halt is written and nothing is sent. A note does not raise the cap. A trajectory whose B request is counted above 272,000 input tokens is refused before its first call (`long_context`), so A1 to A3 are not sent for a B1 that would be refused; the refusal applies again on every resume.
- **Settle.** From the attempt record: a complete response at its reported usage; an error response at $0, since Anthropic does not bill failed requests, while OpenAI's behavior is unverified and covered by reconciliation. No response, a cut stream, an Anthropic stream whose `message_delta` did not report final usage, or incomplete usage holds the full amount for good. A call that settles above its reservation halts the stage once its trajectory finishes; the cap still counts actual settled spend. That halt is recorded even when another halt ends the same trajectory, and at start for any such call left without one by a stop.
- **Price modifiers.** A complete response that shows a documented modifier settles at the larger of its reservation and its cost under the modifier: A's `inference_geo` `us` (1.1x every token), A's one-hour cache writes ($4 per million), B's cached input ($0.20, or $0.10 on `gpt-6.1-sol`) and cache writes (at the 12-hour rate, 2.5x input), B input above 272,000 tokens (2x input and cache, 1.5x output), and B's `flex`, `priority` and `ultrafast` tiers (0.5x, 2x, 6x). A modifier with no documented price, such as an A tier other than `standard`, holds the full reservation. The dispatcher also halts on each of these responses: an A region other than `global`, an A tier other than `standard`, A one-hour cache writes, a B tier other than `default`, B cache tokens, and B input above 272,000 tokens.
- **Prices.** One table, `prices-2026-10-28-r2`, per million tokens: `claude-sonnet-5` $2 input, $2.50 cache write, $0.20 cache read, $10 output; `gpt-6-sol` and `gpt-6.1-sol` $2 and $10.
- **One call at a time.** One process holds the ledger, at most one reservation is open, and a second send in flight is refused.

## Ledgers

Each budget has one ledger per mode, at a fixed place under the data root that every run root shares: `<data-root>/out/ledgers/<mode>/ledger-<budget>.jsonl`. Committed spend therefore adds up across run roots under one data root. Each entry extends a SHA-256 chain over the ledger's lines, and a mark (the ledger id, its last entry and that entry's chain link) records where a ledger stood. `<data-root>/out/ledgers/`, above the mode folders, holds the register `budgets.jsonl`, which lists every ledger created there, and `marks-<budget>.jsonl`, which gets the ledger's mark after every entry. Each run root keeps `ledger_marks-<budget>.jsonl`, written after each trajectory and by the reconcile and correct commands, and each `progress.json` carries the mark of its last entry.

A ledger shorter than any of these marks, from the marks file, the run root or any run root directly under `<data-root>/out`, was rolled back or cut, and one that differs from a mark was replaced or edited; either is refused before anything is sent. So is a mark that names another ledger of the budget, unless the current ledger's header says it replaces that one. Once the register or a mark names a ledger for a budget, a second one, after the first was moved aside or deleted, is created only with `--new-budget-note`, which the new ledger's header keeps with the ids it replaces. `stage.json` names its ledger by budget, file name and id, never by an absolute path, and each run root's `run.json` holds a run id by which the ledger names the run wherever its folder is.

These checks hold within one data root while its register, its marks file or a run root that recorded a mark remains. A run root outside `<data-root>/out` is checked only when it is the one given. A copy of the whole data root carries copies of its ledgers, which no local check can tell apart; the provider-side backstops, prepaid credits and spend limits, still apply.

## Paid mode

Paid mode needs `--paid`, `--confirm-paid-calls` (passed to the HTTPS transport as `confirm_paid_calls`) and `--approval-note`, an existing, non-empty file written after the mentor's written go-ahead, plus `--workspace` and `--openai-project`, the study's Anthropic workspace and OpenAI project. A partial set of paid flags, or paid mode without both scopes, is refused before any transport exists. Paid mode is also refused, before any transport exists, until a provider token-count client exists, because each reservation needs the provider's own pre-send count. With none of the paid flags only the fake transport runs, with no API key read; the report says "every socket refused" only once the refusal hook is installed. `run.json`, `stage.json` and the report's settings record the workspace and project, and each reservation records its provider's scope, so a later stage or a resume under another `--workspace` or `--openai-project` is refused. Every run root shares the budget's ledger and one export covers one scope, so a run whose scope for a provider differs from one the ledger's reservations already name is also refused, before its `run.json` is written. `--approval-note`, `--panel-manifest` and `--export` must name files outside the repository.

## Restart and reconciliation

Each ledger is append-only JSONL, each line fsynced before the next step, so a reservation is on disk before its request leaves. A torn last line, another budget or mode, or other pricing is refused. After a restart, an open reservation counts as spent until matched to its attempt record, or is held if none exists.

At each start the runner checks the ledger against the run root's records: every stage of the budget must have run on this ledger, every attempt they logged must have its reservation, and the ledger must not be shorter than, or differ from, any mark the run root recorded. Otherwise the ledger was rolled back or replaced, and nothing is sent. A slot folder holding a raw response without its attempt record, from a stop between the two writes, gets that record rebuilt from the raw bytes with the dispatcher's own classification, marked as recovered, and the call's reservation then closes from it, so the call is never sent again and no one reads the response to decide anything: bytes that start the provider's stream are taken as HTTP 200, anything else as no response, a transport failure whose reservation is held and whose slot is retried. The workspace header was not kept, so the record lists that check as not applied. A slot whose last attempt is a HALT with no halt record gets one at start and stays refused until a note.

`run_stage.py reconcile` reads a usage export in the runner's own format, not a provider's: CSV with the header `provider,model,input_tokens,cache_write_tokens,cache_read_tokens,output_tokens,cost_usd,period_start,period_end,scope`, or the same rows as JSON, with disjoint token counts; a larger `cost_usd` overrides the priced tokens. Token counts are priced at the table's base prices, so rows covering a call under a documented price modifier must give `cost_usd`. Each row states the export's period, in ISO 8601 with a time zone, and its scope, the Anthropic workspace or the OpenAI project, the same on every row of one provider; the reconcile entry records them. The export is cumulative from the ledger's creation and ends when it was taken. It is refused when:

- it has no rows for a provider that has any call in the ledger, even one whose calls all settled at zero; zero-token rows state that a provider billed nothing;
- a period starts after the ledger's `created_at`, ends before that provider's last settled or held call, or ends after the time of reconciliation;
- a scope is not the study's, as each reservation and the run's `run.json` record it;
- it is identical to an export reconciled before calls that were made since;
- a provider's total is more than 1% below the usage the ledger's own responses reported, which marks it as partial, stale or of another scope, or as pricing calls under a documented modifier at the base prices, which the refusal names. Anthropic's usage data typically appears within 5 minutes of a request, so export again once it is complete.

Usage above the ledger by more than 1% for either provider blocks every later call until `run_stage.py correct --budget` records a note. The note first settles, for each provider, an adjustment of the largest excess of the export over the ledger among the failed reconciliations since the last note, so after it committed spend is no longer below what the provider reported. Usage within the 1% tolerance passes with no adjustment.

In the evaluation, a trajectory does not start once the oldest settled or held call that no passing reconciliation follows is more than 23 hours old, so the daily rule never falls between A1 and its repeats; the refusal is a `reconcile_due` halt: reconcile, note the halt, and resume. A single call is refused at 24 hours as a backstop, and EVAL will not start or resume while such a call is more than 23 hours old.

## Dry run

`scripts/dry_run_stages.py` runs P1 to P3 through the fake transport at central and extreme token counts, reconciling after each stage, then repeats the extreme pilot on one ledger until the cap halts it. Each scenario keeps its ledger in its own `ledgers/` folder, apart from its run folders. The rubric is still a draft, so P3 runs there as a fake-mode rehearsal with the frozen-rubric rule waived, which the summary records; the runner itself never waives it. On the 19 development packs, central committed $16.83 and extreme $47.21; the repeated pilot halted at $94.50, its next $0.58 reservation refused.

## Not done yet

- A provider token-count client; paid mode is refused until it exists.
- Adapters from each provider's usage and cost reports to the runner's usage-export format.
- A record of each ledger outside the data root, which would tell a copy of the data root from the original.
- EVAL on evaluation data: `pre_join` also guards non-development screenshots, so it needs the data owner's path.
