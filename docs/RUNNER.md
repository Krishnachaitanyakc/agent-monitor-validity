# Stage runner

`scripts/run_stage.py` runs every stage one way: the request layer builds both bodies, [spend.py](../judge/spend.py) reserves each call on a ledger, and the existing dispatcher sends A1, A2, A3, then B1. Standard library only; the docstring shows the commands.

## Stages

| Stage | Cases | Slots | Guard | Budget |
|---|---|---|---|---|
| P1 | up to 5 development cases | A1 to A3, B1, cached | `dev_only` | pilot, $95 for P1 to P3 |
| P2 | one prompt pass, up to 20 traces | A1, B1, uncached | `dev_only` | pilot |
| P3 | frozen pass, up to 20 tasks | A1 to A3, B1, cached | `dev_only` | pilot |
| EVAL | the frozen panel | A1 to A3, B1, cached | `pre_join` | evaluation, $285 |

A stage starts only after the previous stage's complete `report.json`, in the same mode, and a passing reconciliation after that stage's last call. P3 and EVAL also refuse a rubric whose status is not `frozen`. EVAL needs `--panel-manifest`, its recorded `--panel-sha256` and `--freeze-tag`: the manifest must match the hash, name the tag and config hash, and list the given packs in order, with the code at the tag and a clean tree. `--b-model` picks judge B's model, `gpt-6-sol` or its first fallback `gpt-6.1-sol`; the model is part of the configuration hash, so a fallback is a configuration change, not a code edit. A resumed stage keeps its packs, mode, configuration, workspace and ledger; a finished one never reruns.

The dispatcher takes all four slots or, for prompt development, A1 and B1 only; any other set is refused.

## Gates

`progress.json` after every trajectory, and `report.json` once every case has ended, show each slot's invalid share by outcome class, the cache hit share of A2 and A3, spend and projected cost against the cap, calls that reached the 32,000-token output cap, halts with their correction notes, and B's capability check. They are for review only: none selects a judge, and a refusal never triggers a fallback. EVAL's report is complete only after a passing reconciliation that follows its last call: reconcile, then run the same command again to write it.

## Spending rules

- **Reserve.** r = (I × 1.02 + 256) × p_in × 1.25 + 32,000 × p_out: $0.47 at 60,000 input tokens, $0.58 at 100,000. I is the provider's own pre-send count. Paid mode is refused until a provider token-count client exists; twice the request layer's estimate is used in fake mode only, when no count is given. The 1.25 prices all input as a cache write. The ledger header keeps the SHA-256 of the price table, the modifiers and these reservation constants, so a code change to any of them refuses a resumed ledger.
- **Admit.** A call goes out only if settled spend, open and held reservations and r stay at or below the cap; otherwise a `spend_cap` halt is written and nothing is sent. A note does not raise the cap. A B call whose input count is above 272,000 tokens is refused before it is reserved (`long_context`).
- **Settle.** From the attempt record: a complete response at its reported usage; an error response at $0, since Anthropic does not bill failed requests, while OpenAI's behaviour is unverified and covered by reconciliation. No response, a cut stream, an Anthropic stream whose `message_delta` did not report final usage, or incomplete usage holds the full amount for good. A call that settles above its reservation halts the stage once its trajectory finishes; the cap still counts actual settled spend.
- **Price modifiers.** A complete response that shows a documented modifier settles at the larger of its reservation and its cost under the modifier: A's `inference_geo` `us` (1.1x every token), A's one-hour cache writes ($4 per million), B's cached input ($0.20, or $0.10 on `gpt-6.1-sol`) and cache writes (at the 12-hour rate, 2.5x input), B input above 272,000 tokens (2x input and cache, 1.5x output), and B's `flex`, `priority` and `ultrafast` tiers (0.5x, 2x, 6x). A modifier with no documented price, such as an A tier other than `standard`, holds the full reservation. The dispatcher also halts on each of these responses.
- **Prices.** One table, `prices-2026-10-28-r2`, per million tokens: `claude-sonnet-5` $2 input, $2.50 cache write, $0.20 cache read, $10 output; `gpt-6-sol` and `gpt-6.1-sol` $2 and $10.
- **One call at a time.** One process holds the ledger, at most one reservation is open, and a second send in flight is refused.

## Ledgers

Each budget has one ledger per mode, at a fixed place under the data root that every run root shares: `<data-root>/out/ledgers/<mode>/ledger-<budget>.jsonl`. Committed spend therefore adds up across run roots, and a new run root never gets a fresh $95 or $285. `budgets.jsonl` in the same folder lists every ledger created there. Once a ledger for a budget exists, a second one, after the first was moved aside or deleted, is created only with `--new-budget-note`, which the new ledger's header keeps with the ids it replaces. `stage.json` names its ledger by budget, file name and id, never by an absolute path, and each run root's `run.json` holds a run id by which the ledger names the run wherever its folder is.

## Paid mode

Paid mode needs `--paid`, `--confirm-paid-calls` (passed to the HTTPS transport as `confirm_paid_calls`) and `--approval-note`, an existing, non-empty file written after the mentor's written go-ahead, plus `--workspace`. A partial set of paid flags, or paid mode without `--workspace`, is refused before any transport exists. Paid mode is also refused, before any transport exists, until a provider token-count client exists, because each reservation needs the provider's own pre-send count. With none of the paid flags only the fake transport runs, with no API key read; the report says "every socket refused" only once the refusal hook is installed. `stage.json` and the report's settings record the workspace ID, so a resume under another `--workspace` is refused. `--approval-note`, `--panel-manifest` and `--export` must name files outside the repository.

## Restart and reconciliation

Each ledger is append-only JSONL, each line fsynced before the next step, so a reservation is on disk before its request leaves. A torn last line, another budget or mode, or other pricing is refused. After a restart, an open reservation counts as spent until matched to its attempt record, or is held if none exists.

At each start the runner checks the ledger against the run root's attempt logs: every stage of the budget must have run on this ledger, every attempt they logged must have its reservation, and the ledger must not be shorter than the last entry a `progress.json` recorded. Otherwise the ledger was rolled back or replaced, and nothing is sent. A slot folder holding a raw response without its attempt record, from a stop between the two writes, is refused before anything is reserved: inspect the file, move it out of the slot folder, and record a note. A slot whose last attempt is a HALT with no halt record gets one at start and stays refused until a note.

`run_stage.py reconcile` reads a usage export in the runner's own format, not a provider's: CSV with the header `provider,model,input_tokens,cache_write_tokens,cache_read_tokens,output_tokens,cost_usd,period_start,period_end,scope`, or the same rows as JSON, with disjoint token counts; a larger `cost_usd` overrides the priced tokens. Each row states the export's period, in ISO 8601 with a time zone, and its scope, the Anthropic workspace or the OpenAI project, the same on every row of one provider; the reconcile entry records them. The export is cumulative from the ledger's creation: one whose period starts after the ledger's `created_at`, or ends before that provider's last settled or held call, is refused. Usage above the ledger by more than 1% for either provider blocks every later call until `run_stage.py correct --budget` records a note. The note first settles, for each provider, an adjustment of the export's excess over the ledger in that failed reconciliation, so committed spend never stays below what a provider reported.

In the evaluation, a call is refused with a `reconcile_due` halt once the oldest settled or held call that no passing reconciliation follows is more than 24 hours old: reconcile, note the halt, and resume. EVAL also will not start or resume while such a call exists.

## Dry run

`scripts/dry_run_stages.py` runs P1 to P3 through the fake transport at central and extreme token counts, reconciling after each stage, then repeats the extreme pilot on one ledger until the cap halts it. Each scenario keeps its ledger in its own `ledgers/` folder, apart from its run folders. The rubric is still a draft, so P3 runs there as a fake-mode rehearsal with the frozen-rubric rule waived, which the summary records; the runner itself never waives it. On the 19 development packs, central committed $16.83 and extreme $47.21; the repeated pilot halted at $94.50, its next $0.58 reservation refused.

## Not done yet

- A provider token-count client; paid mode is refused until it exists.
- Adapters from each provider's usage and cost reports to the runner's usage-export format.
- EVAL on evaluation data: `pre_join` also guards non-development screenshots, so it needs the data owner's path.
