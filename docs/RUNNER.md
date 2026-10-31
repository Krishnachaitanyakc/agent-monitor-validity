# Stage runner

`scripts/run_stage.py` runs every stage one way: the request layer builds both bodies, [spend.py](../judge/spend.py) reserves each call on a ledger, and the existing dispatcher sends A1, A2, A3, then B1. Standard library only; the docstring shows the commands.

## Stages

| Stage | Cases | Slots | Guard | Budget |
|---|---|---|---|---|
| P1 | up to 5 development cases | A1 to A3, B1, cached | `dev_only` | pilot, $95 for P1 to P3 |
| P2 | one prompt pass, up to 20 traces | A1, B1, uncached | `dev_only` | pilot |
| P3 | frozen pass, up to 20 tasks | A1 to A3, B1, cached | `dev_only` | pilot |
| EVAL | the frozen panel | A1 to A3, B1, cached | `pre_join` | evaluation, $285 |

A stage starts only after the previous stage's complete `report.json`, in the same mode, and a passing reconciliation after that stage's last call. EVAL also needs `--panel-manifest`, its recorded `--panel-sha256` and `--freeze-tag`: the manifest must match the hash, name the tag and config hash, and list the given packs in order, with the code at the tag and a clean tree. A resumed stage keeps its packs, mode and configuration; a finished one never reruns.

## Gates

`progress.json` after every trajectory, and `report.json` once every case has ended, show each slot's invalid share by outcome class, the cache hit share of A2 and A3, spend and projected cost against the cap, calls that reached the 32,000-token output cap, halts with their correction notes, and B's capability check. They are for review only: none selects a judge, and a refusal never triggers a fallback.

## Spending rules

- **Reserve.** r = (I × 1.02 + 256) × p_in × 1.25 + 32,000 × p_out: $0.47 at 60,000 input tokens, $0.58 at 100,000. I is the provider's pre-send count when supplied, else twice the request layer's estimate. The 1.25 prices all input as a cache write.
- **Admit.** A call goes out only if settled spend, open and held reservations and r stay at or below the cap; otherwise a `spend_cap` halt is written and nothing is sent. A note does not raise the cap.
- **Settle.** From the attempt record: a complete response at its reported usage, an error response at $0; no response, a cut stream or incomplete usage holds the full amount for good. Settling above the reservation halts the stage.
- **Prices.** One table, `prices-2026-10-28`, per million tokens: `claude-sonnet-5` $2 input, $2.50 cache write, $0.20 cache read, $10 output; `gpt-6-sol` $2 and $10.
- **One call at a time.** One process holds the ledger, at most one reservation is open, and a second send in flight is refused.

## Paid mode

Paid mode needs `--paid`, `--confirm-paid-calls` (passed to the HTTPS transport as `confirm_paid_calls`) and `--approval-note`, an existing, non-empty file written after the mentor's written go-ahead, plus `--workspace`. A partial combination is refused before any transport exists. Otherwise only the fake transport runs, with every socket refused and no API key read. Paid mode reserves with the conservative estimate until a provider token-count client is reviewed.

## Restart and reconciliation

Each budget's ledger is append-only JSONL, each line fsynced before the next step, so a reservation is on disk before its request leaves. A torn last line, another budget or another price table is refused. After a restart, an open reservation counts as spent until matched to its attempt record, or is held if none exists.

`run_stage.py reconcile` reads a provider usage export: CSV with the header `provider,model,input_tokens,cache_write_tokens,cache_read_tokens,output_tokens,cost_usd`, or the same rows as JSON, with disjoint token counts; a larger `cost_usd` overrides the priced tokens. Usage above the ledger by more than 1% for either provider blocks every later call until `run_stage.py correct --budget` records a note. EVAL will not resume while calls over a day old lack a reconciliation.

## Dry run

`scripts/dry_run_stages.py` runs P1 to P3 through the fake transport at central and extreme token counts, reconciling after each stage, then repeats the extreme pilot on one ledger until the cap halts it. On the 19 development packs, central committed $16.83 and extreme $47.21; the repeated pilot halted at $94.50, its next $0.58 reservation refused.

## Not done yet

- A provider token-count client.
- EVAL on evaluation data: `pre_join` also guards non-development screenshots, so it needs the data owner's path.
- Reconciliation inside one long EVAL run; it is checked at each start or resume.
