# Request layer

The request layer turns one pack and its screenshots into the exact bytes each judge receives, sends them in order and logs every attempt. Standard library only. Tests and the dry run use `FakeTransport`; the dry run also refuses every socket connection.

## Modules

- [rubric.py](../judge/rubric.py): rubric v0, a **draft** with its own version and SHA-256: ARB's success question, the five rules and the instruction to answer only in schema S.
- [schema.py](../judge/schema.py): schema S and the parser. Enums ignore ASCII letter case only, so a look-alike letter never matches; an out-of-range `p_failure`, an integer over 18 digits, a rationale over 100 words or another bad covariate voids that field only.
- [images.py](../judge/images.py): the shared 1,024 px fallback, byte-identical on any Python or zlib.
- [requests.py](../judge/requests.py): bodies from the setup note's field table, fit check, hashes, leak checks, examples.
- [streams.py](../judge/streams.py), [outcomes.py](../judge/outcomes.py): stream parsing for both providers and the outcome table.
- [storage.py](../judge/storage.py): the write-once attempt log, halts and correction notes.
- [dispatch.py](../judge/dispatch.py): the dispatcher, response checks, B's capability check and both transports; `HttpsTransport` needs `confirm_paid_calls=True`.
- [dry_run.py](../scripts/dry_run.py): the development dry run; its docstring shows the command.

## Evidence and settings

Both judges get the same parts in the same order: the goal and field notes, each step's pack entry followed by its screenshot as base64 PNG, then the final tree. Text parts are a label line plus canonical JSON and rebuild the pack exactly. The rubric is A's `system` text and B's developer message; every other field follows the table, with `anthropic-version` pinned to `2026-10-21`. No sampling parameter, tool, stop sequence or response chaining is sent, and the dispatcher refuses a body that differs from the table.

If either request breaks a provider limit (for A, a side over 2,000 px with more than 20 images), every screenshot is downscaled for both judges. A pack that still does not fit is ineligible, with its reasons.

## Serial dispatch

A1, A2, A3, then B1, one after another. The A calls send one prepared body, so bytes and settings are identical and no answer reaches a later call. Every slot runs whatever an earlier slot returned. Each attempt logs the seconds since the previous A request started and its cache read and write tokens, so cache hits are measured.

Only a transport failure is retried: same bytes, same slot, at most three retries, then the slot is invalid. A slot that ended is never sent again, so A1 is never replaced.

## Halts

An HTTP 400 that is not a spend or usage limit or a provider content-policy error, a 413, any other invalid-request error, 401, 402, 403, 404, and credit, spend or usage limits halt the run. So does a failed per-response check (model string, service tier, region, one-hour cache tokens, workspace header on A; model string, tier, cache tokens on B) or a repeated response ID. Nothing more is sent until `AttemptLog.record_correction` stores a note for the halt. A halt uses no retry and is never an invalid verdict. A provider content-policy error, one of judge B's policy codes, is FILTERED for its slot and does not halt.

## Outcomes

| Outcome | Judge A | Judge B | Retried |
|---|---|---|---|
| VALID | `end_turn`, schema parses, verdict in enum | `completed`, same | No |
| REFUSAL | stop reason `refusal` | refusal item or part | No |
| TRUNCATED | `max_tokens`, `model_context_window_exceeded` | `incomplete`, not filtered | No |
| FILTERED | none documented | `content_filter` or a policy code | No |
| MALFORMED | schema fails, or unexpected stop reason | schema fails | No |
| ABSTAIN | parses, verdict not in enum | same | No |
| TRANSPORT | no response, cut stream, 408, 429 rate limit, 5xx, 529 | same, with 503 | Yes |
| HALT | see above | see above | No |

A cut stream is TRANSPORT and its partial text is never parsed.

## Reading the example files

The dry run writes `example_request_A.json` and `example_request_B.json` for the `webarena.380` walkthrough pack: endpoint, headers (the credential is added at send time), hashes and size, then `body`, the exact body in sent-byte order with each image's base64 replaced by its length and SHA-256. `placeholders_restore_the_exact_bytes` records that restoring the images reproduces the request hash.

## Hash layers

1. **Pinned source:** ARB files at the pinned revision.
2. **Normalized:** the data path's output files and records.
3. **Pack:** each pack's canonical bytes.
4. **Request:** each body's exact bytes, images included.
5. **Config:** endpoints, headers, models, every body field without the evidence (so rubric and schema text), rubric, schema, parser, pack and builder versions, output cap, image limits, fallback, slots and retry rule.

## Dry run on development data

On Oct 30, at commit `53569bf`, the dry run over 19 development packs built every request twice with identical bytes, and reruns on Python 3.13 and 3.11 matched byte for byte. All 19 fit without the fallback: 3 to 31 screenshots, 13 packs above 20, none wider than 1,280 px. Bodies run from 0.3 to 28.3 MB; estimated input is 6,800 to 59,700 tokens for A, median 51,400. Leak, table and same-evidence checks pass, and the scripted dispatch reached every outcome class, with 6 halts each resumed after a note. Config hash: `7bdf123b4f06bcdc10983bf6b8b1bf57e3a7929646e983e70a769fe0ebe7bce2`.

## Not done yet

- Spending controls, provider token counts and the pilot runner.
- The study workspace ID for A's header check; the dry run uses a stand-in.
