"""Evidence packs and judge requests, built from monitor-view records.

Standard library only. See docs/PACK_WALKTHROUGH.md for the pack layer and
docs/REQUESTS.md for the request layer. The pieces:

- ``acted``: what each action acted on, read from the tree and element properties of the
  step that chose it.
- ``pack``: one monitor-view record -> one pack, the bytes every judge reads, with its
  hash, a private image manifest built from the provenance record, and leak checks.
- ``rubric``: rubric v0 (a draft), sent as system or developer text.
- ``schema``: schema S and the parser for the judges' answers.
- ``images``: screenshot sizes and the shared, deterministic downscale fallback.
- ``requests``: one pack -> the exact request body for each judge, with the fit check,
  request and config hashes, leak checks and redacted examples.
- ``streams``: server-sent event parsing and summaries for both providers.
- ``outcomes``: one finished exchange -> one canonical outcome.
- ``storage``: the write-once attempt log, halts and correction notes.
- ``dispatch``: the serial dispatcher, the per-response checks, the fallback capability
  check, and the fake and HTTPS transports.
- ``spend``: prices, reservations, the spending caps, the write-ahead ledger and
  reconciliation against a provider usage export.
"""
