"""Evidence packs for the judges, built from monitor-view records (see docs/PACK_WALKTHROUGH.md).

Standard library only. The pieces:

- ``acted``: what each action acted on, read from the tree and element properties of the
  step that chose it.
- ``pack``: one monitor-view record -> one pack, the bytes every judge reads, with its
  hash, a private image manifest built from the provenance record, and leak checks.
"""
