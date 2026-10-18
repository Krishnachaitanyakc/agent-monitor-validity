"""Data path for the AgentRewardBench (ARB) study: fetch, normalize, keep labels apart, validate.

Standard library only. The pieces:

- ``constants``: pinned dataset revision, fenced fields, label vocabulary.
- ``ids``: task-ID normalization and split lookup.
- ``fetch``: pinned, size-checked, checksummed downloads with a provenance record.
- ``png``: PNG integrity and dimensions without third-party libraries.
- ``normalize``: one ARB trajectory -> a monitor-view record (what a detector may see)
  plus a provenance record. It never reads labels.
- ``labels``: the separate label store. It refuses test-split labels until the
  predictions are locked.
- ``validate``: required fields, event order, screenshots, missingness, label values.
- ``leakage``: checks that labels, rewards and evaluator verdicts cannot enter the
  monitor view.
"""

NORMALIZER_VERSION = "arb-normalizer-0.1"
