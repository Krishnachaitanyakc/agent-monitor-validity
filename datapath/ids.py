"""Task-ID normalization and split lookup.

ARB file names keep raw task IDs (for example ``visualwebarena.resized.118`` for
Claude and Qwen runs, ``visualwebarena.118`` for GPT-4o runs). Splits are keyed by
the normalized ID, so all agents' runs on one task share a split.
"""

import csv
import hashlib
from pathlib import Path

_SUFFIXES = (".resized", ".improved")


def normalize_task_id(task_id: str) -> str:
    """Strip ARB's ``.resized`` and ``.improved`` markers (split lookup only)."""
    for suffix in _SUFFIXES:
        task_id = task_id.replace(suffix, "")
    return task_id


def trace_id(benchmark: str, agent: str, task_id: str) -> str:
    """Stable trajectory key: one agent's run on one raw task ID."""
    return f"arb/{benchmark}/{agent}/{task_id}"


def trace_key(trace_id_: str) -> str:
    """Opaque key for the monitor view, so the text a judge reads names no agent or task.

    It is a plain hash of the trace ID, chosen for reproducibility. It is not a secret:
    hashing the 1,302 public trace IDs inverts it. Provenance and labels map it back.
    """
    return "t" + hashlib.sha256(trace_id_.encode()).hexdigest()[:16]


class Splits:
    """Lookup from normalized task ID to (split group, split)."""

    def __init__(self, splits_csv: Path):
        self._map = {}
        with open(splits_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = normalize_task_id(row["task_id"])
                if key in self._map and self._map[key][1] != row["split"]:
                    raise ValueError(f"task {key} appears in two splits")
                self._map[key] = (row["benchmark"], row["split"])

    def group(self, task_id: str) -> str:
        """Benchmark group as used by splits.csv, e.g. workarena_l1 or workarena_l2."""
        return self._lookup(task_id)[0]

    def split(self, task_id: str) -> str:
        return self._lookup(task_id)[1]

    def _lookup(self, task_id: str):
        key = normalize_task_id(task_id)
        if key not in self._map:
            raise KeyError(f"task {task_id} (normalized {key}) not in splits.csv")
        return self._map[key]

    def __len__(self) -> int:
        return len(self._map)

    def tasks(self, split: str):
        return sorted(k for k, (_, s) in self._map.items() if s == split)
