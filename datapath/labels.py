"""The separate label store.

Released human labels live only here, never in a monitor-view record. Three rules:

1. Primary row. When a trajectory has several annotation rows, the first row in file
   order is the primary label, which is the rule ARB's own scorer applies. Later rows
   are kept as ``secondary_labels`` with their annotator, never used as the reference.
2. Split guard. Rows whose task is not in the development split are skipped *before any
   label field is accessed*. The CSV reader still parses every row into memory; the
   guard is that no code touches a test row's label fields.
3. No evaluation labels. ``load_labels`` refuses any split other than development, with
   or without a lock file. Reading evaluation labels comes back only in the runner, after
   it validates complete prediction records bound to the frozen panel and run manifest.
"""

import csv
from pathlib import Path

from . import constants as C
from .ids import Splits, normalize_task_id, trace_id, trace_key


class LockedSplitError(PermissionError):
    """Raised when labels outside the development split are requested."""


def load_labels(annotations_csv: Path, splits: Splits, allowed_splits=("dev",),
                lock_file=None) -> dict:
    """Return {trace_key: label record} for development-split trajectories.

    ``allowed_splits`` must be exactly ("dev",) and ``lock_file`` must be None; anything
    else raises ``LockedSplitError`` before the label file is opened. ``splits`` must be
    built from the hash-checked ``data/splits.csv``, since it decides which rows are
    development rows.
    """
    if tuple(allowed_splits) != ("dev",) or lock_file is not None:
        raise LockedSplitError("evaluation label unlocking is disabled; only the development "
                               "split can be read")
    allowed = {"dev"}
    out, counts = {}, {}
    with open(annotations_csv, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if splits.split(row["task_id"]) not in allowed:
                continue                      # skipped before any label field is accessed
            # Annotation rows name the agent by its model directory (e.g. GenericAgent-...),
            # matching the 'agent' field of the cleaned trajectory.
            tid = trace_id(row["benchmark"], row["model_name"], row["task_id"])
            key = trace_key(tid)
            counts[key] = counts.get(key, 0) + 1
            if key in out:                    # first row in file order stays primary
                out[key]["secondary_labels"].append(
                    {"row_index": i, "annotator_name": row["annotator_name"],
                     **{col: row[col] for col in C.LABEL_COLUMNS}})
                continue
            rec = {
                "trace_key": key,
                "trace_id": tid,
                "split": splits.split(row["task_id"]),
                "task_id_normalized": normalize_task_id(row["task_id"]),
                "primary_row_index": i,
                "primary_rule": "first row in file order per (benchmark, model_name, task_id), as ARB's scorer",
            }
            for col in C.LABEL_COLUMNS:
                rec[col] = row[col]
            rec["success_binary"] = C.SUCCESS_BINARY.get(row["trajectory_success"])
            rec["primary_annotator_name"] = row["annotator_name"]
            rec["secondary_labels"] = []
            out[key] = rec
    for key, rec in out.items():
        rec["n_annotation_rows"] = counts[key]
    return out


def annotation_structure(annotations_csv: Path, splits: Splits) -> dict:
    """Label-free row reconciliation over ALL splits: label columns are never read."""
    keys, annotators = {}, {}
    with open(annotations_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            k = (row["benchmark"], row["model_name"], row["task_id"])
            keys[k] = keys.get(k, 0) + 1
            annotators.setdefault(k, []).append(row["annotator_name"])
    multi = [k for k, v in keys.items() if v > 1]
    by_split = {}
    for k in keys:
        s = splits.split(k[2])
        by_split[s] = by_split.get(s, 0) + 1
    multi_groups = {}
    for b, m, _t in multi:
        multi_groups[f"{b} | {m}"] = multi_groups.get(f"{b} | {m}", 0) + 1
    return {
        "rows": sum(keys.values()),
        "trajectories": len(keys),
        "trajectories_by_split": by_split,
        "trajectories_with_extra_rows": len(multi),
        "max_rows_per_trajectory": max(keys.values()) if keys else 0,
        "extra_rows_by_benchmark_and_agent": multi_groups,
        "same_annotator_twice_on_one_trajectory": sum(
            1 for k in multi if len(set(annotators[k])) < len(annotators[k])),
        "distinct_annotator_names": len({a for v in annotators.values() for a in v}),
    }
