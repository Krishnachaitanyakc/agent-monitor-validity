#!/usr/bin/env python3
"""The development table for scripts/draw_allocation.py, and the panel's template exclusion.

    python3 scripts/build_dev_table.py table --counts COUNTS_DIR --out dev_table.json
    python3 scripts/build_dev_table.py exclusion --counts COUNTS_DIR --allocation allocation.json --out exclusion.json

Both read two label-free counts files in COUNTS_DIR. task_cluster_mapping.json has one row
per task with task_id, split, benchmark, image_goal and cluster_key, the task's template.
eligible_minus_exposed.json has eligible_task_ids_after, the panel before template
exclusions, which is what "panel" means below.

``table`` writes one row per development task, leaving out AssistantBench and image-goal
tasks. template_overlaps_panel says whether the task's template holds a panel task, and
only WebArena and VisualWebArena panel tasks count, since only their templates are shared
across tasks.

``exclusion`` writes the record that the frozen panel manifest carries as
template_exclusion: the templates of the 20 prompt tasks (p2) that hold panel tasks, how
many panel tasks those templates hold, and N before and after removing them. The panel
builder removes those tasks and names each remaining task's template in the manifest.

The commands print counts only, and a refusal names no panel task, so no evaluation task
ID reaches the terminal. An output file is written once. Standard library only.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import draw_allocation as DA  # noqa: E402

MAPPING_FILE, ELIGIBLE_FILE = "task_cluster_mapping.json", "eligible_minus_exposed.json"
TEMPLATE_BENCHMARKS = ("webarena", "visualwebarena")
NOT_IN_THE_POOL = "assistantbench"


class BuildRefused(ValueError):
    """The counts files do not agree with each other or with the allocation."""


def _indexed(mapping: list, panel) -> dict:
    """{task_id: row}, once the mapping has no task twice and every panel task is in it, in the test split."""
    by_id = {r["task_id"]: r for r in mapping}
    if len(by_id) != len(mapping):
        raise BuildRefused(f"{len(mapping) - len(by_id)} task IDs appear twice in the mapping")
    absent = set(panel) - set(by_id)
    if absent:
        raise BuildRefused(f"{len(absent)} panel tasks are missing from the mapping")
    other_split = sum(by_id[t]["split"] != "test" for t in set(panel))
    if other_split:
        raise BuildRefused(f"{other_split} panel tasks are not in the test split")
    return by_id


def panel_templates(by_id: dict, panel) -> Counter:
    """How many panel tasks each WebArena or VisualWebArena template holds."""
    return Counter(by_id[t]["cluster_key"] for t in set(panel) if by_id[t]["benchmark"] in TEMPLATE_BENCHMARKS)


def build_table(mapping: list, panel) -> list:
    """The table's rows, sorted by task ID (see the module docstring)."""
    by_id = _indexed(mapping, panel)
    held = panel_templates(by_id, panel)
    pool = [r for r in by_id.values() if r["split"] == "dev" and r["benchmark"] != NOT_IN_THE_POOL and not r["image_goal"]]
    return [{"task_id": r["task_id"], "benchmark": r["benchmark"].split("_")[0],
             "template_overlaps_panel": held[r["cluster_key"]] > 0, "family": r["cluster_key"]}
            for r in sorted(pool, key=lambda r: r["task_id"])]


def template_exclusion(p2, mapping: list, panel) -> dict:
    """The templates of ``p2`` that hold panel tasks, and N before and after removing those tasks."""
    by_id = _indexed(mapping, panel)
    missing = sorted(set(p2) - set(by_id))
    if missing:
        raise BuildRefused(f"these prompt tasks are missing from the mapping: {missing}")
    held = panel_templates(by_id, panel)
    excluded = sorted({by_id[t]["cluster_key"] for t in p2 if held[by_id[t]["cluster_key"]] > 0})
    removed = sum(held[k] for k in excluded)
    before = len(set(panel))
    return {"excluded_templates": excluded, "n_before": before, "excluded_tasks": removed, "n_after": before - removed}


def json_bytes(record) -> bytes:
    return (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")


def load_counts(folder) -> tuple:
    """(the mapping's rows, the panel's task IDs) from a counts folder."""
    folder = Path(folder).expanduser()
    mapping = json.loads((folder / MAPPING_FILE).read_bytes())
    return mapping, json.loads((folder / ELIGIBLE_FILE).read_bytes())["eligible_task_ids_after"]


def _write_once(path, data: bytes) -> None:
    try:
        with open(Path(path).expanduser(), "xb") as f:
            f.write(data)
    except FileExistsError:
        raise SystemExit(f"{path} exists; this output is written once") from None


def cmd_table(args) -> None:
    table = build_table(*load_counts(args.counts))
    free = [r for r in table if not r["template_overlaps_panel"]]
    _write_once(args.out, json_bytes(table))
    print(f"rows {len(table)} {dict(Counter(r['benchmark'] for r in table))}  "
          f"no overlap {len(free)} {dict(Counter(r['benchmark'] for r in free))}")


def cmd_exclusion(args) -> None:
    allocation = DA.verify_allocation(json.loads(Path(args.allocation).expanduser().read_bytes()))
    record = dict(template_exclusion(allocation["p2"], *load_counts(args.counts)),
                  allocation_sha256=allocation["sha256"])
    _write_once(args.out, json_bytes(record))
    print(f"templates {len(record['excluded_templates'])}  n_before {record['n_before']}  "
          f"excluded_tasks {record['excluded_tasks']}  n_after {record['n_after']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    table = sub.add_parser("table", help="write the development table")
    exclusion = sub.add_parser("exclusion", help="write the panel's template exclusion")
    exclusion.add_argument("--allocation", required=True, help="the allocation from scripts/draw_allocation.py")
    for command in (table, exclusion):
        command.add_argument("--counts", required=True, help="the folder of the two counts files")
        command.add_argument("--out", required=True, help="the output file, written once")
    args = parser.parse_args(argv)
    try:
        {"table": cmd_table, "exclusion": cmd_exclusion}[args.command](args)
    except (BuildRefused, DA.DrawRefused) as e:     # these name no panel task
        raise SystemExit(f"refused: {e}") from None
    except (ValueError, KeyError, TypeError) as e:  # a file that is not JSON or lacks a field
        raise SystemExit(f"refused: the input files cannot be read ({type(e).__name__})") from None
    return 0


if __name__ == "__main__":
    sys.exit(main())
