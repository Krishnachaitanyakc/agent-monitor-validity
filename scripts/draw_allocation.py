#!/usr/bin/env python3
"""The development allocation: which development tasks each pilot stage may use.

    python3 scripts/draw_allocation.py --table dev_table.json --seed 169 --out allocation.json \\
        --p1 TASK_1 TASK_2 TASK_3 TASK_4 TASK_5

The table is a JSON list with one row per development task and exactly these fields:
task_id (normalized, as ``datapath.ids.normalize_task_id`` gives it), benchmark (webarena,
visualwebarena or workarena), template_overlaps_panel (true or false) and family. It holds
development tasks only, and no label.

The rule, in order:
  1. P1's five tasks leave the pool.
  2. From the tasks whose template_overlaps_panel is false, 15 are drawn, stratified by
     benchmark: WebArena 8, VisualWebArena 1, WorkArena 6. The one WorkArena task whose
     family contains "infeasible" takes one of the WorkArena 6, unless it is already in P1.
  3. P2 is P1's five plus the 15.
  4. P3 is 20 tasks drawn from the remaining task IDs of the whole table after P2.

One draw: sort the stratum's task IDs, give each the next ``random()`` value of
``random.Random(S)``, sort by that value and take the first ones needed. S is the
big-endian integer of SHA-256 over the text "<seed>:<stratum>", where the stratum is the
benchmark name, or "p3" for P3. Only ``random()`` is used, since Python keeps its sequence
for a given seed across versions and makes no such promise for ``shuffle`` or ``sample``.

The draw is refused when a stratum is short, the table has duplicates or an ID that is not
normalized, the counts do not total, or the P1 tasks are missing. The output file is
written once, with its lists sorted and a SHA-256 over its canonical content, so the same
input gives identical bytes. Standard library only.
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from datapath.ids import normalize_task_id  # noqa: E402

QUOTA = {"webarena": 8, "visualwebarena": 1, "workarena": 6}
P1_TASKS, P2_NEW_TASKS, P3_TASKS = 5, 15, 20
FIELDS = ("task_id", "benchmark", "template_overlaps_panel", "family")
LISTS = ("p1", "p2", "p2_new", "p3")
FORCED_BENCHMARK, FORCED_FAMILY_WORD = "workarena", "infeasible"
P3_STRATUM = "p3"
RULE = ("P1's five tasks leave the pool. From the tasks whose template does not overlap the panel, 15 are drawn, "
        "stratified by benchmark: WebArena 8, VisualWebArena 1, WorkArena 6. The one WorkArena task whose family "
        "contains \"infeasible\" takes one of the WorkArena 6, unless it is already in P1. P2 is P1's five plus the "
        "15. P3 is 20 tasks drawn from the remaining task IDs of the whole table after P2. One draw sorts the "
        "stratum's task IDs, gives each the next random() value of random.Random(S), sorts by that value and takes "
        "the first ones needed, where S is the big-endian integer of SHA-256 over \"<seed>:<stratum>\" and the "
        "stratum is the benchmark name, or \"p3\" for P3.")


class DrawRefused(ValueError):
    """The table, the P1 list, the seed or an allocation file breaks the rule."""


def stratum_seed(seed: int, stratum: str) -> int:
    """The generator seed of one stratum: the big-endian integer of SHA-256("<seed>:<stratum>")."""
    return int.from_bytes(hashlib.sha256(f"{seed}:{stratum}".encode("utf-8")).digest(), "big")


def permuted(task_ids, seed: int, stratum: str) -> list:
    """The task IDs in the stratum's drawn order, the same for any input order."""
    rng = random.Random(stratum_seed(seed, stratum))
    return [t for _, t in sorted((rng.random(), t) for t in sorted(task_ids))]


def _check_table(table) -> dict:
    """{task_id: row} for a table of the documented shape, with no task listed twice."""
    if not isinstance(table, list) or not table:
        raise DrawRefused("the table must be a non-empty JSON list of task rows")
    rows = {}
    for row in table:
        if not isinstance(row, dict) or set(row) != set(FIELDS):
            raise DrawRefused(f"every row has exactly the fields {list(FIELDS)}: {row!r}")
        task = row["task_id"]
        if not isinstance(task, str) or not task or normalize_task_id(task) != task:
            raise DrawRefused(f"a task_id must be a normalized task ID: {task!r}")
        if not isinstance(row["benchmark"], str) or row["benchmark"] not in QUOTA:
            raise DrawRefused(f"{task}: benchmark must be one of {list(QUOTA)}: {row['benchmark']!r}")
        if not isinstance(row["template_overlaps_panel"], bool):
            raise DrawRefused(f"{task}: template_overlaps_panel must be true or false")
        if not isinstance(row["family"], str):
            raise DrawRefused(f"{task}: family must be text")
        if task in rows:
            raise DrawRefused(f"{task} appears twice in the table")
        rows[task] = row
    return rows


def _check_p1(p1, rows: dict) -> list:
    if len(p1) != P1_TASKS or len(set(p1)) != P1_TASKS:
        raise DrawRefused(f"P1 is {P1_TASKS} distinct tasks; {len(set(p1))} distinct were given")
    missing = sorted(set(p1) - set(rows))
    if missing:
        raise DrawRefused(f"P1 tasks are missing from the table: {missing}")
    return sorted(p1)


def _forced_task(rows: dict, p1: list):
    """The WorkArena task the rule forces into P2, or None when it is already in P1."""
    found = sorted(t for t, r in rows.items()
                   if r["benchmark"] == FORCED_BENCHMARK and FORCED_FAMILY_WORD in r["family"])
    if len(found) != 1:
        raise DrawRefused(f"the rule names one {FORCED_BENCHMARK} task whose family contains "
                          f"{FORCED_FAMILY_WORD!r}; the table has {len(found)}")
    task = found[0]
    if task in p1:
        return None
    if rows[task]["template_overlaps_panel"]:
        raise DrawRefused(f"{task} is forced into P2, but its template overlaps the panel")
    return task


def _by_benchmark(task_ids, rows: dict) -> dict:
    found = Counter(rows[t]["benchmark"] for t in task_ids)
    return {b: found[b] for b in QUOTA}


def check_totals(allocation: dict) -> None:
    """Refuse a drawn allocation whose lists do not add up to the rule's sizes."""
    sizes = {name: len(set(allocation[name])) for name in LISTS}
    wanted = {"p1": P1_TASKS, "p2": P1_TASKS + P2_NEW_TASKS, "p2_new": P2_NEW_TASKS, "p3": P3_TASKS}
    if sizes != wanted:
        raise DrawRefused(f"the counts do not total: {sizes}, not {wanted}")
    if allocation["counts_by_benchmark"]["p2_new"] != QUOTA:
        raise DrawRefused(f"the counts do not total: the 15 are {allocation['counts_by_benchmark']['p2_new']}, "
                          f"not {QUOTA}")


def content_sha256(allocation: dict) -> str:
    """SHA-256 of the canonical JSON of every field except the hash itself."""
    content = {k: v for k, v in allocation.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def draw(table, p1, seed: int) -> dict:
    """The allocation for a table, P1's five task IDs and a seed (see the module docstring)."""
    rows = _check_table(table)
    p1 = _check_p1(p1, rows)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise DrawRefused(f"the seed must be a whole number: {seed!r}")
    if sum(QUOTA.values()) != P2_NEW_TASKS:
        raise DrawRefused(f"the counts do not total: the strata add up to {sum(QUOTA.values())}, not {P2_NEW_TASKS}")
    forced = _forced_task(rows, p1)
    p2_new = []
    for benchmark, needed in QUOTA.items():
        pool = [t for t, r in rows.items()
                if r["benchmark"] == benchmark and not r["template_overlaps_panel"] and t not in p1]
        if len(pool) < needed:
            raise DrawRefused(f"the {benchmark} stratum holds {len(pool)} tasks; the rule draws {needed}")
        chosen = [forced] if forced in pool else []
        rest = [t for t in pool if t not in chosen]
        p2_new += chosen + permuted(rest, seed, benchmark)[:needed - len(chosen)]
    p2 = sorted({*p1, *p2_new})
    remaining = set(rows) - set(p2)
    if len(remaining) < P3_TASKS:
        raise DrawRefused(f"{len(remaining)} task IDs remain after P2; P3 draws {P3_TASKS}")
    p3 = sorted(permuted(remaining, seed, P3_STRATUM)[:P3_TASKS])
    allocation = {"p1": p1, "p2": p2, "p2_new": sorted(p2_new), "p3": p3, "seed": seed, "rule": RULE}
    allocation["counts_by_benchmark"] = {name: _by_benchmark(allocation[name], rows) for name in LISTS}
    check_totals(allocation)
    allocation["sha256"] = content_sha256(allocation)
    return allocation


def verify_allocation(allocation) -> dict:
    """The allocation, once it passes the checks a runner makes on the file it is given:
    lists of distinct normalized task IDs, its own hash, P2 as P1 plus the 15, no overlap
    of P3 with P2. Its sizes are left to the draw, since a stage's size is its own cap."""
    if not isinstance(allocation, dict):
        raise DrawRefused("an allocation is a JSON object")
    for name in LISTS:
        ids = allocation.get(name)
        if not isinstance(ids, list) or not all(isinstance(t, str) and t for t in ids):
            raise DrawRefused(f"{name} must be a list of task IDs")
        if len(set(ids)) != len(ids):
            raise DrawRefused(f"{name} lists a task twice")
        if any(normalize_task_id(t) != t for t in ids):
            raise DrawRefused(f"{name} holds a task ID that is not normalized")
    if allocation.get("sha256") != content_sha256(allocation):
        raise DrawRefused("the allocation does not match its own SHA-256")
    p1, p2, p2_new, p3 = (set(allocation[name]) for name in LISTS)
    if p2 != p1 | p2_new:
        raise DrawRefused("p2 is not p1 plus p2_new")
    overlap = sorted(p2 & p3)
    if overlap:
        raise DrawRefused(f"p3 overlaps p2: {overlap}")
    return allocation


def allocation_bytes(allocation: dict) -> bytes:
    return (json.dumps(allocation, indent=2, sort_keys=True) + "\n").encode("utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", required=True, help="the JSON table of development tasks")
    parser.add_argument("--p1", nargs="+", required=True, help="P1's five normalized task IDs")
    parser.add_argument("--seed", type=int, required=True, help="the draw's seed")
    parser.add_argument("--out", required=True, help="the allocation file, written once")
    args = parser.parse_args(argv)
    try:
        allocation = draw(json.loads(Path(args.table).expanduser().read_bytes()), args.p1, args.seed)
    except ValueError as e:                     # a refusal, or a table that is not JSON
        raise SystemExit(f"the draw is refused: {e}") from None
    try:
        with open(Path(args.out).expanduser(), "xb") as f:
            f.write(allocation_bytes(allocation))
    except FileExistsError:
        raise SystemExit(f"{args.out} exists; an allocation is written once") from None
    print(f"p1 {len(allocation['p1'])}  p2 {len(allocation['p2'])}  p3 {len(allocation['p3'])}  "
          f"sha256 {allocation['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
