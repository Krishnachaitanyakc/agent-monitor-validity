#!/usr/bin/env python3
"""ARB data path: one command from the pinned dataset to validated JSONL.

    python3 scripts/arb_datapath.py run --data-root ~/spar169-arb-data \\
        --out ~/spar169-arb-data/out/demo \\
        --trajectory webarena/GenericAgent-gpt-4o-2024-11-20/webarena.370

Steps: file listings -> metadata and label file -> each trajectory and its screenshots
-> normalize -> development-split labels -> validate -> leakage checks -> determinism
check -> outputs. A file-open guard (datapath.fence) is installed first: the label
profile before anything is read, and the development-only profile once the split file
is verified, so evaluation trajectories and screenshots cannot be opened. The
--owner-label-free flag keeps only the label profile; it is for the data owner's own
steps on evaluation data. Every file is checked against the pinned listing's hash; files
already on disk are re-checked, not downloaded again. Trajectories are processed one at
a time in trace-key order, so memory stays at one trajectory.

Outputs go to a staging folder next to --out and are published to --out in one rename,
only after every check passes; --out must not exist yet, so no run ever overwrites or
truncates an earlier output set. They are never committed (the project does not re-host
ARB data):
  monitor_view.jsonl        what a detector may see, one record per trajectory
  provenance.jsonl          source pointers, hashes, split, screenshot files, dropped fields
  labels/labels_dev.jsonl   released labels, development-split records only
  report.json               every check, output hashes, and the run settings

The three JSONL files are byte-identical across runs on the same inputs. The exit code
is 0 only if every check passes; otherwise nothing is published, and the staging folder
with its report.json is left for inspection.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from datapath import NORMALIZER_VERSION, constants as C  # noqa: E402
from datapath import fence, fetch, leakage, validate  # noqa: E402
from datapath.ids import Splits, trace_id, trace_key  # noqa: E402
from datapath.labels import annotation_structure, load_labels  # noqa: E402
from datapath.normalize import SCHEMA_VERSION, check_screenshot_paths, normalize, serialize  # noqa: E402


def _parse_spec(spec: str):
    parts = spec.split("/")
    if len(parts) != 3:
        raise SystemExit(f"--trajectory must be benchmark/agent/task_id, got {spec!r}")
    return tuple(parts)


def _outside_repo(path: Path, flag: str) -> Path:
    path = path.expanduser().resolve()
    if path == REPO or REPO in path.parents:
        raise SystemExit(f"{flag} must be outside the repository: {path}")
    return path


class _Digest:
    """Writes JSONL lines to a file and hashes them as it goes."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "w", encoding="utf-8", newline="\n")
        self.h = hashlib.sha256()

    def write(self, record: dict) -> None:
        line = serialize(record) + "\n"
        self.f.write(line)
        self.h.update(line.encode())

    def close(self) -> str:
        self.f.close()
        return self.h.hexdigest()


def _code_version() -> dict:
    """The repository commit the run used, and whether the working tree had changes."""
    try:
        commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], check=True,
                                capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], check=True,
                                    capture_output=True, text=True).stdout.strip())
        return {"commit": commit, "uncommitted_changes": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "uncommitted_changes": None}


def run(args) -> int:
    data_root = _outside_repo(Path(args.data_root), "--data-root")
    out = _outside_repo(Path(args.out), "--out")
    if out.exists():
        raise SystemExit(f"--out {out} already exists; outputs are never overwritten, "
                         "so choose a new folder")
    fence.install("label_free", data_root, fence.load_config())
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.staging-", dir=out.parent))
    try:
        all_ok = _run_into(args, data_root, staging)
    except BaseException:
        if not any(staging.iterdir()):
            staging.rmdir()
            print(f"run failed; nothing was published to {out}", file=sys.stderr)
        else:
            print(f"run failed; nothing was published to {out}; partial files are in {staging}",
                  file=sys.stderr)
        raise
    if not all_ok:
        print(f"CHECKS FAILED; nothing was published to {out}; the report is {staging / 'report.json'}")
        return 1
    staging.rename(out)
    print(f"ALL CHECKS PASS; outputs published to {out}")
    return 0


def _run_into(args, data_root: Path, out: Path) -> bool:
    """Run the whole data path, writing every output into ``out``; return whether all checks pass."""
    raw_root, listing_dir = data_root / "raw", data_root / "listing"
    specs = [_parse_spec(s) for s in args.trajectory]
    if not specs:
        raise SystemExit("give at least one --trajectory")
    if len(set(specs)) != len(specs):
        raise SystemExit("a --trajectory is given twice")

    # 1. Pinned listings, metadata and the label file (read only through the label store).
    index, listings_from_cache = {}, []
    for top in ("data", "cleaned", "screenshots"):
        if (listing_dir / C.REVISION / f"{top}.json").exists():
            listings_from_cache.append(top)
        index.update(fetch.listing_index(fetch.fetch_listing(top, listing_dir)))
    meta_records = [fetch.fetch_file(p, raw_root, index.get(p))
                    for p in C.META_FILES + (C.LABEL_FILE,)]
    if not getattr(args, "owner_label_free", False):
        fence.install("dev_only", data_root, fence.load_config())
    splits = Splits(raw_root / "data/splits.csv")
    dev_store = load_labels(raw_root / C.LABEL_FILE, splits, allowed_splits=("dev",))

    # 2. One trajectory at a time, in trace-key order.
    specs.sort(key=lambda s: trace_key(trace_id(*s)))
    mv = _Digest(out / "monitor_view.jsonl")
    pv = _Digest(out / "provenance.jsonl")
    lb = _Digest(out / "labels" / "labels_dev.jsonl")
    record_checks, leak_checks, summary, unrecognized = [], [], [], {}
    dev_keys, determinism_ok, redownloads = set(), True, 0
    downloaded = sum(m["downloaded"] for m in meta_records)
    shots_checked = unreferenced = 0
    for bench, agent, task_id in specs:
        path = fetch.find_trajectory(index, bench, agent, task_id)
        rec = fetch.fetch_file(path, raw_root, index[path])
        redownloads += rec["redownloaded_after_failed_check"]
        downloaded += rec["downloaded"]
        source = json.loads((raw_root / path).read_text(encoding="utf-8"))
        if source.get("benchmark") != bench or source.get("agent") != agent:
            raise ValueError(f"{path}: header names {source.get('benchmark')}/{source.get('agent')}")
        check_screenshot_paths(source, task_id)     # before any screenshot is fetched or opened
        referenced, verified = set(), set()
        for step in source.get("steps", []):
            if step.get("screenshot_path"):
                rel = fetch.screenshot_listing_path(step["screenshot_path"])
                referenced.add(rel)
                if rel in index:      # absent from the dataset -> recorded as missing
                    r = fetch.fetch_file(rel, raw_root, index[rel])
                    redownloads += r["redownloaded_after_failed_check"]
                    downloaded += r["downloaded"]
                    verified.add(rel)
                    shots_checked += 1
        shot_dir = f"screenshots/{bench}/{agent}/{task_id}/"
        unreferenced += sum(1 for p in index if p.startswith(shot_dir) and p not in referenced)

        split, group = splits.split(task_id), splits.group(task_id)
        # Whether a file was downloaded on this run is a run event, not provenance;
        # keeping it out of the record keeps provenance identical across runs.
        source_record = {k: v for k, v in rec.items()
                         if k not in ("downloaded", "redownloaded_after_failed_check")}
        view, prov = normalize(source, raw_root, task_id, split, group,
                               source_record=source_record, verified_paths=verified)
        view2, prov2 = normalize(source, raw_root, task_id, split, group,
                                 source_record=source_record, verified_paths=verified)
        determinism_ok &= serialize(view) == serialize(view2) and serialize(prov) == serialize(prov2)
        ids = leakage.identity_strings(source, prov)
        del source, view2, prov2

        check = validate.validate_record(view)
        record_checks.append(check)
        leak_checks.append(leakage.check_record(view, ids))
        mv.write(view)
        pv.write(prov)
        key = prov["trace_key"]
        if split == "dev":
            dev_keys.add(key)
            if key in dev_store:
                lb.write(dev_store[key])
        if prov["unrecognized_top_level_fields"] or prov["unrecognized_step_fields"]:
            unrecognized[key] = [prov["unrecognized_top_level_fields"], prov["unrecognized_step_fields"]]
        row = {"trace_key": key, "trace_id": prov["trace_id"], "split": split,
               "ok": check["ok"] and leak_checks[-1]["ok"],
               "hostname_suffixes_rewritten": prov["hostname_suffixes_rewritten"],
               "dimensions": check["checks"]["V4_screenshots"]["dimensions"],
               "has_label": split == "dev" and key in dev_store}
        if split == "dev":      # lengths and endings of test runs stay out of summaries
            row.update(n_events=prov["n_events"], n_actions=prov["n_actions"],
                       ending=prov["ending"],
                       screenshots=f"{prov['screenshots_present']}/{prov['screenshots_referenced']}")
        summary.append(row)
        del view, prov
    hashes = {"monitor_view.jsonl": mv.close(), "provenance.jsonl": pv.close(),
              "labels/labels_dev.jsonl": lb.close()}

    # 3. Checks over the whole run.
    label_check = validate.validate_labels(dev_store, dev_keys)
    runtime_check = leakage.check_runtime()
    all_ok = (all(c["ok"] for c in record_checks) and label_check["ok"]
              and all(c["ok"] for c in leak_checks) and runtime_check["ok"] and determinism_ok)
    report = {
        "ok": all_ok,
        "settings": {
            "dataset": C.DATASET, "revision": C.REVISION,
            "schema_version": SCHEMA_VERSION, "normalizer_version": NORMALIZER_VERSION,
            "python": platform.python_version(), "trajectories": args.trajectory,
            "code": _code_version(),
            "fence": {"profiles": fence.installed_profiles(),
                      "config_sha256": (fetch.sha256_file(Path(os.environ[fence.CONFIG_ENV]).expanduser())
                                        if os.environ.get(fence.CONFIG_ENV) else None)},
        },
        "outputs_sha256": hashes,
        "inputs": {"listing_files_sha256": {
                       t: fetch.sha256_file(listing_dir / C.REVISION / f"{t}.json")
                       for t in ("data", "cleaned", "screenshots")},
                   "listings_read_from_cache": listings_from_cache,
                   "files_downloaded": downloaded,
                   "metadata_files": meta_records,
                   "screenshots_checked_against_listing": shots_checked,
                   "files_redownloaded_after_failed_check": redownloads,
                   "screenshots_in_dataset_not_referenced_by_steps": unreferenced},
        "summary": summary,
        "split_counts": {"tasks_dev": len(splits.tasks("dev")),
                         "tasks_test": len(splits.tasks("test"))},
        "annotation_structure_label_free": annotation_structure(raw_root / C.LABEL_FILE, splits),
        "checks": {
            "records": record_checks,
            "labels": label_check,
            "leakage_records": leak_checks,
            "leakage_runtime": runtime_check,
            "determinism": {"ok": determinism_ok,
                            "method": "each trajectory normalized twice, bytes compared"},
            "unrecognized_source_fields": unrecognized,
        },
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                     encoding="utf-8")

    for s in summary:
        if s["split"] == "dev":
            print(f"{s['trace_id']:<95} dev  events={s['n_events']:<3} "
                  f"screenshots={s['screenshots']:<6} ending={s['ending']:<14} "
                  f"label={'yes' if s['has_label'] else 'no'}")
        else:
            print(f"{s['trace_id']:<95} test checks={'ok' if s['ok'] else 'FAIL'}")
    print(f"records valid: {sum(c['ok'] for c in record_checks)}/{len(record_checks)}; "
          f"labels: {'ok' if label_check['ok'] else 'FAIL'}; "
          f"leakage: {sum(c['ok'] for c in leak_checks)}/{len(leak_checks)} records, "
          f"runtime {'ok' if runtime_check['ok'] else 'FAIL'}; "
          f"determinism: {'ok' if determinism_ok else 'FAIL'}")
    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="fetch, normalize, label (dev only), validate, check")
    r.add_argument("--data-root", required=True,
                   help="local data directory outside the repository")
    r.add_argument("--out", required=True,
                   help="new output directory outside the repository; must not exist yet")
    r.add_argument("--trajectory", action="append", default=[],
                   help="benchmark/agent/task_id, repeatable")
    r.add_argument("--owner-label-free", action="store_true",
                   help="data owner only: guard the label files but allow evaluation trajectories")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
