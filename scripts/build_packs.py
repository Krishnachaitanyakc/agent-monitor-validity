#!/usr/bin/env python3
"""Evidence packs: one judge input per monitor-view record, plus a private image manifest.

    export SPAR_FENCE_CONFIG=/path/to/local/fence_config.json
    python3 scripts/build_packs.py --data-root ~/spar169-arb-data \\
        --monitor-view ~/spar169-arb-data/out/dev5/monitor_view.jsonl \\
        --provenance ~/spar169-arb-data/out/dev5/provenance.jsonl \\
        --out ~/spar169-arb-data/out/packs-dev5

The development-only file-open guard (datapath.fence) is installed over --data-root
before anything is read, and the command stops unless the fence config is set. It packs
development-split records only: it reads the provenance file first and stops if any
record is from another split. The two input files must list the same trace keys in the
same order, as the data path writes them. Packs are built from the monitor view alone;
provenance supplies only the private image manifest and the leak checks.

Outputs go to a staging folder next to --out and are published to --out in one rename,
only after every check passes; --out must not exist yet. They are never committed:
  packs.jsonl           one line per record: trace key, pack SHA-256 and the pack
  image_manifest.jsonl  private: each screenshot reference -> dataset path, SHA-256, bytes
  report.json           pack version, the three hash layers, acted-element status counts,
                        final-tree truncation counts, and every check

The two JSONL files are byte-identical across runs on the same inputs. The exit code is 0
only if every check passes; otherwise nothing is published.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from datapath import fence, fetch  # noqa: E402
from datapath.normalize import SCHEMA_VERSION  # noqa: E402
from judge import pack as P  # noqa: E402
from judge.acted import STATUSES  # noqa: E402


def _outside_repo(path: Path, flag: str) -> Path:
    path = path.expanduser().resolve()
    if path == REPO or REPO in path.parents:
        raise SystemExit(f"{flag} must be outside the repository: {path}")
    return path


class _Digest:
    """Writes canonical JSONL lines to a file and hashes them as it goes."""

    def __init__(self, path: Path):
        self.f = open(path, "w", encoding="utf-8", newline="\n")
        self.h = hashlib.sha256()

    def write(self, record: dict) -> None:
        line = P.serialize(record) + "\n"
        self.f.write(line)
        self.h.update(line.encode("utf-8"))

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


def _jsonl(path: Path):
    """Yield (line SHA-256, record) for each line, and hash the file from the same bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for raw in f:
            h.update(raw)
            line = raw.decode("utf-8").rstrip("\n")
            yield hashlib.sha256(line.encode("utf-8")).hexdigest(), json.loads(line)
    yield h.hexdigest(), None


def run(args) -> int:
    data_root = _outside_repo(Path(args.data_root), "--data-root")
    out = _outside_repo(Path(args.out), "--out")
    if out.exists():
        raise SystemExit(f"--out {out} already exists; outputs are never overwritten, "
                         "so choose a new folder")
    fence_config = fence.load_config(required=True)
    fence.install("dev_only", data_root, fence_config)
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.staging-", dir=out.parent))
    try:
        all_ok = _run_into(args, staging)
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


def _run_into(args, out: Path) -> bool:
    """Build every pack, writing every output into ``out``; return whether all checks pass."""
    view_path = Path(args.monitor_view).expanduser().resolve()
    prov_path = Path(args.provenance).expanduser().resolve()

    # 1. Provenance first: every record must be from the development split.
    provs, prov_lines = [], {}
    for line_hash, prov in _jsonl(prov_path):
        if prov is None:
            prov_file_hash = line_hash
            break
        provs.append(prov)
        prov_lines[prov["trace_key"]] = line_hash
    for prov in provs:
        if prov.get("split") != "dev":
            raise SystemExit(f"{prov.get('trace_key')}: not a development-split record; "
                             "this command packs development data only")
    if len(prov_lines) != len(provs):
        raise ValueError("the provenance file lists a trace key twice")

    # 2. One record at a time, in file order.
    packs = _Digest(out / "packs.jsonl")
    images = _Digest(out / "image_manifest.jsonl")
    view_lines, pack_hashes, summary, leak_checks = {}, {}, [], []
    totals, cut, determinism_ok, i = Counter(), 0, True, 0
    for line_hash, view in _jsonl(view_path):
        if view is None:
            view_file_hash = line_hash
            break
        if i >= len(provs) or provs[i]["trace_key"] != view.get("trace_key"):
            raise ValueError(f"monitor-view record {i} ({view.get('trace_key')}) does not match "
                             "the provenance record in the same position")
        prov = provs[i]
        i += 1
        key = view["trace_key"]
        pack, data, digest, manifest = P.build(view, prov)
        determinism_ok &= P.build(view, prov)[1] == data
        check = P.check_pack(pack, prov)
        leak_checks.append(dict(check, trace_key=key))
        packs.write({"trace_key": key, "pack_sha256": digest, "pack": pack})
        for entry in manifest:
            images.write(entry)
        counts = Counter(d["status"] for s in pack["steps"] for d in (s["acted_elements"] or []))
        totals.update(counts)
        truncation = pack["final_tree"]["packer_truncation"]
        cut += truncation["applied"]
        view_lines[key], pack_hashes[key] = line_hash, digest
        summary.append({"trace_key": key, "pack_sha256": digest, "ok": check["ok"],
                        "steps": len(pack["steps"]), "screenshot_refs": len(manifest),
                        "acted_element_status_counts": {s: counts[s] for s in STATUSES},
                        "final_tree_packer_truncation": truncation})
        del view, pack, data
    if i != len(provs):
        raise ValueError(f"the provenance file has {len(provs)} records, the monitor view {i}")
    output_hashes = {"packs.jsonl": packs.close(), "image_manifest.jsonl": images.close()}

    # 3. Report.
    all_ok = all(c["ok"] for c in leak_checks) and determinism_ok
    report = {
        "ok": all_ok,
        "settings": {
            "pack_version": P.PACK_VERSION, "monitor_view_schema_version": SCHEMA_VERSION,
            "final_tree_cap_chars": P.FINAL_TREE_CAP,
            "final_tree_kept_head_chars": P.FINAL_TREE_HEAD,
            "final_tree_kept_tail_chars": P.FINAL_TREE_TAIL,
            "python": platform.python_version(), "code": _code_version(),
            "inputs": {"monitor_view": str(view_path), "provenance": str(prov_path)},
            "fence": {"profiles": fence.installed_profiles(),
                      "config_sha256": (fetch.sha256_file(Path(os.environ[fence.CONFIG_ENV]).expanduser())
                                        if os.environ.get(fence.CONFIG_ENV) else None)},
        },
        "hashes": {
            "pinned_source": {
                "note": "ARB source files at the pinned revision, as recorded by the data path",
                "trajectory_sha256": {p["trace_key"]: (p.get("file") or {}).get("sha256") for p in provs},
                "revision": sorted({p.get("revision") for p in provs if p.get("revision")}),
            },
            "normalized": {
                "note": "the data path's outputs that these packs were built from",
                "monitor_view.jsonl": view_file_hash, "provenance.jsonl": prov_file_hash,
                "monitor_view_record_sha256": view_lines,
                "provenance_record_sha256": prov_lines,
            },
            "packs": {
                "note": f"{P.PACK_VERSION}; a pack's hash is the SHA-256 of its canonical bytes",
                "outputs_sha256": output_hashes,
                "pack_sha256": pack_hashes,
            },
        },
        "acted_element_status_counts": {s: totals[s] for s in STATUSES},
        "final_tree_truncation_counts": {"cut_by_packer": cut, "kept_whole": len(summary) - cut},
        "summary": summary,
        "checks": {
            "leakage": leak_checks,
            "determinism": {"ok": determinism_ok, "method": "each pack built twice, bytes compared"},
        },
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                     encoding="utf-8")

    for s in summary:
        c = s["acted_element_status_counts"]
        print(f"{s['trace_key']}  steps={s['steps']:<3} "
              + " ".join(f"{k}={c[k]}" for k in STATUSES)
              + f"  tree={'cut' if s['final_tree_packer_truncation']['applied'] else 'whole'}"
              + f"  pack={s['pack_sha256'][:12]}")
    print(f"packs: {len(summary)}; leakage: {sum(c['ok'] for c in leak_checks)}/{len(leak_checks)}; "
          f"determinism: {'ok' if determinism_ok else 'FAIL'}; "
          + " ".join(f"{k}={totals[k]}" for k in STATUSES))
    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True,
                        help="local data directory the guard covers, outside the repository")
    parser.add_argument("--monitor-view", required=True, help="monitor_view.jsonl from the data path")
    parser.add_argument("--provenance", required=True, help="provenance.jsonl from the same run")
    parser.add_argument("--out", required=True,
                        help="new output directory outside the repository; must not exist yet")
    return run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
