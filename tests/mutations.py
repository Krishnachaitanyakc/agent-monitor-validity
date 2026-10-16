#!/usr/bin/env python3
"""Deliberate breaks that the test suite must catch.

Each mutation copies the code to a temporary folder, makes one exact edit that
reintroduces a leak or a weak check, runs the unit tests there, and expects them to
fail. Run from the repository root:

    python3 tests/mutations.py

Exit code 0 means every mutation was caught.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

MUTATIONS = [
    ("read a test row's label field before the split check", "datapath/labels.py",
     '            if splits.split(row["task_id"]) not in allowed:\n',
     '            _ = row["trajectory_success"]\n            if splits.split(row["task_id"]) not in allowed:\n'),
    ("read every field of a test row before the split check", "datapath/labels.py",
     '            if splits.split(row["task_id"]) not in allowed:\n',
     '            _ = list(row.values())\n            if splits.split(row["task_id"]) not in allowed:\n'),
    ("copy the agent's prompts into the monitor view", "datapath/normalize.py",
     '"elements": step.get("extra_element_properties") or {},',
     '"elements": step.get("extra_element_properties") or {},\n            "prompt": step.get("chat_messages"),'),
    ("remove the schema allowlist", "datapath/leakage.py",
     "    l0 = _unexpected_keys(rec)", "    l0 = []"),
    ("skip the hostname rewrite", "datapath/normalize.py",
     'v, n = HOST_COPY.subn(r"\\1\\2", v)', "n = 0"),
    ("put screenshot hashes back in the monitor view", "datapath/normalize.py",
     'shot = {"width": info["width"],', 'shot = {"sha256": sha256_file(f), "width": info["width"],'),
    ("check LFS downloads by size only", "datapath/fetch.py",
     'size == lfs["size"] and sha == lfs["oid"]', 'size == lfs["size"]'),
    ("check Git blob downloads by size only", "datapath/fetch.py",
     'size == entry["size"] and git_blob_sha1(path) == entry["oid"]', 'size == entry["size"]'),
    ("re-enable evaluation label unlocking", "datapath/labels.py",
     'if tuple(allowed_splits) != ("dev",) or lock_file is not None:', "if False:"),
    ("publish into an existing output folder", "scripts/arb_datapath.py",
     "    if out.exists():", "    if False:"),
    ("write outputs straight into --out without staging", "scripts/arb_datapath.py",
     'staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.staging-", dir=out.parent))', "staging = out"),
    ("fetch screenshots before checking whose they are", "scripts/arb_datapath.py",
     "        check_screenshot_paths(source, task_id)     # before any screenshot is fetched or opened",
     "        pass"),
    ("accept nested values in text fields", "datapath/normalize.py",
     "if isinstance(v, (dict, list, tuple)):", "if False:"),
    ("accept non-finite numbers in element properties", "datapath/leakage.py",
     "return isinstance(v, float) and math.isfinite(v)", "return isinstance(v, float)"),
    ("check only the first screenshot's trajectory and step", "datapath/normalize.py",
     "    for step in steps:\n", "    for step in steps[:1]:\n"),
    ("accept any element property value", "datapath/leakage.py",
     'bad += [f"/events[{i}]/elements/{bid}/{k} (shape)" for k in element_shape_errors(props)]',
     "bad += []"),
    ("check hostnames without URL-unquoting", "datapath/leakage.py",
     "for t in (text, unquoted)", "for t in (text,)"),
    ("remove the label guard from the data-path command", "scripts/arb_datapath.py",
     '    fence.install("label_free", data_root, fence.load_config())', "    pass"),
    ("remove the development-only guard from the data-path command", "scripts/arb_datapath.py",
     '        fence.install("dev_only", data_root, fence.load_config())', "        pass"),
    ("let any caller open the label file", "datapath/fence.py",
     'if kind == "label" and _called_by_reviewed_reader(fetch_only=False):', 'if kind == "label":'),
]


def main() -> int:
    missed = 0
    for name, rel, old, new in MUTATIONS:
        with tempfile.TemporaryDirectory() as tmp:
            for part in ("datapath", "scripts", "tests"):
                shutil.copytree(REPO / part, Path(tmp) / part,
                                ignore=shutil.ignore_patterns("__pycache__"))
            target = Path(tmp) / rel
            text = target.read_text(encoding="utf-8")
            if text.count(old) != 1:
                print(f"SETUP ERROR  {name}: expected exactly one match in {rel}")
                missed += 1
                continue
            target.write_text(text.replace(old, new), encoding="utf-8")
            result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"],
                                    cwd=tmp, capture_output=True, text=True)
            caught = result.returncode != 0
            missed += not caught
            print(f"{'caught' if caught else 'MISSED'}  {name}")
    print(f"{len(MUTATIONS) - missed} of {len(MUTATIONS)} mutations caught")
    return 0 if missed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
