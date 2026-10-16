"""A tripwire on file opens for runs that must not read labels or evaluation data.

``install(profile, data_root, config)`` adds a Python audit hook that raises
``PermissionError`` when the current process opens a guarded path. It sees only file
opens made inside a Python process that installs it (``open``, ``io.open``, ``os.open``
and everything built on them). It does not see other processes, native code that opens
files without Python, or directory listings, so it is a tripwire, not a proof.

Profiles, each guarding everything the previous one does:

- ``label_free``: the label file (``raw/data/annotations.csv`` under the data root, and
  any ``label_files`` in the config), any ``annotations.csv.part`` and any ``judgments``
  folder under the data root, and the config's ``fenced_dirs``, ``fenced_files``,
  ``archive_dirs`` and ``transcript_dirs``.
- ``dev_only``: also trajectories and screenshots of tasks outside the development split
  (under ``raw/cleaned`` and ``raw/screenshots``, recognized by task ID from
  ``raw/data/splits.csv``, read once at install), and the config's ``test_output_dirs``.
- ``pre_join``: also the config's ``eval_output_dirs`` and ``join_dirs``.

Three reviewed readers may open the label file: ``datapath.fetch.fetch_file`` (download
and hash check; the only code that may write or read ``annotations.csv.part``),
``datapath.labels.load_labels`` when called for the development split alone with no lock
file, and ``datapath.labels.annotation_structure``. Nothing else may.

Paths other than those under the data root come from a local JSON file named by the
``SPAR_FENCE_CONFIG`` environment variable. None is written into this repository.
Installing again adds another hook; guards only ever accumulate.
"""

import csv
import json
import os
import sys
from pathlib import Path

from . import fetch, labels
from .ids import normalize_task_id

PROFILES = ("label_free", "dev_only", "pre_join")
CONFIG_ENV = "SPAR_FENCE_CONFIG"
CONFIG_KEYS = ("label_files", "fenced_dirs", "fenced_files", "archive_dirs", "transcript_dirs",
               "test_output_dirs", "eval_output_dirs", "join_dirs")
LABEL_PART_NAME = "annotations.csv.part"

_active = {"profiles": []}


def active_profile():
    """The profile of the most recent install in this process, or None."""
    return _active["profiles"][-1] if _active["profiles"] else None


def installed_profiles() -> list:
    """Every profile installed in this process, in order."""
    return list(_active["profiles"])


def load_config(path=None) -> dict:
    """Read the local fence config named by ``path`` or ``SPAR_FENCE_CONFIG``; {} if neither is set."""
    path = path or os.environ.get(CONFIG_ENV)
    if not path:
        return {}
    config = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    unknown = set(config) - set(CONFIG_KEYS)
    if unknown:
        raise ValueError(f"unknown fence config keys: {sorted(unknown)}")
    return config


def _real(p) -> str:
    return os.path.realpath(os.path.expanduser(os.fsdecode(p)))


def _under(path: str, roots) -> bool:
    return any(path == r or path.startswith(r + os.sep) for r in roots)


def _dev_tasks(splits_csv: Path) -> frozenset:
    with open(splits_csv, newline="", encoding="utf-8") as f:
        return frozenset(normalize_task_id(r["task_id"]) for r in csv.DictReader(f)
                         if r["split"] == "dev")


def _called_by_reviewed_reader(fetch_only: bool) -> bool:
    """Whether a reviewed reader is on the call stack; ``fetch_only`` admits only fetch_file."""
    frame = sys._getframe(2)
    while frame is not None:
        code = frame.f_code
        if code is fetch.fetch_file.__code__:
            return True
        if not fetch_only:
            if code is labels.annotation_structure.__code__:
                return True
            if code is labels.load_labels.__code__:
                loc = frame.f_locals
                return tuple(loc.get("allowed_splits", ())) == ("dev",) and loc.get("lock_file") is None
        frame = frame.f_back
    return False


def install(profile: str, data_root, config: dict = None) -> None:
    """Install the guard for ``profile`` over ``data_root`` and the config's paths."""
    if profile not in PROFILES:
        raise ValueError(f"unknown fence profile {profile!r}; use one of {PROFILES}")
    config = config or {}
    level = PROFILES.index(profile)
    root = Path(_real(data_root))
    raw = root / "raw"
    label_files = frozenset([str(raw / "data" / "annotations.csv")]
                            + [_real(p) for p in config.get("label_files", [])])
    fenced_files = frozenset(_real(p) for p in config.get("fenced_files", []))
    blocked_dirs = [_real(p) for k in ("fenced_dirs", "archive_dirs", "transcript_dirs")
                    for p in config.get(k, [])]
    if level >= 1:
        blocked_dirs += [_real(p) for p in config.get("test_output_dirs", [])]
    if level >= 2:
        blocked_dirs += [_real(p) for k in ("eval_output_dirs", "join_dirs") for p in config.get(k, [])]
    blocked_dirs = tuple(blocked_dirs)
    dev_tasks = _dev_tasks(raw / "data" / "splits.csv") if level >= 1 else frozenset()
    cleaned, shots = str(raw / "cleaned"), str(raw / "screenshots")

    def guarded(path: str):
        name = os.path.basename(path)
        if path in label_files:
            return "label"
        if path in fenced_files:
            return "fenced file"
        in_root = _under(path, [str(root)])
        if name == LABEL_PART_NAME and in_root:
            return "label_part"
        if in_root and "judgments" in Path(path).relative_to(root).parts:
            return "judgments"
        if _under(path, blocked_dirs):
            return "blocked folder"
        if level >= 1:
            if _under(path, [cleaned]) and name.endswith(".json"):
                if normalize_task_id(name[:-len(".json")]) not in dev_tasks:
                    return "evaluation trajectory"
            elif _under(path, [shots]):
                if normalize_task_id(os.path.basename(os.path.dirname(path))) not in dev_tasks:
                    return "evaluation screenshot"
        return None

    def hook(event, args):
        if event != "open" or not args or isinstance(args[0], int):
            return
        path = _real(args[0])
        kind = guarded(path)
        if kind is None:
            return
        if kind == "label" and _called_by_reviewed_reader(fetch_only=False):
            return
        if kind == "label_part" and _called_by_reviewed_reader(fetch_only=True):
            return
        raise PermissionError(f"fence ({profile}): {kind} is guarded: {path}")

    sys.addaudithook(hook)
    _active["profiles"].append(profile)
