"""A tripwire on file opens for runs that must not read labels or evaluation data.

``install(profile, data_root, config)`` adds a Python audit hook that raises
``PermissionError`` when the current process opens a guarded path, or requests a guarded
dataset URL through ``urllib``. It sees only what happens inside a Python process that
installs it. It does not see other processes, native code that opens files without
Python, directory listings, opens made relative to a directory file descriptor, or
deliberate changes to the guard inside the process. It is a tripwire, not a proof.

Profiles, each guarding everything the previous one does:

- ``label_free``: the label file (``raw/data/annotations.csv`` under the data root, and
  any ``label_files`` in the config), any ``annotations.csv.part`` and any ``judgments``
  folder under the data root, and the config's ``fenced_dirs``, ``fenced_files``,
  ``judgment_dirs``, ``archive_dirs`` and ``transcript_dirs``. Dataset URLs under
  ``judgments/`` are refused, and so is the label file's URL outside the download path.
- ``dev_only``: also trajectories and screenshots of tasks outside the development split
  (under ``raw/cleaned`` and ``raw/screenshots``, including partial downloads, recognized
  by task ID from ``raw/data/splits.csv``, read once at install), their dataset URLs, and
  the config's ``test_output_dirs``.
- ``pre_join``: also the config's ``eval_output_dirs`` and ``join_dirs``.

Paths are compared without regard to letter case, and the label files are also
recognized by file identity, so a hard link or a case variant is guarded too.

Only three reviewed readers may open the label file, and only when they are the code that
calls ``open``: ``datapath.fetch.fetch_file`` (its hash check, and its download, which is
the only code that may write or read ``annotations.csv.part``), ``datapath.labels.load_labels``
when called for the development split alone, with no lock file and with the split table
built from the data root's own ``splits.csv``, and ``datapath.labels.annotation_structure``.

Paths other than those under the data root come from a local JSON file named by the
``SPAR_FENCE_CONFIG`` environment variable. None is written into this repository.
Installing again adds another hook; guards only ever accumulate.
"""

import csv
import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from urllib.parse import unquote, urlsplit

from . import fetch, labels
from .ids import Splits, normalize_task_id

# Bound at import, so rebinding these names later in the process does not change the hook.
_getframe, _realpath, _expanduser, _fsdecode, _stat = (
    sys._getframe, os.path.realpath, os.path.expanduser, os.fsdecode, os.stat)
_SPLITS_TYPE = Splits
_CODE = {name: fn.__code__ for name, fn in (
    ("fetch_file", fetch.fetch_file), ("check_file", fetch.check_file),
    ("sha256_file", fetch.sha256_file), ("git_blob_sha1", fetch.git_blob_sha1),
    ("download", fetch._download), ("load_labels", labels.load_labels),
    ("annotation_structure", labels.annotation_structure))}

PROFILES = ("label_free", "dev_only", "pre_join")
CONFIG_ENV = "SPAR_FENCE_CONFIG"
CONFIG_KEYS = ("label_files", "fenced_dirs", "fenced_files", "judgment_dirs", "archive_dirs",
               "transcript_dirs", "test_output_dirs", "eval_output_dirs", "join_dirs")
LABEL_PART_NAME = "annotations.csv.part"
_RESOLVE_PATH = re.compile(r"/resolve/[^/]+/(?P<path>.+)$")

_active = {"profiles": []}


def active_profile():
    """The profile of the most recent install in this process, or None."""
    return _active["profiles"][-1] if _active["profiles"] else None


def installed_profiles() -> list:
    """Every profile installed in this process, in order."""
    return list(_active["profiles"])


def load_config(path=None, required: bool = False) -> dict:
    """Read the local fence config named by ``path`` or ``SPAR_FENCE_CONFIG``.

    Returns {} when neither is set, unless ``required``, in which case it stops the run.
    Every value must be a list of non-empty strings.
    """
    path = path or os.environ.get(CONFIG_ENV)
    if not path:
        if required:
            raise SystemExit(f"set {CONFIG_ENV} to the local fence config file")
        return {}
    config = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("the fence config must be a JSON object")
    unknown = set(config) - set(CONFIG_KEYS)
    if unknown:
        raise ValueError(f"unknown fence config keys: {sorted(unknown)}")
    for key, value in config.items():
        if not (isinstance(value, list) and all(isinstance(v, str) and v for v in value)):
            raise ValueError(f"fence config {key!r} must be a list of non-empty strings")
    return config


def _real(p) -> str:
    return _realpath(_expanduser(_fsdecode(p)))


def _fold(p: str) -> str:
    return unicodedata.normalize("NFC", p).casefold()


def _under(path: str, roots) -> bool:
    return any(path == r or path.startswith(r + os.sep) for r in roots)


def _identity(path: str):
    try:
        st = _stat(path)
    except OSError:
        return None
    return (st.st_dev, st.st_ino)


def _dev_tasks(splits_csv) -> frozenset:
    try:
        with open(splits_csv, newline="", encoding="utf-8") as f:
            return frozenset(normalize_task_id(r["task_id"]) for r in csv.DictReader(f)
                             if r["split"] == "dev")
    except OSError:
        return frozenset()


def _chain(frame, names) -> bool:
    """Whether the frames calling ``frame`` are exactly the named reviewed readers, in order."""
    for name in names:
        frame = frame.f_back
        if frame is None or frame.f_code is not _CODE[name]:
            return False
    return True


def install(profile: str, data_root, config: dict = None) -> None:
    """Install the guard for ``profile`` over ``data_root`` and the config's paths."""
    if profile not in PROFILES:
        raise ValueError(f"unknown fence profile {profile!r}; use one of {PROFILES}")
    config = config or {}
    level = PROFILES.index(profile)
    root = _real(data_root)
    raw = os.path.join(root, "raw")
    splits_csv = os.path.join(raw, "data", "splits.csv")
    label_paths = [os.path.join(raw, "data", "annotations.csv")] + [
        _real(p) for p in config.get("label_files", [])]
    label_folded = frozenset(_fold(p) for p in label_paths)
    fenced_files = [_real(p) for p in config.get("fenced_files", [])]
    fenced_folded = frozenset(_fold(p) for p in fenced_files)
    dir_keys = ["fenced_dirs", "judgment_dirs", "archive_dirs", "transcript_dirs"]
    if level >= 1:
        dir_keys.append("test_output_dirs")
    if level >= 2:
        dir_keys += ["eval_output_dirs", "join_dirs"]
    blocked_dirs = tuple(_fold(_real(p)) for k in dir_keys for p in config.get(k, []))
    root_f = _fold(root)
    cleaned_f = _fold(os.path.join(raw, "cleaned"))
    shots_f = _fold(os.path.join(raw, "screenshots"))
    dev_folded = frozenset(_fold(t) for t in _dev_tasks(splits_csv)) if level >= 1 else frozenset()
    canonical = {}

    def canonical_dev() -> frozenset:
        if "dev" not in canonical:
            canonical["dev"] = _dev_tasks(splits_csv)
        return canonical["dev"]

    def is_test_task(task: str) -> bool:
        return _fold(normalize_task_id(task)) not in dev_folded

    def guarded(path: str):
        f = _fold(path)
        name = os.path.basename(f)
        if f in label_folded:
            return "label"
        ident = _identity(path)
        if ident is not None and ident in {_identity(p) for p in label_paths}:
            return "label"
        if f in fenced_folded or (ident is not None and ident in {_identity(p) for p in fenced_files}):
            return "fenced file"
        in_root = _under(f, [root_f])
        if name == LABEL_PART_NAME and in_root:
            return "label_part"
        if in_root and "judgments" in Path(f).relative_to(root_f).parts:
            return "judgments"
        if _under(f, blocked_dirs):
            return "blocked folder"
        if level >= 1:
            stem = name[:-len(".part")] if name.endswith(".part") else name
            if _under(f, [cleaned_f]) and stem.endswith(".json") and is_test_task(stem[:-len(".json")]):
                return "evaluation trajectory"
            if _under(f, [shots_f]) and is_test_task(os.path.basename(os.path.dirname(f))):
                return "evaluation screenshot"
        return None

    def load_labels_admitted(frame) -> bool:
        loc = frame.f_locals
        if tuple(loc.get("allowed_splits", ())) != ("dev",) or loc.get("lock_file") is not None:
            return False
        sp = loc.get("splits")
        if type(sp) is not _SPLITS_TYPE:
            return False
        claimed_dev = {task for task, (_, split) in sp._map.items() if split == "dev"}
        return claimed_dev <= canonical_dev()

    def opener_admitted(kind: str) -> bool:
        frame = _getframe(2)                      # the code that called open
        code = frame.f_code
        if code in (_CODE["sha256_file"], _CODE["git_blob_sha1"]):
            return _chain(frame, ("check_file", "fetch_file"))
        if kind == "label_part":
            return code is _CODE["download"] and _chain(frame, ("fetch_file",))
        if code is _CODE["annotation_structure"]:
            return True
        if code is _CODE["load_labels"]:
            return load_labels_admitted(frame)
        return False

    def on_stack(name: str) -> bool:
        frame = _getframe(2)
        while frame is not None:
            if frame.f_code is _CODE[name]:
                return True
            frame = frame.f_back
        return False

    def url_refusal(url: str):
        path = unquote(urlsplit(url).path)
        if "/judgments/" in _fold(path) + "/":
            return "judgments URL"
        m = _RESOLVE_PATH.search(path)
        if m is None:
            return None
        dpath = m.group("path")
        if _fold(dpath) == "data/annotations.csv" and not on_stack("fetch_file"):
            return "label URL"
        if level >= 1:
            parts = dpath.split("/")
            if parts[0] == "cleaned" and dpath.endswith(".json") and is_test_task(parts[-1][:-len(".json")]):
                return "evaluation trajectory URL"
            if parts[0] == "screenshots" and len(parts) >= 3 and is_test_task(parts[-2]):
                return "evaluation screenshot URL"
        return None

    def hook(event, args):
        if event == "urllib.Request":
            if args and isinstance(args[0], str):
                kind = url_refusal(args[0])
                if kind is not None:
                    raise PermissionError(f"fence ({profile}): {kind} is guarded: {args[0]}")
            return
        if event != "open" or not args or isinstance(args[0], int):
            return
        path = _real(args[0])
        kind = guarded(path)
        if kind is None:
            return
        if kind in ("label", "label_part") and opener_admitted(kind):
            return
        raise PermissionError(f"fence ({profile}): {kind} is guarded: {path}")

    sys.addaudithook(hook)
    _active["profiles"].append(profile)
