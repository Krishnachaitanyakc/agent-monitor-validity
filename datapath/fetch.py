"""Pinned, hash-checked downloads from the ARB dataset.

Every file resolves against ``constants.REVISION`` and is checked against the pinned
file listing before it is accepted: Git LFS files by the listed SHA-256, small files by
the listed Git blob SHA-1, and both by byte count. A download is written to a ``.part``
file and moved into place only after it passes. A file already on disk is re-checked;
if it fails, it is downloaded again once, and the run stops if the new copy fails too.
"""

import hashlib
import json
import re
import urllib.request
from pathlib import Path

from . import constants as C


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path: Path) -> str:
    """The Git object ID of a file's contents, as the listing reports for non-LFS files."""
    h = hashlib.sha1(b"blob %d\0" % Path(path).stat().st_size)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_url(path: str) -> str:
    return C.RESOLVE_URL.format(dataset=C.DATASET, rev=C.REVISION, path=path)


def fetch_listing(top: str, cache_dir: Path) -> list:
    """Paginated file listing of ``top`` (e.g. 'cleaned') at the pinned revision.

    Cached as ``cache_dir/<REVISION>/<top>.json`` so a cache can never serve another revision.
    """
    cache = cache_dir / C.REVISION / f"{top}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    url = C.TREE_URL.format(dataset=C.DATASET, rev=C.REVISION, path=top)
    entries = []
    while url:
        with urllib.request.urlopen(url, timeout=60) as r:
            entries += json.load(r)
            link = r.headers.get("Link") or ""
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else None
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".json.part")
    tmp.write_text(json.dumps(entries), encoding="utf-8")
    tmp.replace(cache)
    return entries


def listing_index(entries: list) -> dict:
    """{path: listing entry} for the files in a listing."""
    return {e["path"]: e for e in entries if e.get("type") == "file"}


def check_file(path: Path, entry: dict) -> dict:
    """Compare a local file with its listing entry. Returns {'ok', 'method', 'sha256', 'bytes'}."""
    size = path.stat().st_size
    sha = sha256_file(path)
    lfs = entry.get("lfs")
    if lfs:
        ok, method = size == lfs["size"] and sha == lfs["oid"], "lfs-sha256"
    else:
        ok, method = size == entry["size"] and git_blob_sha1(path) == entry["oid"], "git-blob-sha1"
    return {"ok": ok, "method": method, "sha256": sha, "bytes": size}


def _download(path: str, tmp: Path) -> None:
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(resolve_url(path), timeout=120) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def fetch_file(path: str, dest_root: Path, entry: dict) -> dict:
    """Ensure ``dest_root/path`` holds the pinned bytes of ``path``; return its record."""
    if entry is None:
        raise KeyError(f"{path} is not in the pinned listing")
    dest = dest_root / path
    downloaded = False
    if dest.exists():
        check = check_file(dest, entry)
        redownloaded = not check["ok"]
    else:
        check, redownloaded = None, False
    if check is None or not check["ok"]:
        tmp = dest.with_suffix(dest.suffix + ".part")
        _download(path, tmp)
        check = check_file(tmp, entry)
        if not check["ok"]:
            tmp.unlink()
            raise IOError(f"{path}: downloaded bytes do not match the pinned listing")
        tmp.replace(dest)
        downloaded = True
    return {
        "path": path,
        "url": resolve_url(path),
        "revision": C.REVISION,
        "bytes": check["bytes"],
        "sha256": check["sha256"],
        "checked_against_listing": check["method"],
        "downloaded": downloaded,
        "redownloaded_after_failed_check": redownloaded,
    }


def find_trajectory(index: dict, benchmark: str, agent: str, task_id: str) -> str:
    """Path of the cleaned JSON for one (benchmark, agent, raw task_id)."""
    hits = [p for p in index
            if p.startswith(f"cleaned/{benchmark}/{agent}/") and p.endswith(f"/{task_id}.json")]
    if len(hits) != 1:
        raise KeyError(f"expected one cleaned file for {benchmark}/{agent}/{task_id}, found {len(hits)}")
    return hits[0]


def screenshot_listing_path(screenshot_path: str) -> str:
    """Map a step's ``screenshot_path`` to its path in the dataset listing.

    Steps store 'trajectories/screenshots/<bench>/<agent>/<task>/screenshot_step_N.png';
    the dataset stores 'screenshots/<bench>/<agent>/<task>/screenshot_step_N.png'.
    """
    marker = "screenshots/"
    i = screenshot_path.find(marker)
    if i < 0:
        raise ValueError(f"unexpected screenshot path {screenshot_path!r}")
    return screenshot_path[i:]
