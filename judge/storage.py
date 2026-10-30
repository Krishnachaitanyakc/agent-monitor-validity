"""A write-once log of every attempt, halt and correction, kept outside the repository.

Layout under the log root:

    attempts/<trace key>/<config hash>/<judge><repeat>/<attempt>.json   the attempt record
    attempts/<trace key>/<config hash>/<judge><repeat>/<attempt>.raw    the raw response bytes
    halts/<halt id>.json             one per HALT: the error, the slot and the attempt
    corrections/<halt id>.json       the required correction note for that halt
    first_model/<config hash>/<judge>.json      the model string on the first response
    response_ids/<SHA-256 of provider and id>.json   every complete response's id, once

Every file is written once: the bytes go to a temporary file, which is then hard-linked
to its final name, so a file never appears half written and an existing file is never
replaced (``WriteOnceError``). Attempt indices run 1, 2, 3 and so on in each slot, with no
gap or repeat. A run cannot resume while any halt lacks a correction note.
"""

import hashlib
import json
import os
import re
from pathlib import Path

_SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


class WriteOnceError(FileExistsError):
    """A write-once file already exists."""


def write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(tmp, path)
    except FileExistsError:
        raise WriteOnceError(f"{path} already exists; the log is write-once") from None
    finally:
        if tmp.exists():
            tmp.unlink()


def _json(obj) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _name(value: str, what: str) -> str:
    if not isinstance(value, str) or not _SAFE.match(value):
        raise ValueError(f"{what} {value!r} cannot be used in a log path")
    return value


class AttemptLog:
    def __init__(self, root):
        self.root = Path(root)

    # -------------------------------------------------------------- attempts

    def slot_dir(self, trace_key, config_sha256, judge, repeat) -> Path:
        if judge not in ("A", "B") or not (isinstance(repeat, int) and repeat >= 1):
            raise ValueError(f"bad slot {judge}{repeat}")
        return (self.root / "attempts" / _name(trace_key, "trace key")
                / _name(config_sha256, "config hash") / f"{judge}{repeat}")

    def attempts(self, trace_key, config_sha256, judge, repeat) -> list:
        """The slot's attempt records in order; raises if the indices have a gap."""
        folder = self.slot_dir(trace_key, config_sha256, judge, repeat)
        if not folder.exists():
            return []
        numbers = sorted(int(p.stem) for p in folder.glob("*.json"))
        if numbers != list(range(1, len(numbers) + 1)):
            raise ValueError(f"{folder}: attempt indices {numbers} are not 1..{len(numbers)}")
        return [json.loads((folder / f"{n}.json").read_text(encoding="utf-8")) for n in numbers]

    def write_attempt(self, record: dict, raw: bytes) -> Path:
        """Write one attempt: raw bytes, then the record. The index must be the next one."""
        key = (record["trace_key"], record["config_sha256"], record["judge"], record["repeat"])
        n = record["attempt"]
        if n != len(self.attempts(*key)) + 1:
            raise ValueError(f"attempt {n} is not the next index for slot {record['judge']}{record['repeat']}")
        folder = self.slot_dir(*key)
        raw = raw or b""
        record = dict(record, raw_sha256=hashlib.sha256(raw).hexdigest(), raw_bytes=len(raw))
        write_once(folder / f"{n}.raw", raw)
        write_once(folder / f"{n}.json", _json(record))
        return folder / f"{n}.json"

    # -------------------------------------------------------------- halts

    def halts(self) -> list:
        folder = self.root / "halts"
        return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))] if folder.exists() else []

    def write_halt(self, record: dict) -> str:
        halt_id = f"halt-{len(self.halts()) + 1:04d}"
        write_once(self.root / "halts" / f"{halt_id}.json", _json(dict(record, halt_id=halt_id)))
        return halt_id

    def correction(self, halt_id):
        path = self.root / "corrections" / f"{_name(halt_id, 'halt id')}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def unresolved_halts(self) -> list:
        """Halts with no correction note yet; a run may not resume while any remain."""
        return [h for h in self.halts() if self.correction(h["halt_id"]) is None]

    def record_correction(self, halt_id: str, note: str, resume_config_sha256: str, at: str) -> None:
        """Record what was corrected before resuming after one halt. The note is required."""
        if not any(h["halt_id"] == halt_id for h in self.halts()):
            raise ValueError(f"no halt {halt_id}")
        if not (isinstance(note, str) and note.strip()):
            raise ValueError("a correction note is required before resuming")
        write_once(self.root / "corrections" / f"{_name(halt_id, 'halt id')}.json",
                   _json({"halt_id": halt_id, "note": note, "resume_config_sha256": resume_config_sha256,
                          "recorded_at": at}))

    # -------------------------------------------------------------- response identity

    def first_model(self, config_sha256, judge):
        path = self.root / "first_model" / _name(config_sha256, "config hash") / f"{judge}.json"
        return json.loads(path.read_text(encoding="utf-8"))["model"] if path.exists() else None

    def set_first_model(self, config_sha256, judge, model, attempt_ref) -> None:
        write_once(self.root / "first_model" / _name(config_sha256, "config hash") / f"{judge}.json",
                   _json({"model": model, "first_attempt": attempt_ref}))

    def claim_response_id(self, provider: str, response_id: str, attempt_ref: str) -> bool:
        """Record a complete response's provider id; False if it was seen before."""
        digest = hashlib.sha256(f"{provider}\n{response_id}".encode("utf-8")).hexdigest()
        try:
            write_once(self.root / "response_ids" / f"{digest}.json",
                       _json({"provider": provider, "id": response_id, "attempt": attempt_ref}))
        except WriteOnceError:
            return False
        return True
