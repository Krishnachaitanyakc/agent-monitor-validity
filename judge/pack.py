"""One monitor-view record -> one evidence pack, the bytes every judge reads.

A pack holds:
- the goal and any image URLs found in it;
- for each step: the step number, URL, the action exactly as recorded, what the action
  acted on (``acted.describe_action`` on that step's own tree and element properties),
  the agent's reasoning, labelled as the agent's own claim, the error recorded in that
  step's observation, labelled as being about the previous action, and an opaque
  screenshot reference;
- the final step's full accessibility tree, capped at ``FINAL_TREE_CAP`` characters: the
  first 60% and the last 40% of the cap, joined by an explicit marker. What the packer cut
  is recorded under ``packer_truncation``. The pack says nothing about how the run ended.

A pack is built from the monitor-view record alone, which carries no labels, rewards or
checker output. It carries no trace key, file path, task ID, agent or model name,
screenshot hash or byte count. A screenshot reference maps to the dataset path and
SHA-256 only in the private image manifest, which is built from the provenance record and
used to attach the exact image bytes.

``PACK_VERSION`` versions this format apart from the monitor-view schema, so a change to
packs never changes the normalized records or their hashes.
"""

import hashlib
import json

from datapath import constants as C
from datapath.leakage import identity_strings
from datapath.normalize import SCHEMA_VERSION

from .acted import describe_action

PACK_VERSION = "pack-0.1"
FINAL_TREE_CAP = 32000
FINAL_TREE_HEAD = FINAL_TREE_CAP * 3 // 5             # 60% of the cap: 19,200 characters
FINAL_TREE_TAIL = FINAL_TREE_CAP - FINAL_TREE_HEAD    # 40% of the cap: 12,800 characters

FIELD_NOTES = {
    "acted_elements": (
        "The elements this step's action names, looked up in this step's own accessibility "
        "tree and element properties, which describe the page the agent saw when it chose the "
        "action. resolved: one tree line has the ID. ambiguous: several do, and all are listed. "
        "nonvisible: no tree line has the ID, but the element properties list it. absent: this "
        "step has no record of the ID. no_target: the action names no element. unparsed: the "
        "action could not be read. Null when the step has no action."),
    "error_reported_for_previous_action": (
        "The error recorded in this step's observation, exactly as recorded. It reports the "
        "outcome of the previous step's action, not of this step's action."),
    "final_tree": (
        "The final step's accessibility tree. A tree longer than the cap keeps its first 60% "
        "and last 40% of the cap around a marker, and packer_truncation records the cut. The "
        "cut is made by the packer and is not part of the run."),
    "reasoning_claimed_by_agent": (
        "The agent's own account, written when it chose this step's action. It is the "
        "agent's claim, not a record of what happened."),
    "screenshot": (
        "A reference to this step's screenshot, which is attached separately. Null when the "
        "step has none."),
}

# Keys a pack never has at any depth: bookkeeping that stays in provenance, and every
# fenced or label key of the monitor view.
PRIVATE_KEYS = frozenset({"trace_key", "trace_id", "task_id", "task_id_normalized", "split",
                          "path", "sha256", "bytes", "ending"}) | C.FORBIDDEN_KEYS


def serialize(obj) -> str:
    """Canonical one-line JSON (sorted keys, compact) so identical packs give identical bytes.

    Non-finite numbers (NaN, infinity) are not valid JSON and raise ValueError.
    """
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False)


def sha256_text(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def image_ref(step: int) -> str:
    """The opaque reference a pack uses for a step's screenshot."""
    return f"image-{step}"


def cap_tree(text):
    """(kept text, packer_truncation record) for the final step's tree."""
    n = len(text or "")
    if n <= FINAL_TREE_CAP:
        return text, {"applied": False, "original_chars": n, "kept_head_chars": n,
                      "kept_tail_chars": 0, "omitted_chars": 0, "marker": None}
    omitted = n - FINAL_TREE_CAP
    marker = f"\n[the packer cut {omitted} characters of this tree here]\n"
    return (text[:FINAL_TREE_HEAD] + marker + text[-FINAL_TREE_TAIL:],
            {"applied": True, "original_chars": n, "kept_head_chars": FINAL_TREE_HEAD,
             "kept_tail_chars": FINAL_TREE_TAIL, "omitted_chars": omitted, "marker": marker})


def build_pack(view: dict) -> dict:
    """The pack for one monitor-view record of schema ``SCHEMA_VERSION``."""
    if view.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"monitor-view schema {view.get('schema_version')!r}; "
                         f"this packer reads {SCHEMA_VERSION!r}")
    events = view.get("events") or []
    if not events or [e.get("step") for e in events] != list(range(len(events))):
        raise ValueError("events must be steps 0, 1, 2 and so on")
    steps = []
    for event in events:
        obs = event.get("observation") or {}
        action = event.get("action")
        # This step's own tree and element properties: the page the agent saw when it chose the action.
        acted = describe_action(action, obs.get("axtree"), event.get("elements")) if action else None
        steps.append({
            "step": event["step"],
            "url": event.get("url"),
            "action": action,
            "acted_elements": acted,
            "reasoning_claimed_by_agent": event.get("reasoning"),
            "error_reported_for_previous_action": event.get("last_action_error"),
            "screenshot": image_ref(event["step"]) if (obs.get("screenshot") or {}).get("png_ok") else None,
        })
    last = events[-1]
    text, cut = cap_tree((last.get("observation") or {}).get("axtree"))
    task = view.get("task") or {}
    return {
        "pack_version": PACK_VERSION,
        "field_notes": dict(FIELD_NOTES),
        "goal": {"text": task.get("goal"), "image_urls": list(task.get("goal_image_urls") or [])},
        "steps": steps,
        "final_tree": {"step": last["step"], "text": text, "packer_truncation": cut},
    }


def image_manifest(pack: dict, provenance: dict, pack_hash: str) -> list:
    """Private: one entry per screenshot reference in the pack, from the provenance record.

    Each entry maps the reference to the screenshot's dataset path, SHA-256 and byte count,
    so the exact image bytes can be attached later. Raises ValueError when a referenced
    step has no present, hashed screenshot in provenance.
    """
    files = {}
    for f in provenance.get("screenshot_files") or []:
        if f.get("step") in files:
            raise ValueError(f"provenance lists two screenshots for step {f.get('step')}")
        files[f.get("step")] = f
    out = []
    for step in pack["steps"]:
        if step["screenshot"] is None:
            continue
        f = files.get(step["step"])
        if not (f and f.get("present") and f.get("sha256")):
            raise ValueError(f"step {step['step']}: the pack references a screenshot that "
                             "provenance does not list as present")
        out.append({"trace_key": provenance["trace_key"], "pack_sha256": pack_hash,
                    "ref": step["screenshot"], "step": step["step"], "path": f["path"],
                    "sha256": f["sha256"], "bytes": f["bytes"]})
    return out


def build(view: dict, provenance: dict):
    """(pack, pack bytes, pack SHA-256, image manifest) for one record and its provenance."""
    if provenance.get("trace_key") != view.get("trace_key"):
        raise ValueError(f"provenance {provenance.get('trace_key')!r} does not belong to "
                         f"monitor view {view.get('trace_key')!r}")
    pack = build_pack(view)
    data = serialize(pack)
    digest = sha256_text(data)
    return pack, data, digest, image_manifest(pack, provenance, digest)


def private_strings(provenance: dict) -> list:
    """Strings from a provenance record that must never appear in its pack."""
    agent = provenance.get("agent") or ""
    out = identity_strings({"agent": agent}, provenance)
    if agent.startswith("GenericAgent-"):
        out.append(agent[len("GenericAgent-"):])         # the model part of the agent name
    f = provenance.get("file") or {}
    out += [f.get("path"), f.get("sha256")]
    for s in provenance.get("screenshot_files") or []:
        out += [s.get("path"), s.get("sha256")]
    return sorted({s for s in out if isinstance(s, str) and s})


def _keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _keys(v)


def check_pack(pack: dict, provenance: dict) -> dict:
    """Leak checks on one pack, given its own provenance record.

    Fails when a private or fenced key appears at any depth, a fenced field name or the
    label file's name appears anywhere, or any identifier, path or hash from the provenance
    record (agent, model, trace ID, raw and normalized task ID, the source file's path and
    SHA-256, each screenshot's path and SHA-256) appears anywhere.
    """
    text = serialize(pack)
    keys = sorted({k for k in _keys(pack) if k in PRIVATE_KEYS})
    substrings = [s for s in C.FORBIDDEN_SUBSTRINGS if s in text]
    found = [s for s in private_strings(provenance) if s in text]
    return {"ok": not (keys or substrings or found), "private_keys": keys,
            "forbidden_substrings": substrings, "provenance_strings": found}
