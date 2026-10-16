"""One cleaned ARB trajectory -> a monitor-view record plus a provenance record.

The monitor view is everything a detector may see:
- the task goal;
- the ordered events, each with the agent's action, reasoning, URL, open pages,
  focused element, last action error, and its observation (screenshot dimensions and
  PNG status, full and pruned accessibility trees, element properties).

It carries no labels, rewards or evaluator outputs, and no text that names the agent
or the task. Its key is an opaque ``trace_key``. The readable trace ID, the raw task
ID, the screenshot paths, and the screenshot hashes and byte counts, which are file
bookkeeping rather than evidence, live in the provenance record. This keeps identifiers
out of what a judge reads; it does not hide a run from a person who looks it up, since
the key, a screenshot or the goal text can be matched against the public dataset.
ARB hosts each site twice, and the agents split across the two copies, so the copy
suffix in hostnames (``wa-forum-xl-2.mcgill-nlp.org``) is rewritten to the site name
(``wa-forum.mcgill-nlp.org``) in every text field. Fenced fields are listed in
``constants``; this module never imports the label store.

The provenance record carries the source pointers, checksums, the split, the
screenshot files, and which fields were dropped and why. Rewards and labels are never
copied into it. ``ending`` is filled for development runs only, since it stands in for
the checker's verdict.
"""

import json
import re
from pathlib import Path

from . import NORMALIZER_VERSION
from . import constants as C
from .fetch import screenshot_listing_path, sha256_file
from .ids import normalize_task_id, trace_id, trace_key
from .png import inspect_png

SCHEMA_VERSION = "0.2-arb"
_IMAGE_URL = re.compile(r"https?://\S+?\.(?:png|jpe?g|gif|webp)\b", re.IGNORECASE)
# ARB's hosted copies of the WebArena and VisualWebArena sites: <site>-xl-<copy>.mcgill-nlp.org
HOST_COPY = re.compile(r"\b(v?wa-[a-z0-9-]+?)-xl-\d+(\.mcgill-nlp\.org)\b")
_KEPT_TOP = {"benchmark", "goal", "steps"}
_KEPT_STEP = {
    "num", "reasoning", "action", "screenshot_path", "url", "open_pages_urls",
    "focused_element", "last_action_error", "axtree", "axtree_pruned",
    "extra_element_properties",
}


class _Deid:
    """Rewrites hostname copy suffixes and counts the rewrites."""

    def __init__(self):
        self.count = 0

    def __call__(self, v):
        if v is None:
            return None
        v = v if isinstance(v, str) else str(v)
        v, n = HOST_COPY.subn(r"\1\2", v)
        self.count += n
        return v


def normalize(source: dict, data_root: Path, task_id: str, split: str, split_group: str,
              source_record: dict = None, verified_paths: set = None):
    """Return (monitor_view, provenance) for one parsed cleaned-trajectory JSON.

    ``task_id`` is the raw task ID from the cleaned file name. Every screenshot a step
    points to must name this trajectory's benchmark, agent, task and step number;
    otherwise ``ValueError`` is raised before anything is returned. Screenshots are read from
    ``data_root / <dataset path>``; when ``verified_paths`` is given, a screenshot not
    in it is treated as missing even if a file exists on disk.
    """
    benchmark, agent = source["benchmark"], source["agent"]
    check_screenshot_paths(source, task_id)
    tid = trace_id(benchmark, agent, task_id)
    key = trace_key(tid)
    deid = _Deid()
    goal = deid(source.get("goal"))

    events, screenshot_files = [], []
    for step in source.get("steps", []):
        shot = None
        sp = step.get("screenshot_path")
        if sp:
            rel = screenshot_listing_path(sp)
            f = data_root / rel
            present = f.exists() and (verified_paths is None or rel in verified_paths)
            file_rec = {"step": step.get("num"), "path": rel, "present": present}
            if present:
                info = inspect_png(f)
                shot = {"width": info["width"], "height": info["height"],
                        "png_ok": info["ok"], "png_error": info["error"]}
                file_rec.update(sha256=sha256_file(f), bytes=info["bytes"])
            else:
                shot = {"missing": True}
            screenshot_files.append(file_rec)
        events.append({
            "step": step.get("num"),
            "action": deid(step.get("action")),
            "reasoning": deid(step.get("reasoning")),
            "url": deid(step.get("url")),
            "open_pages_urls": [deid(u) for u in step.get("open_pages_urls") or []],
            "focused_element": deid(step.get("focused_element")),
            "last_action_error": deid(step.get("last_action_error")),
            "observation": {
                "screenshot": shot,
                "axtree": deid(step.get("axtree")),
                "axtree_pruned": deid(step.get("axtree_pruned")),
            },
            "elements": step.get("extra_element_properties") or {},
        })

    monitor_view = {
        "trace_key": key,
        "schema_version": SCHEMA_VERSION,
        "task": {
            "goal": goal,
            "goal_image_urls": sorted(set(_IMAGE_URL.findall(goal or ""))),
        },
        "events": events,
    }

    steps = source.get("steps") or []
    seen_step_keys = set().union(*(s.keys() for s in steps)) if steps else set()
    provenance = {
        "trace_key": key,
        "trace_id": tid,
        "source": "agentrewardbench",
        "dataset": C.DATASET,
        "revision": C.REVISION,
        "file": source_record,
        "source_url": (source_record or {}).get("url"),
        "benchmark": benchmark,
        "split_group": split_group,
        "split": split,
        "agent": agent,
        "task_id": task_id,
        "task_id_normalized": normalize_task_id(task_id),
        "n_events": len(events),
        "n_actions": sum(1 for e in events if e["action"]),
        "ending": ending_type(events) if split == "dev" else None,
        "hostname_suffixes_rewritten": deid.count,
        "screenshot_files": screenshot_files,
        "screenshots_referenced": len(screenshot_files),
        "screenshots_present": sum(1 for s in screenshot_files if s["present"]),
        "dropped_top_level_fields": {k: v for k, v in C.FENCED_TOP_LEVEL.items() if k in source},
        "dropped_step_fields": {k: v for k, v in C.FENCED_STEP.items() if k in seen_step_keys},
        "unrecognized_top_level_fields": sorted(set(source) - set(C.FENCED_TOP_LEVEL) - _KEPT_TOP),
        "unrecognized_step_fields": sorted(seen_step_keys - set(C.FENCED_STEP) - _KEPT_STEP),
        "stratum": "natural",
        "task_family": split_group,
        "normalizer_version": NORMALIZER_VERSION,
    }
    return monitor_view, provenance


def ending_type(events: list) -> str:
    """'agent_message', 'step_cap', or 'environment' (ended before the cap without a message).

    Recorded in provenance for development runs only; see ``constants.STEP_CAP``.
    """
    actions = [e["action"] for e in events if e["action"]]
    if actions and actions[-1].lstrip().startswith(tuple(a + "(" for a in C.AGENT_STOP_ACTIONS)):
        return "agent_message"
    if len(actions) >= C.STEP_CAP:
        return "step_cap"
    return "environment"


_SHOT_PATH = re.compile(
    r"(?:^|/)screenshots/(?P<benchmark>[^/]+)/(?P<agent>[^/]+)/(?P<task>[^/]+)"
    r"/screenshot_step_(?P<step>\d+)\.png$")


def check_screenshot_paths(source: dict, task_id: str) -> None:
    """Raise ValueError unless every step's screenshot belongs to this trajectory and step.

    The path must have the form .../screenshots/<benchmark>/<agent>/<task>/screenshot_step_<n>.png,
    with the trajectory's own benchmark, agent and raw task ID, and n equal to the step number.
    """
    steps = source.get("steps", [])
    for step in steps:
        sp = step.get("screenshot_path")
        if not sp:
            continue
        m = _SHOT_PATH.search(sp)
        if m is None:
            raise ValueError(f"step {step.get('num')}: unexpected screenshot path {sp!r}")
        expected = {"benchmark": source.get("benchmark"), "agent": source.get("agent"),
                    "task": task_id, "step": str(step.get("num"))}
        wrong = {k: m.group(k) for k in expected if m.group(k) != expected[k]}
        if wrong:
            raise ValueError(f"step {step.get('num')}: screenshot {sp!r} does not belong to "
                             f"this trajectory and step (mismatched {sorted(wrong)})")


def serialize(record: dict) -> str:
    """Canonical one-line JSON (sorted keys) so identical inputs give identical bytes."""
    return json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
