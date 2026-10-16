"""Checks that labels, rewards and evaluator verdicts cannot enter the monitor view.

- L0 allowlist: every key at every level of a monitor-view record is one the schema
  defines, so an unknown field cannot ride along, and every element property has its
  supported shape (see ``element_shape_errors``);
- L1 no fenced or label key at any depth;
- L2 no fenced field name or label-file name anywhere in the serialized record;
- L3 no agent or task identifier: agent name, model name, trace ID, raw and normalized
  task ID, the ``task_<n>`` folder name ARB uses for goal images, or a hosted-site copy
  suffix that shows which group of agents ran;
- L4 the normalizer module does not import the label store;
- L5 the package that bundles ARB's labels is not importable in the interpreter running
  this check. The judge runner must run ``check_runtime`` in its own environment too.

The label store itself is checked by unit tests with a canary value.
"""

import ast
import importlib.util
import inspect
import re
import urllib.parse

from . import constants as C
from . import normalize as normalize_module
from .ids import normalize_task_id
from .normalize import serialize

# Any trace of a hosted-site copy suffix, in any case, also after URL-unquoting. Broader
# than the rewrite on purpose, so a form the rewrite misses fails the check.
_COPY_TRACE = re.compile(r"v?wa-[a-z0-9-]+-xl-\d", re.IGNORECASE)

ALLOWED = {
    "top": {"trace_key", "schema_version", "task", "events"},
    "task": {"goal", "goal_image_urls"},
    "event": {"step", "action", "reasoning", "url", "open_pages_urls", "focused_element",
              "last_action_error", "observation", "elements"},
    "observation": {"screenshot", "axtree", "axtree_pruned"},
    "screenshot": {"width", "height", "png_ok", "png_error", "missing"},
    "element": {"visibility", "bbox", "clickable", "set_of_marks"},
}


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def element_shape_errors(props: dict) -> list:
    """Shape problems in one element's properties.

    Supported shapes: ``bbox`` is null or a list of exactly 4 numbers; ``visibility`` is a
    number; ``clickable`` and ``set_of_marks`` are booleans. Booleans do not count as numbers.
    """
    errors = []
    if "bbox" in props:
        b = props["bbox"]
        if not (b is None or (isinstance(b, list) and len(b) == 4 and all(_is_number(x) for x in b))):
            errors.append("bbox")
    if "visibility" in props and not _is_number(props["visibility"]):
        errors.append("visibility")
    for k in ("clickable", "set_of_marks"):
        if k in props and not isinstance(props[k], bool):
            errors.append(k)
    return errors


def _walk_keys(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield f"{path}/{k}", k
            yield from _walk_keys(v, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_keys(v, f"{path}[{i}]")


def _unexpected_keys(rec: dict) -> list:
    bad = [f"/{k}" for k in rec if k not in ALLOWED["top"]]
    bad += [f"/task/{k}" for k in (rec.get("task") or {}) if k not in ALLOWED["task"]]
    for i, e in enumerate(rec.get("events") or []):
        bad += [f"/events[{i}]/{k}" for k in e if k not in ALLOWED["event"]]
        obs = e.get("observation") or {}
        bad += [f"/events[{i}]/observation/{k}" for k in obs if k not in ALLOWED["observation"]]
        shot = obs.get("screenshot") or {}
        bad += [f"/events[{i}]/observation/screenshot/{k}" for k in shot
                if k not in ALLOWED["screenshot"]]
        for bid, props in (e.get("elements") or {}).items():
            if not isinstance(props, dict):
                bad.append(f"/events[{i}]/elements/{bid}")
                continue
            bad += [f"/events[{i}]/elements/{bid}/{k}" for k in props if k not in ALLOWED["element"]]
            bad += [f"/events[{i}]/elements/{bid}/{k} (shape)" for k in element_shape_errors(props)]
    return bad


def check_record(rec: dict, identity_strings=()) -> dict:
    l0 = _unexpected_keys(rec)
    # Direct children of an element map are element ids (data, not field names).
    keys = [(p, k) for p, k in _walk_keys(rec) if not p.rsplit("/", 1)[0].endswith("/elements")]
    l1 = [p for p, k in keys if k in C.FORBIDDEN_KEYS]
    text = serialize(rec)
    l2 = [s for s in C.FORBIDDEN_SUBSTRINGS if s in text]
    l3 = sorted({s for s in identity_strings if s and s in text})
    unquoted = urllib.parse.unquote(urllib.parse.unquote(text))
    l3 += sorted({m.group(0) for t in (text, unquoted) for m in _COPY_TRACE.finditer(t)})
    return {"trace_key": rec.get("trace_key"), "ok": not (l0 or l1 or l2 or l3),
            "L0_unexpected_keys": l0, "L1_forbidden_keys": l1,
            "L2_forbidden_substrings": l2, "L3_identifiers": l3}


def identity_strings(source: dict, provenance: dict) -> list:
    """Strings that would identify the agent or the task if they reached the monitor view."""
    task_id = provenance.get("task_id") or ""
    out = [source.get("agent"), source.get("model"), provenance.get("trace_id"),
           task_id, normalize_task_id(task_id)]
    num = re.search(r"\.(\d+)$", task_id)
    if num:
        out.append(f"task_{num.group(1)}/")
    return [s for s in out if isinstance(s, str) and s]


def check_runtime() -> dict:
    src = inspect.getsource(normalize_module)
    imported = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    l4 = "labels" in imported or any(m.endswith(".labels") for m in imported)
    l5 = importlib.util.find_spec(C.LABEL_BUNDLING_PACKAGE) is not None
    return {"ok": not (l4 or l5),
            "L4_normalizer_imports_label_store": l4,
            "L5_label_bundling_package_importable": l5}
