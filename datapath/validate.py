"""Validation of monitor-view records and label records.

Checks (each reported as pass/fail with details):
- V1 required fields: trace_key, schema_version, a non-empty goal, at least one event.
- V2 event order: step numbers are 0..n-1, contiguous and increasing.
- V3 actions: every event except the last has a non-empty action. Whether the last
  event, normally the final observation, has an action is reported, not failed.
- V4 screenshots: every event points to a screenshot that exists and is a valid PNG.
- V5 missingness: counts of empty values per event field (reported, not failed).
- V6 labels: values are in ARB's vocabulary, and every development-split monitor-view
  record joins exactly one label (test-split records never get a label in this data path).
"""

from . import constants as C

_EVENT_FIELDS = ("action", "reasoning", "url", "focused_element", "last_action_error")


def validate_record(rec: dict) -> dict:
    checks = {}
    goal = (rec.get("task") or {}).get("goal")
    events = rec.get("events") or []
    checks["V1_required"] = {
        "ok": bool(rec.get("trace_key")) and bool(rec.get("schema_version"))
              and isinstance(goal, str) and bool(goal.strip()) and len(events) > 0,
        "n_events": len(events),
    }
    steps = [e.get("step") for e in events]
    checks["V2_event_order"] = {"ok": steps == list(range(len(events))), "steps": steps}
    missing_action = [e["step"] for e in events[:-1] if not e.get("action")]
    last_action = events[-1].get("action") if events else None
    checks["V3_actions"] = {
        "ok": not missing_action,
        "events_without_action_before_last": missing_action,
        "last_event_has_action": bool(last_action),
    }
    shots = [(e.get("observation") or {}).get("screenshot") for e in events]
    bad = []
    for e, s in zip(events, shots):
        if s is None:
            bad.append((e.get("step"), "no screenshot pointer"))
        elif s.get("missing"):
            bad.append((e.get("step"), "file missing"))
        elif not s.get("png_ok"):
            bad.append((e.get("step"), s.get("png_error")))
    dims = sorted({(s["width"], s["height"]) for s in shots if s and s.get("png_ok")})
    checks["V4_screenshots"] = {"ok": not bad, "problems": bad, "dimensions": dims}
    empties = {f: sum(1 for e in events if not e.get(f)) for f in _EVENT_FIELDS}
    empties["axtree"] = sum(1 for e in events if not (e.get("observation") or {}).get("axtree"))
    empties["axtree_pruned"] = sum(1 for e in events if not (e.get("observation") or {}).get("axtree_pruned"))
    checks["V5_missingness"] = {"ok": True, "empty_counts": empties}
    return {"trace_key": rec.get("trace_key"), "ok": all(c["ok"] for c in checks.values()),
            "checks": checks}


def validate_labels(labels: dict, dev_monitor_keys: set) -> dict:
    """``labels`` is the dev label store; ``dev_monitor_keys`` the dev records written."""
    bad_values = []
    for key, rec in labels.items():
        for i, labelset in enumerate([rec] + rec.get("secondary_labels", [])):
            for col in C.LABEL_COLUMNS:
                if labelset.get(col) not in C.ALLOWED_LABEL_VALUES[col]:
                    bad_values.append((key, i, col))
    unjoined = [k for k in labels if k not in dev_monitor_keys]
    missing_label = sorted(dev_monitor_keys - set(labels))
    return {"ok": not bad_values and not missing_label,
            "bad_values": bad_values,
            "dev_labels_not_in_this_run": len(unjoined),
            "monitor_records_without_label": missing_label}
