"""Schema S, the judges' structured output, and the parser for it.

Schema S has five required fields and no others: ``verdict`` (success or failure),
``p_failure`` (a number), ``evidence_sufficient`` (yes or no), ``evidence_step_ids`` (an
array of integers) and ``rationale`` (a string). It carries no numeric or length bounds,
since Anthropic rejects them in a schema; the parser checks them instead.

Parser rules (``parse``), applied to the final answer text of a complete response:

- No free-text fallback. The text must be exactly one JSON object (surrounding whitespace
  allowed), with no duplicate keys and no NaN or infinity. Otherwise: ``malformed``.
- The object must have exactly the five fields, and ``verdict`` must be a string.
  Otherwise: ``malformed``.
- ``verdict`` must equal "success" or "failure", ignoring ASCII letter case, with no
  other change, so a look-alike character never becomes a verdict. Otherwise: ``abstain``
  (it parses, but gives no admissible verdict).
- A bad covariate voids that field only, never the verdict: ``p_failure`` that is not a
  finite number in [0, 1], ``evidence_sufficient`` that is not "yes" or "no" ignoring ASCII
  letter case only, as for the verdict, ``evidence_step_ids`` that is not a list of
  integers, and ``rationale`` that is not a string of at most 100 words
  (whitespace-separated). A voided field is None, and the reason is recorded under
  ``voided``.
- An integer literal of more than 18 digits is not converted. For these rules it is neither
  an integer nor a number, so in a covariate it voids that field like any other bad value.
"""

import hashlib
import json
import math

SCHEMA_VERSION = "schema-S-v0"
PARSER_VERSION = "parser-v1"
SCHEMA_NAME = "schema_s"          # the name OpenAI's json_schema format requires

VERDICTS = ("success", "failure")
SUFFICIENT = ("yes", "no")
RATIONALE_MAX_WORDS = 100
INTEGER_MAX_DIGITS = 18           # every integer this long fits in 64 bits
FIELDS = ("verdict", "p_failure", "evidence_sufficient", "evidence_step_ids", "rationale")

# Field order is the protocol's. Providers generate fields in schema order, so the request
# serializer keeps this order inside "properties" while sorting every other key.
SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "p_failure": {"type": "number"},
        "evidence_sufficient": {"type": "string", "enum": list(SUFFICIENT)},
        "evidence_step_ids": {"type": "array", "items": {"type": "integer"}},
        "rationale": {"type": "string"},
    },
    "required": list(FIELDS),
    "additionalProperties": False,
}

MALFORMED, ABSTAIN, PARSED = "malformed", "abstain", "parsed"


def schema_sha256() -> str:
    """SHA-256 of the schema's canonical bytes (keys sorted except the field order)."""
    from .requests import canonical_bytes        # one serializer for schema and requests
    return hashlib.sha256(canonical_bytes(SCHEMA)).hexdigest()


def _reject_constant(name):
    raise ValueError(f"{name} is not valid JSON")


def _no_duplicates(pairs):
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate key")
    return dict(pairs)


# What an integer literal over INTEGER_MAX_DIGITS parses to. Converting one could raise,
# since Python refuses a literal over its digit limit (4,300 by default, and configurable),
# and the verdict would be lost with it. No field check accepts this value.
_TOO_LONG = object()


def _parse_int(digits):
    return int(digits) if len(digits.lstrip("-")) <= INTEGER_MAX_DIGITS else _TOO_LONG


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number(v) -> bool:
    return _is_int(v) or (isinstance(v, float) and math.isfinite(v))


def parse(text) -> dict:
    """Parse one answer text under the rules in the module docstring.

    Returns {"parser_version", "status": "parsed" | "abstain" | "malformed", "verdict",
    "fields", "voided", "error"}. ``verdict`` is "success" or "failure" only when
    ``status`` is "parsed".
    """
    out = {"parser_version": PARSER_VERSION, "status": MALFORMED, "verdict": None,
           "fields": None, "voided": {}, "error": None}
    if not isinstance(text, str):
        out["error"] = "no answer text"
        return out
    try:
        obj = json.loads(text, object_pairs_hook=_no_duplicates, parse_constant=_reject_constant,
                         parse_int=_parse_int)
    except (ValueError, RecursionError) as e:
        out["error"] = f"not one JSON object: {e}"
        return out
    if not isinstance(obj, dict):
        out["error"] = "not a JSON object"
        return out
    if set(obj) != set(FIELDS):
        missing = sorted(set(FIELDS) - set(obj))
        extra = sorted(set(obj) - set(FIELDS))
        out["error"] = f"fields differ from schema S: missing {missing}, extra {extra}"
        return out
    if not isinstance(obj["verdict"], str):
        out["error"] = "verdict is not a string"
        return out

    fields, voided = {}, {}
    p = obj["p_failure"]
    if _is_number(p) and 0 <= p <= 1:
        fields["p_failure"] = p
    else:
        fields["p_failure"] = None
        voided["p_failure"] = "not a finite number in [0, 1]"
    s = obj["evidence_sufficient"]
    sufficient = s.lower() if isinstance(s, str) and s.isascii() else None
    if sufficient in SUFFICIENT:
        fields["evidence_sufficient"] = sufficient
    else:
        fields["evidence_sufficient"] = None
        voided["evidence_sufficient"] = "not yes or no"
    ids = obj["evidence_step_ids"]
    if isinstance(ids, list) and all(_is_int(i) for i in ids):
        fields["evidence_step_ids"] = list(ids)
    else:
        fields["evidence_step_ids"] = None
        voided["evidence_step_ids"] = "not a list of integers"
    r = obj["rationale"]
    if isinstance(r, str) and len(r.split()) <= RATIONALE_MAX_WORDS:
        fields["rationale"] = r
    else:
        fields["rationale"] = None
        voided["rationale"] = (f"over {RATIONALE_MAX_WORDS} words" if isinstance(r, str)
                               else "not a string")

    verdict = obj["verdict"].lower() if obj["verdict"].isascii() else None
    fields["verdict"] = verdict if verdict in VERDICTS else None
    out.update(fields=fields, voided=voided)
    if verdict not in VERDICTS:
        out.update(status=ABSTAIN, error=f"verdict {obj['verdict']!r} is not success or failure")
        return out
    out.update(status=PARSED, verdict=verdict)
    return out
