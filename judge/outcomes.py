"""One finished exchange -> one canonical outcome, per the setup note's outcome table.

An exchange is what a transport returns for one attempt: the HTTP status (None when no
response arrived), the response headers, the raw response bytes (an SSE stream on
success, a JSON error body otherwise) and a transport error, if any.

| Outcome   | Judge A (Anthropic Messages)                | Judge B (OpenAI Responses)                  |
|-----------|---------------------------------------------|---------------------------------------------|
| VALID     | ``end_turn``; schema parses; verdict in enum | ``completed``; schema parses; verdict in enum |
| REFUSAL   | stop reason ``refusal``                      | a ``refusal`` output item or content part    |
| TRUNCATED | ``max_tokens``, ``model_context_window_exceeded`` | ``incomplete`` for any reason but a filter |
| FILTERED  | none documented                              | ``incomplete`` with ``content_filter``, or a policy error code |
| MALFORMED | complete, schema fails, or an unexpected stop reason | complete, schema fails                |
| ABSTAIN   | parses, verdict not success or failure        | same                                        |
| TRANSPORT | no response, a cut stream, 408, 429 rate limit, 5xx, 529, other errors | same, including 503 |
| HALT      | 401, 402, 403, 404; 400 or 413; any invalid-request error; spend, usage or billing limits | 401, 402, 403, 404; 400 or 413; invalid-request and image errors; credit, spend and usage limits |

Only TRANSPORT is retried, and it is decided without reading any model output: a stream
that did not reach its end is TRANSPORT and its partial content is never parsed. HALT is
not an outcome of the slot. It stops the run, and ``halt_kind`` says why: ``auth``,
``billing``, ``permission``, ``not_found``, ``spend_limit`` or ``request_error``. A request
or configuration error halts because identical retries cannot succeed, and it is never an
invalid verdict. Every non-VALID terminal outcome is invalid for its slot.

An attempt that cannot be classified, because reading it raised, has the halt kind
``classification_error`` (``classification_failure``).
"""

import json
from collections import namedtuple

from . import schema
from .streams import anthropic_summary, openai_summary

VALID, REFUSAL, TRUNCATED, FILTERED = "VALID", "REFUSAL", "TRUNCATED", "FILTERED"
MALFORMED, ABSTAIN, TRANSPORT, HALT = "MALFORMED", "ABSTAIN", "TRANSPORT", "HALT"
TERMINAL = (VALID, REFUSAL, TRUNCATED, FILTERED, MALFORMED, ABSTAIN)
OUTCOMES = TERMINAL + (TRANSPORT, HALT)

Exchange = namedtuple("Exchange", "status headers raw transport_error first_byte_at",
                      defaults=(None,))

# Anthropic: the message of a 400 at a spend limit you set, and of the 429 at a tier's cap.
A_SPEND_MESSAGES = ("You have reached your specified API usage limits",
                    "You have reached your specified workspace API usage limits",
                    "You have reached your API usage limits")
A_SPEND_CODES = frozenset({"enforced_spend_limit_reached"})

B_BILLING_CODES = frozenset({"credit_balance_exhausted"})
B_SPEND_CODES = frozenset({"organization_spend_limit_exceeded", "project_spend_limit_exceeded",
                           "organization_usage_limit_exceeded", "insufficient_quota"})
B_POLICY_CODES = frozenset({"invalid_prompt", "bio_policy", "misalignment_policy_violation",
                            "image_content_policy_violation", "content_filter"})
B_TRANSIENT_CODES = frozenset({"server_error", "rate_limit_exceeded", "server_is_overloaded",
                               "slow_down", "vector_store_timeout"})
B_REQUEST_CODES = frozenset({"data_residency_mismatch", "invalid_image", "invalid_image_format",
                             "invalid_base64_image", "invalid_image_url", "image_too_large",
                             "image_too_small", "image_parse_error", "invalid_image_mode",
                             "image_file_too_large", "unsupported_image_media_type",
                             "empty_image_file", "failed_to_download_image", "image_file_not_found"})
IMAGE_ERROR_CODES = frozenset(c for c in B_REQUEST_CODES if "image" in c)
# OpenAI marks key, permission and not-found errors by the HTTP status alone; these codes
# name them in a body whose status was not kept.
B_ACCESS_CODES = {"invalid_api_key": "auth", "unsupported_country_region_territory": "permission",
                  "model_not_found": "not_found"}
B_TRANSIENT_TYPES = frozenset({"server_error"})
CLASSIFICATION_ERROR = "classification_error"


def _result(outcome, detail, halt_kind=None, summary=None, parsed=None, error=None) -> dict:
    return {"outcome": outcome, "retryable": outcome == TRANSPORT, "halt_kind": halt_kind,
            "detail": detail, "summary": summary, "parse": parsed, "error": error}


def _error_body(raw: bytes) -> dict:
    try:
        obj = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return {}
    err = obj.get("error") if isinstance(obj, dict) else None
    return err if isinstance(err, dict) else {}


def _error_record(err: dict) -> dict:
    code = err.get("code")
    if code is None and isinstance(err.get("details"), dict):
        code = err["details"].get("error_code")
    return {"type": err.get("type"), "code": code, "message": (err.get("message") or "")[:500]}


def _anthropic_error(status, err: dict):
    """(outcome, halt kind, detail) for an Anthropic error, from HTTP or the stream."""
    etype, msg = err.get("type"), err.get("message") or ""
    code = (err.get("details") or {}).get("error_code") if isinstance(err.get("details"), dict) else None
    label = f"{status if status is not None else 'stream'} {etype}"
    if status == 401 or etype == "authentication_error":
        return HALT, "auth", label
    if status == 402 or etype == "billing_error":
        return HALT, "billing", label
    if status == 403 or etype == "permission_error":
        return HALT, "permission", label
    if status == 404 or etype == "not_found_error":
        return HALT, "not_found", label
    if code in A_SPEND_CODES or msg.startswith(A_SPEND_MESSAGES):
        return HALT, "spend_limit", label
    if status in (400, 413) or etype in ("invalid_request_error", "request_too_large"):
        return HALT, "request_error", label
    return TRANSPORT, None, label


def _openai_error(status, err: dict):
    """(outcome, halt kind, detail) for an OpenAI error, from HTTP or the stream."""
    code, etype = err.get("code"), err.get("type")
    label = f"{status if status is not None else 'stream'} {code or etype}"
    if status == 401:
        return HALT, "auth", label
    if status == 402 or code in B_BILLING_CODES:
        return HALT, "billing", label
    if status == 403:
        return HALT, "permission", label
    if status == 404:
        return HALT, "not_found", label
    if code in B_SPEND_CODES or etype == "insufficient_quota":
        return HALT, "spend_limit", label
    if code in B_POLICY_CODES:
        return FILTERED, None, label
    if status in (400, 413) or etype == "invalid_request_error" or code in B_REQUEST_CODES:
        return HALT, "request_error", label
    return TRANSPORT, None, label


def _openai_body_error(err: dict):
    """(outcome, halt kind, detail) for an OpenAI error body whose status was not kept. The
    status alone marks a 401, 402, 403, 404, 400 or 413 error as one that halts, so a body
    that would be TRANSPORT halts unless it is a known transient error. A code in
    ``B_ACCESS_CODES`` names its halt kind."""
    outcome, kind, label = _openai_error(None, err)
    code, etype = err.get("code"), err.get("type")
    if code in B_ACCESS_CODES:
        return HALT, B_ACCESS_CODES[code], label
    if outcome == TRANSPORT and code not in B_TRANSIENT_CODES and etype not in B_TRANSIENT_TYPES:
        return HALT, "request_error", label
    return outcome, kind, label


def _parsed_outcome(text):
    parsed = schema.parse(text)
    outcome = {schema.PARSED: VALID, schema.ABSTAIN: ABSTAIN}.get(parsed["status"], MALFORMED)
    return outcome, parsed


def anthropic_outcome(ex: Exchange) -> dict:
    if ex.status is not None and not 200 <= ex.status < 300:
        err = _error_body(ex.raw)
        outcome, kind, detail = _anthropic_error(ex.status, err)
        return _result(outcome, detail, kind, error=_error_record(err))
    if ex.status is None:
        return _result(TRANSPORT, f"no response: {ex.transport_error}")
    s = anthropic_summary(ex.raw)
    if s["error"] is not None:
        outcome, kind, detail = _anthropic_error(None, s["error"])
        return _result(outcome, detail, kind, summary=s, error=_error_record(s["error"]))
    if not s["complete"]:
        return _result(TRANSPORT, "stream ended without a stop reason"
                       + (f": {ex.transport_error}" if ex.transport_error else ""), summary=s)
    reason = s["stop_reason"]
    if reason == "end_turn":
        outcome, parsed = _parsed_outcome(s["text"])
        return _result(outcome, f"end_turn, {parsed['status']}", summary=s, parsed=parsed)
    if reason == "refusal":
        return _result(REFUSAL, "stop reason refusal", summary=s)
    if reason in ("max_tokens", "model_context_window_exceeded"):
        return _result(TRUNCATED, f"stop reason {reason}", summary=s)
    return _result(MALFORMED, f"unexpected stop reason {reason}", summary=s)


def openai_outcome(ex: Exchange) -> dict:
    if ex.status is not None and not 200 <= ex.status < 300:
        err = _error_body(ex.raw)
        outcome, kind, detail = _openai_error(ex.status, err)
        return _result(outcome, detail, kind, error=_error_record(err))
    if ex.status is None:
        return _result(TRANSPORT, f"no response: {ex.transport_error}")
    s = openai_summary(ex.raw)
    if s["error"] is not None:
        outcome, kind, detail = _openai_error(None, s["error"])
        return _result(outcome, detail, kind, summary=s, error=_error_record(s["error"]))
    if not s["complete"]:
        return _result(TRANSPORT, "stream ended without a terminal response event"
                       + (f": {ex.transport_error}" if ex.transport_error else ""), summary=s)
    if s["terminal_event"] == "response.failed":
        err = s["response_error"] or {}
        code = err.get("code")
        if code in B_TRANSIENT_CODES or code is None:
            return _result(TRANSPORT, f"response failed: {code}", summary=s, error=_error_record(err))
        outcome, kind, detail = _openai_error(None, err)
        return _result(outcome, f"response failed: {detail}", kind, summary=s, error=_error_record(err))
    if s["terminal_event"] == "response.incomplete":
        if s["incomplete_reason"] == "content_filter":
            return _result(FILTERED, "incomplete: content_filter", summary=s)
        return _result(TRUNCATED, f"incomplete: {s['incomplete_reason']}", summary=s)
    if s["status"] != "completed":
        return _result(MALFORMED, f"completed event with status {s['status']}", summary=s)
    if s["refusal"] is not None:
        return _result(REFUSAL, "refusal output", summary=s)
    outcome, parsed = _parsed_outcome(s["text"])
    return _result(outcome, f"completed, {parsed['status']}", summary=s, parsed=parsed)


def classification_failure(status, e: Exception) -> dict:
    """The outcome of an attempt whose classification raised. That is a defect in the
    reader and says nothing about the model, so the run stops whichever outcome this is.
    An error status means there is no model response, so it is a HALT, and the slot resumes
    with its next attempt after the note. Otherwise the response may be complete and paid
    for: it is MALFORMED, which ends the slot, so it is never sent again. Only the
    exception's type is kept, since its message can quote the response."""
    outcome = HALT if status is not None and not 200 <= status < 300 else MALFORMED
    return _result(outcome, f"classification raised {type(e).__name__}", CLASSIFICATION_ERROR)


def canonical_outcome(provider: str, exchange: Exchange) -> dict:
    """The canonical outcome of one attempt: {"outcome", "retryable", "halt_kind", "detail",
    "summary", "parse", "error"}."""
    if provider == "anthropic":
        return anthropic_outcome(exchange)
    if provider == "openai":
        return openai_outcome(exchange)
    raise ValueError(f"unknown provider {provider!r}")


def error_body_outcome(provider: str, raw: bytes):
    """The outcome of ``raw`` as a provider's JSON error body whose HTTP status was not
    kept; None when it is not an error body. It is classified as the same error in a
    stream is, which for Anthropic is enough, since its error types mark what its statuses
    mark. An OpenAI body that would be TRANSPORT halts unless it is a known transient error
    (``_openai_body_error``)."""
    err = _error_body(raw)
    if not err:
        return None
    if provider == "anthropic":
        outcome, kind, label = _anthropic_error(None, err)
    elif provider == "openai":
        outcome, kind, label = _openai_body_error(err)
    else:
        raise ValueError(f"unknown provider {provider!r}")
    return _result(outcome, "error body, status not kept: " + label.removeprefix("stream "), kind,
                   error=_error_record(err))
