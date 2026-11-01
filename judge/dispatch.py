"""The serial dispatcher for one trajectory, its transports and the response checks.

**Order.** A1, then A2, then A3, then B1, one after another. A1, A2 and A3 send the same
``PreparedRequest``, built before dispatch, so their bytes and settings are identical and
no answer can reach a later call. Every slot runs whatever an earlier slot returned; only
a halt stops the sequence. The gap between the start of one A request and the next is
recorded, since a repeat reads the prompt cache only within its five-minute lifetime, and
every response's cache read and write token counts are logged.

**Retries.** Only TRANSPORT is retried: the same bytes, the same slot, at most three
retries with exponential backoff (a server's ``retry-after`` is honored when longer, up
to two minutes). A slot still without a complete response after three retries is
INVALID. Any terminal outcome ends the slot, and a slot that ended is never sent again.

**Halt.** A HALT outcome, a failed per-response check, or a repeated provider response id
is logged as a HALT attempt and a halt record, and the dispatcher stops at once. It will
not start again while any halt lacks a correction note (``AttemptLog.record_correction``).

**Per-response checks**, on every complete model response, each a halt on failure:
A: the model string equals the first one recorded for this configuration;
``usage.service_tier`` is ``standard``; ``usage.inference_geo`` is ``global``; no one-hour
cache tokens; the ``anthropic-workspace-id`` header is the study workspace when one is
given. B: the model string equals the first recorded; ``service_tier`` is ``default``;
``cached_tokens`` and ``cache_write_tokens`` are 0; ``input_tokens`` is at most 272,000,
above which the provider prices the call at its long-context rate. A field that is not
reported fails.

**Recovery.** A stop between the two writes of an attempt leaves its raw response without
its record. ``recover_attempt`` rebuilds that record from the bytes as the dispatcher
would have classified them, and nothing is chosen by reading it. The status and headers
were not kept. Bytes that start the provider's stream are taken as HTTP 200, so a
complete response is recorded as it was and a slot it ended is never sent again, while a
cut stream is TRANSPORT. A JSON error body is classified as the same error in a stream
(``outcomes.error_body_outcome``): a request, key, billing, permission, not-found or
spend-limit error halts until a correction note, and a transient one is TRANSPORT.
Anything else is taken as no response (TRANSPORT), whose slot is retried under the rule
above. The workspace header check is listed as not applied.

**Transports.** ``FakeTransport`` returns scripted exchanges and records what it was sent;
the tests and the dry run use only it. ``HttpsTransport`` sends the exact body bytes with
``urllib`` and reads the event stream; it refuses to exist unless paid calls are
confirmed, and no test or dry run constructs it.
"""

import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.client import HTTPException

from . import requests as R
from .outcomes import (ABSTAIN, HALT, IMAGE_ERROR_CODES, MALFORMED, REFUSAL, TERMINAL,
                       TRANSPORT, VALID, Exchange, canonical_outcome, error_body_outcome)
from .streams import parse_sse

CACHE_LIFETIME_S = 300
B_LONG_CONTEXT_TOKENS = 272000               # judge B input above this is priced at the long-context rate
PROMPT_SLOTS = (("A", 1), ("B", 1))          # prompt development: A1 and B1 only
BACKOFF_BASE_S = 2.0
RETRY_AFTER_CAP_S = 120.0
KEPT_HEADERS = ("request-id", "anthropic-workspace-id", "x-request-id", "openai-project",
                "retry-after", "content-type")


class HaltPending(RuntimeError):
    """A halt has no correction note yet, so the run may not resume."""


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="milliseconds")


# ---------------------------------------------------------------- transports


class FakeTransport:
    """Returns scripted exchanges in order and records every request it is sent."""

    def __init__(self, script=()):
        self.script = list(script)
        self.sent = []

    def add(self, *exchanges) -> None:
        self.script.extend(exchanges)

    def send(self, request: R.PreparedRequest) -> Exchange:
        self.sent.append(request)
        if not self.script:
            raise AssertionError("the fake transport has no scripted exchange left")
        return self.script.pop(0)


def read_stream(fileobj, clock=time.time):
    """(raw bytes, transport error or None, first byte time) from a streamed response."""
    chunks, first = [], None
    try:
        while True:
            chunk = fileobj.read1(65536) if hasattr(fileobj, "read1") else fileobj.read(65536)
            if not chunk:
                break
            if first is None:
                first = clock()
            chunks.append(chunk)
    except (OSError, HTTPException, ValueError) as e:
        return b"".join(chunks), f"{type(e).__name__}: {e}", first
    return b"".join(chunks), None, first


class HttpsTransport:
    """Sends one request over HTTPS with ``urllib`` and reads its event stream.

    The credential comes from the named environment variable at send time and is the only
    header added to the request. SDKs are not used, so nothing retries behind the runner.
    """

    KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}

    def __init__(self, confirm_paid_calls: bool = False, timeout_s: float = 900.0, clock=time.time):
        if confirm_paid_calls is not True:
            raise ValueError("HttpsTransport sends paid requests; pass confirm_paid_calls=True "
                             "only in an approved paid run")
        self.timeout_s, self.clock = timeout_s, clock

    def send(self, request: R.PreparedRequest) -> Exchange:
        key = os.environ.get(self.KEY_ENV[request.provider])
        if not key:
            raise RuntimeError(f"set {self.KEY_ENV[request.provider]}")
        headers = dict(request.headers)
        if request.provider == "anthropic":
            headers["x-api-key"] = key
        else:
            headers["authorization"] = f"Bearer {key}"
        req = urllib.request.Request(request.url, data=request.body, headers=headers, method="POST")
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout_s)
        except urllib.error.HTTPError as e:
            raw, err, first = read_stream(e, self.clock)
            return Exchange(e.code, {k.lower(): v for k, v in e.headers.items()}, raw, err, first)
        except (urllib.error.URLError, OSError, HTTPException) as e:
            return Exchange(None, {}, b"", f"{type(e).__name__}: {e}", None)
        with resp:
            raw, err, first = read_stream(resp, self.clock)
            return Exchange(resp.status, {k.lower(): v for k, v in resp.headers.items()}, raw, err, first)


# ---------------------------------------------------------------- checks


def response_checks(provider: str, summary: dict, headers: dict, first_model, workspace) -> list:
    """Failed per-response checks for one complete response (see the module docstring)."""
    fails = []
    if first_model is not None and summary.get("model") != first_model:
        fails.append(f"model {summary.get('model')!r} is not the first recorded {first_model!r}")
    u = summary.get("usage") or {}
    if provider == "anthropic":
        if u.get("service_tier") != "standard":
            fails.append(f"usage.service_tier is {u.get('service_tier')!r}, not 'standard'")
        if u.get("inference_geo") != "global":
            fails.append(f"usage.inference_geo is {u.get('inference_geo')!r}, not 'global'")
        if u.get("cache_write_1h_tokens") != 0:
            fails.append(f"one-hour cache tokens are {u.get('cache_write_1h_tokens')!r}, not 0")
        if workspace is not None and headers.get("anthropic-workspace-id") != workspace:
            fails.append("the anthropic-workspace-id header is not the study workspace")
    else:
        if summary.get("service_tier") != "default":
            fails.append(f"service_tier is {summary.get('service_tier')!r}, not 'default'")
        if u.get("cache_read_tokens") != 0 or u.get("cache_write_tokens") != 0:
            fails.append(f"cache tokens read {u.get('cache_read_tokens')!r}, written "
                         f"{u.get('cache_write_tokens')!r}; both must be 0")
        if not isinstance(u.get("input_tokens"), int) or u["input_tokens"] > B_LONG_CONTEXT_TOKENS:
            fails.append(f"input_tokens is {u.get('input_tokens')!r}, not at most {B_LONG_CONTEXT_TOKENS}")
    return fails


def _identity_checks(log, provider: str, judge: str, config_sha256: str, summary: dict, headers: dict, workspace,
                     ref: str) -> list:
    """The per-response checks of a complete response, with the first model string and the
    response id recorded as the dispatcher records them."""
    first = log.first_model(config_sha256, judge)
    checks = response_checks(provider, summary, headers, first, workspace)
    if not checks and first is None:
        log.set_first_model(config_sha256, judge, summary.get("model"), ref)
    if summary.get("id") and not log.claim_response_id(provider, summary["id"], ref):
        checks.append(f"provider response id {summary['id']!r} was seen before")
    return checks


def _backoff(retry: int, headers: dict) -> float:
    wait = BACKOFF_BASE_S * 2 ** (retry - 1)
    try:
        wait = max(wait, min(float(headers.get("retry-after")), RETRY_AFTER_CAP_S))
    except (TypeError, ValueError):
        pass
    return wait


def _slot_done(records: list) -> bool:
    if not records:
        return False
    if records[-1]["outcome"] in TERMINAL:
        return True
    return records[-1]["outcome"] == TRANSPORT and sum(r["outcome"] == TRANSPORT for r in records) > R.MAX_RETRIES


def _slot_result(records: list) -> dict:
    last = records[-1]
    return {"outcome": last["outcome"], "valid": last["outcome"] == VALID,
            "verdict": (last.get("parse") or {}).get("verdict") if last["outcome"] == VALID else None,
            "attempts": len(records), "transport_attempts": sum(r["outcome"] == TRANSPORT for r in records)}


# ---------------------------------------------------------------- dispatch


def dispatch_trajectory(trace_key: str, requests: dict, config_sha256: str, transport, log,
                        workspace, sleep=time.sleep, clock=time.time, slots=R.SLOTS) -> dict:
    """Run slots A1, A2, A3 and B1 for one trajectory, serially (see the module docstring).

    ``requests`` maps "A" and "B" to the ``PreparedRequest`` built for this trajectory.
    ``workspace`` is the study's Anthropic workspace ID, or None where no real workspace
    exists (the dry run). ``slots`` is either all four, ``R.SLOTS``, or ``PROMPT_SLOTS``,
    A1 then B1, which prompt development sends; any other set is refused. Returns
    {"halted": bool, "halt_id", "slots": {slot: result}}.
    """
    slots = tuple(slots)
    if slots not in (R.SLOTS, PROMPT_SLOTS):
        raise ValueError(f"slots {slots} are neither {R.SLOTS} nor {PROMPT_SLOTS}")
    pending = log.unresolved_halts()
    if pending:
        raise HaltPending(f"{len(pending)} halt(s) need a correction note before resuming: "
                          + ", ".join(h["halt_id"] for h in pending))
    for judge in ("A", "B"):
        request = requests[judge]
        if request.judge != judge or R.sha256_bytes(request.body) != request.sha256:
            raise ValueError(f"the {judge} request is not a prepared {judge} request")
        body = json.loads(request.body)
        problems = R.check_body(judge, body, caching="cache_control" in body)
        if problems:
            raise ValueError(f"the {judge} request differs from the request-field table: {problems}")
    results, last_a_start = {}, None
    for judge, repeat in slots:
        slot = f"{judge}{repeat}"
        request = requests[judge]
        records = log.attempts(trace_key, config_sha256, judge, repeat)
        if _slot_done(records):
            results[slot] = _slot_result(records)
            continue
        while True:
            attempt = len(records) + 1
            ref = f"{trace_key}/{slot}/{attempt}"
            started = clock()
            exchange = transport.send(request)
            finished = clock()
            oc = canonical_outcome(request.provider, exchange)
            summary = oc["summary"] or {}
            headers = {k.lower(): v for k, v in (exchange.headers or {}).items()}
            checks = []
            if oc["outcome"] in TERMINAL and summary.get("complete"):
                checks = _identity_checks(log, request.provider, judge, config_sha256, summary, headers, workspace,
                                          ref)
            outcome, halt_kind = oc["outcome"], oc["halt_kind"]
            if checks:
                outcome, halt_kind = HALT, "response_check"
            since = None if judge != "A" or last_a_start is None else round(started - last_a_start, 3)
            record = {
                "trace_key": trace_key, "config_sha256": config_sha256, "judge": judge, "repeat": repeat,
                "slot": slot, "attempt": attempt, "provider": request.provider,
                "request_sha256": request.sha256, "request_bytes": len(request.body),
                "model_requested": request.model,
                "started_at": _iso(started), "finished_at": _iso(finished),
                "first_byte_after_s": (round(exchange.first_byte_at - started, 3)
                                       if exchange.first_byte_at is not None else None),
                "seconds_since_previous_a_start": since,
                "within_cache_lifetime": None if since is None else since <= CACHE_LIFETIME_S,
                "http_status": exchange.status, "transport_error": exchange.transport_error,
                "response_headers": {k: headers[k] for k in KEPT_HEADERS if k in headers},
                "outcome": outcome, "halt_kind": halt_kind, "detail": oc["detail"],
                "response_outcome": oc["outcome"], "failed_checks": checks, "error": oc["error"],
                "response_id": summary.get("id"), "response_model": summary.get("model"),
                "stop": summary.get("stop_reason") or summary.get("terminal_event"),
                "usage": summary.get("usage") or {}, "parse": oc["parse"],
            }
            log.write_attempt(record, exchange.raw)
            records.append(record)
            if judge == "A":
                last_a_start = started
            if outcome == HALT:
                halt_id = log.write_halt({"trace_key": trace_key, "config_sha256": config_sha256,
                                          "slot": slot, "attempt": attempt, "halt_kind": halt_kind,
                                          "detail": oc["detail"], "failed_checks": checks,
                                          "error": oc["error"], "http_status": exchange.status,
                                          "at": _iso(finished),
                                          "resume_rule": "record a correction note; the slot resumes "
                                                         "with its next attempt index"})
                return {"halted": True, "halt_id": halt_id, "slots": results}
            if oc["retryable"]:
                retries = sum(r["outcome"] == TRANSPORT for r in records)
                if retries <= R.MAX_RETRIES:
                    sleep(_backoff(retries, headers))
                    continue
            results[slot] = _slot_result(records)
            break
    return {"halted": False, "halt_id": None, "slots": results}


# ---------------------------------------------------------------- recovery

# A stream may open with an error event; a JSON error body is never an event.
STREAM_STARTS = {"anthropic": ("message_start", "error"),
                 "openai": ("response.created", "response.in_progress", "response.queued", "error")}


def _stream_started(provider: str, raw: bytes) -> bool:
    events = parse_sse(raw)
    if not events:
        return False
    try:
        first = json.loads(events[0][1])
    except ValueError:
        return False
    return isinstance(first, dict) and first.get("type") in STREAM_STARTS[provider]


def recover_attempt(log, reservation: dict) -> dict:
    """Write and return the record of the attempt ``reservation`` paid for, rebuilt from its
    raw response, whose record was never written (see the module docstring)."""
    trace_key, config_sha256 = reservation["trace_key"], reservation["config_sha256"]
    judge, repeat, attempt, provider = (reservation["judge"], reservation["repeat"], reservation["attempt"],
                                        reservation["provider"])
    slot = f"{judge}{repeat}"
    raw = (log.slot_dir(trace_key, config_sha256, judge, repeat) / f"{attempt}.raw").read_bytes()
    status = 200 if _stream_started(provider, raw) else None
    exchange = Exchange(status, {}, raw, None if status else "the response's status was not kept", None)
    error_body = None if status else error_body_outcome(provider, raw)
    oc = error_body or canonical_outcome(provider, exchange)
    summary = oc["summary"] or {}
    checks, ref = [], f"{trace_key}/{slot}/{attempt}"
    if oc["outcome"] in TERMINAL and summary.get("complete"):
        first = log.first_model(config_sha256, judge)
        checks = response_checks(provider, summary, {}, first, None)
        if not checks and first is None:
            log.set_first_model(config_sha256, judge, summary.get("model"), ref)
        rid = summary.get("id")             # the stopped process may have claimed it for this attempt
        if rid and not log.claim_response_id(provider, rid, ref) and not log.owns_response_id(provider, rid, ref):
            checks.append(f"provider response id {rid!r} was seen before")
    outcome, halt_kind = oc["outcome"], oc["halt_kind"]
    if checks:
        outcome, halt_kind = HALT, "response_check"
    record = {
        "trace_key": trace_key, "config_sha256": config_sha256, "judge": judge, "repeat": repeat, "slot": slot,
        "attempt": attempt, "provider": provider, "request_sha256": reservation["request_sha256"],
        "request_bytes": None, "model_requested": reservation["model"], "started_at": None, "finished_at": None,
        "first_byte_after_s": None, "seconds_since_previous_a_start": None, "within_cache_lifetime": None,
        "http_status": status, "transport_error": exchange.transport_error, "response_headers": {},
        "outcome": outcome, "halt_kind": halt_kind, "detail": oc["detail"], "response_outcome": oc["outcome"],
        "failed_checks": checks, "error": oc["error"], "response_id": summary.get("id"),
        "response_model": summary.get("model"), "stop": summary.get("stop_reason") or summary.get("terminal_event"),
        "usage": summary.get("usage") or {}, "parse": oc["parse"],
        "recovered": {"from": f"{attempt}.raw",
                      "http_status": ("inferred from the stream" if status else
                                      "unknown; an error body, classified as in a stream" if error_body else
                                      "unknown; taken as no response"),
                      "checks_not_applied": ["anthropic-workspace-id header"] if provider == "anthropic" else []},
    }
    log.complete_attempt(record)
    return record


# ---------------------------------------------------------------- capability check


def capability_check(records: list, model: str) -> dict:
    """The fallback check for judge B on its first responses: access, image acceptance,
    strict structured output and a stable model string. Refusals are excluded: a refusal
    is a recorded outcome and never a reason to change judges. No invalid-rate threshold
    is used.

    ``records`` are B attempt records in dispatch order. Each check is "ok", "failed" or
    "unknown" (no response yet that shows it).
    """
    access = images = strict = "unknown"
    models = []
    for r in records:
        if r["outcome"] == REFUSAL or r.get("response_outcome") == REFUSAL:
            continue
        code = (r.get("error") or {}).get("code")
        if r.get("halt_kind") in ("auth", "permission", "not_found"):
            access = "failed"
        if code in IMAGE_ERROR_CODES:
            images = "failed"
        if r.get("response_model"):
            models.append(r["response_model"])
            if access == "unknown":
                access = "ok"
            if images == "unknown":
                images = "ok"
        rc = r.get("response_outcome")
        if strict == "unknown" and rc in (VALID, ABSTAIN):
            strict = "ok"
        elif strict == "unknown" and rc == MALFORMED:
            strict = "failed"
    stable = "unknown" if not models else ("ok" if len(set(models)) == 1 else "failed")
    checks = {"access": access, "images_accepted": images, "strict_output": strict,
              "stable_model_string": stable}
    failed = sorted(k for k, v in checks.items() if v == "failed")
    candidates = R.B_FALLBACK_MODELS if model == R.B_MODEL else ()
    return {"model": model, "checks": checks, "failed": failed,
            "fallback_indicated": bool(failed) and bool(candidates),
            "next_model": candidates[0] if failed and candidates else None,
            "models_seen": sorted(set(models))}
