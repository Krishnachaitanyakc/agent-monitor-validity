"""The serial dispatcher for one trajectory, its transports and the response checks.

**Order.** A1, then A2, then A3, then B1, one after another. A1, A2 and A3 send the same
``PreparedRequest``, built before dispatch, so their bytes and settings are identical and
no answer can reach a later call. Every slot runs whatever an earlier slot returned; only
a halt stops the sequence. The gap between the start of one A request and the next is
recorded, since a repeat reads the prompt cache only within its five-minute lifetime, and
every response's cache read and write token counts are logged.

**Retries.** Only TRANSPORT is retried: the same bytes, the same slot, at most three
retries with exponential backoff (a server's ``retry-after`` is honoured when longer, up
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
``cached_tokens`` and ``cache_write_tokens`` are 0. A field that is not reported fails.

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
                       TRANSPORT, VALID, Exchange, canonical_outcome)

CACHE_LIFETIME_S = 300
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
    return fails


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
    exists (the dry run). ``slots`` runs a subset of the four in the same order, starting
    with A1; prompt development sends A1 and B1 only. Returns {"halted": bool, "halt_id",
    "slots": {slot: result}}.
    """
    slots = tuple(slots)
    if not slots or slots[0] != ("A", 1) or slots != tuple(s for s in R.SLOTS if s in slots):
        raise ValueError(f"slots {slots} are not A1 followed by others of {R.SLOTS} in order")
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
                first = log.first_model(config_sha256, judge)
                checks = response_checks(request.provider, summary, headers, first, workspace)
                if not checks and first is None:
                    log.set_first_model(config_sha256, judge, summary.get("model"), ref)
                if summary.get("id") and not log.claim_response_id(request.provider, summary["id"], ref):
                    checks.append(f"provider response id {summary['id']!r} was seen before")
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
