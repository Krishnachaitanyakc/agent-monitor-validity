"""Server-sent event streams from both providers: parsing, summaries and synthetic streams.

``parse_sse`` follows the SSE format: lines end in LF, CRLF or CR; a blank line ends an
event; ``event:`` names it, ``data:`` lines are joined with LF, a line starting with a
colon is a comment, and one space after the colon is dropped. An event that is still open
when the bytes end is discarded, so a cut stream never yields a half event.

``anthropic_summary`` reads a Messages stream: ``message_start`` (id, model, usage),
content blocks and their deltas, ``message_delta`` (stop reason, cumulative usage),
``message_stop``, ``ping`` and ``error``. ``openai_summary`` reads a Responses stream:
``response.created`` and the other lifecycle events, the terminal ``response.completed``,
``response.incomplete`` or ``response.failed`` (each carrying the full response), and
``error``. Other event types are counted and otherwise ignored.

A summary is ``complete`` only when the stream reached its provider's end: for Anthropic,
``message_start``, a stop reason and ``message_stop`` with no error event; for OpenAI, a
terminal response event with no error event. The answer text is extracted only from a
complete stream; partial content is never read.

The synthetic stream builders at the end produce provider-shaped streams for the tests
and the development dry run. They are never sent anywhere.
"""

import json

# ---------------------------------------------------------------- parsing


def parse_sse(raw: bytes) -> list:
    """[(event name, data string), ...] for every finished event in ``raw``."""
    text = raw.decode("utf-8", errors="replace")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    events, name, data = [], None, []
    # The last element follows the final line break; an unterminated line is dropped.
    for line in lines[:-1]:
        if line == "":
            if data:
                events.append((name or "message", "\n".join(data)))
            name, data = None, []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            name = value
        elif field == "data":
            data.append(value)
    return events


def _json_events(raw: bytes):
    bad = 0
    out = []
    for name, data in parse_sse(raw):
        try:
            obj = json.loads(data)
        except ValueError:
            bad += 1
            continue
        if isinstance(obj, dict):
            out.append((name, obj))
        else:
            bad += 1
    return out, bad


def _int(v):
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def anthropic_summary(raw: bytes) -> dict:
    """The final state of a Messages stream (see the module docstring)."""
    events, bad = _json_events(raw)
    s = {"provider": "anthropic", "complete": False, "id": None, "model": None, "stop_reason": None,
         "stop_details": None, "text": None, "usage_raw": {}, "usage": {}, "error": None,
         "events": len(events), "unparsed_events": bad, "unknown_event_types": []}
    blocks, started, stopped = {}, False, False
    usage = {}
    for _, obj in events:
        t = obj.get("type")
        if t == "message_start":
            started = True
            msg = obj.get("message") or {}
            s["id"], s["model"] = msg.get("id"), msg.get("model")
            usage = dict(msg.get("usage") or {})
        elif t == "content_block_start":
            block = obj.get("content_block") or {}
            blocks[obj.get("index")] = {"type": block.get("type"), "text": block.get("text") or ""}
        elif t == "content_block_delta":
            delta = obj.get("delta") or {}
            block = blocks.get(obj.get("index"))
            if block is not None and delta.get("type") == "text_delta":
                block["text"] += delta.get("text") or ""
        elif t == "message_delta":
            delta = obj.get("delta") or {}
            if delta.get("stop_reason") is not None:
                s["stop_reason"] = delta.get("stop_reason")
                s["stop_details"] = delta.get("stop_details")
            for k, v in (obj.get("usage") or {}).items():
                if v is not None:
                    usage[k] = v                     # cumulative counts replace earlier ones
        elif t == "message_stop":
            stopped = True
        elif t == "error":
            s["error"] = obj.get("error") or {"type": None}
        elif t not in ("ping", "content_block_stop"):
            s["unknown_event_types"].append(t)
    s["usage_raw"] = usage
    cache = usage.get("cache_creation") if isinstance(usage.get("cache_creation"), dict) else {}
    s["usage"] = {"input_tokens": _int(usage.get("input_tokens")),
                  "output_tokens": _int(usage.get("output_tokens")),
                  "cache_read_tokens": _int(usage.get("cache_read_input_tokens")),
                  "cache_write_tokens": _int(usage.get("cache_creation_input_tokens")),
                  "cache_write_5m_tokens": _int(cache.get("ephemeral_5m_input_tokens")),
                  "cache_write_1h_tokens": _int(cache.get("ephemeral_1h_input_tokens")),
                  "service_tier": usage.get("service_tier"), "inference_geo": usage.get("inference_geo")}
    s["complete"] = started and stopped and s["stop_reason"] is not None and s["error"] is None
    if s["complete"]:
        s["text"] = "".join(b["text"] for i, b in sorted(blocks.items(), key=lambda kv: kv[0] if isinstance(kv[0], int) else -1)
                            if b["type"] == "text")
    return s


TERMINAL_EVENTS = ("response.completed", "response.incomplete", "response.failed")


def openai_summary(raw: bytes) -> dict:
    """The final state of a Responses stream (see the module docstring)."""
    events, bad = _json_events(raw)
    s = {"provider": "openai", "complete": False, "terminal_event": None, "status": None, "id": None,
         "model": None, "service_tier": None, "incomplete_reason": None, "response_error": None,
         "text": None, "refusal": None, "usage_raw": {}, "usage": {}, "error": None,
         "events": len(events), "unparsed_events": bad}
    final = None
    for _, obj in events:
        t = obj.get("type")
        if t in ("response.created", "response.in_progress", "response.queued"):
            r = obj.get("response") or {}
            s["id"], s["model"] = r.get("id") or s["id"], r.get("model") or s["model"]
        elif t in TERMINAL_EVENTS:
            final = obj.get("response") or {}
            s["terminal_event"] = t
        elif t == "error":
            s["error"] = {"code": obj.get("code"), "message": obj.get("message"), "param": obj.get("param")}
    if final is not None:
        s["id"], s["model"] = final.get("id") or s["id"], final.get("model") or s["model"]
        s["status"], s["service_tier"] = final.get("status"), final.get("service_tier")
        s["incomplete_reason"] = (final.get("incomplete_details") or {}).get("reason")
        s["response_error"] = final.get("error")
        usage = final.get("usage") or {}
        s["usage_raw"] = usage
        details = usage.get("input_tokens_details") or {}
        out_details = usage.get("output_tokens_details") or {}
        s["usage"] = {"input_tokens": _int(usage.get("input_tokens")),
                      "output_tokens": _int(usage.get("output_tokens")),
                      "cache_read_tokens": _int(details.get("cached_tokens")),
                      "cache_write_tokens": _int(details.get("cache_write_tokens")),
                      "reasoning_tokens": _int(out_details.get("reasoning_tokens"))}
    s["complete"] = final is not None and s["error"] is None
    if s["complete"] and s["terminal_event"] == "response.completed":
        texts, refusals = [], []
        for item in final.get("output") or []:
            if item.get("type") == "refusal":
                refusals.append(item.get("refusal") or "")
            if item.get("type") != "message":
                continue
            for part in item.get("content") or []:
                if part.get("type") == "output_text":
                    texts.append(part.get("text") or "")
                elif part.get("type") == "refusal":
                    refusals.append(part.get("refusal") or "")
        s["text"] = "".join(texts)
        s["refusal"] = "\n".join(refusals) if refusals else None
    return s


# ---------------------------------------------------------------- synthetic streams


def sse(events) -> bytes:
    """Encode [(event name, object), ...] as an SSE stream, as both providers send it."""
    return "".join(f"event: {name}\ndata: {json.dumps(obj)}\n\n" for name, obj in events).encode("utf-8")


def anthropic_events(text, stop_reason="end_turn", model="claude-sonnet-5", msg_id="msg_synthetic",
                     usage=None, service_tier="standard", inference_geo="global", thinking=True):
    """A Messages stream: an omitted thinking block, one text block, then the stop."""
    u = {"input_tokens": 1000, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
         "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
         "output_tokens": 1, "service_tier": service_tier, "inference_geo": inference_geo}
    u.update(usage or {})
    events = [("message_start", {"type": "message_start", "message": {
        "id": msg_id, "type": "message", "role": "assistant", "content": [], "model": model,
        "stop_reason": None, "stop_sequence": None, "usage": u}}), ("ping", {"type": "ping"})]
    index = 0
    if thinking:
        events += [("content_block_start", {"type": "content_block_start", "index": 0,
                                            "content_block": {"type": "thinking", "thinking": "", "signature": ""}}),
                   ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                            "delta": {"type": "thinking_delta", "thinking": ""}}),
                   ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                            "delta": {"type": "signature_delta", "signature": "c2ln"}}),
                   ("content_block_stop", {"type": "content_block_stop", "index": 0})]
        index = 1
    if text is not None:
        half = len(text) // 2
        events += [("content_block_start", {"type": "content_block_start", "index": index,
                                            "content_block": {"type": "text", "text": ""}})]
        for piece in (text[:half], text[half:]):
            events.append(("content_block_delta", {"type": "content_block_delta", "index": index,
                                                   "delta": {"type": "text_delta", "text": piece}}))
        events.append(("content_block_stop", {"type": "content_block_stop", "index": index}))
    events += [("message_delta", {"type": "message_delta",
                                  "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                                  "usage": {"output_tokens": 250}}),
               ("message_stop", {"type": "message_stop"})]
    return events


def openai_events(text, status="completed", model="gpt-6-sol", resp_id="resp_synthetic",
                  incomplete_reason=None, refusal=None, error=None, usage=None, service_tier="default"):
    """A Responses stream: created, in progress, the text deltas, then one terminal event."""
    u = {"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
         "output_tokens": 250, "output_tokens_details": {"reasoning_tokens": 200}, "total_tokens": 1250}
    u.update(usage or {})
    base = {"id": resp_id, "object": "response", "model": model, "service_tier": service_tier,
            "store": False, "error": None, "incomplete_details": None, "output": [], "usage": None}
    events = [("response.created", {"type": "response.created", "sequence_number": 0,
                                    "response": dict(base, status="in_progress")}),
              ("response.in_progress", {"type": "response.in_progress", "sequence_number": 1,
                                        "response": dict(base, status="in_progress")})]
    content = []
    if text is not None:
        content.append({"type": "output_text", "text": text, "annotations": []})
        events.append(("response.output_text.delta", {"type": "response.output_text.delta", "item_id": "msg_1",
                                                      "output_index": 1, "content_index": 0, "delta": text,
                                                      "sequence_number": 2}))
    if refusal is not None:
        content.append({"type": "refusal", "refusal": refusal})
    output = [{"id": "rs_1", "type": "reasoning", "summary": []}]
    if content:
        output.append({"id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
                       "content": content})
    final = dict(base, status=status, output=output, usage=u,
                 incomplete_details={"reason": incomplete_reason} if incomplete_reason else None,
                 error=error)
    name = {"completed": "response.completed", "incomplete": "response.incomplete",
            "failed": "response.failed"}[status]
    events.append((name, {"type": name, "sequence_number": 3, "response": final}))
    return events
