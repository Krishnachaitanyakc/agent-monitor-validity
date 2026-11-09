"""Exact request bodies for judge A (Anthropic Messages) and judge B (OpenAI Responses).

One pack and its screenshots give one body per judge. The bodies are the exact bytes a
transport sends; nothing is added at send time except the credential header and, on an
OpenAI request, the project header.

**Evidence.** Both judges receive the same evidence parts in the same order: a text part
with the pack's version, field notes and goal; for each step, a text part with that
step's pack entry, followed by its screenshot when it has one; and a text part with the
final tree. Each text part is a fixed label line plus the canonical JSON of that piece of
the pack, so the parts rebuild the pack exactly. The judges differ only in wrapping.

**Settings.** Every field follows the setup note's request-field table (``A_SETTINGS`` and
``B_SETTINGS``). No sampling parameter, tool, stop sequence, response chaining, cache key
or fallback list is ever sent; ``check_body`` refuses a body whose fields differ from the
table.

**Serialization and hashes.** ``canonical_bytes`` is compact UTF-8 JSON with keys sorted
at every level, except inside a JSON Schema ``properties`` object, which keeps schema S's
field order because providers generate fields in schema order. A request's hash is the
SHA-256 of its body bytes. The config hash covers the endpoint, headers and every body
field of both judges (the bodies with the evidence left out, which include the rubric
and the schema), the rubric, schema, parser, pack and builder versions, the token caps,
the image limits and fallback, the slots and the retry rule.

**Fit.** A request fits when every provider limit below holds. If either judge's request
does not fit, every screenshot is downscaled to a 1,024 px long edge, identically for
both judges, and both bodies are rebuilt; if a request still does not fit, the pack is
ineligible and the reasons are recorded.
"""

import base64
import hashlib
import json
import math
from collections import namedtuple
from pathlib import Path

from datapath import constants as C

from . import images as I
from . import pack as P
from .rubric import RUBRIC_SHA256, RUBRIC_STATUS, RUBRIC_TEXT, RUBRIC_VERSION
from .schema import PARSER_VERSION, SCHEMA, SCHEMA_NAME, SCHEMA_VERSION

REQUESTS_VERSION = "requests-v0"
OUTPUT_CAP = 32000

A_MODEL = "claude-sonnet-5"
B_MODEL = "gpt-6-sol"
B_FALLBACK_MODELS = ("gpt-6.1-sol",)       # first fallback, only on a failed capability check

A_ENDPOINT = "https://api.anthropic.com/v1/messages"
B_ENDPOINT = "https://api.openai.com/v1/responses"
# Pinned API version: the version Anthropic's Python SDK 2.0.0 sends.
ANTHROPIC_VERSION = "2026-10-21"
A_HEADERS = (("anthropic-version", ANTHROPIC_VERSION), ("content-type", "application/json"))
B_HEADERS = (("content-type", "application/json"),)

# The request-field table, apart from model, system or input, and messages.
A_SETTINGS = {
    "max_tokens": OUTPUT_CAP,
    "stream": True,
    "thinking": {"type": "adaptive"},
    "output_config": {"effort": "high", "format": {"type": "json_schema", "schema": SCHEMA}},
    "cache_control": {"type": "ephemeral"},        # omitted in prompt development
    "inference_geo": "global",
    "service_tier": "standard_only",
}
B_SETTINGS = {
    "max_output_tokens": OUTPUT_CAP,
    "stream": True,
    "reasoning": {"effort": "medium", "mode": "standard"},
    "text": {"format": {"type": "json_schema", "name": SCHEMA_NAME, "schema": SCHEMA, "strict": True},
             "verbosity": "medium"},
    "prompt_cache_options": {"mode": "explicit"},  # and no breakpoint, so nothing is cached
    "service_tier": "default",
    "store": False,
}
NEVER_SENT = frozenset({"temperature", "top_p", "top_k", "top_logprobs", "tools", "tool_choice",
                        "stop_sequences", "stop", "fallbacks", "previous_response_id",
                        "prompt_cache_key", "prompt_cache_retention", "conversation", "seed"})

# Provider limits checked before any send. Anthropic: at most 600 images per request on a
# 1M-context model, 8,000 px per side, at most 2,000 px per side when a request has more
# than 20 images, 10 MB of base64 per image and 128 MB per request. OpenAI: at most 1,500
# images, 30,000 patches of 32 px per image and 512 MB per request. MB counts 10**6 bytes.
A_LIMITS = {"max_images": 600, "max_side": 8000, "many_images_above": 20, "many_images_max_side": 2000,
            "max_image_base64_bytes": 10 * 10**6, "max_body_bytes": 128 * 10**6}
B_LIMITS = {"max_images": 1500, "patch_px": 32, "max_patches_per_image": 30000,
            "max_body_bytes": 512 * 10**6}
FALLBACK_LONG_EDGE = 1024

PACK_KEYS = frozenset({"pack_version", "field_notes", "goal", "steps", "final_tree"})
SLOTS = (("A", 1), ("A", 2), ("A", 3), ("B", 1))
MAX_RETRIES = 3

# Input-token estimates for the report only; the provider's own count replaces them.
CHARS_PER_TOKEN = 2.5        # Anthropic: about 2.5M characters per 1M tokens

PreparedRequest = namedtuple("PreparedRequest", "judge provider model url headers body sha256")


# ---------------------------------------------------------------- serialization

def _canon(obj, keep_order=False):
    if isinstance(obj, dict):
        items = list(obj.items()) if keep_order else sorted(obj.items())
        return {k: _canon(v, keep_order=(k == "properties" and isinstance(v, dict) and not keep_order))
                for k, v in items}
    if isinstance(obj, (list, tuple)):
        return [_canon(v) for v in obj]
    return obj


def canonical(obj):
    """``obj`` with its keys in canonical order (see the module docstring)."""
    return _canon(obj)


def canonical_bytes(obj) -> bytes:
    return json.dumps(_canon(obj), ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------- evidence

def evidence_parts(pack: dict) -> list:
    """The ordered evidence parts: ("text", str) or ("image", screenshot reference)."""
    if set(pack) != PACK_KEYS or pack.get("pack_version") != P.PACK_VERSION:
        raise ValueError(f"not a {P.PACK_VERSION} pack: keys {sorted(pack)}")
    parts = [("text", "Task goal and field notes:\n" + P.serialize(
        {k: pack[k] for k in ("pack_version", "field_notes", "goal")}))]
    for step in pack["steps"]:
        parts.append(("text", f"Step {step['step']}:\n" + P.serialize(step)))
        if step["screenshot"] is not None:
            parts.append(("image", step["screenshot"]))
    parts.append(("text", "Final page:\n" + P.serialize({"final_tree": pack["final_tree"]})))
    return parts


def rebuild_pack(texts: list) -> dict:
    """The pack rebuilt from the text parts of a request, in order (for checks)."""
    pack = json.loads(texts[0].split("\n", 1)[1])
    pack["steps"] = [json.loads(t.split("\n", 1)[1]) for t in texts[1:-1]]
    pack.update(json.loads(texts[-1].split("\n", 1)[1]))
    return pack


def read_images(entries: list, data_root) -> dict:
    """{reference: PNG bytes} for one pack's image-manifest entries.

    Each file is read from ``<data_root>/raw/<path>`` and must match the manifest's byte
    count and SHA-256. The caller installs the development-only guard first.
    """
    out = {}
    for e in entries:
        if e["ref"] in out:
            raise ValueError(f"the manifest lists {e['ref']} twice")
        data = (Path(data_root) / "raw" / e["path"]).read_bytes()
        if len(data) != e["bytes"] or sha256_bytes(data) != e["sha256"]:
            raise ValueError(f"{e['ref']}: the file does not match the manifest")
        I.png_size(data)
        out[e["ref"]] = data
    return out


def _image_list(parts, images: dict) -> list:
    refs = [ref for kind, ref in parts if kind == "image"]
    if len(set(refs)) != len(refs) or set(refs) != set(images):
        raise ValueError(f"screenshots {sorted(images)} do not match the pack's references {refs}")
    return refs


# ---------------------------------------------------------------- bodies

def _a_content(parts, images):
    out = []
    for kind, value in parts:
        if kind == "text":
            out.append({"type": "text", "text": value})
        else:
            out.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                    "data": base64.b64encode(images[value]).decode("ascii")}})
    return out


def _b_content(parts, images):
    out = []
    for kind, value in parts:
        if kind == "text":
            out.append({"type": "input_text", "text": value})
        else:
            out.append({"type": "input_image", "detail": "original",
                        "image_url": "data:image/png;base64," + base64.b64encode(images[value]).decode("ascii")})
    return out


def body_a(content: list, caching: bool = True) -> dict:
    body = {"model": A_MODEL, "system": [{"type": "text", "text": RUBRIC_TEXT}],
            "messages": [{"role": "user", "content": content}]}
    body.update(A_SETTINGS)
    if not caching:
        del body["cache_control"]
    return body


def body_b(content: list, model: str = B_MODEL) -> dict:
    body = {"model": model,
            "input": [{"role": "developer", "content": [{"type": "input_text", "text": RUBRIC_TEXT}]},
                      {"role": "user", "content": content}]}
    body.update(B_SETTINGS)
    return body


def prepared(judge: str, body: dict) -> PreparedRequest:
    data = canonical_bytes(body)
    provider, url, headers = (("anthropic", A_ENDPOINT, A_HEADERS) if judge == "A"
                              else ("openai", B_ENDPOINT, B_HEADERS))
    return PreparedRequest(judge, provider, body["model"], url, headers, data, sha256_bytes(data))


def expected_fields(judge: str, caching: bool = True) -> set:
    if judge == "A":
        return {"model", "system", "messages"} | set(A_SETTINGS) - ({"cache_control"} if not caching else set())
    return {"model", "input"} | set(B_SETTINGS)


def check_body(judge: str, body: dict, caching: bool = True) -> list:
    """Differences between one body and the request-field table; empty when it matches."""
    problems = []
    keys = set(body)
    if keys & NEVER_SENT:
        problems.append(f"never-sent fields present: {sorted(keys & NEVER_SENT)}")
    if keys != expected_fields(judge, caching):
        problems.append(f"fields {sorted(keys)} differ from the table")
    settings = A_SETTINGS if judge == "A" else B_SETTINGS
    for k, v in settings.items():
        if k in body and canonical_bytes(body[k]) != canonical_bytes(v):
            problems.append(f"{k} differs from the table")
    if judge == "A":
        if body.get("model") != A_MODEL:
            problems.append("model is not judge A's")
        if body.get("system") != [{"type": "text", "text": RUBRIC_TEXT}]:
            problems.append("system text is not the rubric")
        msgs = body.get("messages") or []
        if len(msgs) != 1 or msgs[0].get("role") != "user":
            problems.append("not exactly one user turn")
    else:
        if body.get("model") not in (B_MODEL,) + B_FALLBACK_MODELS:
            problems.append("model is not judge B's or its fallback")
        items = body.get("input") or []
        if (len(items) != 2 or items[0] != {"role": "developer",
                                            "content": [{"type": "input_text", "text": RUBRIC_TEXT}]}
                or items[1].get("role") != "user"):
            problems.append("input is not the rubric as developer text plus one user turn")
    return problems


def neutral_content(judge: str, body: dict) -> list:
    """The evidence in one body as provider-neutral parts: ("text", s) or ("image", base64)."""
    content = body["messages"][0]["content"] if judge == "A" else body["input"][1]["content"]
    out = []
    for block in content:
        if block["type"] in ("text", "input_text"):
            out.append(("text", block["text"]))
        elif block["type"] == "image":
            out.append(("image", block["source"]["data"]))
        elif block["type"] == "input_image":
            prefix = "data:image/png;base64,"
            out.append(("image", block["image_url"][len(prefix):]
                        if block["image_url"].startswith(prefix) else None))
        else:
            out.append((block["type"], None))
    return out


# ---------------------------------------------------------------- fit

def _image_problems(judge, sizes, b64_lengths) -> list:
    problems = []
    if judge == "A":
        lim = A_LIMITS
        if len(sizes) > lim["max_images"]:
            problems.append(f"A: {len(sizes)} images, above {lim['max_images']}")
        side = lim["many_images_max_side"] if len(sizes) > lim["many_images_above"] else lim["max_side"]
        big = [i for i, (w, h) in enumerate(sizes) if max(w, h) > side]
        if big:
            problems.append(f"A: {len(big)} of {len(sizes)} images exceed {side} px per side")
        heavy = [i for i, n in enumerate(b64_lengths) if n > lim["max_image_base64_bytes"]]
        if heavy:
            problems.append(f"A: {len(heavy)} images exceed {lim['max_image_base64_bytes']} base64 bytes")
    else:
        lim = B_LIMITS
        if len(sizes) > lim["max_images"]:
            problems.append(f"B: {len(sizes)} images, above {lim['max_images']}")
        px = lim["patch_px"]
        big = [i for i, (w, h) in enumerate(sizes)
               if math.ceil(w / px) * math.ceil(h / px) > lim["max_patches_per_image"]]
        if big:
            problems.append(f"B: {len(big)} images exceed {lim['max_patches_per_image']} patches")
    return problems


def fit_problems(judge: str, request: PreparedRequest, sizes: list, b64_lengths: list) -> list:
    """Every provider limit that one request breaks; empty when it fits."""
    problems = _image_problems(judge, sizes, b64_lengths)
    cap = (A_LIMITS if judge == "A" else B_LIMITS)["max_body_bytes"]
    if len(request.body) > cap:
        problems.append(f"{judge}: body is {len(request.body)} bytes, above {cap}")
    return problems


def _build_both(parts, images, b_model, caching):
    a = prepared("A", body_a(_a_content(parts, images), caching))
    b = prepared("B", body_b(_b_content(parts, images), b_model))
    return a, b


def prepare(pack: dict, images: dict, b_model: str = B_MODEL, caching: bool = True) -> dict:
    """Both judges' requests for one pack, with the fit check and the shared fallback.

    Returns {"eligible", "reasons", "fallback": {"applied", "triggered_by", "long_edge",
    "error"}, "requests": {"A": PreparedRequest, "B": PreparedRequest} or None,
    "image_sizes": [(width, height) as sent], "native_sizes": [...]}.
    """
    parts = evidence_parts(pack)
    refs = _image_list(parts, images)
    native = [I.png_size(images[r]) for r in refs]

    def attempt(imgs):
        a, b = _build_both(parts, imgs, b_model, caching)
        sizes = [I.png_size(imgs[r]) for r in refs]
        lengths = [4 * math.ceil(len(imgs[r]) / 3) for r in refs]
        problems = fit_problems("A", a, sizes, lengths) + fit_problems("B", b, sizes, lengths)
        return a, b, sizes, problems

    a, b, sizes, problems = attempt(images)
    out = {"eligible": True, "reasons": [], "requests": {"A": a, "B": b}, "image_sizes": sizes,
           "native_sizes": native,
           "fallback": {"applied": False, "triggered_by": [], "long_edge": FALLBACK_LONG_EDGE, "error": None}}
    if not problems:
        return out
    out["fallback"]["triggered_by"] = problems
    try:
        small = {r: I.fit_long_edge(images[r], FALLBACK_LONG_EDGE) for r in refs}
    except ValueError as e:
        out.update(eligible=False, requests=None,
                   reasons=problems + [f"the fallback cannot downscale a screenshot: {e}"])
        out["fallback"]["error"] = str(e)
        return out
    a, b, sizes, problems = attempt(small)
    out["fallback"]["applied"] = True
    out["image_sizes"] = sizes
    if problems:
        out.update(eligible=False, requests=None, reasons=problems)
    else:
        out["requests"] = {"A": a, "B": b}
    return out


# ---------------------------------------------------------------- config

def config_record(b_model: str = B_MODEL, caching: bool = True) -> dict:
    """Everything that fixes how requests are built and sent, apart from the evidence."""
    return {
        "requests_version": REQUESTS_VERSION, "pack_version": P.PACK_VERSION,
        "rubric": {"version": RUBRIC_VERSION, "status": RUBRIC_STATUS, "sha256": RUBRIC_SHA256},
        "schema": {"version": SCHEMA_VERSION, "sha256": hashlib.sha256(canonical_bytes(SCHEMA)).hexdigest()},
        "parser_version": PARSER_VERSION, "output_cap_tokens": OUTPUT_CAP,
        "judges": {
            "A": {"provider": "anthropic", "endpoint": A_ENDPOINT, "headers": dict(A_HEADERS),
                  "model": A_MODEL, "body_without_evidence": body_a([], caching)},
            "B": {"provider": "openai", "endpoint": B_ENDPOINT, "headers": dict(B_HEADERS),
                  "model": b_model, "body_without_evidence": body_b([], b_model)},
        },
        "images": {"media_type": "image/png", "a_limits": A_LIMITS, "b_limits": B_LIMITS,
                   "fallback_long_edge": FALLBACK_LONG_EDGE, "fallback": "both judges, every screenshot"},
        "slots": [f"{j}{r}" for j, r in SLOTS], "max_retries": MAX_RETRIES, "caching": caching,
    }


def config_sha256(b_model: str = B_MODEL, caching: bool = True) -> str:
    return sha256_bytes(canonical_bytes(config_record(b_model, caching)))


# ---------------------------------------------------------------- checks and examples

IMAGE_PLACEHOLDER = "<base64 PNG omitted: {n} bytes, SHA-256 {sha}>"


def redacted(judge: str, body: dict) -> dict:
    """A copy of ``body`` with each image's base64 replaced by its length and SHA-256."""
    body = json.loads(canonical_bytes(body))
    content = body["messages"][0]["content"] if judge == "A" else body["input"][1]["content"]
    for block in content:
        if block["type"] == "image":
            d = block["source"]["data"]
            block["source"]["data"] = IMAGE_PLACEHOLDER.format(n=len(d), sha=sha256_bytes(d.encode("ascii")))
        elif block["type"] == "input_image":
            prefix, d = block["image_url"].split(",", 1)
            block["image_url"] = prefix + "," + IMAGE_PLACEHOLDER.format(
                n=len(d), sha=sha256_bytes(d.encode("ascii")))
    return body


def restore(judge: str, redacted_body: dict, images_b64: list) -> bytes:
    """Body bytes rebuilt from a redacted body and the base64 images, in order (for checks)."""
    body = json.loads(json.dumps(redacted_body))
    content = body["messages"][0]["content"] if judge == "A" else body["input"][1]["content"]
    it = iter(images_b64)
    for block in content:
        if block["type"] == "image":
            block["source"]["data"] = next(it)
        elif block["type"] == "input_image":
            block["image_url"] = block["image_url"].split(",", 1)[0] + "," + next(it)
    return canonical_bytes(body)


def _keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _keys(v)


def check_request(judge: str, body_bytes: bytes, private_strings=()) -> dict:
    """Leak checks on one request body, with its image data left out of the text search.

    Fails when a private or fenced key appears at any depth of the body (apart from the
    top-level ``model``, which names the judge), a fenced field
    name or the label file's name appears anywhere, or any private string (identifier,
    path or hash from provenance or the image manifest) appears anywhere.
    """
    body = json.loads(body_bytes)
    text = canonical_bytes(redacted(judge, body)).decode("utf-8")
    # The top-level "model" names the judge; every other key at any depth is checked.
    keys = sorted({k for k in _keys({k: v for k, v in body.items() if k != "model"})
                   if k in P.PRIVATE_KEYS})
    substrings = [s for s in C.FORBIDDEN_SUBSTRINGS if s in text]
    found = [s for s in private_strings if s and s in text]
    return {"ok": not (keys or substrings or found), "private_keys": keys,
            "forbidden_substrings": substrings, "private_strings": found}


def manifest_strings(entries: list) -> list:
    """Strings from a pack's image-manifest entries that must never reach a request."""
    out = []
    for e in entries:
        out += [e.get("trace_key"), e.get("pack_sha256"), e.get("path"), e.get("sha256")]
    return sorted({s for s in out if isinstance(s, str) and s})


def token_estimate(judge: str, request: PreparedRequest, sizes: list) -> dict:
    """A rough input-token estimate: text characters / 2.5, plus image tokens.

    A: ceil(w/28) x ceil(h/28) per image, capped at 4,784 (Anthropic's high-resolution
    tier). B: 32-px patches per image, an assumption until the first paid day measures it.
    """
    body = json.loads(request.body)
    chars = len(RUBRIC_TEXT) + len(canonical_bytes(SCHEMA).decode("utf-8"))
    chars += sum(len(v) for kind, v in neutral_content(judge, body) if kind == "text")
    if judge == "A":
        image_tokens = sum(min(4784, math.ceil(w / 28) * math.ceil(h / 28)) for w, h in sizes)
    else:
        image_tokens = sum(math.ceil(w / 32) * math.ceil(h / 32) for w, h in sizes)
    text_tokens = math.ceil(chars / CHARS_PER_TOKEN)
    return {"text_chars": chars, "text_tokens": text_tokens, "image_tokens": image_tokens,
            "input_tokens": text_tokens + image_tokens}
