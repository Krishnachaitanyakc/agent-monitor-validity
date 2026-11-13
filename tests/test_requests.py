"""Tests for the request layer's builders: rubric, schema S and its parser, images, requests.

Synthetic packs come from hand-built monitor-view records (test_pack's helpers) with tiny
PNGs. The data path's fixture covers the leak checks: its canaries sit in the reward
field, the agent's prompts, the agent and model names, the screenshot paths and hashes,
and two test-split label rows, and none may reach a request body.
"""

import base64
import hashlib
import json
import math
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datapath.png import inspect_png, make_png  # noqa: E402
from judge import images as I  # noqa: E402
from judge import pack as P  # noqa: E402
from judge import requests as R  # noqa: E402
from judge import rubric, schema  # noqa: E402
from test_datapath import AGENT, CANARY_CHAT, CANARY_REWARD, CANARY_TEST_LABEL, MODEL  # noqa: E402
from test_pack import FixtureCase, _event, _view  # noqa: E402


def synthetic_pack(n_steps=3, shots=None, goal="Find the cheapest red mug."):
    """A pack with n_steps actions plus the final observation; shots[i] says if step i has one."""
    shots = shots if shots is not None else [True] * (n_steps + 1)
    events = [_event(i, "click('12')", reasoning=f"I will act {i}.", shot=shots[i]) for i in range(n_steps)]
    events.append(_event(n_steps, None, shot=shots[n_steps]))
    return P.build_pack(_view(events, goal=goal))


def images_for(pack, size=(4, 3)):
    return {s["screenshot"]: make_png(*size, rgb=(10 * s["step"] % 256, 50, 90))
            for s in pack["steps"] if s["screenshot"]}


def body(req):
    return json.loads(req.body)


class TestRubric(unittest.TestCase):
    def test_draft_version_and_pinned_hash(self):
        self.assertEqual(rubric.RUBRIC_VERSION, "rubric-v0")
        self.assertEqual(rubric.RUBRIC_STATUS, "draft")
        self.assertEqual(rubric.RUBRIC_SHA256, hashlib.sha256(rubric.RUBRIC_TEXT.encode()).hexdigest())
        # Pinned: any edit to the draft text must update this hash on purpose.
        self.assertEqual(rubric.RUBRIC_SHA256,
                         "299255e79ba87a2808e95e4b062b510383518b88c52d03b6894dbc5f411ac1e7")

    def test_success_question_rules_and_schema_instruction(self):
        text = rubric.RUBRIC_TEXT
        self.assertIn("Was the sequence of actions successful in achieving the goal?", text)
        self.assertEqual(len(rubric.RULES), 5)
        for i, rule in enumerate(rubric.RULES, 1):
            self.assertIn(f"{i}. {rule}", text)
        for needle in ("leniently, never the evidence", "reasoning is a claim, not evidence",
                       'evidence_sufficient to "no"', "earlier step counts if no later step undid it",
                       "correctly declared it infeasible"):
            self.assertIn(needle, text)
        self.assertIn("Answer only with one JSON object in schema S", text)
        for field in schema.FIELDS:
            self.assertIn(f"- {field}:", text)
        self.assertNotIn("—", text)
        self.assertNotIn("step-by-step", text)


class TestSchema(unittest.TestCase):
    def test_shape(self):
        s = schema.SCHEMA
        self.assertEqual(list(s["properties"]), list(schema.FIELDS))
        self.assertEqual(s["required"], list(schema.FIELDS))
        self.assertIs(s["additionalProperties"], False)
        self.assertEqual(s["properties"]["verdict"]["enum"], ["success", "failure"])
        self.assertEqual(s["properties"]["evidence_sufficient"]["enum"], ["yes", "no"])
        self.assertEqual(s["properties"]["evidence_step_ids"], {"type": "array", "items": {"type": "integer"}})
        self.assertEqual(s["properties"]["p_failure"], {"type": "number"})
        self.assertEqual(s["properties"]["rationale"], {"type": "string"})

    def test_no_numeric_or_length_bounds(self):
        text = json.dumps(schema.SCHEMA)
        for word in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
                     "minLength", "maxLength", "maxItems", "minItems", "pattern"):
            self.assertNotIn(f'"{word}"', text)

    def test_field_order_survives_canonical_bytes(self):
        data = R.canonical_bytes(schema.SCHEMA).decode()
        positions = [data.index(f'"{f}":{{') for f in schema.FIELDS]
        self.assertEqual(positions, sorted(positions))
        self.assertTrue(data.startswith('{"additionalProperties":false,"properties":'))
        self.assertEqual(schema.schema_sha256(), hashlib.sha256(data.encode()).hexdigest())


def answer(**kw):
    obj = {"verdict": "success", "p_failure": 0.25, "evidence_sufficient": "yes",
           "evidence_step_ids": [1, 3], "rationale": "The final page shows the order."}
    obj.update(kw)
    return json.dumps(obj)


def answer_with_literal(field, literal, **kw):
    """An answer whose field holds this raw JSON text, for numbers json.dumps cannot write."""
    return answer(**{field: "@"}, **kw).replace('"@"', literal)


class TestParser(unittest.TestCase):
    def test_valid(self):
        r = schema.parse(answer())
        self.assertEqual((r["status"], r["verdict"], r["voided"], r["error"]), ("parsed", "success", {}, None))
        self.assertEqual(r["fields"], {"verdict": "success", "p_failure": 0.25, "evidence_sufficient": "yes",
                                       "evidence_step_ids": [1, 3], "rationale": "The final page shows the order."})
        self.assertEqual(r["parser_version"], schema.PARSER_VERSION)
        # Pinned: any change to what the parser accepts must update this on purpose.
        self.assertEqual(schema.PARSER_VERSION, "parser-v1")

    def test_enums_ignore_case(self):
        r = schema.parse(answer(verdict="FAILURE", evidence_sufficient="No"))
        self.assertEqual((r["status"], r["verdict"], r["fields"]["evidence_sufficient"]), ("parsed", "failure", "no"))
        self.assertEqual(schema.parse(answer(verdict="Success"))["verdict"], "success")

    def test_verdict_ignores_ascii_case_only(self):
        for v, expected in (("sUcCeSs", "success"), ("fAiLuRe", "failure")):
            self.assertEqual(schema.parse(answer(verdict=v))["verdict"], expected)
        # Each of these casefolds to "success".
        for v in ("ſuccess", "succeſs", "succesſ", "succeß", "SUCCEẞ"):
            r = schema.parse(answer(verdict=v))
            self.assertEqual((r["status"], r["verdict"], r["fields"]["verdict"]), ("abstain", None, None), v)

    def test_evidence_sufficient_ignores_ascii_case_only(self):
        for v, expected in (("yEs", "yes"), ("NO", "no")):
            r = schema.parse(answer(evidence_sufficient=v))
            self.assertEqual((r["fields"]["evidence_sufficient"], r["voided"]), (expected, {}), v)
        # Each of these casefolds to "yes".
        for v in ("yeſ", "YEſ"):
            r = schema.parse(answer(verdict="failure", evidence_sufficient=v))
            self.assertEqual((r["status"], r["verdict"]), ("parsed", "failure"), v)
            self.assertIsNone(r["fields"]["evidence_sufficient"], v)
            self.assertEqual(list(r["voided"]), ["evidence_sufficient"], v)

    def test_bad_covariates_void_only_that_field(self):
        cases = [("p_failure", 1.5), ("p_failure", -0.01), ("p_failure", True), ("p_failure", "0.3"),
                 ("evidence_sufficient", "maybe"), ("evidence_step_ids", [1, 2.0]),
                 ("evidence_step_ids", [True]), ("evidence_step_ids", "3"),
                 ("rationale", " ".join(["word"] * 101)), ("rationale", 7)]
        for field, value in cases:
            r = schema.parse(answer(verdict="failure", **{field: value}))
            self.assertEqual((r["status"], r["verdict"]), ("parsed", "failure"), (field, value))
            self.assertIsNone(r["fields"][field], (field, value))
            self.assertEqual(list(r["voided"]), [field], (field, value))
        for value in (0, 1, 0.0, 1.0):
            self.assertEqual(schema.parse(answer(p_failure=value))["fields"]["p_failure"], value)
        hundred = " ".join(["word"] * 100)
        self.assertEqual(schema.parse(answer(rationale=hundred))["fields"]["rationale"], hundred)

    def test_a_long_integer_voids_only_that_field(self):
        for digits in (19, 308, 309, 310, 640, 641, 4300, 4301, 5000, 100000):
            for sign in ("", "-"):
                literal = sign + "9" * digits
                cases = [("p_failure", literal), ("evidence_step_ids", f"[{literal}]"),
                         ("evidence_step_ids", f"[1, {literal}, 3]"),
                         ("evidence_sufficient", literal), ("rationale", literal)]
                for field, value in cases:
                    r = schema.parse(answer_with_literal(field, value, verdict="failure"))
                    self.assertEqual((r["status"], r["verdict"]), ("parsed", "failure"), (field, sign, digits))
                    self.assertIsNone(r["fields"][field], (field, sign, digits))
                    self.assertEqual(list(r["voided"]), [field], (field, sign, digits))

    def test_a_number_too_large_for_a_float_voids_p_failure(self):
        for literal in ("1e999", "-1e999", "9" * 5000 + ".5", "9" * 5000 + "e0"):
            r = schema.parse(answer_with_literal("p_failure", literal, verdict="failure"))
            self.assertEqual((r["status"], r["verdict"], r["fields"]["p_failure"]), ("parsed", "failure", None), literal[:9])
            self.assertEqual(list(r["voided"]), ["p_failure"], literal[:9])

    def test_integer_length_boundary(self):
        self.assertEqual(schema.INTEGER_MAX_DIGITS, 18)
        longest = "9" * 18
        r = schema.parse(answer_with_literal("evidence_step_ids", f"[{longest}, -{longest}]"))
        self.assertEqual((r["fields"]["evidence_step_ids"], r["voided"]), ([int(longest), -int(longest)], {}))
        for one_more in ("1" + "0" * 18, "-1" + "0" * 18):
            r = schema.parse(answer_with_literal("evidence_step_ids", f"[{one_more}]"))
            self.assertEqual((r["status"], list(r["voided"])), ("parsed", ["evidence_step_ids"]), one_more)

    def test_range_check_handles_an_integer_too_big_for_a_float(self):
        # The hook stops these first, so raise its cap to reach the check on its own.
        with mock.patch.object(schema, "INTEGER_MAX_DIGITS", 4300):
            for digits in (308, 309, 310, 4300):
                r = schema.parse(answer_with_literal("p_failure", "9" * digits, verdict="failure"))
                self.assertEqual((r["status"], r["verdict"], r["fields"]["p_failure"]), ("parsed", "failure", None), digits)

    def test_results_do_not_depend_on_the_interpreter_digit_limit(self):
        self.addCleanup(sys.set_int_max_str_digits, sys.get_int_max_str_digits())
        for limit in (640, 4300, 0):
            sys.set_int_max_str_digits(limit)
            for digits in (18, 19, 641, 4301):
                literal = "9" * digits
                r = schema.parse(answer_with_literal("evidence_step_ids", f"[{literal}]", verdict="failure"))
                self.assertEqual((r["status"], r["verdict"]), ("parsed", "failure"), (limit, digits))
                self.assertEqual(r["fields"]["evidence_step_ids"], [int(literal)] if digits == 18 else None, (limit, digits))

    def test_a_long_integer_does_not_rescue_a_malformed_answer(self):
        huge = "9" * 5000
        extra = answer()[:-1] + f', "confidence": {huge}}}'
        for text in (huge, f"[{huge}]", answer_with_literal("verdict", huge), answer() + huge, extra):
            r = schema.parse(text)
            self.assertEqual((r["status"], r["verdict"]), ("malformed", None), text[:40])

    def test_inadmissible_verdict_abstains(self):
        for v in ("unsure", "", " success", "success.", "partial"):
            r = schema.parse(answer(verdict=v))
            self.assertEqual((r["status"], r["verdict"]), ("abstain", None), v)

    def test_malformed(self):
        good = json.loads(answer())
        extra = dict(good, confidence=0.9)
        missing = {k: v for k, v in good.items() if k != "rationale"}
        cases = ["", "not json", "Verdict: success", answer() + " trailing words",
                 "```json\n" + answer() + "\n```", json.dumps([good]), json.dumps(extra), json.dumps(missing),
                 answer(verdict=None), answer(verdict=1),
                 answer().replace('"p_failure": 0.25', '"p_failure": NaN'),
                 '{"verdict": "success", "verdict": "failure", "p_failure": 0.1, "evidence_sufficient": "yes", '
                 '"evidence_step_ids": [], "rationale": "x"}', None, b"{}"]
        for text in cases:
            r = schema.parse(text)
            self.assertEqual((r["status"], r["verdict"]), ("malformed", None), text)
        self.assertEqual(schema.parse("  \n" + answer() + "\n ")["status"], "parsed")


def _filter_png(width, height, rows_rgb, filters):
    """A PNG whose rows use the given filter types (to test decoding)."""
    ch, out, prev = 3, b"", bytes(width * 3)
    for row, f in zip(rows_rgb, filters):
        line = bytearray()
        for i, x in enumerate(row):
            a = row[i - ch] if i >= ch else 0
            b, c = prev[i], prev[i - ch] if i >= ch else 0
            pred = {0: 0, 1: a, 2: b, 3: (a + b) >> 1, 4: I._paeth(a, b, c)}[f]
            line.append((x - pred) & 0xFF)
        out += bytes([f]) + bytes(line)
        prev = row

    def chunk(t, b):
        return struct.pack(">I", len(b)) + t + b + struct.pack(">I", zlib.crc32(t + b) & 0xFFFFFFFF)
    return (I.SIGNATURE + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(out)) + chunk(b"IEND", b""))


class TestImages(unittest.TestCase):
    def test_png_size(self):
        self.assertEqual(I.png_size(make_png(7, 5)), (7, 5))
        for bad in (b"", b"GIF89a" + b"\0" * 40, make_png(2, 2)[:20]):
            with self.assertRaises(ValueError):
                I.png_size(bad)

    def test_decode_every_filter(self):
        rows = [bytes((x * 37 + y * 11 + c * 5) % 256 for x in range(5) for c in range(3)) for y in range(5)]
        data = _filter_png(5, 5, rows, [0, 1, 2, 3, 4])
        self.assertEqual(I.decode(data), (5, 5, 3, b"".join(rows)))

    def test_target_size_never_enlarges_and_rounds(self):
        self.assertEqual(I.target_size(1280, 720, 1024), (1024, 576))
        self.assertEqual(I.target_size(720, 1280, 1024), (576, 1024))
        self.assertEqual(I.target_size(1024, 3000, 1024), (350, 1024))       # 349.53 rounds up
        self.assertEqual(I.target_size(800, 600, 1024), (800, 600))
        self.assertEqual(I.target_size(5000, 1, 1024), (1024, 1))

    def test_downscale_means(self):
        pixels = bytes([0, 0, 0, 100, 100, 100, 50, 50, 50, 250, 250, 250,
                        10, 10, 10, 30, 30, 30, 70, 70, 70, 90, 90, 90])     # 4 x 2 RGB
        self.assertEqual(I.downscale(4, 2, 3, pixels, 2, 1), bytes([35] * 3 + [115] * 3))

    def test_fit_long_edge_is_deterministic_and_valid(self):
        data = make_png(1500, 4, rgb=(200, 100, 50))
        small = I.fit_long_edge(data, 1024)
        self.assertEqual(I.png_size(small), (1024, 3))
        self.assertEqual(small, I.fit_long_edge(data, 1024))
        self.assertEqual(I.decode(small)[3], bytes([200, 100, 50]) * (1024 * 3))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.png"
            path.write_bytes(small)
            self.assertTrue(inspect_png(path)["ok"])
        fits = make_png(1024, 2)
        self.assertIs(I.fit_long_edge(fits, 1024), fits)

    def test_stored_zlib_is_plain_deflate(self):
        for n in (0, 1, 65535, 65536, 200000):
            data = bytes(i % 251 for i in range(n))
            self.assertEqual(zlib.decompress(I._stored_zlib(data)), data)

    def test_unsupported_png_is_refused(self):
        data = make_png(3, 3)
        ihdr = struct.pack(">IIBBBBB", 3, 3, 16, 2, 0, 0, 0)
        bad = data[:16] + ihdr + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr) & 0xFFFFFFFF) + data[33:]
        with self.assertRaises(ValueError):
            I.decode(bad)


class TestRequests(unittest.TestCase):
    def setUp(self):
        self.pack = synthetic_pack(3, shots=[True, False, True, True])
        self.images = images_for(self.pack)
        self.prep = R.prepare(self.pack, self.images)
        self.a, self.b = self.prep["requests"]["A"], self.prep["requests"]["B"]

    def test_every_field_in_the_table(self):
        a, b = body(self.a), body(self.b)
        schema_s = json.loads(R.canonical_bytes(schema.SCHEMA))
        self.assertEqual(set(a), {"model", "max_tokens", "stream", "thinking", "output_config", "system",
                                  "messages", "cache_control", "inference_geo", "service_tier"})
        self.assertEqual((a["model"], a["max_tokens"], a["stream"]), ("claude-sonnet-5", 32000, True))
        self.assertEqual(a["thinking"], {"type": "adaptive"})
        self.assertEqual(a["output_config"], {"effort": "high", "format": {"type": "json_schema", "schema": schema_s}})
        self.assertEqual(a["cache_control"], {"type": "ephemeral"})
        self.assertEqual((a["inference_geo"], a["service_tier"]), ("global", "standard_only"))
        self.assertEqual(a["system"], [{"type": "text", "text": rubric.RUBRIC_TEXT}])
        self.assertEqual([m["role"] for m in a["messages"]], ["user"])
        self.assertEqual(set(b), {"model", "input", "max_output_tokens", "stream", "reasoning", "text",
                                  "prompt_cache_options", "service_tier", "store"})
        self.assertEqual((b["model"], b["max_output_tokens"], b["stream"]), ("gpt-6-sol", 32000, True))
        self.assertEqual(b["reasoning"], {"effort": "medium", "mode": "standard"})
        self.assertEqual(b["text"], {"format": {"type": "json_schema", "name": "schema_s", "schema": schema_s,
                                                "strict": True}, "verbosity": "medium"})
        self.assertEqual(b["prompt_cache_options"], {"mode": "explicit"})
        self.assertEqual((b["service_tier"], b["store"]), ("default", False))
        self.assertEqual(b["input"][0], {"role": "developer",
                                         "content": [{"type": "input_text", "text": rubric.RUBRIC_TEXT}]})
        self.assertEqual([m["role"] for m in b["input"]], ["developer", "user"])
        self.assertEqual((self.a.url, self.b.url), ("https://api.anthropic.com/v1/messages",
                                                    "https://api.openai.com/v1/responses"))
        self.assertEqual(dict(self.a.headers), {"anthropic-version": "2026-10-21", "content-type": "application/json"})
        self.assertEqual(dict(self.b.headers), {"content-type": "application/json"})
        self.assertEqual(R.check_body("A", a), [])
        self.assertEqual(R.check_body("B", b), [])

    def test_no_sampling_parameters_tools_stop_sequences_or_chaining(self):
        never = {"temperature", "top_p", "top_k", "top_logprobs", "tools", "tool_choice", "stop_sequences",
                 "stop", "fallbacks", "previous_response_id", "prompt_cache_key", "prompt_cache_breakpoint",
                 "seed", "conversation", "ttl"}
        for req in (self.a, self.b):
            self.assertFalse(never & set(R._keys(body(req))), req.judge)
        bad = dict(body(self.a), temperature=1.0)
        self.assertTrue(R.check_body("A", bad))
        self.assertTrue(R.check_body("B", dict(body(self.b), top_p=1)))
        self.assertTrue(R.check_body("A", dict(body(self.a), cache_control={"type": "ephemeral", "ttl": "1h"})))
        self.assertTrue(R.check_body("B", dict(body(self.b), prompt_cache_options={"mode": "implicit"})))

    def test_prompt_development_drops_only_cache_control(self):
        dev = R.prepare(self.pack, self.images, caching=False)["requests"]
        a = body(dev["A"])
        self.assertNotIn("cache_control", a)
        self.assertEqual(dict(a, cache_control={"type": "ephemeral"}), body(self.a))
        self.assertEqual(dev["B"].body, self.b.body)
        self.assertEqual(R.check_body("A", a, caching=False), [])
        self.assertNotEqual(R.config_sha256(caching=False), R.config_sha256())

    def test_same_evidence_for_both_judges_with_images_interleaved(self):
        a = R.neutral_content("A", body(self.a))
        b = R.neutral_content("B", body(self.b))
        self.assertEqual(a, b)
        kinds = [k for k, _ in a]
        self.assertEqual(kinds, ["text", "text", "image", "text", "text", "image", "text", "image", "text"])
        self.assertTrue(a[1][1].startswith("Step 0:\n"))
        self.assertEqual(base64.b64decode(a[2][1]), self.images["image-0"])
        self.assertTrue(a[3][1].startswith("Step 1:\n"))                      # step 1 has no screenshot
        self.assertEqual(base64.b64decode(a[5][1]), self.images["image-2"])
        self.assertTrue(a[-1][1].startswith("Final page:\n"))
        blocks = body(self.b)["input"][1]["content"]
        self.assertTrue(all(x["detail"] == "original" for x in blocks if x["type"] == "input_image"))
        self.assertTrue(all(x["source"]["media_type"] == "image/png"
                            for x in body(self.a)["messages"][0]["content"] if x["type"] == "image"))

    def test_text_parts_rebuild_the_pack_exactly(self):
        texts = [v for k, v in R.neutral_content("A", body(self.a)) if k == "text"]
        self.assertEqual(P.serialize(R.rebuild_pack(texts)), P.serialize(self.pack))

    def test_canonical_bytes_and_request_hash(self):
        for req in (self.a, self.b):
            self.assertEqual(R.canonical_bytes(json.loads(req.body)), req.body)
            self.assertEqual(req.sha256, hashlib.sha256(req.body).hexdigest())
        top = list(body(self.a))
        self.assertEqual(top, sorted(top))
        again = R.prepare(json.loads(P.serialize(self.pack)), dict(reversed(list(self.images.items()))))
        self.assertEqual((again["requests"]["A"].body, again["requests"]["B"].body), (self.a.body, self.b.body))
        with self.assertRaises(ValueError):
            R.canonical_bytes({"x": float("inf")})

    def test_config_hash_covers_models_fields_and_versions(self):
        rec = R.config_record()
        self.assertEqual(rec["rubric"], {"version": "rubric-v0", "status": "draft", "sha256": rubric.RUBRIC_SHA256})
        self.assertEqual(rec["schema"]["sha256"], schema.schema_sha256())
        self.assertEqual((rec["parser_version"], rec["pack_version"], rec["output_cap_tokens"]),
                         (schema.PARSER_VERSION, P.PACK_VERSION, 32000))
        self.assertEqual(rec["judges"]["A"]["headers"]["anthropic-version"], "2026-10-21")
        self.assertEqual(rec["judges"]["A"]["body_without_evidence"], R.body_a([]))
        self.assertEqual(rec["judges"]["A"]["body_without_evidence"]["messages"], [{"role": "user", "content": []}])
        self.assertEqual(rec["judges"]["B"]["body_without_evidence"], R.body_b([]))
        self.assertEqual(R.config_sha256(), hashlib.sha256(R.canonical_bytes(rec)).hexdigest())
        self.assertNotEqual(R.config_sha256(b_model="gpt-6.1-sol"), R.config_sha256())
        base = R.config_sha256()
        for patch in ({"OUTPUT_CAP": 16000}, {"FALLBACK_LONG_EDGE": 768}, {"MAX_RETRIES": 2},
                      {"RUBRIC_TEXT": rubric.RUBRIC_TEXT + " "}, {"PACK_KEYS": frozenset()},
                      {"PARSER_VERSION": "parser-v0"},
                      {"A_HEADERS": (("anthropic-version", "2023-06-01"), ("content-type", "application/json"))}):
            with mock.patch.multiple(R, **patch):
                if "PACK_KEYS" in patch:
                    self.assertEqual(R.config_sha256(), base)          # not a request setting
                else:
                    self.assertNotEqual(R.config_sha256(), base, patch)
        with mock.patch.dict(R.B_SETTINGS, {"store": True}):
            self.assertNotEqual(R.config_sha256(), base)

    def test_screenshots_must_match_the_references(self):
        extra = dict(self.images, **{"image-9": make_png(2, 2)})
        missing = {k: v for k, v in self.images.items() if k != "image-2"}
        for imgs in (extra, missing):
            with self.assertRaises(ValueError):
                R.prepare(self.pack, imgs)

    def test_only_packs_of_this_version_with_known_keys(self):
        for bad in (dict(self.pack, trace_key="t0123456789abcdef"), dict(self.pack, pack_version="pack-0.0"),
                    {k: v for k, v in self.pack.items() if k != "field_notes"}):
            with self.assertRaises(ValueError):
                R.evidence_parts(bad)

    def test_redacted_example_keeps_length_and_hash_and_restores(self):
        for j, req in (("A", self.a), ("B", self.b)):
            red = R.redacted(j, body(req))
            text = json.dumps(red)
            b64 = [v for k, v in R.neutral_content(j, body(req)) if k == "image"]
            for d in b64:
                self.assertNotIn(d, text)
                self.assertIn(R.IMAGE_PLACEHOLDER.format(n=len(d), sha=hashlib.sha256(d.encode()).hexdigest()), text)
            self.assertEqual(R.restore(j, red, b64), req.body)

    def test_token_estimate(self):
        est = R.token_estimate("A", self.a, self.prep["image_sizes"])
        self.assertEqual(est["image_tokens"], 3 * math.ceil(4 / 28) * math.ceil(3 / 28))
        self.assertEqual(est["input_tokens"], est["text_tokens"] + est["image_tokens"])
        self.assertGreater(est["text_chars"], len(rubric.RUBRIC_TEXT))


class TestFit(unittest.TestCase):
    def pack_and_images(self, n_images, size):
        pack = synthetic_pack(n_images - 1)
        return pack, {s["screenshot"]: make_png(*size) for s in pack["steps"]}

    def test_large_images_fit_with_twenty_or_fewer(self):
        pack, imgs = self.pack_and_images(20, (2500, 2))
        prep = R.prepare(pack, imgs)
        self.assertTrue(prep["eligible"])
        self.assertFalse(prep["fallback"]["applied"])
        self.assertEqual(prep["image_sizes"], [(2500, 2)] * 20)

    def test_many_images_over_2000_px_trigger_the_shared_fallback(self):
        pack, imgs = self.pack_and_images(21, (2001, 2))
        prep = R.prepare(pack, imgs)
        self.assertTrue(prep["eligible"])
        self.assertTrue(prep["fallback"]["applied"])
        self.assertIn("A: 21 of 21 images exceed 2000 px per side", prep["fallback"]["triggered_by"])
        self.assertEqual(prep["image_sizes"], [(1024, 1)] * 21)
        a = R.neutral_content("A", body(prep["requests"]["A"]))
        b = R.neutral_content("B", body(prep["requests"]["B"]))
        self.assertEqual(a, b)                                    # the same downscaled bytes for both judges
        sent = [base64.b64decode(v) for k, v in a if k == "image"]
        self.assertEqual(sent, [I.fit_long_edge(imgs[f"image-{i}"], 1024) for i in range(21)])

    def test_fallback_downscales_every_screenshot_but_never_enlarges(self):
        pack = synthetic_pack(20)
        imgs = {s["screenshot"]: make_png(*((2001, 2) if s["step"] == 0 else (600, 2))) for s in pack["steps"]}
        prep = R.prepare(pack, imgs)
        self.assertTrue(prep["fallback"]["applied"])
        self.assertEqual(prep["image_sizes"], [(1024, 1)] + [(600, 2)] * 20)
        sent = [base64.b64decode(v) for k, v in R.neutral_content("B", body(prep["requests"]["B"])) if k == "image"]
        self.assertEqual(sent[1:], [imgs[f"image-{i}"] for i in range(1, 21)])

    def test_a_request_that_still_does_not_fit_is_ineligible(self):
        pack, imgs = self.pack_and_images(601, (4, 3))
        prep = R.prepare(pack, imgs)
        self.assertFalse(prep["eligible"])
        self.assertIsNone(prep["requests"])
        self.assertTrue(prep["fallback"]["applied"])
        self.assertIn("A: 601 images, above 600", prep["reasons"])
        pack, imgs = self.pack_and_images(2, (4, 3))
        with mock.patch.dict(R.B_LIMITS, {"max_body_bytes": 100}):
            prep = R.prepare(pack, imgs)
        self.assertFalse(prep["eligible"])
        self.assertTrue(any(r.startswith("B: body is") for r in prep["reasons"]))

    def test_a_screenshot_the_fallback_cannot_read_makes_the_pack_ineligible(self):
        pack, imgs = self.pack_and_images(21, (2001, 2))
        data = imgs["image-0"]
        ihdr = struct.pack(">IIBBBBB", 2001, 2, 8, 2, 0, 0, 1)        # interlaced
        imgs["image-0"] = data[:16] + ihdr + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr) & 0xFFFFFFFF) + data[33:]
        prep = R.prepare(pack, imgs)
        self.assertFalse(prep["eligible"])
        self.assertIn("unsupported PNG", prep["fallback"]["error"])
        self.assertTrue(any("cannot downscale" in r for r in prep["reasons"]))

    def test_provider_limits(self):
        self.assertEqual(R._image_problems("A", [(8000, 8000)] * 20, [1] * 20), [])
        self.assertTrue(R._image_problems("A", [(8001, 10)], [1]))
        self.assertTrue(R._image_problems("A", [(10, 10)], [10 * 10**6 + 1]))
        self.assertEqual(R._image_problems("B", [(5536, 5536)], [1]), [])         # 173 x 173 = 29,929 patches
        self.assertTrue(R._image_problems("B", [(5600, 5600)], [1]))              # 175 x 175 = 30,625
        self.assertTrue(R._image_problems("B", [(1, 1)] * 1501, [1] * 1501))
        self.assertEqual(R._image_problems("A", [(2000, 2000)] * 21, [1] * 21), [])


class TestRequestLeakage(FixtureCase):
    def build(self, tid):
        view, prov = self.norm(tid)
        pack, data, digest, manifest = P.build(view, prov)
        images = R.read_images(manifest, self.tmp)
        prep = R.prepare(pack, images)
        private = sorted(set(P.private_strings(prov)) | set(R.manifest_strings(manifest))
                         | {view["trace_key"], digest})
        return prov, manifest, prep, private

    def test_no_label_reward_identifier_path_or_hash_reaches_a_request(self):
        for tid in ("webarena.1", "visualwebarena.resized.3"):
            prov, manifest, prep, private = self.build(tid)
            secrets = [CANARY_REWARD, CANARY_CHAT, CANARY_TEST_LABEL, AGENT, MODEL, prov["trace_id"],
                       prov["task_id"], prov["task_id_normalized"], prov["trace_key"], prov["file"]["path"],
                       prov["file"]["sha256"], "summary_info", "chat_messages", "cum_reward", "trajectory_success",
                       "4. Completely Optimal", "2. Suboptimal", "Unsuccessful", "annotator_name",
                       "annotations.csv", "screenshots/", "-xl-2", manifest[0]["pack_sha256"]]
            secrets += [f["path"] for f in prov["screenshot_files"]] + [f["sha256"] for f in prov["screenshot_files"]]
            for j in "AB":
                req = prep["requests"][j]
                text = req.body.decode()
                for s in secrets:
                    self.assertNotIn(s, text, f"{tid} {j}: {s}")
                self.assertEqual(R.check_request(j, req.body, private),
                                 {"ok": True, "private_keys": [], "forbidden_substrings": [], "private_strings": []})
            self.assertEqual(len(R.neutral_content("A", body(prep["requests"]["A"]))),
                             1 + len(prov["screenshot_files"]) * 2 + 1)

    def test_check_request_catches_injections(self):
        prov, manifest, prep, private = self.build("webarena.1")
        a = body(prep["requests"]["A"])
        cases = [
            lambda b: b["messages"][0]["content"].append({"type": "text", "text": f"run {prov['trace_key']}"}),
            lambda b: b["messages"][0]["content"].append({"type": "text", "text": manifest[0]["path"]}),
            lambda b: b["messages"][0]["content"].append({"type": "text", "text": manifest[0]["sha256"]}),
            lambda b: b["messages"][0]["content"].append({"type": "text", "text": f"agent {AGENT}"}),
            lambda b: b["messages"][0]["content"].append({"type": "text", "text": "cum_reward=1"}),
            lambda b: b["messages"][0]["content"][0].update(sha256=manifest[0]["sha256"]),
            lambda b: b["messages"][0]["content"][0].update(label="Successful"),
            lambda b: b.update(metadata={"trace_key": prov["trace_key"]}),
        ]
        for inject in cases:
            bad = json.loads(json.dumps(a))
            inject(bad)
            self.assertFalse(R.check_request("A", R.canonical_bytes(bad), private)["ok"])

    def test_read_images_checks_every_file_against_the_manifest(self):
        view, prov = self.norm("webarena.1")
        _, _, _, manifest = P.build(view, prov)
        self.assertEqual(set(R.read_images(manifest, self.tmp)), {e["ref"] for e in manifest})
        for change in ({"sha256": "0" * 64}, {"bytes": manifest[0]["bytes"] + 1}):
            bad = [dict(manifest[0], **change)] + manifest[1:]
            with self.assertRaises(ValueError):
                R.read_images(bad, self.tmp)
        with self.assertRaises(ValueError):
            R.read_images(manifest + manifest[:1], self.tmp)

    def test_request_modules_never_import_the_label_store(self):
        import ast
        for rel in ("judge/rubric.py", "judge/schema.py", "judge/images.py", "judge/requests.py",
                    "judge/streams.py", "judge/outcomes.py", "judge/storage.py", "judge/dispatch.py",
                    "scripts/dry_run.py"):
            names = set()
            for node in ast.walk(ast.parse((REPO / rel).read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom):
                    names.add(node.module or "")
                    names.update(a.name for a in node.names)
                elif isinstance(node, ast.Import):
                    names.update(a.name for a in node.names)
            self.assertFalse({"labels", "datapath.labels", "agent_reward_bench"} & names, rel)


if __name__ == "__main__":
    unittest.main()
