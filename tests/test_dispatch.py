"""Tests for streams, outcomes, the write-once log and the serial dispatcher.

Every response here is a synthetic, provider-shaped stream or error body served by
FakeTransport. No test builds the HTTPS transport's connection or sends anything.
"""

import io
import json
import os
import shutil
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from judge import dispatch as D  # noqa: E402
from judge import requests as R  # noqa: E402
from judge.outcomes import (ABSTAIN, CLASSIFICATION_ERROR, FILTERED, HALT, MALFORMED, REFUSAL,  # noqa: E402
                            TRANSPORT, TRUNCATED, VALID, Exchange, canonical_outcome)
from judge.storage import AttemptLog, WriteOnceError, write_once  # noqa: E402
from judge.streams import (anthropic_events, anthropic_summary, openai_events,  # noqa: E402
                           openai_summary, parse_sse, sse)
from test_requests import answer, answer_with_literal, images_for, synthetic_pack  # noqa: E402

KEY = "t0123456789abcdef"
WS = "wrkspc_test"
CANARY_ANSWER = "CANARY_A1_ANSWER_4e1"


class Fakes:
    """Synthetic exchanges with unique response ids."""

    def __init__(self):
        self.n = 0

    def _id(self, p):
        self.n += 1
        return f"{p}_{self.n}"

    def a(self, text=None, stop="end_turn", model=R.A_MODEL, usage=None, headers=None, **kw):
        text = answer() if text is None and stop == "end_turn" else text
        h = {"anthropic-workspace-id": WS} if headers is None else headers
        return Exchange(200, h, sse(anthropic_events(text, stop_reason=stop, model=model, msg_id=self._id("msg"),
                                                     usage=usage, **kw)), None, None)

    def b(self, text=None, status="completed", headers=None, **kw):
        if text is None and status == "completed" and kw.get("refusal") is None:
            text = answer()
        return Exchange(200, headers or {}, sse(openai_events(text, status=status, resp_id=self._id("resp"), **kw)),
                        None, None)


def a_err(status, etype, message="x", code=None, headers=None):
    err = {"type": etype, "message": message}
    if code:
        err["details"] = {"error_code": code}
    return Exchange(status, headers or {}, json.dumps({"type": "error", "error": err}).encode(), None, None)


def b_err(status, code, etype="invalid_request_error", message="x"):
    return Exchange(status, {}, json.dumps({"error": {"message": message, "type": etype, "param": None,
                                                      "code": code}}).encode(), None, None)


NO_RESPONSE = Exchange(None, {}, b"", "TimeoutError: timed out", None)


class TestSSE(unittest.TestCase):
    def test_parse_rules(self):
        raw = (b": comment\r\nevent: one\r\ndata: {\"a\":\r\ndata:1}\r\n\r\n"
               b"data: plain\rdata:  two spaces\r\r"
               b"event: open\ndata: never finished\n")
        self.assertEqual(parse_sse(raw), [("one", '{"a":\n1}'), ("message", "plain\n two spaces")])
        self.assertEqual(parse_sse(b"event: x\ndata: y\n\n\n\n"), [("x", "y")])
        self.assertEqual(parse_sse(b"event: x\n\n"), [])

    def test_anthropic_summary(self):
        text = answer(rationale="Uses usage counts.")
        s = anthropic_summary(sse(anthropic_events(text, msg_id="msg_1", usage={"cache_read_input_tokens": 900})))
        self.assertTrue(s["complete"])
        self.assertEqual((s["id"], s["model"], s["stop_reason"], s["text"]), ("msg_1", "claude-sonnet-5", "end_turn", text))
        self.assertEqual((s["usage"]["cache_read_tokens"], s["usage"]["output_tokens"],
                          s["usage"]["cache_write_1h_tokens"], s["usage"]["service_tier"],
                          s["usage"]["inference_geo"]), (900, 250, 0, "standard", "global"))
        self.assertTrue(s["usage_final"])
        events = anthropic_events(text)
        for _, obj in events:
            if obj.get("type") == "message_delta":
                del obj["usage"]
        s = anthropic_summary(sse(events))
        self.assertEqual((s["complete"], s["usage_final"], s["usage"]["output_tokens"]), (True, False, 1))

    def test_a_cut_stream_is_incomplete_and_its_text_is_never_read(self):
        raw = sse(anthropic_events(answer()))
        for cut in (len(raw) - 1, len(raw) // 2, raw.index(b"message_delta") - 7):
            s = anthropic_summary(raw[:cut])
            self.assertFalse(s["complete"], cut)
            self.assertIsNone(s["text"], cut)

    def test_openai_summary(self):
        text = answer()
        s = openai_summary(sse(openai_events(text, resp_id="resp_1")))
        self.assertTrue(s["complete"])
        self.assertEqual((s["id"], s["model"], s["status"], s["service_tier"], s["text"], s["refusal"]),
                         ("resp_1", "gpt-6-sol", "completed", "default", text, None))
        self.assertEqual((s["usage"]["cache_read_tokens"], s["usage"]["cache_write_tokens"]), (0, 0))
        raw = sse(openai_events(text))
        cut = openai_summary(raw[:raw.index(b"event: response.completed")])
        self.assertFalse(cut["complete"])
        self.assertIsNone(cut["text"])


class TestOutcomes(unittest.TestCase):
    def setUp(self):
        self.f = Fakes()

    def a(self, ex):
        return canonical_outcome("anthropic", ex)

    def b(self, ex):
        return canonical_outcome("openai", ex)

    def test_anthropic_terminal_outcomes(self):
        cases = [(self.f.a(), VALID), (self.f.a(text="", stop="refusal"), REFUSAL),
                 (self.f.a(stop="max_tokens"), TRUNCATED), (self.f.a(stop="model_context_window_exceeded"), TRUNCATED),
                 (self.f.a(text='{"verdict": "success"}'), MALFORMED), (self.f.a(text="success"), MALFORMED),
                 (self.f.a(text=answer(verdict="unclear")), ABSTAIN),
                 (self.f.a(stop="stop_sequence"), MALFORMED), (self.f.a(stop="pause_turn"), MALFORMED)]
        for ex, expected in cases:
            oc = self.a(ex)
            self.assertEqual(oc["outcome"], expected, oc["detail"])
            self.assertFalse(oc["retryable"])
        oc = self.a(self.f.a(text=answer(verdict="Failure", p_failure=3)))
        self.assertEqual((oc["outcome"], oc["parse"]["verdict"], oc["parse"]["fields"]["p_failure"]),
                         (VALID, "failure", None))

    def test_a_long_integer_in_a_covariate_keeps_the_verdict(self):
        for digits in (400, 5000):
            for field, value in (("p_failure", "9" * digits), ("evidence_step_ids", "[" + "9" * digits + "]")):
                text = answer_with_literal(field, value, verdict="Failure")
                for oc in (self.a(self.f.a(text=text)), self.b(self.f.b(text=text))):
                    self.assertEqual((oc["outcome"], oc["parse"]["verdict"], oc["parse"]["fields"][field]),
                                     (VALID, "failure", None), (field, digits))

    def test_openai_terminal_outcomes(self):
        cases = [(self.f.b(), VALID), (self.f.b(text=None, refusal="I can't help."), REFUSAL),
                 (self.f.b(text=answer(), refusal="partly refused"), REFUSAL),
                 (self.f.b(status="incomplete", incomplete_reason="max_output_tokens"), TRUNCATED),
                 (self.f.b(status="incomplete", incomplete_reason="content_filter"), FILTERED),
                 (self.f.b(status="failed", error={"code": "invalid_prompt", "message": "flagged"}), FILTERED),
                 (self.f.b(status="failed", error={"code": "bio_policy", "message": "x"}), FILTERED),
                 (b_err(400, "invalid_prompt"), FILTERED),
                 (self.f.b(text="not json"), MALFORMED), (self.f.b(text=answer(verdict="maybe")), ABSTAIN)]
        for ex, expected in cases:
            oc = self.b(ex)
            self.assertEqual(oc["outcome"], expected, oc["detail"])
            self.assertFalse(oc["retryable"])

    def test_request_and_configuration_errors_halt(self):
        for ex in (a_err(400, "invalid_request_error", "messages: bad"), a_err(413, "request_too_large"),
                   a_err(422, "invalid_request_error")):
            oc = self.a(ex)
            self.assertEqual((oc["outcome"], oc["halt_kind"], oc["retryable"]), (HALT, "request_error", False))
        for ex in (b_err(400, "unsupported_parameter"), b_err(413, None, etype="request_too_large"),
                   b_err(400, None), b_err(422, None, etype="invalid_request_error")):
            oc = self.b(ex)
            self.assertEqual((oc["outcome"], oc["halt_kind"]), (HALT, "request_error"), ex)
        oc = self.b(self.f.b(status="failed", error={"code": "invalid_image", "message": "bad image"}))
        self.assertEqual((oc["outcome"], oc["halt_kind"]), (HALT, "request_error"))

    def test_access_billing_credit_and_spend_errors_halt(self):
        cases = [(a_err(401, "authentication_error"), "auth"), (a_err(402, "billing_error"), "billing"),
                 (a_err(403, "permission_error"), "permission"), (a_err(404, "not_found_error"), "not_found"),
                 (a_err(400, "invalid_request_error", "You have reached your specified API usage limits. "
                        "You will regain access on 2026-12-01."), "spend_limit"),
                 (a_err(400, "invalid_request_error", "You have reached your specified workspace API usage limits."),
                  "spend_limit"),
                 (a_err(429, "rate_limit_error", "Monthly cap reached.", code="enforced_spend_limit_reached"),
                  "spend_limit"),
                 (a_err(429, "rate_limit_error", "You have reached your API usage limits: tier cap."), "spend_limit")]
        for ex, kind in cases:
            oc = self.a(ex)
            self.assertEqual((oc["outcome"], oc["halt_kind"]), (HALT, kind), ex)
        for code, kind in (("credit_balance_exhausted", "billing"), ("organization_spend_limit_exceeded", "spend_limit"),
                           ("project_spend_limit_exceeded", "spend_limit"),
                           ("organization_usage_limit_exceeded", "spend_limit")):
            oc = self.b(b_err(429, code, etype="insufficient_quota"))
            self.assertEqual((oc["outcome"], oc["halt_kind"]), (HALT, kind), code)
        for status, kind in ((401, "auth"), (402, "billing"), (403, "permission"), (404, "not_found")):
            self.assertEqual(self.b(b_err(status, None, etype="x"))["halt_kind"], kind)

    def test_transport_failures_are_retryable(self):
        raw = sse(anthropic_events(answer()))
        stream_err = sse(anthropic_events(None)[:2] + [("error", {"type": "error", "error": {
            "type": "overloaded_error", "message": "Overloaded"}})])
        for ex in (NO_RESPONSE, a_err(408, "timeout_error"), a_err(429, "rate_limit_error", "slow down"),
                   a_err(500, "api_error"), a_err(504, "timeout_error"), a_err(529, "overloaded_error"),
                   Exchange(200, {}, raw[:len(raw) // 2], "IncompleteRead", None), Exchange(200, {}, b"", None, None),
                   Exchange(200, {}, stream_err, None, None), a_err(409, "conflict_error")):
            oc = self.a(ex)
            self.assertEqual((oc["outcome"], oc["retryable"]), (TRANSPORT, True), oc["detail"])
            self.assertIsNone(oc["parse"])
        braw = sse(openai_events(answer()))
        for ex in (NO_RESPONSE, b_err(429, "rate_limit_exceeded", etype="requests"), b_err(500, None, etype="server_error"),
                   b_err(503, "server_is_overloaded", etype="service_unavailable_error"),
                   Exchange(200, {}, braw[:braw.index(b"event: response.completed")], None, None),
                   Exchange(200, {}, sse([("error", {"type": "error", "code": "server_error", "message": "x"})]), None, None),
                   self.f.b(status="failed", error={"code": "server_error", "message": "x"})):
            oc = self.b(ex)
            self.assertEqual((oc["outcome"], oc["retryable"]), (TRANSPORT, True), oc["detail"])
            self.assertIsNone(oc["parse"])

    def test_a_cut_stream_holding_a_whole_answer_is_still_transport(self):
        raw = sse(anthropic_events(answer()))
        cut = raw[:raw.index(b"event: message_stop")]
        oc = self.a(Exchange(200, {}, cut, "connection reset", None))
        self.assertEqual((oc["outcome"], oc["parse"], oc["summary"]["text"]), (TRANSPORT, None, None))
        stopless = raw.replace(b'"stop_reason": "end_turn"', b'"stop_reason": null')
        self.assertEqual(self.a(Exchange(200, {}, stopless, None, None))["outcome"], TRANSPORT)


class LogCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.log = AttemptLog(self.tmp / "log")
        self.f = Fakes()
        pack = synthetic_pack(2)
        prep = R.prepare(pack, images_for(pack))
        self.reqs = prep["requests"]
        self.config = R.config_sha256()
        self.sleeps = []
        self.t = [1000.0]

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def clock(self):
        self.t[0] += 1.5
        return self.t[0]

    def run_slots(self, *exchanges, workspace=WS, reqs=None, key=KEY, project=None):
        transport = D.FakeTransport(exchanges)
        result = D.dispatch_trajectory(key, reqs or self.reqs, self.config, transport, self.log, workspace,
                                       sleep=self.sleeps.append, clock=self.clock, project=project)
        return result, transport

    def records(self, slot, key=KEY):
        return self.log.attempts(key, self.config, slot[0], int(slot[1]))


class TestStorage(LogCase):
    def record(self, attempt=1, judge="A", repeat=1):
        return {"trace_key": KEY, "config_sha256": self.config, "judge": judge, "repeat": repeat,
                "attempt": attempt, "outcome": VALID}

    def test_write_once_refuses_to_overwrite(self):
        path = self.tmp / "x" / "f.json"
        write_once(path, b"one")
        with self.assertRaises(WriteOnceError):
            write_once(path, b"two")
        self.assertEqual(path.read_bytes(), b"one")
        self.assertEqual(sorted(p.name for p in path.parent.iterdir()), ["f.json"])

    def test_a_record_holding_a_lone_surrogate_is_written_and_reads_back(self):
        self.log.write_attempt(dict(self.record(1), detail="caf\u00e9 \ud83d end"), b"raw")
        self.assertEqual(self.log.attempts(KEY, self.config, "A", 1)[0]["detail"], "caf\u00e9 \ud83d end")
        data = (self.log.slot_dir(KEY, self.config, "A", 1) / "1.json").read_bytes()
        self.assertIn("caf\u00e9".encode("utf-8"), data)                    # other text stays as it was

    def test_attempts_are_write_once_and_contiguous(self):
        self.log.write_attempt(self.record(1), b"raw one")
        with self.assertRaises(ValueError):
            self.log.write_attempt(self.record(1), b"again")
        with self.assertRaises(ValueError):
            self.log.write_attempt(self.record(3), b"gap")
        self.log.write_attempt(self.record(2), b"raw two")
        recs = self.log.attempts(KEY, self.config, "A", 1)
        self.assertEqual([r["attempt"] for r in recs], [1, 2])
        self.assertEqual(recs[0]["raw_bytes"], 7)
        folder = self.log.slot_dir(KEY, self.config, "A", 1)
        self.assertEqual((folder / "1.raw").read_bytes(), b"raw one")
        with self.assertRaises(WriteOnceError):
            write_once(folder / "2.json", b"{}")
        (folder / "2.json").rename(folder / "5.json")
        with self.assertRaises(ValueError):
            self.log.attempts(KEY, self.config, "A", 1)

    def test_raw_bytes_come_first_and_keep_the_next_index(self):
        path = self.log.write_raw(KEY, self.config, "A", 1, 1, b"raw one")
        self.assertEqual(path.read_bytes(), b"raw one")
        self.assertEqual(self.log.attempts(KEY, self.config, "A", 1), [])
        with self.assertRaises(WriteOnceError):
            self.log.write_raw(KEY, self.config, "A", 1, 1, b"again")
        with self.assertRaises(ValueError):
            self.log.write_raw(KEY, self.config, "A", 1, 2, b"gap")
        self.log.complete_attempt(self.record(1))
        self.assertEqual(self.log.attempts(KEY, self.config, "A", 1)[0]["raw_bytes"], 7)
        self.assertEqual(self.log.write_raw(KEY, self.config, "A", 1, 2, None).read_bytes(), b"")

    def test_unsafe_names_are_refused(self):
        for key in ("../x", ".hidden", "a/b", ""):
            with self.assertRaises(ValueError):
                self.log.slot_dir(key, self.config, "A", 1)
        with self.assertRaises(ValueError):
            self.log.slot_dir(KEY, self.config, "C", 1)

    def test_halts_need_a_written_correction(self):
        hid = self.log.write_halt({"slot": "A1", "halt_kind": "request_error"})
        self.assertEqual([h["halt_id"] for h in self.log.unresolved_halts()], [hid])
        for note in ("", "   ", None):
            with self.assertRaises(ValueError):
                self.log.record_correction(hid, note, self.config, at="now")
        with self.assertRaises(ValueError):
            self.log.record_correction("halt-0099", "fixed", self.config, at="now")
        self.log.record_correction(hid, "Removed the bad field.", self.config, at="now")
        self.assertEqual(self.log.unresolved_halts(), [])
        with self.assertRaises(WriteOnceError):
            self.log.record_correction(hid, "a second note", self.config, at="now")

    def test_response_ids_and_first_model_are_write_once(self):
        self.assertTrue(self.log.claim_response_id("anthropic", "msg_1", "ref"))
        self.assertFalse(self.log.claim_response_id("anthropic", "msg_1", "ref2"))
        self.assertTrue(self.log.claim_response_id("openai", "msg_1", "ref3"))
        self.log.set_first_model(self.config, "A", "claude-sonnet-5", "ref")
        self.assertEqual(self.log.first_model(self.config, "A"), "claude-sonnet-5")
        with self.assertRaises(WriteOnceError):
            self.log.set_first_model(self.config, "A", "other", "ref")


class TestDispatch(LogCase):
    def test_serial_order_and_byte_identical_repeats(self):
        result, t = self.run_slots(self.f.a(), self.f.a(), self.f.a(), self.f.b())
        self.assertFalse(result["halted"])
        self.assertEqual([r.judge for r in t.sent], ["A", "A", "A", "B"])
        self.assertEqual(t.sent[0].body, t.sent[1].body)
        self.assertEqual(t.sent[1].body, t.sent[2].body)
        self.assertIs(t.sent[0], self.reqs["A"])
        self.assertEqual(t.sent[3].body, self.reqs["B"].body)
        self.assertEqual({s: r["outcome"] for s, r in result["slots"].items()},
                         {"A1": VALID, "A2": VALID, "A3": VALID, "B1": VALID})
        recs = [self.records(s)[0] for s in ("A1", "A2", "A3", "B1")]
        self.assertEqual({r["request_sha256"] for r in recs[:3]}, {self.reqs["A"].sha256})
        self.assertEqual([r["seconds_since_previous_a_start"] for r in recs], [None, 3.0, 3.0, None])
        self.assertEqual([r["within_cache_lifetime"] for r in recs], [None, True, True, None])

    def test_only_all_four_slots_or_a1_and_b1_are_accepted(self):
        transport = D.FakeTransport([self.f.a(), self.f.b()])
        result = D.dispatch_trajectory(KEY, self.reqs, self.config, transport, self.log, WS, sleep=self.sleeps.append,
                                       clock=self.clock, slots=(("A", 1), ("B", 1)))
        self.assertEqual([r.judge for r in transport.sent], ["A", "B"])
        self.assertEqual(sorted(result["slots"]), ["A1", "B1"])
        self.assertEqual(self.records("A2"), [])
        self.assertEqual(D.PROMPT_SLOTS, (("A", 1), ("B", 1)))
        for bad in ((), (("A", 1),), (("A", 1), ("A", 2)), (("A", 1), ("A", 2), ("A", 3)),
                    (("A", 1), ("A", 2), ("B", 1)), (("A", 1), ("A", 3), ("B", 1)), (("B", 1),),
                    (("A", 2), ("A", 1)), (("A", 1), ("A", 1)), (("A", 1), ("A", 4)),
                    (("A", 1), ("B", 1), ("A", 2))):
            transport = D.FakeTransport([self.f.a()])
            with self.assertRaises(ValueError):
                D.dispatch_trajectory("t2222222222222222", self.reqs, self.config, transport, self.log, WS,
                                      slots=bad)
            self.assertEqual(transport.sent, [])

    def test_no_answer_reaches_a_later_call(self):
        first = self.f.a(text=answer(rationale=CANARY_ANSWER))
        _, t = self.run_slots(first, self.f.a(), self.f.a(), self.f.b())
        for req in t.sent[1:]:
            self.assertNotIn(CANARY_ANSWER.encode(), req.body)
        self.assertEqual(self.records("A1")[0]["parse"]["fields"]["rationale"], CANARY_ANSWER)

    def test_repeats_run_whatever_a1_returned(self):
        firsts = [[self.f.a(text=answer(verdict="success"))], [self.f.a(text=answer(verdict="failure"))],
                  [self.f.a(text="", stop="refusal")], [self.f.a(text="garbage")],
                  [self.f.a(text=answer(verdict="unsure"))], [self.f.a(stop="max_tokens")],
                  [NO_RESPONSE] * 4]
        for i, first in enumerate(firsts):
            self.log = AttemptLog(self.tmp / f"log{i}")
            result, t = self.run_slots(*first, self.f.a(), self.f.a(), self.f.b())
            self.assertEqual([r.judge for r in t.sent], ["A"] * (len(first) + 2) + ["B"], i)
            self.assertEqual(set(result["slots"]), {"A1", "A2", "A3", "B1"})

    def test_transport_retries_same_bytes_same_slot(self):
        result, t = self.run_slots(NO_RESPONSE, a_err(529, "overloaded_error"), self.f.a(), self.f.a(), self.f.a(),
                                   self.f.b())
        recs = self.records("A1")
        self.assertEqual([r["outcome"] for r in recs], [TRANSPORT, TRANSPORT, VALID])
        self.assertEqual([r["attempt"] for r in recs], [1, 2, 3])
        self.assertEqual({r.body for r in t.sent[:3]}, {self.reqs["A"].body})
        self.assertEqual(self.sleeps, [2.0, 4.0])
        self.assertEqual(result["slots"]["A1"]["transport_attempts"], 2)

    def test_at_most_three_retries_then_invalid(self):
        result, t = self.run_slots(NO_RESPONSE, NO_RESPONSE, NO_RESPONSE, NO_RESPONSE, self.f.a(), self.f.a(),
                                   self.f.b())
        self.assertEqual([r["outcome"] for r in self.records("A1")], [TRANSPORT] * 4)
        self.assertEqual(result["slots"]["A1"], {"outcome": TRANSPORT, "valid": False, "verdict": None,
                                                 "attempts": 4, "transport_attempts": 4})
        self.assertEqual(self.sleeps, [2.0, 4.0, 8.0])
        self.assertEqual(len(t.sent), 7)
        self.assertFalse(t.script)

    def test_retry_after_is_honored_up_to_a_cap(self):
        self.run_slots(a_err(429, "rate_limit_error", headers={"retry-after": "30"}),
                       a_err(429, "rate_limit_error", headers={"retry-after": "9999"}), self.f.a(), self.f.a(),
                       self.f.a(), self.f.b())
        self.assertEqual(self.sleeps, [30.0, 120.0])

    def test_terminal_outcomes_are_never_retried(self):
        exchanges = {REFUSAL: self.f.a(text="", stop="refusal"), TRUNCATED: self.f.a(stop="max_tokens"),
                     MALFORMED: self.f.a(text="{}"), ABSTAIN: self.f.a(text=answer(verdict="?"))}
        for i, (outcome, ex) in enumerate(exchanges.items()):
            self.log = AttemptLog(self.tmp / f"t{i}")
            result, t = self.run_slots(ex, self.f.a(), self.f.a(), self.f.b())
            self.assertEqual(len(t.sent), 4)
            self.assertEqual([r["outcome"] for r in self.records("A1")], [outcome])
        self.assertEqual(self.sleeps, [])

    def test_a_400_halts_everything_and_resumes_only_after_a_correction(self):
        result, t = self.run_slots(self.f.a(), a_err(400, "invalid_request_error", "bad field"))
        self.assertTrue(result["halted"])
        self.assertEqual(len(t.sent), 2)
        self.assertEqual(self.records("A2")[0]["outcome"], HALT)
        self.assertEqual(self.records("A2")[0]["halt_kind"], "request_error")
        halt = self.log.halts()[0]
        self.assertEqual((halt["slot"], halt["halt_kind"], halt["http_status"]), ("A2", "request_error", 400))
        with self.assertRaises(D.HaltPending):
            self.run_slots(self.f.a())
        self.log.record_correction(result["halt_id"], "Fixed the request builder.", self.config, at="now")
        result, t = self.run_slots(self.f.a(), self.f.a(), self.f.b())
        self.assertFalse(result["halted"])
        self.assertEqual([r.judge for r in t.sent], ["A", "A", "B"])           # A1 is not sent again
        self.assertEqual([r["outcome"] for r in self.records("A2")], [HALT, VALID])
        self.assertEqual(len(self.records("A1")), 1)

    def test_halts_do_not_use_up_retries(self):
        result, _ = self.run_slots(NO_RESPONSE, a_err(402, "billing_error"))
        self.log.record_correction(result["halt_id"], "Added credit.", self.config, at="now")
        result, t = self.run_slots(NO_RESPONSE, NO_RESPONSE, NO_RESPONSE, self.f.a(), self.f.a(), self.f.b())
        self.assertEqual([r["outcome"] for r in self.records("A1")], [TRANSPORT, HALT, TRANSPORT, TRANSPORT, TRANSPORT])
        self.assertEqual(result["slots"]["A1"]["outcome"], TRANSPORT)
        self.assertEqual(self.sleeps, [2.0, 4.0, 8.0])

    def test_response_checks_halt(self):
        bad = [lambda f: f.a(model="claude-sonnet-5-20261101"),
               lambda f: f.a(service_tier="priority"),
               lambda f: f.a(inference_geo="us"),
               lambda f: f.a(usage={"cache_creation": {"ephemeral_5m_input_tokens": 0,
                                                       "ephemeral_1h_input_tokens": 512}}),
               lambda f: f.a(usage={"cache_creation": None}),
               lambda f: f.a(service_tier=None),
               lambda f: f.a(headers={"anthropic-workspace-id": "wrkspc_other"})]
        for i, make in enumerate(bad):
            self.log = AttemptLog(self.tmp / f"c{i}")
            self.log.set_first_model(self.config, "A", R.A_MODEL, "earlier run")
            result, _ = self.run_slots(make(self.f))
            self.assertTrue(result["halted"], i)
            rec = self.records("A1")[0]
            self.assertEqual((rec["outcome"], rec["halt_kind"], rec["response_outcome"]), (HALT, "response_check", VALID))
            self.assertTrue(rec["failed_checks"], i)
        for i, make in enumerate([lambda f: f.b(model="gpt-6-sol-2026-11-01"), lambda f: f.b(service_tier="flex"),
                                  lambda f: f.b(usage={"input_tokens_details": {"cached_tokens": 64,
                                                                                "cache_write_tokens": 0}}),
                                  lambda f: f.b(usage={"input_tokens_details": {"cached_tokens": 0,
                                                                                "cache_write_tokens": 64}}),
                                  lambda f: f.b(usage={"input_tokens_details": {}}),
                                  lambda f: f.b(usage={"input_tokens": 272001}),
                                  lambda f: f.b(usage={"input_tokens": None})]):
            self.log = AttemptLog(self.tmp / f"b{i}")
            self.log.set_first_model(self.config, "B", R.B_MODEL, "earlier run")
            result, _ = self.run_slots(self.f.a(), self.f.a(), self.f.a(), make(self.f))
            self.assertTrue(result["halted"], i)
            self.assertEqual(self.records("B1")[0]["halt_kind"], "response_check")
        self.log = AttemptLog(self.tmp / "b-at-limit")
        result, _ = self.run_slots(self.f.a(), self.f.a(), self.f.a(), self.f.b(usage={"input_tokens": 272000}))
        self.assertFalse(result["halted"])
        self.assertIn("input_tokens is 272001, not at most 272000",
                      D.response_checks("openai", openai_summary(sse(openai_events(
                          answer(), usage={"input_tokens": 272001}))), {}, None, None))

    def test_a_response_that_reports_another_openai_project_halts(self):
        result, _ = self.run_slots(self.f.a(), self.f.a(), self.f.a(),
                                   self.f.b(headers={"OpenAI-Project": "proj_other"}), project="proj_study")
        self.assertTrue(result["halted"])
        rec = self.records("B1")[0]
        self.assertEqual((rec["outcome"], rec["halt_kind"], rec["response_outcome"]), (HALT, "response_check", VALID))
        self.assertEqual(rec["failed_checks"], ["the openai-project header is not the study project"])
        self.assertEqual(rec["response_headers"]["openai-project"], "proj_other")
        self.assertEqual(self.log.unresolved_halts()[0]["halt_kind"], "response_check")

    def test_a_response_that_reports_no_other_openai_project_passes(self):
        cases = [({"openai-project": "proj_study"}, "proj_study"), ({}, "proj_study"),
                 ({"openai-project": "proj_x"}, None)]
        for i, (headers, project) in enumerate(cases):
            self.log = AttemptLog(self.tmp / f"p{i}")
            result, _ = self.run_slots(self.f.a(), self.f.a(), self.f.a(), self.f.b(headers=headers), project=project)
            self.assertFalse(result["halted"], i)

    def test_the_project_check_reads_only_openai_responses(self):
        headers = {"anthropic-workspace-id": WS, "openai-project": "proj_other"}
        result, _ = self.run_slots(self.f.a(headers=headers), self.f.a(), self.f.a(), self.f.b(), project="proj_study")
        self.assertFalse(result["halted"])

    def test_a_rebuilt_b_response_lists_the_project_check_as_not_applied(self):
        self.log.write_raw(KEY, self.config, "B", 1, 1, self.f.b().raw)
        rec = D.recover_attempt(self.log, {"trace_key": KEY, "config_sha256": self.config, "judge": "B", "repeat": 1,
                                           "attempt": 1, "provider": "openai", "model": R.B_MODEL,
                                           "request_sha256": self.reqs["B"].sha256})
        self.assertEqual(rec["recovered"]["checks_not_applied"], ["openai-project header"])

    def test_first_model_string_is_recorded_then_enforced(self):
        self.run_slots(self.f.a(), self.f.a(), self.f.a(), self.f.b())
        self.assertEqual(self.log.first_model(self.config, "A"), R.A_MODEL)
        self.assertEqual(self.log.first_model(self.config, "B"), R.B_MODEL)
        result, _ = self.run_slots(self.f.a(), self.f.a(model="claude-sonnet-5-changed"), key="t1111111111111111")
        self.assertTrue(result["halted"])
        self.assertEqual(self.records("A2", key="t1111111111111111")[0]["halt_kind"], "response_check")

    def test_a_repeated_response_id_halts(self):
        ex = self.f.a()
        result, _ = self.run_slots(ex, ex)
        self.assertTrue(result["halted"])
        self.assertIn("seen before", self.records("A2")[0]["failed_checks"][0])

    def test_a_response_is_on_disk_before_anything_reads_it(self):
        ex, seen = self.f.a(), []

        def classify(provider, exchange):
            seen.append((self.log.slot_dir(KEY, self.config, "A", 1) / "1.raw").read_bytes())
            raise KeyboardInterrupt

        with mock.patch.object(D, "canonical_outcome", classify):
            with self.assertRaises(KeyboardInterrupt):
                self.run_slots(ex)
        self.assertEqual(seen, [ex.raw])
        self.assertEqual(self.records("A1"), [])

    def test_an_exception_while_a_response_is_read_ends_its_slot_and_halts(self):
        with mock.patch.object(D, "canonical_outcome", side_effect=RuntimeError("quoted text")):
            result, t = self.run_slots(self.f.a())
        self.assertTrue(result["halted"])
        self.assertEqual(len(t.sent), 1)
        rec = self.records("A1")[0]
        self.assertEqual((rec["outcome"], rec["response_outcome"], rec["halt_kind"], rec["detail"], rec["parse"]),
                         (MALFORMED, MALFORMED, CLASSIFICATION_ERROR, "classification raised RuntimeError", None))
        self.assertTrue(D._slot_done([rec]))
        halt = self.log.halts()[0]
        self.assertEqual((halt["halt_id"], halt["slot"], halt["attempt"], halt["halt_kind"], halt["http_status"]),
                         (result["halt_id"], "A1", 1, CLASSIFICATION_ERROR, 200))
        with self.assertRaises(D.HaltPending):
            self.run_slots(self.f.a())
        self.log.record_correction(result["halt_id"], "Read the raw response.", self.config, at="now")
        result, t = self.run_slots(self.f.a(), self.f.a(), self.f.b())
        self.assertFalse(result["halted"])
        self.assertEqual([r.judge for r in t.sent], ["A", "A", "B"])
        self.assertEqual(result["slots"]["A1"], {"outcome": MALFORMED, "valid": False, "verdict": None,
                                                 "attempts": 1, "transport_attempts": 0})

    def test_an_identity_check_that_raises_ends_the_slot_too(self):
        with mock.patch.object(AttemptLog, "claim_response_id", side_effect=OSError("disk")):
            result, _ = self.run_slots(self.f.a())
        self.assertTrue(result["halted"])
        self.assertEqual([(r["outcome"], r["detail"]) for r in self.records("A1")],
                         [(MALFORMED, "classification raised OSError")])

    def test_an_error_status_that_cannot_be_read_halts_and_its_slot_resumes(self):
        unreadable = Exchange(400, {}, json.dumps({"type": "error", "error": {"type": "x", "message": 5}}).encode(),
                              None, None)
        result, _ = self.run_slots(unreadable)
        self.assertTrue(result["halted"])
        rec = self.records("A1")[0]
        self.assertEqual((rec["outcome"], rec["halt_kind"], rec["http_status"]), (HALT, CLASSIFICATION_ERROR, 400))
        self.assertFalse(D._slot_done([rec]))
        self.assertIn("next attempt index", self.log.halts()[0]["resume_rule"])
        self.log.record_correction(result["halt_id"], "Read the error body.", self.config, at="now")
        result, t = self.run_slots(self.f.a(), self.f.a(), self.f.a(), self.f.b())
        self.assertEqual([r.judge for r in t.sent], ["A", "A", "A", "B"])
        self.assertEqual([r["outcome"] for r in self.records("A1")], [HALT, VALID])

    def test_a_value_the_log_cannot_store_ends_the_slot_and_halts(self):
        result, _ = self.run_slots(self.f.a(service_tier=float("nan")))
        self.assertTrue(result["halted"])
        rec = self.records("A1")[0]
        self.assertEqual((rec["outcome"], rec["halt_kind"], rec["detail"], rec["usage"]),
                         (MALFORMED, CLASSIFICATION_ERROR, "classification raised ValueError", {}))

    def test_a_model_that_is_not_a_string_halts_and_is_never_recorded_as_the_first(self):
        for n, model in enumerate((["gpt-6-sol"], {"id": "gpt-6-sol"}, 5, None)):
            self.log = AttemptLog(self.tmp / f"log-{n}")
            events = openai_events(answer(), resp_id=f"resp_{n}")
            for _, obj in events:
                if "response" in obj:
                    obj["response"]["model"] = model
            result, _ = self.run_slots(self.f.a(), self.f.a(), self.f.a(), Exchange(200, {}, sse(events), None, None))
            self.assertTrue(result["halted"], model)
            rec = self.records("B1")[0]
            self.assertEqual((rec["outcome"], rec["halt_kind"], rec["failed_checks"]),
                             (HALT, "response_check", [f"model {model!r} is not a string"]), model)
            self.assertIsNone(self.log.first_model(self.config, "B"), model)

    def test_a_lone_surrogate_in_the_answer_is_stored_and_the_verdict_kept(self):
        text = answer_with_literal("rationale", r'"caf\u00e9 \ud83d end"', verdict="failure")
        result, _ = self.run_slots(self.f.a(text=text), self.f.a(), self.f.a(), self.f.b())
        self.assertFalse(result["halted"])
        rec = self.records("A1")[0]
        self.assertEqual((rec["outcome"], rec["parse"]["verdict"], rec["parse"]["fields"]["rationale"]),
                         (VALID, "failure", "caf\u00e9 \ud83d end"))
        data = (self.log.slot_dir(KEY, self.config, "A", 1) / "1.json").read_bytes()
        self.assertIn("caf\u00e9".encode("utf-8"), data)                    # other text stays as it was

    def test_cache_tokens_are_recorded(self):
        self.run_slots(self.f.a(usage={"cache_creation_input_tokens": 5000}),
                       self.f.a(usage={"cache_read_input_tokens": 5000}),
                       self.f.a(usage={"cache_creation_input_tokens": 5000}), self.f.b())
        usage = [self.records(s)[0]["usage"] for s in ("A1", "A2", "A3", "B1")]
        self.assertEqual([(u["cache_read_tokens"], u["cache_write_tokens"]) for u in usage],
                         [(0, 5000), (5000, 0), (0, 5000), (0, 0)])

    def test_requests_that_differ_from_the_table_are_never_sent(self):
        bad = dict(json.loads(self.reqs["A"].body), temperature=0)
        data = R.canonical_bytes(bad)
        reqs = dict(self.reqs, A=self.reqs["A"]._replace(body=data, sha256=R.sha256_bytes(data)))
        with self.assertRaises(ValueError):
            self.run_slots(self.f.a(), reqs=reqs)
        stale = dict(self.reqs, B=self.reqs["B"]._replace(body=self.reqs["B"].body + b" "))
        transport = D.FakeTransport()
        with self.assertRaises(ValueError):
            D.dispatch_trajectory(KEY, stale, self.config, transport, self.log, WS)
        self.assertEqual(transport.sent, [])


class TestCapability(LogCase):
    def b_records(self, *exchanges):
        recs = []
        for ex in exchanges:
            self.n_logs = getattr(self, "n_logs", 0) + 1
            self.log = AttemptLog(self.tmp / f"cap{self.n_logs}")
            result, _ = self.run_slots(self.f.a(), self.f.a(), self.f.a(), ex)
            if result["halted"]:
                self.log.record_correction(result["halt_id"], "test", self.config, at="now")
            recs += self.records("B1")
        return recs

    def test_valid_first_response_passes(self):
        out = D.capability_check(self.b_records(self.f.b(), self.f.b()), R.B_MODEL)
        self.assertEqual(out["checks"], {"access": "ok", "images_accepted": "ok", "strict_output": "ok",
                                         "stable_model_string": "ok"})
        self.assertFalse(out["fallback_indicated"])

    def test_refusals_are_never_a_reason_to_change_judges(self):
        recs = self.b_records(self.f.b(text=None, refusal="no"), self.f.b(text=None, refusal="no"))
        out = D.capability_check(recs, R.B_MODEL)
        self.assertEqual(out["failed"], [])
        self.assertFalse(out["fallback_indicated"])
        self.assertEqual(out["checks"]["strict_output"], "unknown")
        out = D.capability_check(recs + self.b_records(self.f.b()), R.B_MODEL)
        self.assertEqual((out["checks"]["strict_output"], out["fallback_indicated"]), ("ok", False))

    def test_failures_point_to_the_first_fallback(self):
        cases = [(self.f.b(text="not json"), "strict_output"),
                 (self.f.b(status="failed", error={"code": "invalid_image", "message": "x"}), "images_accepted"),
                 (b_err(404, "model_not_found", etype="invalid_request_error"), "access")]
        for ex, check in cases:
            out = D.capability_check(self.b_records(ex), R.B_MODEL)
            self.assertIn(check, out["failed"], check)
            self.assertEqual((out["fallback_indicated"], out["next_model"]), (True, "gpt-6.1-sol"), check)
        out = D.capability_check(self.b_records(self.f.b(text="not json")), "gpt-6.1-sol")
        self.assertEqual((out["fallback_indicated"], out["next_model"]), (False, None))

    def test_a_changing_model_string_fails_stability(self):
        recs = self.b_records(self.f.b())
        changed = dict(recs[0], response_model="gpt-6-sol-b")
        out = D.capability_check(recs + [changed], R.B_MODEL)
        self.assertEqual(out["checks"]["stable_model_string"], "failed")

    def test_a_model_that_is_not_a_string_shows_nothing_about_the_judge(self):
        for model in (["gpt-6-sol"], {"id": "gpt-6-sol"}, 5):
            recs = self.b_records(self.f.b(model=model))
            self.assertEqual((recs[0]["outcome"], recs[0]["response_model"]), (HALT, model), model)
            out = D.capability_check(recs, R.B_MODEL)
            self.assertEqual((out["models_seen"], out["checks"]["stable_model_string"], out["failed"]),
                             ([], "unknown", []), model)
            out = D.capability_check(recs + self.b_records(self.f.b()), R.B_MODEL)
            self.assertEqual((out["models_seen"], out["checks"]["stable_model_string"]), ([R.B_MODEL], "ok"), model)

    def test_an_unclassified_response_is_not_a_capability_failure(self):
        real = D.canonical_outcome

        def classify(provider, exchange):
            if provider == "openai":
                raise RuntimeError("defect")
            return real(provider, exchange)

        with mock.patch.object(D, "canonical_outcome", classify):
            recs = self.b_records(self.f.b())
        self.assertEqual(recs[0]["halt_kind"], CLASSIFICATION_ERROR)
        out = D.capability_check(recs, R.B_MODEL)
        self.assertEqual((out["checks"]["strict_output"], out["failed"]), ("unknown", []))

    def test_no_invalid_rate_threshold(self):
        recs = self.b_records(self.f.b(), *[self.f.b(status="incomplete", incomplete_reason="max_output_tokens")] * 3)
        out = D.capability_check(recs, R.B_MODEL)
        self.assertFalse(out["fallback_indicated"])


class TestHttpsTransport(unittest.TestCase):
    def test_refuses_to_exist_without_paid_call_confirmation(self):
        for value in (False, None, 1, "yes"):
            with self.assertRaises(ValueError):
                D.HttpsTransport(confirm_paid_calls=value, openai_project="proj_study")

    def test_refuses_to_exist_without_the_openai_project(self):
        for value in (None, ""):
            with self.assertRaisesRegex(ValueError, "pass openai_project"):
                D.HttpsTransport(confirm_paid_calls=True, openai_project=value)

    def sent_by_transport(self, request):
        """The urllib request a send builds and the exchange it returns. ``urlopen`` is
        replaced, so nothing leaves, and the credential is a placeholder."""
        built = []

        class Response(io.BytesIO):
            status = 200
            headers = {"OpenAI-Project": "proj_study", "x-request-id": "req_1"}

        def urlopen(req, timeout):
            built.append(req)
            return Response(b"data: {}\n\n")

        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "placeholder", "OPENAI_API_KEY": "placeholder"}):
            with mock.patch.object(D.urllib.request, "urlopen", urlopen):
                exchange = D.HttpsTransport(confirm_paid_calls=True, openai_project="proj_study").send(request)
        self.assertEqual(len(built), 1)
        return built[0], exchange

    def test_every_openai_request_names_the_configured_project_at_send_time(self):
        pack = synthetic_pack(1)
        reqs = R.prepare(pack, images_for(pack))["requests"]
        sent, exchange = self.sent_by_transport(reqs["B"])
        self.assertEqual(sent.get_header("Openai-project"), "proj_study")
        self.assertEqual((sent.get_header("Authorization"), sent.data), ("Bearer placeholder", reqs["B"].body))
        self.assertEqual(exchange.headers["openai-project"], "proj_study")
        self.assertEqual(dict(reqs["B"].headers), {"content-type": "application/json"})
        sent, _ = self.sent_by_transport(reqs["A"])
        self.assertIsNone(sent.get_header("Openai-project"))

    def test_sends_and_token_counts_carry_the_fixed_user_agent_for_both_providers(self):
        pack = synthetic_pack(1)
        reqs = R.prepare(pack, images_for(pack))["requests"]
        for judge in ("A", "B"):
            for request in (reqs[judge], D.count_request(reqs[judge])):
                sent, _ = self.sent_by_transport(request)
                self.assertEqual(sent.get_header("User-agent"), D.USER_AGENT)
            self.assertNotIn("user-agent", dict(reqs[judge].headers))

    def test_the_project_is_not_part_of_the_prepared_request_or_the_configuration(self):
        self.assertEqual(dict(R.B_HEADERS), {"content-type": "application/json"})
        self.assertEqual(R.config_record()["judges"]["B"]["headers"], {"content-type": "application/json"})
        self.assertNotIn("proj_", json.dumps(R.config_record()))

    def test_read_stream(self):
        raw = sse(anthropic_events(answer()))
        times = iter([5.0, 6.0])
        self.assertEqual(D.read_stream(io.BytesIO(raw), clock=lambda: next(times)), (raw, None, 5.0))

        class Broken(io.BytesIO):
            def read1(self, n=-1):
                if self.tell() > 100:
                    raise ConnectionResetError("reset")
                return super().read1(50)
        data, err, _ = D.read_stream(Broken(raw))
        self.assertEqual(data, raw[:150])
        self.assertIn("ConnectionResetError", err)


class HttpResponse(io.BytesIO):
    """What ``urlopen`` returns."""

    def __init__(self, raw, status=200, headers=None):
        super().__init__(raw)
        self.status, self.headers = status, headers or {}


class CutResponse(HttpResponse):
    """A response that loses its connection after its bytes."""

    def read1(self, n=-1):
        data = super().read1(n)
        if not data:
            raise ConnectionResetError("reset")
        return data


class FakeHttp:
    """Stands in for ``urlopen``: each request gets the next scripted answer and is
    recorded, so nothing leaves. An answer is an exception, a response or (status, headers,
    raw bytes)."""

    def __init__(self, *script):
        self.script, self.requests = list(script), []

    def __call__(self, req, timeout):
        self.requests.append(req)
        answer = self.script.pop(0)
        if isinstance(answer, Exception):
            raise answer
        if isinstance(answer, HttpResponse):
            return answer
        status, headers, raw = answer
        if status >= 400:
            raise urllib.error.HTTPError(req.full_url, status, "error", headers, io.BytesIO(raw))
        return HttpResponse(raw, status, headers)


def error_answer(status, etype, message="x", code=None, headers=None):
    err = {"type": etype, "message": message, "code": code}
    return status, headers or {}, json.dumps({"type": "error", "error": err}).encode()


class TestHttpsTokenCounter(unittest.TestCase):
    WORKSPACE, PROJECT = "wrkspc_study", "proj_study"
    A_REPLY, B_REPLY = b'{"input_tokens":61234}', b'{"object":"response.input_tokens","input_tokens":70123}'
    A_OK = (200, {"anthropic-workspace-id": "wrkspc_study"}, A_REPLY)
    B_OK = (200, {"x-request-id": "req_1"}, B_REPLY)
    RETRIES = (2.0, 4.0, 8.0)

    def setUp(self):
        pack = synthetic_pack(1)
        self.reqs = R.prepare(pack, images_for(pack))["requests"]
        self.sleeps = []

    def counted(self, request, *script, keys=("ANTHROPIC_API_KEY", "OPENAI_API_KEY")):
        """(the count or the ``TokenCountError`` raised, the requests made)."""
        http = FakeHttp(*script)
        env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")}
        with mock.patch.dict(os.environ, {**env, **dict.fromkeys(keys, "placeholder")}, clear=True):
            with mock.patch.object(D.urllib.request, "urlopen", http):
                transport = D.HttpsTransport(confirm_paid_calls=True, openai_project=self.PROJECT)
                counter = D.HttpsTokenCounter(transport, self.WORKSPACE, sleep=self.sleeps.append)
                try:
                    return counter.count(request), http.requests
                except D.TokenCountError as e:
                    return e, http.requests

    def assert_raises_once(self, judge, answer):
        """The ``TokenCountError`` that one answer raises, with nothing tried again."""
        self.sleeps.clear()
        error, sent = self.counted(self.reqs[judge], answer, self.A_OK if judge == "A" else self.B_OK)
        self.assertIsInstance(error, D.TokenCountError, answer)
        self.assertEqual((len(sent), self.sleeps), (1, []), answer)
        return error

    def test_refuses_to_exist_without_the_workspace(self):
        for value in (None, ""):
            with self.assertRaisesRegex(ValueError, "pass workspace"):
                D.HttpsTokenCounter(D.HttpsTransport(confirm_paid_calls=True, openai_project="p"), value)

    def test_each_count_is_one_post_to_its_provider_s_documented_endpoint(self):
        for judge, ok, url, count in (("A", self.A_OK, "https://api.anthropic.com/v1/messages/count_tokens", 61234),
                                      ("B", self.B_OK, "https://api.openai.com/v1/responses/input_tokens", 70123)):
            got, sent = self.counted(self.reqs[judge], ok)
            self.assertEqual(got, count)
            self.assertEqual([(r.full_url, r.get_method()) for r in sent], [(url, "POST")])

    def test_a_count_is_the_body_that_will_be_sent_less_only_the_fields_its_endpoint_does_not_list(self):
        unlisted = {"A": {"max_tokens", "stream", "inference_geo", "service_tier"},
                    "B": {"max_output_tokens", "stream", "prompt_cache_options", "service_tier", "store"}}
        pack = synthetic_pack(2)
        for caching in (True, False):
            reqs = R.prepare(pack, images_for(pack), caching=caching)["requests"]
            for judge, ok in (("A", self.A_OK), ("B", self.B_OK)):
                sent_body = json.loads(reqs[judge].body)
                self.assertLessEqual(unlisted[judge], set(sent_body))
                wanted = {k: v for k, v in sent_body.items() if k not in unlisted[judge]}
                _, [asked] = self.counted(reqs[judge], ok)
                self.assertEqual(json.loads(asked.data), wanted)
                self.assertEqual(asked.data, R.canonical_bytes(wanted))
                self.assertEqual(json.loads(reqs[judge].body), sent_body)

    def test_every_field_left_in_a_count_is_one_its_endpoint_lists(self):
        listed = {"A": {"messages", "model", "cache_control", "output_config", "speed", "system", "thinking",
                        "tool_choice", "tools"},
                  "B": {"conversation", "input", "instructions", "model", "parallel_tool_calls", "personality",
                        "previous_response_id", "reasoning", "text", "tool_choice", "tools", "truncation"}}
        for judge in "AB":
            self.assertLessEqual(set(json.loads(D.count_request(self.reqs[judge]).body)), listed[judge])
        self.assertIn("cache_control", json.loads(D.count_request(self.reqs["A"]).body))
        self.assertIn("reasoning", json.loads(D.count_request(self.reqs["B"]).body))

    def test_a_count_carries_the_credential_version_and_project_of_the_send(self):
        _, [a] = self.counted(self.reqs["A"], self.A_OK)
        self.assertEqual((a.get_header("X-api-key"), a.get_header("Anthropic-version")),
                         ("placeholder", R.ANTHROPIC_VERSION))
        self.assertIsNone(a.get_header("Openai-project"))
        _, [b] = self.counted(self.reqs["B"], self.B_OK)
        self.assertEqual((b.get_header("Authorization"), b.get_header("Openai-project")),
                         ("Bearer placeholder", self.PROJECT))

    def test_a_malformed_or_missing_count_raises_and_is_not_asked_again(self):
        headers = self.A_OK[1]
        bodies = (b"", b"not json", b"[]", b"null", b"{}", b'{"input_tokens":null}', b'{"input_tokens":"12"}',
                  b'{"input_tokens":12.0}', b'{"input_tokens":true}', b'{"input_tokens":0}', b'{"input_tokens":-1}',
                  b'{"input_tokens":' + b"9" * 5000 + b"}", b"[" * 200000)
        for raw in bodies:
            self.assert_raises_once("A", (200, headers, raw))
        for raw in (b'{"input_tokens":5}', b'{"object":"response","input_tokens":5}'):
            self.assertIn("response.input_tokens", str(self.assert_raises_once("B", (200, {}, raw))))

    def test_a_request_access_or_spend_error_raises_and_is_not_asked_again(self):
        for status, etype in ((400, "invalid_request_error"), (401, "authentication_error"), (402, "billing_error"),
                              (403, "permission_error"), (404, "not_found_error"), (413, "request_too_large")):
            error = self.assert_raises_once("A", error_answer(status, etype, "the reason"))
            self.assertEqual((error.status, error.error["type"], error.error["message"]), (status, etype, "the reason"))
        self.assert_raises_once("A", error_answer(429, "rate_limit_error", "You have reached your API usage limits"))
        self.assert_raises_once("B", error_answer(429, "insufficient_quota", code="insufficient_quota"))
        self.assert_raises_once("B", error_answer(400, "invalid_request_error", code="content_filter"))

    def test_a_count_that_raises_while_it_is_read_records_only_the_exception_s_type(self):
        with mock.patch.object(D.HttpsTokenCounter, "_read", side_effect=ValueError("the reply was: quoted text")):
            error, sent = self.counted(self.reqs["A"], self.A_OK, self.A_OK)
        self.assertIsInstance(error, D.TokenCountError)
        self.assertEqual((str(error), len(sent)),
                         ("the anthropic token count failed: reading the response raised ValueError", 1))

    def test_a_transport_failure_is_asked_again_under_the_dispatcher_s_rule(self):
        for failure in (urllib.error.URLError("down"), error_answer(429, "rate_limit_error"),
                        error_answer(408, "timeout_error"), error_answer(500, "api_error"),
                        error_answer(529, "overloaded_error")):
            self.sleeps.clear()
            got, sent = self.counted(self.reqs["A"], failure, failure, failure, self.A_OK)
            self.assertEqual((got, len(sent), tuple(self.sleeps)), (61234, 4, self.RETRIES), failure)
            self.sleeps.clear()
            error, sent = self.counted(self.reqs["A"], *[failure] * 6)
            self.assertIsInstance(error, D.TokenCountError)
            self.assertEqual((len(sent), tuple(self.sleeps)), (4, self.RETRIES), failure)

    def test_a_retry_honors_retry_after_up_to_the_dispatcher_s_cap(self):
        for wait, slept in (("30", 30.0), ("500", 120.0), ("soon", 2.0)):
            self.sleeps.clear()
            answer = error_answer(429, "rate_limit_error", headers={"Retry-After": wait})
            got, _ = self.counted(self.reqs["A"], answer, self.A_OK)
            self.assertEqual((got, self.sleeps), (61234, [slept]))

    def test_a_count_from_a_response_that_was_cut_is_not_taken(self):
        got, sent = self.counted(self.reqs["A"], CutResponse(self.A_REPLY, 200, self.A_OK[1]), self.A_OK)
        self.assertEqual((got, len(sent), self.sleeps), (61234, 2, [2.0]))

    def test_a_count_from_another_scope_raises_and_is_not_asked_again(self):
        for headers in ({"anthropic-workspace-id": "wrkspc_other"}, {}):
            self.assertIn("workspace", str(self.assert_raises_once("A", (200, headers, self.A_REPLY))))
        answer = (200, {"OpenAI-Project": "proj_other"}, self.B_REPLY)
        self.assertIn("project", str(self.assert_raises_once("B", answer)))

    def test_an_openai_count_may_name_the_study_project_or_none(self):
        for headers in ({}, {"OpenAI-Project": self.PROJECT}):
            self.assertEqual(self.counted(self.reqs["B"], (200, headers, self.B_REPLY))[0], 70123)

    def test_a_missing_credential_raises_before_anything_leaves(self):
        error, sent = self.counted(self.reqs["A"], self.A_OK, keys=("OPENAI_API_KEY",))
        self.assertIsInstance(error, D.TokenCountError)
        self.assertIn("set ANTHROPIC_API_KEY", str(error))
        self.assertEqual(sent, [])

    def test_an_exception_while_sending_raises_with_only_its_type(self):
        error, sent = self.counted(self.reqs["A"], ValueError("Invalid header value b'the-key'"))
        self.assertIsInstance(error, D.TokenCountError)
        self.assertIn("ValueError", str(error))
        self.assertNotIn("the-key", str(error))
        self.assertEqual(len(sent), 1)


if __name__ == "__main__":
    unittest.main()
