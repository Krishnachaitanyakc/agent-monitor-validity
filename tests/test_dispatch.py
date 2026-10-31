"""Tests for streams, outcomes, the write-once log and the serial dispatcher.

Every response here is a synthetic, provider-shaped stream or error body served by
FakeTransport. No test builds the HTTPS transport's connection or sends anything.
"""

import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from judge import dispatch as D  # noqa: E402
from judge import requests as R  # noqa: E402
from judge.outcomes import (ABSTAIN, FILTERED, HALT, MALFORMED, REFUSAL, TRANSPORT,  # noqa: E402
                            TRUNCATED, VALID, Exchange, canonical_outcome)
from judge.storage import AttemptLog, WriteOnceError, write_once  # noqa: E402
from judge.streams import (anthropic_events, anthropic_summary, openai_events,  # noqa: E402
                           openai_summary, parse_sse, sse)
from test_requests import answer, images_for, synthetic_pack  # noqa: E402

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

    def b(self, text=None, status="completed", **kw):
        if text is None and status == "completed" and kw.get("refusal") is None:
            text = answer()
        return Exchange(200, {}, sse(openai_events(text, status=status, resp_id=self._id("resp"), **kw)), None, None)


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

    def run_slots(self, *exchanges, workspace=WS, reqs=None, key=KEY):
        transport = D.FakeTransport(exchanges)
        result = D.dispatch_trajectory(key, reqs or self.reqs, self.config, transport, self.log, workspace,
                                       sleep=self.sleeps.append, clock=self.clock)
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

    def test_a_slot_subset_runs_in_order_and_nothing_else_is_accepted(self):
        transport = D.FakeTransport([self.f.a(), self.f.b()])
        result = D.dispatch_trajectory(KEY, self.reqs, self.config, transport, self.log, WS, sleep=self.sleeps.append,
                                       clock=self.clock, slots=(("A", 1), ("B", 1)))
        self.assertEqual([r.judge for r in transport.sent], ["A", "B"])
        self.assertEqual(sorted(result["slots"]), ["A1", "B1"])
        self.assertEqual(self.records("A2"), [])
        for bad in ((), (("B", 1),), (("A", 2), ("A", 1)), (("A", 1), ("A", 1)), (("A", 1), ("A", 4)),
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

    def test_retry_after_is_honoured_up_to_a_cap(self):
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
                                  lambda f: f.b(usage={"input_tokens_details": {}})]):
            self.log = AttemptLog(self.tmp / f"b{i}")
            self.log.set_first_model(self.config, "B", R.B_MODEL, "earlier run")
            result, _ = self.run_slots(self.f.a(), self.f.a(), self.f.a(), make(self.f))
            self.assertTrue(result["halted"], i)
            self.assertEqual(self.records("B1")[0]["halt_kind"], "response_check")

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

    def test_no_invalid_rate_threshold(self):
        recs = self.b_records(self.f.b(), *[self.f.b(status="incomplete", incomplete_reason="max_output_tokens")] * 3)
        out = D.capability_check(recs, R.B_MODEL)
        self.assertFalse(out["fallback_indicated"])


class TestHttpsTransport(unittest.TestCase):
    def test_refuses_to_exist_without_paid_call_confirmation(self):
        for value in (False, None, 1, "yes"):
            with self.assertRaises(ValueError):
                D.HttpsTransport(confirm_paid_calls=value)

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


if __name__ == "__main__":
    unittest.main()
