"""Tests for the spending controls in judge/spend.py.

Every response is a synthetic, provider-shaped stream served by FakeTransport or built in
memory. Nothing is sent anywhere, and no test builds the HTTPS transport.
"""

import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from decimal import Decimal
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datapath.png import make_png  # noqa: E402
from judge import dispatch as D  # noqa: E402
from judge import requests as R  # noqa: E402
from judge import spend as S  # noqa: E402
from judge.outcomes import TRANSPORT, VALID, Exchange  # noqa: E402
from judge.storage import AttemptLog  # noqa: E402
from judge.streams import anthropic_events, openai_events, sse  # noqa: E402
from test_dispatch import KEY, WS, NO_RESPONSE, a_err  # noqa: E402
from test_requests import answer, images_for, synthetic_pack  # noqa: E402

A, B = R.A_MODEL, R.B_MODEL
CAP = S.CAPS["pilot"]
START, END = "2026-01-01T00:00:00Z", "2099-01-01T00:00:00Z"        # a period that covers every test ledger


def a_raw(inp=0, write=0, read=0, out=3000, stop="end_turn", msg_id="msg_1", one_hour=0, tier="standard",
          geo="global", final_usage=True):
    usage = {"input_tokens": inp, "cache_creation_input_tokens": write, "cache_read_input_tokens": read,
             "cache_creation": {"ephemeral_5m_input_tokens": write - one_hour, "ephemeral_1h_input_tokens": one_hour}}
    events = anthropic_events(answer() if stop == "end_turn" else None, stop_reason=stop, msg_id=msg_id, usage=usage,
                              service_tier=tier, inference_geo=geo)
    for _, obj in events:
        if obj.get("type") == "message_delta":
            if final_usage:
                obj["usage"] = {"output_tokens": out}
            else:
                del obj["usage"]
    return sse(events)


def b_raw(inp=60000, out=3000, resp_id="resp_1", cached=0, written=0, tier="default"):
    usage = {"input_tokens": inp, "input_tokens_details": {"cached_tokens": cached, "cache_write_tokens": written},
             "output_tokens": out, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": inp + out}
    return sse(openai_events(answer(), resp_id=resp_id, usage=usage, service_tier=tier))


def fields(provider="anthropic", model=A, **kw):
    return dict({"provider": provider, "model": model, "run": "run", "stage": "P1", "trace_key": KEY,
                 "slot": "A1"}, **kw)


class LedgerCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.path = self.tmp / "ledger-pilot.jsonl"

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def ledger(self, budget="pilot"):
        led = S.Ledger(self.path, budget)
        self.addCleanup(led.close)
        return led

    def lines(self):
        return [json.loads(x) for x in self.path.read_text().splitlines()]


class TestPrices(unittest.TestCase):
    def test_one_versioned_table(self):
        self.assertEqual(S.PRICE_TABLE_VERSION, "prices-2026-10-28-r2")
        self.assertEqual(S.PRICES[A], {"provider": "anthropic", "input": "2", "cache_write": "2.50",
                                       "cache_read": "0.20", "output": "10"})
        for model in (B, "gpt-6.1-sol"):
            self.assertEqual(S.PRICES[model], {"provider": "openai", "input": "2", "cache_write": None,
                                               "cache_read": None, "output": "10"})
        self.assertEqual(sorted(S.PRICES), sorted((A, B) + R.B_FALLBACK_MODELS))
        self.assertEqual(S.CAPS, {"pilot": Decimal("95"), "evaluation": Decimal("285")})

    def test_the_documented_modifiers(self):
        a, b = S.MODIFIERS["anthropic"], S.MODIFIERS["openai"]
        self.assertEqual((a["inference_geo"], a["service_tier"], a["cache_write_1h_per_million"]),
                         ({"global": "1", "us": "1.1"}, {"standard": "1"}, {A: "4"}))
        self.assertEqual(b["service_tier"], {"default": "1", "flex": "0.5", "priority": "2", "ultrafast": "6"})
        self.assertEqual((b["cache_read_per_million"], b["cache_write_per_million"]),
                         ({B: "0.20", "gpt-6.1-sol": "0.10"}, {B: "5", "gpt-6.1-sol": "5"}))
        self.assertEqual(b["long_context"], {"above_input_tokens": 272000, "input_and_cache": "2", "output": "1.5"})

    def test_setup_note_costs_per_call(self):
        cost = S.call_cost
        for (i, o), (write, repeat, plain) in {(60000, 3000): ("0.18", "0.042", "0.15"),
                                               (100000, 20000): ("0.45", "0.22", "0.40")}.items():
            self.assertEqual(cost(A, {"input": 0, "cache_write": i, "cache_read": 0, "output": o}), Decimal(write))
            self.assertEqual(cost(A, {"input": 0, "cache_write": 0, "cache_read": i, "output": o}), Decimal(repeat))
            self.assertEqual(cost(A, {"input": i, "cache_write": 0, "cache_read": 0, "output": o}), Decimal(plain))
            self.assertEqual(cost(B, {"input": i, "cache_write": 0, "cache_read": 0, "output": o}), Decimal(plain))
        # B has no caching, so a cached token it reports is priced as input.
        self.assertEqual(cost(B, {"input": 0, "cache_write": 0, "cache_read": 1000, "output": 0}), Decimal("0.002"))

    def test_a_model_outside_the_table_cannot_be_reserved(self):
        for model in ("gpt-5.6-sol", "claude-sonnet-5.5"):
            with self.assertRaises(ValueError):
                S.reservation(model, 1000, 32000)


class TestReservation(unittest.TestCase):
    def test_the_formula(self):
        # The setup note: $0.47 at 60,000 input tokens and $0.58 at 100,000.
        self.assertEqual(S.reservation(A, 60000, 32000), Decimal("0.47364"))
        self.assertEqual(S.reservation(A, 100000, 32000), Decimal("0.57564"))
        self.assertEqual(S.reservation(B, 100000, 32000), Decimal("0.57564"))
        for i in (0, 1, 999, 51400, 250000):
            expected = (Decimal(i) * Decimal("1.02") + 256) * Decimal("0.000002") * Decimal("1.25") + Decimal("0.32")
            self.assertEqual(S.reservation(A, i, 32000), expected)
        self.assertEqual(S.reservation(A, 0, 1000), Decimal("256") * Decimal("0.0000025") + Decimal("0.01"))

    def test_a_call_inside_its_count_and_cap_never_costs_more_than_its_reservation(self):
        for i in (1, 1024, 60000, 100000):
            r = S.reservation(A, i, 32000)
            worst = S.call_cost(A, {"input": 0, "cache_write": i, "cache_read": 0, "output": 32000})
            self.assertLessEqual(worst, r)
            self.assertGreater(r - worst, S.reservation(A, 0, 1) - Decimal("0.00001"))

    def test_bad_counts_are_refused(self):
        for i, o in ((-1, 32000), (None, 32000), (1.5, 32000), (1000, 0)):
            with self.assertRaises(ValueError):
                S.reservation(A, i, o)

    def test_the_fake_mode_estimate_never_underestimates_the_request_layers(self):
        self.assertEqual(S.ESTIMATE_FACTOR, 2)
        cases = [(synthetic_pack(1), (4, 3)), (synthetic_pack(4), (300, 200)),
                 (synthetic_pack(3, shots=[True, False, True, False]), (1280, 720)),
                 (synthetic_pack(2, goal="Find " + "the cheapest red mug " * 400), (90, 2000))]
        for pack, size in cases:
            images = {ref: make_png(*size, rgb=(1, 2, 3)) for ref in images_for(pack)}
            prep = R.prepare(pack, images)
            for j in "AB":
                est = R.token_estimate(j, prep["requests"][j], prep["image_sizes"])["input_tokens"]
                basis = S.estimated_input_tokens(est)
                self.assertGreaterEqual(basis, est)
                self.assertEqual(basis, 2 * est)
                self.assertGreaterEqual(S.reservation(prep["requests"][j].model, basis, 32000),
                                        S.reservation(prep["requests"][j].model, est, 32000))

    def test_the_output_cap_comes_from_the_body(self):
        pack = synthetic_pack(1)
        reqs = R.prepare(pack, images_for(pack))["requests"]
        self.assertEqual([S.max_output_tokens(j, json.loads(reqs[j].body)) for j in "AB"], [32000, 32000])


class TestCharge(unittest.TestCase):
    def test_complete_responses_settle_at_their_usage(self):
        kind, cost, tokens, _ = S.charge("anthropic", A, 200, a_raw(inp=10, write=60000, read=0, out=3000))
        self.assertEqual((kind, cost), (S.SETTLE, Decimal("0.18002")))
        self.assertEqual(tokens, {"input": 10, "cache_write": 60000, "cache_read": 0, "output": 3000})
        kind, cost, _, _ = S.charge("anthropic", A, 200, a_raw(read=60000, out=3000))
        self.assertEqual((kind, cost), (S.SETTLE, Decimal("0.042")))
        kind, cost, _, _ = S.charge("openai", B, 200, b_raw(60000, 3000))
        self.assertEqual((kind, cost), (S.SETTLE, Decimal("0.15")))
        kind, cost, _, _ = S.charge("anthropic", A, 200, a_raw(inp=60000, out=32000, stop="max_tokens"))
        self.assertEqual((kind, cost), (S.SETTLE, Decimal("0.44")))

    def test_errors_settle_at_zero_and_unknown_usage_holds(self):
        self.assertEqual(S.charge("anthropic", A, 400, b'{"type":"error"}')[:2], (S.SETTLE, Decimal(0)))
        self.assertEqual(S.charge("openai", B, 529, b"")[:2], (S.SETTLE, Decimal(0)))
        self.assertEqual(S.charge("anthropic", A, None, b"")[0], S.HOLD)
        raw = a_raw(write=60000)
        self.assertEqual(S.charge("anthropic", A, 200, raw[:len(raw) // 2])[0], S.HOLD)
        no_usage = sse(openai_events(answer(), usage={"input_tokens": None}))
        self.assertEqual(S.charge("openai", B, 200, no_usage)[0], S.HOLD)
        no_split = sse(anthropic_events(answer(), usage={"cache_creation_input_tokens": 500, "cache_creation": {}}))
        self.assertEqual(S.charge("anthropic", A, 200, no_split)[0], S.HOLD)
        self.assertIn("Anthropic does not bill failed requests; OpenAI's behavior is unverified",
                      S.charge("openai", B, 400, b"")[3])

    def test_an_anthropic_stream_without_final_usage_holds(self):
        kind, cost, _, reason = S.charge("anthropic", A, 200, a_raw(inp=60000, out=3000, final_usage=False))
        self.assertEqual((kind, cost), (S.HOLD, None))
        self.assertIn("message_delta did not report final usage", reason)
        self.assertEqual(S.charge("anthropic", A, 200, a_raw(inp=60000, out=3000))[:2], (S.SETTLE, Decimal("0.15")))


class TestModifiers(unittest.TestCase):
    """A documented modifier settles at the larger of the reservation and its cost under it."""

    r = S.reservation(A, 60000, 32000)

    def charge(self, provider, model, raw, reserved=None):
        return S.charge(provider, model, 200, raw, reserved=self.r if reserved is None else reserved)

    def test_an_a_region_other_than_global(self):
        kind, cost, _, detail = self.charge("anthropic", A, a_raw(inp=60000, out=3000, geo="us"))
        self.assertEqual((kind, cost), (S.SETTLE, self.r))
        self.assertIn("inference_geo us at 1.1x", detail)
        kind, cost, _, _ = self.charge("anthropic", A, a_raw(inp=60000, out=3000, geo="us"), reserved=Decimal(0))
        self.assertEqual(cost, Decimal("0.165"))
        kind, cost, _, detail = self.charge("anthropic", A, a_raw(inp=60000, out=3000, geo="eu"))
        self.assertEqual((kind, cost), (S.HOLD, None))
        self.assertIn("no documented price: inference_geo 'eu'", detail)

    def test_a_service_tier_other_than_standard_or_default(self):
        for tier in ("priority", "batch", None):
            self.assertEqual(self.charge("anthropic", A, a_raw(inp=60000, tier=tier))[0], S.HOLD, tier)
        for tier, plain in (("priority", Decimal("0.30")), ("ultrafast", Decimal("0.90")), ("flex", Decimal("0.075"))):
            kind, cost, _, _ = self.charge("openai", B, b_raw(60000, 3000, tier=tier), reserved=Decimal(0))
            self.assertEqual((kind, cost), (S.SETTLE, plain), tier)
            self.assertEqual(self.charge("openai", B, b_raw(60000, 3000, tier=tier))[1], max(self.r, plain))
        self.assertEqual(self.charge("openai", B, b_raw(60000, 3000, tier="scale"))[0], S.HOLD)
        self.assertEqual(self.charge("openai", B, b_raw(60000, 3000))[1], Decimal("0.15"))    # default: as reported

    def test_b_cache_read_or_write_tokens(self):
        kind, cost, tokens, detail = self.charge("openai", B, b_raw(60000, 3000, cached=50000, written=10000),
                                                 reserved=Decimal(0))
        self.assertEqual(tokens, {"input": 0, "cache_write": 10000, "cache_read": 50000, "output": 3000})
        self.assertEqual(cost, Decimal("0.09"))                  # $0.20 and $5 per million, plus output
        self.assertIn("50000 cached and 10000 cache-write", detail)
        self.assertEqual(self.charge("openai", B, b_raw(60000, 3000, cached=50000, written=10000))[1], self.r)
        _, cost, _, _ = S.charge("openai", "gpt-6.1-sol", 200, b_raw(60000, 3000, cached=60000), reserved=Decimal(0))
        self.assertEqual(cost, Decimal("0.036"))                 # $0.10 per million on the fallback
        self.assertEqual(self.charge("openai", B, b_raw(60000, 3000, cached=60001))[0], S.HOLD)

    def test_a_one_hour_cache_writes(self):
        kind, cost, tokens, detail = self.charge("anthropic", A, a_raw(write=60000, one_hour=60000, out=3000),
                                                 reserved=Decimal(0))
        self.assertEqual((kind, cost), (S.SETTLE, Decimal("0.27")))      # $4 per million, plus output
        self.assertIn("60000 one-hour cache write tokens at $4 per million", detail)
        _, cost, _, _ = self.charge("anthropic", A, a_raw(write=60000, one_hour=30000, out=3000), reserved=Decimal(0))
        self.assertEqual(cost, Decimal("0.225"))
        self.assertEqual(self.charge("anthropic", A, a_raw(write=60000, one_hour=60000, out=3000))[1], self.r)

    def test_b_input_above_272000_tokens(self):
        kind, cost, _, detail = self.charge("openai", B, b_raw(300000, 3000))
        self.assertEqual((kind, cost), (S.SETTLE, Decimal("1.245")))     # 2x input, 1.5x output
        self.assertIn("above 272000", detail)
        self.assertEqual(self.charge("openai", B, b_raw(272000, 3000), reserved=Decimal(0))[1], Decimal("0.574"))

    def test_the_ledger_settles_a_modified_call_at_its_reservation_or_more(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        with S.Ledger(tmp / "ledger-pilot.jsonl", "pilot") as led:
            rid = led.reserve(fields(), self.r)
            entry = led.close_reservation(rid, ({"http_status": 200}, a_raw(inp=60000, out=3000, geo="us")))
            self.assertEqual((entry["type"], Decimal(entry["cost_usd"]), entry["over_reservation"]),
                             ("settle", self.r, False))
            rid = led.reserve(fields("openai", B), Decimal("0.5"))
            entry = led.close_reservation(rid, ({"http_status": 200}, b_raw(300000, 3000)))
            self.assertEqual((Decimal(entry["cost_usd"]), entry["over_reservation"]), (Decimal("1.245"), True))


class TestLedger(LedgerCase):
    def test_header_sequence_and_append_only_lines(self):
        led = self.ledger()
        rid = led.reserve(fields(), Decimal("0.5"))
        led.close_reservation(rid, None)
        rows = self.lines()
        self.assertEqual([r["type"] for r in rows], ["ledger", "reserve", "hold"])
        self.assertEqual([r["seq"] for r in rows], [0, 1, 2])
        self.assertEqual((rows[0]["budget"], rows[0]["cap_usd"], rows[0]["price_table"],
                          rows[0]["pricing"]["reservation"]["estimate_factor"], rows[0]["pricing_sha256"]),
                         ("pilot", "95", S.PRICE_TABLE_VERSION, 2, S.pricing_sha256()))
        self.assertEqual(len(rows[0]["ledger_id"]), 16)
        self.assertEqual(rows[1]["committed_before_usd"], "0")

    def test_admission_at_the_cap_boundary(self):
        led = self.ledger()
        r = S.reservation(A, 60000, 32000)
        led.close_reservation(led.reserve(fields(), CAP - r), None)
        rid = led.reserve(fields(), r)                       # lands exactly on the cap: admitted
        led.close_reservation(rid, None)
        self.assertEqual(led.totals()["committed"], CAP)
        with self.assertRaises(S.SpendHalt) as cm:
            led.reserve(fields(), Decimal("0.000001"))
        self.assertEqual(cm.exception.kind, "spend_cap")
        self.assertEqual(self.lines()[-1]["type"], "refuse")
        self.assertEqual(led.open_reservations(), [])

        other = S.Ledger(self.tmp / "other.jsonl", "pilot")
        self.addCleanup(other.close)
        other.close_reservation(other.reserve(fields(), CAP - r + Decimal("0.000001")), None)
        with self.assertRaises(S.SpendHalt):
            other.reserve(fields(), r)                        # one millionth of a dollar over
        self.assertEqual(other.totals()["committed"], CAP - r + Decimal("0.000001"))

    def test_the_evaluation_cap(self):
        led = S.Ledger(self.tmp / "ledger-evaluation.jsonl", "evaluation")
        self.addCleanup(led.close)
        led.close_reservation(led.reserve(fields(), Decimal("284.5")), None)
        led.close_reservation(led.reserve(fields(), Decimal("0.5")), None)
        with self.assertRaises(S.SpendHalt):
            led.reserve(fields(), Decimal("0.01"))

    def test_one_reservation_open_at_a_time_and_one_process_per_ledger(self):
        led = self.ledger()
        led.reserve(fields(), Decimal("0.5"))
        with self.assertRaises(S.ConcurrentCall):
            led.reserve(fields(), Decimal("0.5"))
        with self.assertRaises(S.LedgerError):
            S.Ledger(self.path, "pilot")

    def test_an_unusable_ledger_is_refused(self):
        self.ledger().close()
        with self.assertRaises(S.LedgerError):
            S.Ledger(self.path, "evaluation")                # another budget's file
        with open(self.path, "ab") as f:
            f.write(b'{"type": "reserve", "seq": 1')        # a torn last line
        with self.assertRaises(S.LedgerError):
            S.Ledger(self.path, "pilot")
        self.path.write_text(json.dumps(dict(self.lines_of_header(), pricing_sha256="0" * 64)) + "\n")
        with self.assertRaises(S.LedgerError):
            S.Ledger(self.path, "pilot")

    def test_a_changed_reservation_constant_refuses_a_resumed_ledger(self):
        self.ledger().close()
        for name, value in (("ESTIMATE_FACTOR", 3), ("WRITE_FACTOR", Decimal("1.2")), ("INPUT_MARGIN", Decimal("1")),
                            ("INPUT_PAD_TOKENS", 0), ("OUTPUT_CAP_RULE", "max_output is 16,000"),
                            ("B_INPUT_LIMIT", 300000)):
            with mock.patch.object(S, name, value):
                with self.assertRaises(S.LedgerError, msg=name):
                    S.Ledger(self.path, "pilot")
        with mock.patch.dict(S.PRICES, {"gpt-9": {"provider": "openai", "input": "1", "cache_write": None,
                                                  "cache_read": None, "output": "1"}}):
            with self.assertRaises(S.LedgerError):
                S.Ledger(self.path, "pilot")
        self.ledger().close()                                # unchanged values open it again

    def lines_of_header(self):
        return json.loads(self.path.read_text().splitlines()[0])

    def test_restart_counts_an_open_reservation_until_it_is_matched(self):
        led = self.ledger()
        r = S.reservation(A, 100000, 32000)
        led.reserve(fields(), r)
        led.close()                                          # the process stops mid-call
        led = self.ledger()
        self.assertEqual(led.totals(), {"settled": 0, "held": 0, "open": r, "committed": r})
        with self.assertRaises(S.ConcurrentCall):
            led.reserve(fields(), Decimal("0.5"))
        closed = led.recover(find=lambda entry: None)
        self.assertEqual([e["type"] for e in closed], ["hold"])
        self.assertEqual(led.totals(), {"settled": 0, "held": r, "open": 0, "committed": r})
        led.close_reservation(led.reserve(fields(), CAP - r), None)
        with self.assertRaises(S.SpendHalt):
            led.reserve(fields(), Decimal("0.000001"))

    def test_restart_settles_from_the_recorded_attempt(self):
        pack = synthetic_pack(1)
        req = R.prepare(pack, images_for(pack))["requests"]["A"]
        log_root = self.tmp / "log"
        config = R.config_sha256()
        led = self.ledger()
        f = fields(trace_key=KEY, config_sha256=config, judge="A", repeat=1, attempt=1,
                   request_sha256=req.sha256, log_root=str(log_root))
        rid = led.reserve(f, S.reservation(A, 60000, 32000))
        led.close()
        AttemptLog(log_root).write_attempt({"trace_key": KEY, "config_sha256": config, "judge": "A", "repeat": 1,
                                            "attempt": 1, "request_sha256": req.sha256, "http_status": 200},
                                           a_raw(write=60000, out=3000))
        led = self.ledger()
        closed = led.recover()
        self.assertEqual([(e["type"], e["id"], Decimal(e["cost_usd"])) for e in closed],
                         [("settle", rid, Decimal("0.18"))])
        self.assertEqual(led.totals()["committed"], Decimal("0.18"))

    def test_a_ledger_moved_with_its_run_folder_still_recovers(self):
        pack = synthetic_pack(1)
        req = R.prepare(pack, images_for(pack))["requests"]["A"]
        config, before, after = R.config_sha256(), self.tmp / "before", self.tmp / "after"
        led = S.Ledger(before / "ledger-pilot.jsonl", "pilot")
        f = fields(trace_key=KEY, config_sha256=config, judge="A", repeat=1, attempt=1, request_sha256=req.sha256,
                   log_root="run/P1/attempt_log")
        led.reserve(f, S.reservation(A, 60000, 32000))
        led.close()
        AttemptLog(before / "run/P1/attempt_log").write_attempt(
            {"trace_key": KEY, "config_sha256": config, "judge": "A", "repeat": 1, "attempt": 1,
             "request_sha256": req.sha256, "http_status": 200}, a_raw(read=60000, out=3000))
        before.rename(after)
        with S.Ledger(after / "ledger-pilot.jsonl", "pilot") as led:
            self.assertEqual([e["type"] for e in led.recover()], ["settle"])
            self.assertEqual(led.totals()["committed"], Decimal("0.042"))


class MeteredCase(LedgerCase):
    def setUp(self):
        super().setUp()
        self.log = AttemptLog(self.tmp / "log")
        pack = synthetic_pack(2)
        self.reqs = R.prepare(pack, images_for(pack))["requests"]
        self.config = R.config_sha256()
        self.inputs = {j: {"request_sha256": self.reqs[j].sha256, "input_tokens": 60000,
                           "input_source": "provider_count", "max_output_tokens": 32000} for j in "AB"}
        self.n = 0

    def ex_a(self, **kw):
        self.n += 1
        return Exchange(200, {"anthropic-workspace-id": WS}, a_raw(msg_id=f"msg_{self.n}", **kw), None, None)

    def ex_b(self):
        self.n += 1
        return Exchange(200, {}, b_raw(resp_id=f"resp_{self.n}"), None, None)

    def dispatch(self, inner, led, key=KEY):
        metered = S.MeteredTransport(inner, led, "run", "P1", self.tmp / "log")
        metered.begin(key, self.config, self.log, R.SLOTS, self.inputs)
        try:
            return D.dispatch_trajectory(key, self.reqs, self.config, metered, self.log, WS, sleep=lambda s: None)
        finally:
            metered.finish()


class TestMetered(MeteredCase):
    def test_each_attempt_settles_from_its_record_and_a_timeout_holds(self):
        led = self.ledger()
        inner = D.FakeTransport([NO_RESPONSE, self.ex_a(write=60000), self.ex_a(read=60000),
                                 a_err(529, "overloaded_error"), self.ex_a(read=60000), self.ex_b()])
        result = self.dispatch(inner, led)
        self.assertFalse(result["halted"])
        rows = [r for r in led.reservations.values()]
        self.assertEqual([(r["slot"], r["attempt"], r["state"]) for r in rows],
                         [("A1", 1, "held"), ("A1", 2, "settled"), ("A2", 1, "settled"), ("A3", 1, "settled"),
                          ("A3", 2, "settled"), ("B1", 1, "settled")])
        self.assertEqual([r["cost"] for r in rows if r["state"] == "settled"],
                         [Decimal("0.18"), Decimal("0.042"), Decimal(0), Decimal("0.042"), Decimal("0.15")])
        self.assertEqual(led.totals()["committed"], Decimal("0.414") + S.reservation(A, 60000, 32000))
        self.assertEqual(len(inner.sent), len(rows))
        self.assertEqual([self.log.attempts(KEY, self.config, "A", 1)[i]["outcome"] for i in (0, 1)],
                         [TRANSPORT, VALID])

    def test_the_reservation_is_on_disk_before_the_request_leaves(self):
        led = self.ledger()
        fsyncs, seen = [], []
        real_fsync = os.fsync

        def counting_fsync(fd):
            fsyncs.append(fd)
            return real_fsync(fd)

        test = self
        ledger_fd = led._f.fileno()

        class Inner:
            def __init__(self, script):
                self.script = list(script)

            def send(self, request):
                last = test.lines()[-1]
                seen.append((last["type"], last["request_sha256"] == request.sha256, fsyncs.count(ledger_fd)))
                return self.script.pop(0)

        with mock.patch.object(S.os, "fsync", counting_fsync):
            self.dispatch(Inner([self.ex_a(write=60000), self.ex_a(read=60000), self.ex_a(read=60000),
                                 self.ex_b()]), led)
        self.assertEqual([s[:2] for s in seen], [("reserve", True)] * 4)
        # Each send follows the fsync of its own reservation (and of the previous call's settlement).
        self.assertEqual([s[2] for s in seen], [1, 3, 5, 7])

    def test_the_cap_halts_before_the_call_that_would_pass_it(self):
        led = self.ledger()
        r = S.reservation(A, 60000, 32000)
        # Room for A1 (settles at $0.18) and A2 ($0.042), not for A3.
        led.close_reservation(led.reserve(fields(), CAP - Decimal("0.18") - r - Decimal("0.01")), None)
        inner = D.FakeTransport([self.ex_a(write=60000), self.ex_a(read=60000), self.ex_a(read=60000), self.ex_b()])
        with self.assertRaises(S.SpendHalt) as cm:
            self.dispatch(inner, led)
        self.assertEqual((cm.exception.kind, cm.exception.entry["slot"]), ("spend_cap", "A3"))
        self.assertEqual(len(inner.sent), 2)
        self.assertEqual(len(self.log.attempts(KEY, self.config, "A", 3)), 0)
        self.assertLessEqual(led.totals()["committed"], CAP)
        self.assertEqual(led.open_reservations(), [])
        self.assertEqual(self.lines()[-1]["type"], "refuse")

    def test_calls_go_out_one_at_a_time(self):
        led = self.ledger()
        test = self

        class Reentrant:
            def send(self, request):
                return test.metered.send(request)

        self.metered = S.MeteredTransport(Reentrant(), led, "run", "P1", self.tmp / "log")
        self.metered.begin(KEY, self.config, self.log, R.SLOTS, self.inputs)
        with self.assertRaises(S.ConcurrentCall):
            self.metered.send(self.reqs["A"])

        entered, release, errors = threading.Event(), threading.Event(), []

        class Slow:
            def send(self, request):
                entered.set()
                release.wait(5)
                return test.ex_a(write=60000)

        led2 = S.Ledger(self.tmp / "two.jsonl", "pilot")
        self.addCleanup(led2.close)
        metered = S.MeteredTransport(Slow(), led2, "run", "P1", self.tmp / "log2")
        metered.begin(KEY, self.config, AttemptLog(self.tmp / "log2"), R.SLOTS, self.inputs)
        t = threading.Thread(target=metered.send, args=(self.reqs["A"],))
        t.start()
        self.assertTrue(entered.wait(5))
        try:
            metered.send(self.reqs["A"])
        except S.ConcurrentCall as e:
            errors.append(e)
        release.set()
        t.join()
        self.assertEqual(len(errors), 1)
        self.assertEqual(len([e for e in led2.entries if e["type"] == "reserve"]), 1)

    def test_a_transport_that_raises_holds_its_reservation(self):
        led = self.ledger()

        class Broken:
            def send(self, request):
                raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            self.dispatch(Broken(), led)
        self.assertEqual([r["state"] for r in led.reservations.values()], ["held"])
        self.assertEqual(self.log.attempts(KEY, self.config, "A", 1), [])

    def test_a_b_call_above_272000_input_tokens_is_refused_before_it_is_reserved(self):
        led = self.ledger()
        self.inputs["B"] = dict(self.inputs["B"], input_tokens=272001)
        inner = D.FakeTransport([self.ex_a(write=60000), self.ex_a(read=60000), self.ex_a(read=60000)])
        with self.assertRaises(S.SpendHalt) as cm:
            self.dispatch(inner, led)
        self.assertEqual((cm.exception.kind, cm.exception.entry["slot"]), ("long_context", "B1"))
        self.assertEqual([r.judge for r in inner.sent], ["A", "A", "A"])
        self.assertEqual([e["slot"] for e in led.entries if e["type"] == "reserve"], ["A1", "A2", "A3"])
        self.inputs["B"] = dict(self.inputs["B"], input_tokens=272000)
        self.assertEqual(led.reserve(dict(fields("openai", B), input_tokens=272000), Decimal("0.5"))[0], "r")

    def test_a_request_must_be_the_one_priced(self):
        led = self.ledger()
        metered = S.MeteredTransport(D.FakeTransport([self.ex_a()]), led, "run", "P1", self.tmp / "log")
        metered.begin(KEY, self.config, self.log, R.SLOTS, dict(self.inputs, A=dict(self.inputs["A"],
                                                                                     request_sha256="0" * 64)))
        with self.assertRaises(ValueError):
            metered.send(self.reqs["A"])
        self.assertEqual(led.reservations, {})


class TestReconcile(LedgerCase):
    def settled(self, led, provider, model, raw):
        rid = led.reserve(fields(provider, model), Decimal("1"))
        led.close_reservation(rid, ({"http_status": 200}, raw))

    def export(self, name, rows):
        path = self.tmp / name
        if name.endswith(".json"):
            path.write_text(json.dumps({"rows": rows}))
        else:
            lines = [",".join(S.EXPORT_FIELDS)] + [",".join(str(r[k]) for k in S.EXPORT_FIELDS) for r in rows]
            path.write_text("\n".join(lines) + "\n")
        return S.read_usage_export(path)

    @staticmethod
    def row(provider, model, inp=0, write=0, read=0, out=0, cost="", start=START, end=END, scope=None):
        return {"provider": provider, "model": model, "input_tokens": inp, "cache_write_tokens": write,
                "cache_read_tokens": read, "output_tokens": out, "cost_usd": cost, "period_start": start,
                "period_end": end, "scope": scope or ("wrkspc_study" if provider == "anthropic" else "proj_study")}

    def test_csv_and_json_exports_read_the_same(self):
        rows = [self.row("anthropic", A, write=60000, out=3000), self.row("openai", B, inp=60000, out=3000, cost="0.2")]
        from_csv, sha_csv = self.export("u.csv", rows)
        from_json, sha_json = self.export("u.json", rows)
        self.assertEqual(from_csv, from_json)
        self.assertNotEqual(sha_csv, sha_json)
        self.assertEqual([S.export_usd(r) for r in from_csv], [Decimal("0.18"), Decimal("0.2")])
        self.assertEqual((from_csv[0]["period_start"], from_csv[0]["scope"]), (START, "wrkspc_study"))

    def test_bad_exports_are_refused(self):
        good = self.row("anthropic", A, out=1)
        for rows in ([dict(good, model="claude-sonnet-5.5")], [dict(good, provider="openai")],
                     [dict(good, output_tokens=-1)], [dict(good, output_tokens="1.5")], [dict(good, cost_usd="-1")],
                     [dict(good, period_start="")], [dict(good, scope=" ")], [dict(good, period_end="soon")],
                     [dict(good, period_end="2026-06-01T00:00:00")], [dict(good, period_end="2025-01-01T00:00:00Z")],
                     [good, dict(good, scope="wrkspc_other")], [good, dict(good, period_start="2026-02-01T00:00:00Z")]):
            with self.assertRaises(ValueError):
                self.export("bad.json", rows)
        (self.tmp / "bad.csv").write_text("provider,model,tokens\n")
        with self.assertRaises(ValueError):
            S.read_usage_export(self.tmp / "bad.csv")
        no_period = ",".join(f for f in S.EXPORT_FIELDS if f not in S.COVERAGE_FIELDS)
        (self.tmp / "bad.csv").write_text(no_period + "\nanthropic,claude-sonnet-5,0,0,0,1,\n")
        with self.assertRaises(ValueError):
            S.read_usage_export(self.tmp / "bad.csv")
        (self.tmp / "bad.txt").write_text("")
        with self.assertRaises(ValueError):
            S.read_usage_export(self.tmp / "bad.txt")

    def test_usage_above_the_ledger_by_more_than_one_percent_halts(self):
        led = self.ledger()
        self.settled(led, "anthropic", A, a_raw(inp=100000, out=0))          # $0.20
        self.settled(led, "openai", B, b_raw(100000, 0))                     # $0.20
        self.assertFalse(led.reconciled_after_last_call())
        for inp, ok in ((100000, True), (101000, True), (101001, False)):
            rows, sha = self.export("u.csv", [self.row("anthropic", A, inp=inp), self.row("openai", B, inp=100000)])
            if not ok:
                break
            self.assertTrue(led.reconcile(rows, sha)["ok"], inp)
        self.assertTrue(led.reconciled_after_last_call())
        entry = led.reconcile(rows, sha)
        self.assertFalse(entry["ok"])
        self.assertEqual(entry["providers"]["anthropic"]["excess_share"], "0.010010")
        self.assertEqual(entry["coverage"], {"anthropic": {"period_start": START, "period_end": END,
                                                           "scope": "wrkspc_study"},
                                             "openai": {"period_start": START, "period_end": END,
                                                        "scope": "proj_study"}})
        self.assertFalse(led.reconciled_after_last_call())
        with self.assertRaises(S.SpendHalt) as cm:
            led.reserve(fields(), Decimal("0.01"))
        self.assertEqual(cm.exception.kind, "reconciliation")
        good, sha = self.export("ok.csv", [self.row("anthropic", A, inp=100000), self.row("openai", B, inp=100000)])
        self.assertTrue(led.reconcile(good, sha)["ok"])
        with self.assertRaises(S.SpendHalt):                                 # a later pass is not a correction
            led.reserve(fields(), Decimal("0.01"))
        with self.assertRaises(ValueError):
            led.record_correction("  ")
        led.record_correction("The export included another project's usage; re-exported the study project.")
        led.reserve(fields(), Decimal("0.01"))
        led.close()
        reopened = self.ledger()
        self.assertIsNone(reopened.blocking_reconciliation())
        self.assertEqual(reopened.adjusted(), Decimal("0.002002"))     # the failed export's anthropic excess

    def test_a_correction_settles_what_the_provider_reported_above_the_ledger(self):
        led = self.ledger()
        self.settled(led, "anthropic", A, a_raw(inp=100000, out=0))          # $0.20
        self.settled(led, "openai", B, b_raw(100000, 0))                     # $0.20
        rows, sha = self.export("u.csv", [self.row("anthropic", A, inp=150000), self.row("openai", B, inp=95000)])
        self.assertFalse(led.reconcile(rows, sha)["ok"])
        real_append = led._append

        def stop_before_the_note(entry):
            if entry["type"] == "correction":
                raise KeyboardInterrupt
            return real_append(entry)

        with mock.patch.object(led, "_append", stop_before_the_note):
            with self.assertRaises(KeyboardInterrupt):
                led.record_correction("Checked the export.")
        led.close()
        led = self.ledger()
        self.assertIsNotNone(led.blocking_reconciliation())
        note = led.record_correction("Checked the export: the usage is the study's.")
        adjust = [e for e in led.entries if e["type"] == "adjust"]
        self.assertEqual([(e["provider"], Decimal(e["cost_usd"]), e["reconcile_seq"]) for e in adjust],
                         [("anthropic", Decimal("0.1"), note["reconcile_seq"])])   # once, and none for openai
        self.assertIsNone(led.blocking_reconciliation())
        self.assertEqual(led.totals(), {"settled": Decimal("0.5"), "held": 0, "open": 0, "committed": Decimal("0.5")})
        self.assertEqual(Decimal(led.summary()["budget_adjusted_usd"]), Decimal("0.1"))
        self.assertTrue(led.reconcile(rows, sha)["ok"])
        led.close()
        led = self.ledger()
        self.assertEqual(led.totals()["committed"], Decimal("0.5"))
        led.close()
        lines = self.path.read_text().splitlines()
        stray = {"type": "adjust", "provider": "openai", "cost_usd": "1", "reconcile_seq": 1, "at": "now",
                 "seq": len(lines)}
        self.path.write_text("\n".join(lines + [json.dumps(stray)]) + "\n")
        with self.assertRaises(S.LedgerError):                               # an adjustment answers a failure only
            S.Ledger(self.path, "pilot")

    def test_an_export_must_cover_the_ledger_from_its_creation_to_its_last_call(self):
        now = [1800000000.0]
        led = S.Ledger(self.path, "pilot", clock=lambda: now[0])
        self.addCleanup(led.close)
        now[0] += 60
        self.settled(led, "anthropic", A, a_raw(inp=100000, out=0))
        last = led.entries[-1]["at"]
        late, _ = self.export("late.csv", [self.row("anthropic", A, inp=100000, start=last)])
        with self.assertRaisesRegex(ValueError, "after the ledger was created"):
            led.reconcile(late, "late")
        early, _ = self.export("early.csv", [self.row("anthropic", A, inp=100000, end=led.created_at)])
        with self.assertRaisesRegex(ValueError, "before the ledger's last anthropic call"):
            led.reconcile(early, "early")
        exact, _ = self.export("exact.csv", [self.row("anthropic", A, inp=100000, start=led.created_at, end=last)])
        self.assertEqual(led.reconcile(exact, "exact")["coverage"]["anthropic"]["period_end"], last)

    def test_an_export_missing_a_provider_is_refused(self):
        led = self.ledger()
        self.settled(led, "anthropic", A, a_raw(inp=1000, out=0))
        self.settled(led, "openai", B, b_raw(1000, 0))
        rows, sha = self.export("u.csv", [self.row("anthropic", A, inp=1000)])
        with self.assertRaises(ValueError):
            led.reconcile(rows, sha)
        rows, sha = self.export("u.csv", [self.row("anthropic", A, inp=1000), self.row("openai", B, inp=1000)])
        self.assertTrue(led.reconcile(rows, sha)["ok"])


class TestDailyReconciliation(LedgerCase):
    def test_the_evaluation_refuses_a_call_while_one_more_than_a_day_old_is_unreconciled(self):
        now = [1800000000.0]
        led = S.Ledger(self.tmp / "ledger-evaluation.jsonl", "evaluation", clock=lambda: now[0])
        self.addCleanup(led.close)
        led.close_reservation(led.reserve(fields(), Decimal("0.5")), ({"http_status": 200}, a_raw(inp=1000, out=0)))
        now[0] += S.DAILY_RECONCILE_S                                  # exactly a day: still admitted
        led.close_reservation(led.reserve(fields(), Decimal("0.5")), None)
        now[0] += 1
        self.assertEqual(led.reconcile_due()["seq"], 1)
        with self.assertRaises(S.SpendHalt) as cm:
            led.reserve(fields(), Decimal("0.5"))
        self.assertEqual((cm.exception.kind, led.entries[-1]["oldest_unreconciled"]), ("reconcile_due", "r000001"))
        rows = [{"provider": "anthropic", "model": A, "tokens": {"input": 1000, "cache_write": 0, "cache_read": 0,
                                                                  "output": 0}, "cost_usd": None,
                 "period_start": START, "period_end": END, "scope": "wrkspc_study"}]
        self.assertTrue(led.reconcile(rows, "daily")["ok"])
        self.assertIsNone(led.reconcile_due())
        led.reserve(fields(), Decimal("0.5"))

    def test_the_pilot_has_no_daily_rule(self):
        now = [1800000000.0]
        led = S.Ledger(self.path, "pilot", clock=lambda: now[0])
        self.addCleanup(led.close)
        led.close_reservation(led.reserve(fields(), Decimal("0.5")), None)
        now[0] += 3 * S.DAILY_RECONCILE_S
        led.reserve(fields(), Decimal("0.5"))


class TestBudgetLedger(unittest.TestCase):
    """One ledger per budget at one fixed place, and no silent second ledger."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.folder = self.tmp / "ledgers"

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def open(self, **kw):
        led = S.open_budget_ledger(self.folder, "pilot", "fake", **kw)
        self.addCleanup(led.close)
        return led

    def register(self):
        return [json.loads(x) for x in (self.folder / S.REGISTER_NAME).read_text().splitlines()]

    def test_one_ledger_is_created_once_and_shared(self):
        with self.assertRaisesRegex(S.LedgerError, "no pilot ledger"):
            S.open_budget_ledger(self.folder, "pilot", "fake")
        led = self.open(create=True)
        first = led.ledger_id
        led.close_reservation(led.reserve(fields(), Decimal("1")), None)
        led.close()
        again = self.open(create=True)
        self.assertEqual((again.ledger_id, again.totals()["committed"], again.mode), (first, Decimal("1"), "fake"))
        self.assertEqual([(e["budget"], e["ledger_id"], e["file"]) for e in self.register()],
                         [("pilot", first, "ledger-pilot.jsonl")])
        again.close()
        with self.assertRaisesRegex(S.LedgerError, "not a paid one"):
            S.open_budget_ledger(self.folder, "pilot", "paid")
        evaluation = S.open_budget_ledger(self.folder, "evaluation", "fake", create=True)
        self.addCleanup(evaluation.close)
        self.assertNotEqual(evaluation.ledger_id, first)

    def test_a_second_ledger_for_a_budget_needs_a_new_budget_note(self):
        led = self.open(create=True)
        first = led.ledger_id
        led.close()
        path = self.folder / "ledger-pilot.jsonl"
        for gone in ("moved", "deleted"):
            if gone == "moved":
                path.rename(self.folder / "ledger-pilot-old.jsonl")
            else:
                (self.folder / "ledger-pilot-old.jsonl").unlink()
            with self.assertRaisesRegex(S.LedgerError, "was created here before"):
                S.open_budget_ledger(self.folder, "pilot", "fake", create=True)
            with self.assertRaisesRegex(S.LedgerError, "was created here before"):
                S.open_budget_ledger(self.folder, "pilot", "fake", create=True, new_budget_note="  ")
            self.assertFalse(path.exists())
        led = self.open(create=True, new_budget_note="A new pilot budget, approved in writing on the study channel.")
        self.assertEqual((led.header["new_budget_note"], led.header["replaces"]),
                         ("A new pilot budget, approved in writing on the study channel.", [first]))
        self.assertEqual([e["ledger_id"] for e in self.register()], [first, led.ledger_id])
        led.close()
        with self.assertRaisesRegex(S.LedgerError, "moved aside"):
            S.open_budget_ledger(self.folder, "pilot", "fake", new_budget_note="Another one.")

    def test_an_unregistered_ledger_in_the_fixed_place_is_refused(self):
        self.open(create=True).close()
        path = self.folder / "ledger-pilot.jsonl"
        path.unlink()
        S.Ledger(path, "pilot", mode="fake").close()                   # made by hand, outside the register
        with self.assertRaisesRegex(S.LedgerError, "not the pilot ledger registered"):
            S.open_budget_ledger(self.folder, "pilot", "fake")


class TestSimulatedPilot(LedgerCase):
    """A pilot of 5, 60 and 40 cases (P2 three passes, P3 four repeats), at extreme token
    counts, against the $95 cap."""

    def test_a_pilot_of_5_60_40_cases_at_extreme_tokens_halts_in_p3_below_the_cap(self):
        led = self.ledger()
        i, o = 100000, 20000
        write = ("anthropic", A, lambda n: a_raw(write=i, out=o, msg_id=n))
        read = ("anthropic", A, lambda n: a_raw(read=i, out=o, msg_id=n))
        plain = ("anthropic", A, lambda n: a_raw(inp=i, out=o, msg_id=n))
        judge_b = ("openai", B, lambda n: b_raw(i, o, resp_id=n))
        plan = ([("P1", t, call) for t in range(5) for call in (write, read, read, judge_b)]
                + [("P2", t, call) for t in range(60) for call in (plain, judge_b)]
                + [("P3", t, call) for t in range(40) for call in (write, read, read, read, read, judge_b)])
        halted = None
        for n, (stage, task, (provider, model, raw)) in enumerate(plan):
            try:
                rid = led.reserve(fields(provider, model, stage=stage), S.reservation(model, i, 32000))
            except S.SpendHalt as e:
                halted = (stage, task, e.entry)
                break
            led.close_reservation(rid, ({"http_status": 200}, raw(f"id_{n}")))
        self.assertIsNotNone(halted)
        stage, task, refused = halted
        self.assertEqual(stage, "P3")
        self.assertIn(task, range(20, 25))
        spent = led.totals()["committed"]
        self.assertLessEqual(spent, CAP)
        self.assertGreater(spent + Decimal(refused["amount_usd"]), CAP)
        self.assertEqual(led.totals()["held"], 0)


if __name__ == "__main__":
    unittest.main()
