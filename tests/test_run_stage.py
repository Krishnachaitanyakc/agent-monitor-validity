"""Tests for the stage runner, scripts/run_stage.py.

The runner's core runs on synthetic packs through the fake transport, with a frozen
rubric stand-in where P3 or EVAL runs. The commands run in fresh processes on the data
path's fixture, since an installed guard cannot be removed; there the rubric is the real
draft. Where paid mode is checked, the HTTPS transport is a stand-in that cannot send, or
the real one with ``urlopen`` replaced by a fake HTTP layer, so nothing leaves.
"""

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
from decimal import Decimal
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_stage as RS  # noqa: E402
from judge import dispatch as D  # noqa: E402
from judge import pack as P  # noqa: E402
from judge import requests as R  # noqa: E402
from judge import schema  # noqa: E402
from judge import spend as S  # noqa: E402
from judge.outcomes import HALT, Exchange, canonical_outcome, error_body_outcome  # noqa: E402
from judge.storage import AttemptLog  # noqa: E402
from judge.streams import sse  # noqa: E402
from test_pack import FixtureCase  # noqa: E402
from test_requests import images_for, synthetic_pack  # noqa: E402

CAP = S.CAPS["pilot"]
KEY_ENVS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")


def items(n, tag):
    """n synthetic items; each item's image entries are its screenshots themselves."""
    out = []
    for i in range(n):
        pack = synthetic_pack(2, goal=f"Find the cheapest mug, case {tag}{i}.")
        out.append(RS.Item(f"t{tag}{i:015d}"[:17], pack, P.sha256_text(P.serialize(pack)), images_for(pack), []))
    return out


def ineligible_item(tag):
    """An item whose 601 images are above judge A's limit, even after the shared fallback."""
    pack = synthetic_pack(600, goal=f"Find the cheapest mug, case {tag}.")
    return RS.Item(f"t{tag}{0:015d}"[:17], pack, P.sha256_text(P.serialize(pack)), images_for(pack), [])


def reserves(led):
    return [e for e in led.entries if e["type"] == "reserve"]


class StandInHttps(D.HttpsTransport):
    """Of the HTTPS transport's type, so paid mode's checks can be reached, but never set up
    and never able to send."""

    def __init__(self, openai_project="p"):
        self.sent, self.openai_project = [], openai_project

    def send(self, request):
        raise AssertionError("no test sends a paid request")


class RunnerCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "run"
        self.ledgers = self.tmp / "ledgers"
        frozen = mock.patch.object(R, "RUBRIC_STATUS", "frozen")      # a frozen rubric stand-in
        frozen.start()
        self.addCleanup(frozen.stop)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def ledger(self, path=None, budget="pilot", clock=time.time, mode="fake"):
        led = S.Ledger(path or RS.ledger_path(self.ledgers, budget), budget, clock, mode=mode)
        self.addCleanup(led.close)
        return led

    def stage(self, stage, its, led, profile="central", events=None, root=None, provider=None, **kw):
        """Run a stage through the fake provider, which adds what it billed to the ledger's
        usage export, as the stage command does."""
        provider = provider or RS.ScriptedProvider(profile, events=events)
        provider.export = RS.export_path(led.path.parent, led.budget, led.ledger_id)
        provider.period_start, provider.clock = led.created_at, kw.get("clock", time.time)
        try:
            report = RS.run_stage(stage, root or self.root, its, mode="fake", transport=provider, ledger=led,
                                  read_images=lambda e: e, workspace=RS.FAKE_WORKSPACE, counter=provider, **kw)
        finally:
            provider.write_export()
        return report, provider

    @staticmethod
    def reconcile(led, *_):
        """Reconcile the ledger with the fake provider's usage export so far."""
        rows, sha = S.read_usage_export(RS.export_path(led.path.parent, led.budget, led.ledger_id))
        return led.reconcile(rows, sha)

    def run_id(self, root=None):
        return json.loads(((root or self.root) / "run.json").read_text())["run_id"]

    def pilot(self, led, sizes=(5, 20, 20), profile="central", root=None):
        reports = {}
        for stage, n in zip(("P1", "P2", "P3"), sizes):
            report, provider = self.stage(stage, items(n, stage[1]), led, profile, root=root)
            reports[stage] = report
            if not report["complete"]:
                break
            self.reconcile(led, provider)
        return reports


class TestPilot(RunnerCase):
    def test_pilot_of_5_20_20_cases_costs_16_35_central_and_48_25_extreme(self):
        for profile, total in (("central", "16.35"), ("extreme", "48.25")):
            self.root, self.ledgers = self.tmp / profile, self.tmp / f"ledgers-{profile}"
            led = self.ledger()
            reports = self.pilot(led, profile=profile)
            self.assertTrue(all(r["complete"] for r in reports.values()), profile)
            self.assertEqual(led.totals(), {"settled": Decimal(total), "held": 0, "open": 0,
                                            "committed": Decimal(total)})
            self.assertLessEqual(led.totals()["committed"], CAP)
            self.assertEqual([len(r["gates"]["spend"]["refusals"]) for r in reports.values()], [0, 0, 0])
        self.assertEqual(reports["P2"]["gates"]["spend"]["stage_spent_usd"], "16.0000000")

    def test_an_extreme_pilot_repeated_on_one_ledger_halts_before_the_cap(self):
        led = self.ledger(self.tmp / "shared.jsonl")
        halted = None
        for k in (1, 2, 3):
            reports = self.pilot(led, profile="extreme", root=self.tmp / f"pilot-{k}")
            last = list(reports.values())[-1]
            if not last["complete"]:
                halted = (k, last)
                break
        self.assertIsNotNone(halted)
        k, report = halted
        self.assertEqual((k, report["stage"], report["stopped"]["kind"]), (2, "P3", "spend_cap"))
        committed = led.totals()["committed"]
        refusal = report["gates"]["spend"]["refusals"][0]
        self.assertLessEqual(committed, CAP)
        self.assertGreater(committed + Decimal(refusal["amount_usd"]), CAP)
        last = reserves(led)[-1]
        self.assertEqual((last["run"], last["run_root"], last["log_root"]),
                         (self.run_id(self.tmp / "pilot-2"), "pilot-2", "P3/attempt_log"))
        sent = AttemptLog(self.tmp / last["run_root"] / last["log_root"]).attempts(
            last["trace_key"], last["config_sha256"], last["judge"], last["repeat"])
        self.assertEqual(len(sent), 1)
        # The refused call was never sent, and a note does not raise the cap.
        log = AttemptLog(self.tmp / "pilot-2" / "P3" / "attempt_log")
        halt = log.halts()[0]
        self.assertEqual((halt["halt_kind"], halt["slot"]), ("spend_cap", refusal["slot"]))
        log.record_correction(halt["halt_id"], "Checked the ledger; the cap stands.", R.config_sha256(), at="now")
        again, provider = self.stage("P3", items(20, "3"), led, "extreme", root=self.tmp / "pilot-2")
        self.assertEqual(again["stopped"]["kind"], "spend_cap")
        self.assertEqual(provider.sent, [])
        self.assertEqual(led.totals()["committed"], committed)

    def test_p1_sends_four_slots_serially_and_p2_sends_a1_and_b1_uncached(self):
        led = self.ledger()
        report, provider = self.stage("P1", items(2, "a"), led)
        self.assertEqual([r.judge for r in provider.sent], ["A", "A", "A", "B"] * 2)
        self.assertEqual(len({r.body for r in provider.sent[:3]}), 1)
        self.assertEqual(report["settings"]["config_sha256"], R.config_sha256(caching=True))
        self.reconcile(led, provider)
        report, provider = self.stage("P2", items(3, "b"), led)
        self.assertEqual([r.judge for r in provider.sent], ["A", "B"] * 3)
        self.assertTrue(all(b'"cache_control"' not in r.body for r in provider.sent if r.judge == "A"))
        self.assertEqual(report["settings"]["config_sha256"], R.config_sha256(caching=False))
        self.assertEqual(list(report["gates"]["invalid_share"]), ["A1", "B1"])
        self.assertIn("not_applicable", report["gates"]["cache_hit_share"])

    def test_the_gates_are_reported(self):
        led = self.ledger()
        its = items(4, "a")
        events = {(its[0].trace_key, "A2", 1): "cache_miss", (its[1].trace_key, "B1", 1): "refusal",
                  (its[2].trace_key, "A3", 1): "output_cap", (its[3].trace_key, "B1", 1): "output_cap",
                  (its[3].trace_key, "A1", 1): "timeout"}
        report, _ = self.stage("P1", its, led, events=events)
        g = report["gates"]
        self.assertTrue(report["complete"])
        self.assertEqual({s: v["invalid"] for s, v in g["invalid_share"].items()}, {"A1": 0, "A2": 0, "A3": 1, "B1": 2})
        self.assertEqual(g["invalid_share"]["B1"]["by_outcome"]["REFUSAL"], {"n": 1, "share": 0.25})
        self.assertEqual(g["invalid_share"]["B1"]["by_outcome"]["TRUNCATED"], {"n": 1, "share": 0.25})
        self.assertEqual((g["cache_hit_share"]["A2"]["hit"], g["cache_hit_share"]["A2"]["miss"]), (3, 1))
        self.assertEqual(g["cache_hit_share"]["A3"]["hit_share"], 1.0)
        self.assertEqual(sorted((c["trace_key"], c["slot"]) for c in g["output_cap_reached"]),
                         [(its[2].trace_key, "A3"), (its[3].trace_key, "B1")])
        self.assertEqual((g["spend"]["held"], g["spend"]["reservations"]), (1, 17))
        self.assertEqual(g["spend"]["cap_usd"], "95")
        self.assertIsNotNone(g["spend"]["projected_stage_usd"])
        self.assertIn("never", g["note"])
        self.assertEqual(g["capability_check_b"]["acted_on"], False)

    def test_refusals_never_trigger_a_fallback(self):
        led = self.ledger()
        its = items(3, "a")
        report, provider = self.stage("P1", its, led, events={(i.trace_key, "B1", 1): "refusal" for i in its})
        cap = report["gates"]["capability_check_b"]
        self.assertEqual((cap["failed"], cap["fallback_indicated"]), ([], False))
        self.assertEqual({r.model for r in provider.sent if r.judge == "B"}, {R.B_MODEL})

    def test_the_b_fallback_is_a_configuration_change(self):
        led = self.ledger()
        its = items(1, "a")
        report, provider = self.stage("P1", its, led, b_model="gpt-6.1-sol")
        self.assertTrue(report["complete"])
        self.assertEqual({r.model for r in provider.sent if r.judge == "B"}, {"gpt-6.1-sol"})
        self.assertEqual(report["settings"]["config_sha256"], R.config_sha256(b_model="gpt-6.1-sol"))
        self.assertNotEqual(report["settings"]["config_sha256"], R.config_sha256())
        self.assertEqual(report["settings"]["b_model"], "gpt-6.1-sol")
        self.assertEqual([e["model"] for e in reserves(led) if e["judge"] == "B"], ["gpt-6.1-sol"])
        self.assertEqual(led.totals()["committed"], Decimal("0.414"))

    def test_a_scripted_halt_stops_the_stage_until_a_note_is_recorded(self):
        led = self.ledger()
        its = items(3, "a")
        report, _ = self.stage("P1", its, led, events={(its[1].trace_key, "A2", 1): "http_400"})
        self.assertEqual((report["complete"], report["stopped"]["kind"]), (False, "request_error"))
        self.assertFalse((self.root / "P1" / "report.json").exists())
        with self.assertRaises(RS.StageRefused):
            self.stage("P1", its, led)
        log = AttemptLog(self.root / "P1" / "attempt_log")
        log.record_correction(report["stopped"]["halt_id"], "Fixed the request.", R.config_sha256(), at="now")
        report, provider = self.stage("P1", its, led)
        self.assertTrue(report["complete"])
        self.assertEqual([r.judge for r in provider.sent], ["A", "A", "B"] + ["A", "A", "A", "B"])
        self.assertEqual(report["gates"]["halts"][0]["correction"], "Fixed the request.")

    def test_an_interrupted_call_stays_counted_and_its_slot_is_sent_again(self):
        led = self.ledger()
        its = items(1, "a")

        class Crash(RS.ScriptedProvider):
            def send(self, request):
                if len(self.sent) == 1:
                    self.sent.append(request)
                    raise KeyboardInterrupt
                return super().send(request)

        with self.assertRaises(KeyboardInterrupt):
            self.stage("P1", its, led, provider=Crash("central"))
        r = S.reservation(R.A_MODEL, 60000, 32000)
        self.assertEqual(led.totals()["held"], r)
        led.close()
        led = self.ledger()
        report, provider = self.stage("P1", its, led)
        self.assertTrue(report["complete"])
        a2 = [e for e in reserves(led) if e["slot"] == "A2"]
        self.assertEqual([(e["attempt"], led.reservations[e["id"]]["state"]) for e in a2],
                         [(1, "held"), (1, "settled")])
        self.assertEqual(led.totals()["held"], r)

    def test_an_ineligible_pilot_case_is_recorded_and_the_stage_goes_on(self):
        led = self.ledger()
        bad = ineligible_item("x")
        report, provider = self.stage("P1", [bad, *items(1, "a")], led)
        self.assertTrue(report["complete"], report["stopped"])
        self.assertEqual(report["cases"], {"planned": 5, "given": 2, "ended": 2, "ineligible": [bad.trace_key]})
        self.assertEqual([r.judge for r in provider.sent], ["A", "A", "A", "B"])
        self.assertEqual(report["gates"]["halts"], [])

    def test_a_call_that_costs_more_than_its_reservation_halts_once_its_trajectory_finishes(self):
        led = self.ledger()

        class Low(RS.ScriptedProvider):
            def count(self, request):
                return 10

        report, provider = self.stage("P1", items(2, "a"), led, provider=Low("extreme"))
        self.assertEqual(report["stopped"]["kind"], "over_reservation")
        self.assertTrue(report["gates"]["spend"]["over_reservation"])
        self.assertEqual([r.judge for r in provider.sent], ["A", "A", "A", "B"])     # the whole first trajectory
        self.assertEqual(led.totals()["committed"], Decimal("1.29"))                 # actual settled spend


class Low(RS.ScriptedProvider):
    """A provider whose pre-send count is far below what it bills."""

    def count(self, request):
        return 10


class TestOverReservationAndLongContext(RunnerCase):
    def test_an_over_reservation_halt_lost_in_a_stop_is_recorded_at_start(self):
        led = self.ledger()
        its = items(2, "a")
        real = AttemptLog.write_halt

        def stop(log, record):
            if record["halt_kind"] == "over_reservation":
                raise KeyboardInterrupt
            return real(log, record)

        with mock.patch.object(AttemptLog, "write_halt", stop):
            with self.assertRaises(KeyboardInterrupt):
                self.stage("P1", its, led, provider=Low("extreme"))
        log = AttemptLog(self.root / "P1" / "attempt_log")
        self.assertEqual(log.halts(), [])
        provider = Low("extreme")
        with self.assertRaisesRegex(RS.StageRefused, "halts without a correction note"):
            self.stage("P1", its, led, provider=provider)
        self.assertEqual(provider.sent, [])
        halts = log.halts()
        self.assertEqual([(h["halt_kind"], h["recorded_at_start"], len(h["reservations"])) for h in halts],
                         [("over_reservation", True, 2)])               # A1 and B1
        with self.assertRaisesRegex(RS.StageRefused, "halts without a correction note"):
            self.stage("P1", its, led, provider=provider)                    # recorded once
        self.assertEqual(len(log.halts()), 1)

    def test_an_over_reservation_is_halted_even_when_another_halt_ends_the_trajectory(self):
        led = self.ledger()
        r = S.reservation(R.A_MODEL, 10, 32000)
        filler = CAP - Decimal("0.45") - r + Decimal("0.001")              # room for A1, not for A2 after it
        led.close_reservation(led.reserve({"provider": "anthropic", "model": R.A_MODEL, "run": "earlier",
                                           "stage": "P1"}, filler), None)
        report, provider = self.stage("P1", items(1, "a"), led, provider=Low("extreme"))
        self.assertEqual((report["stopped"]["kind"], report["stopped"]["slot"]), ("spend_cap", "A2"))
        self.assertEqual([r.judge for r in provider.sent], ["A"])
        self.assertEqual(sorted(h["kind"] for h in report["gates"]["halts"]), ["over_reservation", "spend_cap"])

        led2 = self.ledger(self.tmp / "other" / "ledger-pilot.jsonl")
        its = items(1, "b")
        report, _ = self.stage("P1", its, led2, root=self.tmp / "run-2",
                               provider=Low("extreme", events={(its[0].trace_key, "B1", 1): "http_400"}))
        self.assertEqual((report["stopped"]["kind"], report["stopped"]["slot"]), ("request_error", "B1"))
        self.assertEqual(sorted(h["kind"] for h in report["gates"]["halts"]), ["over_reservation", "request_error"])

    def test_a_b_error_response_stays_held_until_a_passing_reconciliation_releases_it(self):
        led = self.ledger()
        its = items(1, "a")
        report, _ = self.stage("P1", its, led, events={(its[0].trace_key, "B1", 1): "http_400"})
        self.assertEqual((report["stopped"]["kind"], report["stopped"]["slot"]), ("request_error", "B1"))
        b1 = next(e for e in reserves(led) if e["slot"] == "B1")
        r = S.reservation(R.B_MODEL, 60000, 32000)
        self.assertEqual((led.reservations[b1["id"]]["state"], led.totals()["held"]), ("held", r))
        self.assertEqual(report["gates"]["spend"]["held"], 1)
        committed = led.totals()["committed"]
        entry = self.reconcile(led)                                 # the fake provider billed nothing for it
        self.assertTrue(entry["ok"])
        self.assertEqual((led.entries[-1]["type"], led.entries[-1]["id"], led.entries[-1]["reconcile_seq"]),
                         ("release", b1["id"], entry["seq"]))
        self.assertEqual((led.reservations[b1["id"]]["state"], led.totals()["held"]), ("settled", 0))
        self.assertEqual(led.totals()["committed"], committed - r)
        self.assertTrue(led.reconciled_after_last_call())

    def test_an_error_in_a_b_stream_stays_held_until_a_passing_reconciliation_releases_it(self):
        class InStream(RS.ScriptedProvider):
            def _exchange(self, request, stage, ref, kind):
                if ref.endswith("_B1_1"):
                    return Exchange(200, {}, sse([("response.created", {"type": "response.created", "response": {
                        "id": f"resp_{ref}", "model": request.model}}), ("error", {
                            "type": "error", "code": "server_error", "message": "x"})]), None, None)
                return super()._exchange(request, stage, ref, kind)

        led = self.ledger()
        report, _ = self.stage("P1", items(1, "a"), led, provider=InStream("central"))
        self.assertTrue(report["complete"], report["stopped"])
        b1 = [e for e in reserves(led) if e["slot"] == "B1"]
        self.assertEqual([led.reservations[e["id"]]["state"] for e in b1], ["held", "settled"])
        self.assertEqual([e["until_reconciled"] for e in led.entries if e["type"] == "hold"], [True])
        held = led.totals()["held"]
        committed = led.totals()["committed"]
        entry = self.reconcile(led)                                 # the fake provider billed nothing for it
        self.assertTrue(entry["ok"])
        self.assertEqual((led.entries[-1]["type"], led.entries[-1]["id"], led.entries[-1]["reconcile_seq"]),
                         ("release", b1[0]["id"], entry["seq"]))
        self.assertEqual((led.totals()["held"], led.totals()["committed"]), (0, committed - held))

    def test_a_trajectory_whose_b_count_is_above_272000_tokens_is_refused_before_anything_is_sent(self):
        led = self.ledger()

        class Long(RS.ScriptedProvider):
            def count(self, request):
                return 300000 if request.judge == "B" else 60000

        its = items(2, "a")
        report, provider = self.stage("P1", its, led, provider=Long("central"))
        self.assertEqual((report["stopped"]["kind"], report["stopped"]["slot"], report["stopped"]["trace_key"]),
                         ("long_context", "B1", its[0].trace_key))
        self.assertEqual((provider.sent, reserves(led)), ([], []))
        log = AttemptLog(self.root / "P1" / "attempt_log")
        log.record_correction(report["stopped"]["halt_id"], "Checked the count.", R.config_sha256(), at="now")
        report, provider = self.stage("P1", its, led, provider=Long("central"))
        self.assertEqual((report["stopped"]["kind"], provider.sent), ("long_context", []))


class TestGating(RunnerCase):
    def test_each_stage_needs_the_previous_report_and_a_reconciliation(self):
        led = self.ledger()
        with self.assertRaisesRegex(RS.StageRefused, "P1's report"):
            self.stage("P2", items(1, "b"), led)
        _, provider = self.stage("P1", items(1, "a"), led)
        with self.assertRaisesRegex(RS.StageRefused, "reconcile"):
            self.stage("P2", items(1, "b"), led)
        self.reconcile(led, provider)
        with self.assertRaisesRegex(RS.StageRefused, "P2's report"):
            self.stage("P3", items(1, "c"), led)
        with self.assertRaisesRegex(RS.StageRefused, "already complete"):
            self.stage("P1", items(1, "a"), led)
        with self.assertRaisesRegex(RS.StageRefused, "1 to 20"):
            self.stage("P2", items(21, "b"), led)
        report, _ = self.stage("P2", items(1, "b"), led)
        self.assertTrue(report["complete"])
        self.assertEqual(report["previous_report"], str(self.root / "P1" / "report.json"))

    def test_a_run_keeps_one_mode_and_a_resumed_stage_its_inputs_and_workspace(self):
        led = self.ledger()
        its = items(2, "a")
        report, _ = self.stage("P1", its, led, events={(its[1].trace_key, "A1", 1): "http_400"})
        log = AttemptLog(self.root / "P1" / "attempt_log")
        log.record_correction(report["stopped"]["halt_id"], "Noted.", R.config_sha256(), at="now")
        with self.assertRaisesRegex(RS.StageRefused, "same ones"):
            self.stage("P1", its[:1], led)
        paid, https = self.ledger(self.tmp / "paid" / "ledger-pilot.jsonl", mode="paid"), StandInHttps()
        with self.assertRaisesRegex(RS.StageRefused, "same ones"):
            RS.run_stage("P1", self.root, its, mode="paid", transport=https, ledger=paid,
                         read_images=lambda e: e, workspace="w", openai_project="p",
                         counter=D.HttpsTokenCounter(https, "w"))
        provider = RS.ScriptedProvider()
        with self.assertRaisesRegex(RS.StageRefused, "same ones"):
            RS.run_stage("P1", self.root, its, mode="fake", transport=provider, ledger=led, read_images=lambda e: e,
                         workspace="wrkspc_another", counter=provider)
        self.assertEqual(provider.sent, [])
        record = json.loads((self.root / "P1" / "stage.json").read_text())
        self.assertEqual((record["workspace"], record["ledger"]), (RS.FAKE_WORKSPACE, {
            "budget": "pilot", "file": "ledger-pilot.jsonl", "id": led.ledger_id}))
        self.assertEqual(report["settings"]["workspace"], RS.FAKE_WORKSPACE)
        with self.assertRaises(ValueError):
            RS.run_stage("EVAL", self.root, its, mode="fake", transport=D.FakeTransport(), ledger=led,
                         read_images=lambda e: e, workspace="w")

    def test_every_stage_of_a_run_keeps_the_workspace_and_project_of_its_first(self):
        led = self.ledger()
        report, provider = self.stage("P1", items(1, "a"), led)
        self.reconcile(led)
        record = json.loads((self.root / "run.json").read_text())
        self.assertEqual((record["workspace"], record["openai_project"]), (RS.FAKE_WORKSPACE, RS.FAKE_PROJECT))
        for scope in ({"workspace": "wrkspc_other"}, {"workspace": RS.FAKE_WORKSPACE, "openai_project": "proj_other"}):
            provider = RS.ScriptedProvider()
            kw = dict({"workspace": RS.FAKE_WORKSPACE}, **scope)
            with self.assertRaisesRegex(RS.StageRefused, "every stage of a run uses the same ones"):
                RS.run_stage("P2", self.root, items(1, "b"), mode="fake", transport=provider, ledger=led,
                             read_images=lambda e: e, counter=provider, **kw)
            self.assertEqual(provider.sent, [])
        self.assertFalse((self.root / "P2").exists())
        self.assertEqual({e["scope"] for e in reserves(led)}, {RS.FAKE_WORKSPACE, RS.FAKE_PROJECT})
        report, _ = self.stage("P2", items(1, "b"), led)
        self.assertEqual((report["complete"], report["settings"]["openai_project"]), (True, RS.FAKE_PROJECT))

    def test_a_response_from_another_openai_project_halts_the_stage(self):
        led = self.ledger()
        its = items(1, "a")
        report, _ = self.stage("P1", its, led, provider=RS.ScriptedProvider(project="proj_other"))
        self.assertEqual((report["stopped"]["kind"], report["stopped"]["slot"]), ("response_check", "B1"))
        self.assertFalse(report["complete"])
        rec = AttemptLog(self.root / "P1" / "attempt_log").attempts(its[0].trace_key, R.config_sha256(), "B", 1)[0]
        self.assertEqual(rec["response_headers"]["openai-project"], "proj_other")
        self.assertEqual(rec["failed_checks"], ["the openai-project header is not the study project"])
        other = self.ledger(self.tmp / "other" / "ledger-pilot.jsonl")
        report, _ = self.stage("P1", items(1, "b"), other, root=self.tmp / "run-2")
        self.assertTrue(report["complete"], report["stopped"])       # the same project passes

    def test_a_second_run_root_under_other_scopes_on_the_same_ledger_is_refused_before_it_starts(self):
        led = self.ledger()
        self.stage("P1", items(1, "a"), led)
        self.reconcile(led)
        second, made = self.tmp / "run-2", len(led.entries)
        for scope, flag in (({"workspace": "wrkspc_other"}, "--workspace"),
                            ({"openai_project": "proj_other"}, "--openai-project")):
            provider = RS.ScriptedProvider()
            kw = dict({"workspace": RS.FAKE_WORKSPACE}, **scope)
            with self.assertRaisesRegex(RS.StageRefused, f"ran under \\[.+\\], not .+; .+ fix {flag}"):
                RS.run_stage("P1", second, items(1, "b"), mode="fake", transport=provider, ledger=led,
                             read_images=lambda e: e, counter=provider, **kw)
            self.assertEqual(provider.sent, [])
        self.assertFalse((second / "run.json").exists())
        self.assertEqual(len(led.entries), made)
        report, _ = self.stage("P1", items(1, "b"), led, root=second)
        self.assertTrue(report["complete"])
        self.assertTrue(self.reconcile(led)["ok"])

    def test_the_transport_and_the_ledger_must_match_the_mode(self):
        its = items(1, "a")
        provider = RS.ScriptedProvider()

        class Plain:
            sent = []

            def send(self, request):
                raise AssertionError("never sent")

        cases = [("fake", Plain(), self.ledger(), "fake mode runs only a fake transport"),
                 ("paid", provider, self.ledger(self.tmp / "p" / "ledger-pilot.jsonl", mode="paid"),
                  "paid mode runs only the HTTPS transport"),
                 ("fake", provider, self.ledger(self.tmp / "q" / "ledger-pilot.jsonl", mode="paid"), "a paid ledger"),
                 ("fake", provider, self.ledger(self.tmp / "r" / "ledger-pilot.jsonl", mode=None), "a None ledger")]
        for mode, transport, led, message in cases:
            with self.assertRaisesRegex(ValueError, message):
                RS.run_stage("P1", self.root, its, mode=mode, transport=transport, ledger=led,
                             read_images=lambda e: e, workspace="w", openai_project="p", counter=provider)
        self.assertEqual(provider.sent, [])
        self.assertFalse(self.root.exists())

    def test_p3_and_eval_refuse_a_rubric_that_is_not_frozen(self):
        with mock.patch.object(R, "RUBRIC_STATUS", "draft"):
            led = self.ledger()
            for stage in ("P1", "P2"):
                report, provider = self.stage(stage, items(1, stage[1]), led)
                self.assertTrue(report["complete"])
                self.reconcile(led, provider)
            with self.assertRaisesRegex(RS.StageRefused, "frozen rubric"):
                self.stage("P3", items(1, "3"), led)
            self.assertFalse((self.root / "P3").exists())
            with self.assertRaisesRegex(RS.StageRefused, "frozen rubric"):
                self.stage("EVAL", items(1, "e"), self.ledger(budget="evaluation"))
            paid = self.ledger(self.tmp / "paid" / "ledger-pilot.jsonl", mode="paid")
            with self.assertRaisesRegex(ValueError, "waived only with the fake transport"):
                RS.run_stage("P3", self.root, items(1, "3"), mode="paid", transport=StandInHttps(), ledger=paid,
                             read_images=lambda e: e, workspace="w", openai_project="p", counter=RS.ScriptedProvider(),
                             rubric_check=False)
            report, _ = self.stage("P3", items(1, "3"), led, rubric_check=False)
            self.assertTrue(report["complete"])
            self.assertEqual((report["settings"]["rubric_status"], report["settings"]["rubric_check"]),
                             ("draft", "waived in a fake-mode dry run"))


class TestRunRecords(RunnerCase):
    def test_a_model_change_or_a_repeated_response_id_after_p1_halts_a_later_stage(self):
        led = self.ledger()
        report, provider = self.stage("P1", items(1, "a"), led)
        self.reconcile(led, provider)

        class Renamed(RS.ScriptedProvider):
            def _exchange(self, request, stage, ref, kind):
                ex = super()._exchange(request, stage, ref, kind)
                return ex._replace(raw=ex.raw.replace(b'"model": "claude-sonnet-5"', b'"model": "claude-sonnet-5-1"'))

        report, _ = self.stage("P2", items(1, "b"), led, provider=Renamed("central"))
        self.assertEqual((report["stopped"]["kind"], report["stopped"]["slot"]), ("response_check", "A1"))
        halt = AttemptLog(self.root / "P2" / "attempt_log").halts()[0]
        self.assertIn("is not the first recorded 'claude-sonnet-5'", halt["failed_checks"][0])

        class Reused(RS.ScriptedProvider):
            def _exchange(self, request, stage, ref, kind):
                return super()._exchange(request, stage, ref.replace("P2_", "P1_", 1), kind)

        root, its = self.tmp / "run-2", items(1, "c")
        report, provider = self.stage("P1", its, led, root=root)
        self.reconcile(led, provider)
        report, _ = self.stage("P2", its, led, root=root, provider=Reused("central"))
        self.assertEqual((report["stopped"]["kind"], report["stopped"]["slot"]), ("response_check", "A1"))
        halt = AttemptLog(root / "P2" / "attempt_log").halts()[0]
        self.assertIn("was seen before", halt["failed_checks"][0])

    def stop_between_writes(self, slot, cut=False):
        """complete_attempt that, for ``slot``, leaves the raw response (or half of it) on disk
        and stops before the record."""
        real = AttemptLog.complete_attempt

        def write(log, record):
            if record["slot"] == slot:
                folder = log.slot_dir(record["trace_key"], record["config_sha256"], record["judge"], record["repeat"])
                raw = folder / f"{record['attempt']}.raw"
                if cut:
                    raw.write_bytes(raw.read_bytes()[:raw.stat().st_size // 2])
                raise KeyboardInterrupt
            return real(log, record)
        return mock.patch.object(AttemptLog, "complete_attempt", write)

    def a1(self, key, root=None):
        return AttemptLog((root or self.root) / "P1" / "attempt_log").attempts(key, R.config_sha256(), "A", 1)

    def test_a_parser_exception_keeps_the_response_and_a1_is_not_sent_again(self):
        led = self.ledger()
        its = items(2, "a")
        key = its[0].trace_key
        with mock.patch.object(schema, "parse", side_effect=RuntimeError("the answer was: quoted text")):
            report, first = self.stage("P1", its, led)
        self.assertEqual((report["complete"], report["stopped"]["kind"], report["stopped"]["slot"]),
                         (False, "classification_error", "A1"))
        self.assertEqual([r.judge for r in first.sent], ["A"])
        records = self.a1(key)
        self.assertEqual([(r["attempt"], r["outcome"], r["halt_kind"], r["detail"]) for r in records],
                         [(1, "MALFORMED", "classification_error", "classification raised RuntimeError")])
        folder = AttemptLog(self.root / "P1" / "attempt_log").slot_dir(key, R.config_sha256(), "A", 1)
        self.assertEqual(R.sha256_bytes((folder / "1.raw").read_bytes()), records[0]["raw_sha256"])
        self.assertTrue((folder / "1.raw").stat().st_size > 0)
        provider = RS.ScriptedProvider()
        with self.assertRaisesRegex(RS.StageRefused, "halts without a correction note"):
            self.stage("P1", its, led, provider=provider)
        self.assertEqual(provider.sent, [])
        log = AttemptLog(self.root / "P1" / "attempt_log")
        self.assertIn("never sent again", log.halts()[0]["resume_rule"])
        log.record_correction(report["stopped"]["halt_id"], "Read the raw response.", R.config_sha256(), at="now")
        report, second = self.stage("P1", its, led)
        self.assertTrue(report["complete"], report["stopped"])
        self.assertEqual([r.judge for r in second.sent], ["A", "A", "B"] + ["A", "A", "A", "B"])    # no second A1
        self.assertEqual([e["attempt"] for e in reserves(led) if e["slot"] == "A1" and e["trace_key"] == key], [1])
        self.assertEqual(len(self.a1(key)), 1)
        self.assertEqual(report["gates"]["invalid_share"]["A1"]["by_outcome"]["MALFORMED"]["n"], 1)
        a1 = next(e for e in reserves(led) if e["slot"] == "A1" and e["trace_key"] == key)
        self.assertEqual(led.reservations[a1["id"]]["state"], "settled")                  # its usage could be read

    def test_bytes_that_stop_the_classifier_and_the_pricing_hold_the_reservation(self):
        class Unreadable(RS.ScriptedProvider):
            def _exchange(self, request, stage, ref, kind):
                if ref.endswith("_A1_1"):
                    raw = sse([("message_start", {"type": "message_start", "message": 5})])
                    return Exchange(200, {}, raw, None, None)
                return super()._exchange(request, stage, ref, kind)

        led = self.ledger()
        its = items(1, "a")
        report, first = self.stage("P1", its, led, provider=Unreadable("central"))
        self.assertEqual(report["stopped"]["kind"], "classification_error")
        self.assertEqual(len(first.sent), 1)
        self.assertEqual([(r["outcome"], r["http_status"], r["detail"]) for r in self.a1(its[0].trace_key)],
                         [("MALFORMED", 200, "classification raised AttributeError")])
        a1 = next(e for e in reserves(led) if e["slot"] == "A1")
        self.assertEqual(led.reservations[a1["id"]]["state"], "held")
        self.assertEqual([e["reason"] for e in led.entries if e["type"] == "hold" and e["id"] == a1["id"]],
                         ["pricing the response raised AttributeError, so its usage is unknown"])
        AttemptLog(self.root / "P1" / "attempt_log").record_correction(
            report["stopped"]["halt_id"], "Read the raw response.", R.config_sha256(), at="now")
        report, second = self.stage("P1", its, led)
        self.assertTrue(report["complete"], report["stopped"])
        self.assertEqual([r.judge for r in second.sent], ["A", "A", "B"])

    def test_a_stop_while_a_response_is_classified_loses_nothing(self):
        led = self.ledger()
        its = items(1, "a")
        with mock.patch.object(schema, "parse", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.stage("P1", its, led)
        self.assertEqual(led.open_reservations()[0]["slot"], "A1")               # left for the rebuild
        report, provider = self.stage("P1", its, led)
        self.assertTrue(report["complete"], report["stopped"])
        self.assertEqual([r.judge for r in provider.sent], ["A", "A", "B"])
        self.assertEqual([(r["attempt"], r["outcome"]) for r in self.a1(its[0].trace_key)], [(1, "VALID")])
        self.assertEqual(len([e for e in reserves(led) if e["slot"] == "A1"]), 1)

    def test_a_parser_exception_during_the_rebuild_is_recorded_and_a1_is_not_sent_again(self):
        led = self.ledger()
        its = items(1, "a")
        with self.stop_between_writes("A1"):
            with self.assertRaises(KeyboardInterrupt):
                self.stage("P1", its, led)
        provider = RS.ScriptedProvider()
        with mock.patch.object(schema, "parse", side_effect=RuntimeError("defect")):
            with self.assertRaisesRegex(RS.StageRefused, "halts without a correction note"):
                self.stage("P1", its, led, provider=provider)
        self.assertEqual(provider.sent, [])
        rec = self.a1(its[0].trace_key)[0]
        self.assertEqual((rec["outcome"], rec["halt_kind"], rec["http_status"], rec["recovered"]["http_status"]),
                         ("MALFORMED", "classification_error", 200, "inferred from the stream"))
        log = AttemptLog(self.root / "P1" / "attempt_log")
        halts = log.halts()
        self.assertEqual([(h["slot"], h["halt_kind"], h["recorded_at_start"]) for h in halts],
                         [("A1", "classification_error", True)])
        log.record_correction(halts[0]["halt_id"], "Read the raw response.", R.config_sha256(), at="now")
        report, provider = self.stage("P1", its, led)
        self.assertTrue(report["complete"], report["stopped"])
        self.assertEqual([r.judge for r in provider.sent], ["A", "A", "B"])
        self.assertEqual(len(self.a1(its[0].trace_key)), 1)

    def test_a_rebuild_that_cannot_tell_the_status_says_so(self):
        led = self.ledger()
        its = items(1, "a")
        with self.stop_between_writes("A1"):
            with self.assertRaises(KeyboardInterrupt):
                self.stage("P1", its, led)
        provider = RS.ScriptedProvider()
        with mock.patch.object(D, "_stream_started", side_effect=RuntimeError("defect")):
            with self.assertRaisesRegex(RS.StageRefused, "halts without a correction note"):
                self.stage("P1", its, led, provider=provider)
        rec = self.a1(its[0].trace_key)[0]
        self.assertEqual((rec["outcome"], rec["http_status"], rec["recovered"]["http_status"]),
                         ("MALFORMED", None, "unknown; its classification raised"))

    def test_a_classification_halt_lost_in_a_stop_is_recorded_at_start(self):
        led = self.ledger()
        its = items(1, "a")
        with mock.patch.object(schema, "parse", side_effect=RuntimeError("defect")):
            with mock.patch.object(AttemptLog, "write_halt", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    self.stage("P1", its, led)
        log = AttemptLog(self.root / "P1" / "attempt_log")
        self.assertEqual(log.halts(), [])
        provider = RS.ScriptedProvider()
        with self.assertRaisesRegex(RS.StageRefused, "halts without a correction note"):
            self.stage("P1", its, led, provider=provider)
        self.assertEqual(provider.sent, [])
        halt = log.halts()[0]
        self.assertEqual((halt["slot"], halt["halt_kind"], halt["recorded_at_start"]),
                         ("A1", "classification_error", True))
        self.assertIn("never sent again", halt["resume_rule"])

    def test_a_raw_response_without_its_record_is_rebuilt_and_never_sent_again(self):
        for slot in ("A1", "A2"):
            self.root = self.tmp / f"run-{slot}"
            led = self.ledger(self.tmp / f"ledgers-{slot}" / "ledger-pilot.jsonl")
            its = items(1, "a")
            with self.stop_between_writes(slot):
                with self.assertRaises(KeyboardInterrupt):
                    self.stage("P1", its, led)
            made = len(reserves(led))
            self.assertEqual(led.open_reservations()[0]["slot"], slot)
            report, provider = self.stage("P1", its, led)
            self.assertTrue(report["complete"], report["stopped"])
            rest = ["A", "A", "A", "B"][int(slot[1]):]
            self.assertEqual([r.judge for r in provider.sent], rest)          # the stopped slot is not resent
            self.assertEqual(len(reserves(led)), made + len(rest))
            log = AttemptLog(self.root / "P1" / "attempt_log")
            records = log.attempts(its[0].trace_key, R.config_sha256(), "A", int(slot[1]))
            self.assertEqual(len(records), 1)
            rec = records[0]
            self.assertEqual((rec["outcome"], rec["http_status"], rec["recovered"]["http_status"]),
                             ("VALID", 200, "inferred from the stream"))
            self.assertEqual(rec["response_id"], f"msg_P1_{its[0].trace_key}_{slot}_1")
            self.assertEqual(rec["recovered"]["checks_not_applied"], ["anthropic-workspace-id header"])
            stopped = next(e for e in reserves(led) if e["slot"] == slot)
            self.assertEqual(led.reservations[stopped["id"]]["state"], "settled")
            self.assertEqual(report["attempts_rebuilt_at_start"],
                             [{"trace_key": its[0].trace_key, "slot": slot, "attempt": 1, "outcome": "VALID",
                               "http_status": 200}])
            self.assertEqual(led.totals()["held"], 0)

    def test_a_cut_raw_response_without_its_record_is_a_transport_failure_and_is_retried(self):
        led = self.ledger()
        its = items(1, "a")
        with self.stop_between_writes("A2", cut=True):
            with self.assertRaises(KeyboardInterrupt):
                self.stage("P1", its, led)
        report, provider = self.stage("P1", its, led)
        self.assertTrue(report["complete"], report["stopped"])
        self.assertEqual([r.judge for r in provider.sent], ["A", "A", "A", "B"][1:])
        records = AttemptLog(self.root / "P1" / "attempt_log").attempts(its[0].trace_key, R.config_sha256(), "A", 2)
        self.assertEqual([(r["outcome"], r["http_status"]) for r in records], [("TRANSPORT", 200), ("VALID", 200)])
        a2 = [e for e in reserves(led) if e["slot"] == "A2"]
        self.assertEqual([led.reservations[e["id"]]["state"] for e in a2], ["held", "settled"])

    def test_a_raw_error_body_without_its_record_halts_as_the_dispatcher_would_and_is_not_resent(self):
        led = self.ledger()
        its = items(1, "a")
        with self.stop_between_writes("A1"):
            with self.assertRaises(KeyboardInterrupt):
                self.stage("P1", its, led, events={(its[0].trace_key, "A1", 1): "http_400"})
        provider = RS.ScriptedProvider()
        with self.assertRaisesRegex(RS.StageRefused, "halts without a correction note"):
            self.stage("P1", its, led, provider=provider)
        self.assertEqual(provider.sent, [])
        log = AttemptLog(self.root / "P1" / "attempt_log")
        rec = log.attempts(its[0].trace_key, R.config_sha256(), "A", 1)[0]
        self.assertEqual((rec["outcome"], rec["halt_kind"], rec["http_status"], rec["recovered"]["http_status"]),
                         ("HALT", "request_error", None, "unknown; an error body, classified by its error fields"))
        a1 = next(e for e in reserves(led) if e["slot"] == "A1")
        self.assertEqual(led.reservations[a1["id"]]["state"], "held")              # the status was not kept
        self.assertEqual([e["reason"] for e in led.entries if e["type"] == "hold" and e["id"] == a1["id"]],
                         ["an error body whose status was not kept; held, since the provider may still bill the "
                          "request"])
        halts = log.halts()
        self.assertEqual([(h["slot"], h["attempt"], h["halt_kind"], h["recorded_at_start"]) for h in halts],
                         [("A1", 1, "request_error", True)])
        log.record_correction(halts[0]["halt_id"], "Fixed the request.", R.config_sha256(), at="now")
        report, provider = self.stage("P1", its, led)
        self.assertTrue(report["complete"], report["stopped"])
        self.assertEqual([r.judge for r in provider.sent], ["A", "A", "A", "B"])
        self.assertEqual([r["outcome"] for r in log.attempts(its[0].trace_key, R.config_sha256(), "A", 1)],
                         ["HALT", "VALID"])

        class Busy(RS.ScriptedProvider):
            def _exchange(self, request, stage, ref, kind):
                if request.judge == "B":
                    body = {"error": {"message": "Rate limit reached.", "type": "requests", "param": None,
                                      "code": "rate_limit_exceeded"}}
                    return Exchange(429, {}, json.dumps(body).encode(), None, None)
                return super()._exchange(request, stage, ref, kind)

        # A transient error body is a transport failure, and its slot is retried.
        self.root, its = self.tmp / "run-2", items(1, "b")
        led = self.ledger(self.tmp / "ledgers-2" / "ledger-pilot.jsonl")
        with self.stop_between_writes("B1"):
            with self.assertRaises(KeyboardInterrupt):
                self.stage("P1", its, led, provider=Busy("central"))
        report, provider = self.stage("P1", its, led)
        self.assertTrue(report["complete"], report["stopped"])
        self.assertEqual([r.judge for r in provider.sent], ["B"])
        b1 = AttemptLog(self.root / "P1" / "attempt_log").attempts(its[0].trace_key, R.config_sha256(), "B", 1)
        self.assertEqual([(r["outcome"], r["http_status"]) for r in b1], [("TRANSPORT", None), ("VALID", 200)])

    def test_a_raw_openai_error_body_that_would_be_a_transport_failure_halts_unless_known_transient(self):
        def body(etype, code):
            return json.dumps({"error": {"message": "x", "type": etype, "param": None, "code": code}}).encode()

        # OpenAI marks these by the status alone; without it they still halt, under the kind a code names.
        for status, etype, code, kind in ((403, "request_forbidden", "unsupported_country_region_territory",
                                           "permission"),
                                          (401, "invalid_request_error", "invalid_api_key", "auth"),
                                          (404, "invalid_request_error", "model_not_found", "not_found"),
                                          (403, "unrecognized_error", None, "request_error"),
                                          (400, "unrecognized_error", "unrecognized_code", "request_error")):
            live = canonical_outcome("openai", Exchange(status, {}, body(etype, code), None, None))
            rebuilt = error_body_outcome("openai", body(etype, code))
            self.assertEqual(live["outcome"], HALT, status)
            self.assertEqual((rebuilt["outcome"], rebuilt["halt_kind"]), (HALT, kind), (status, etype, code))
        for etype, code in (("requests", "rate_limit_exceeded"), ("server_error", None),
                            ("service_unavailable_error", "server_is_overloaded")):
            rebuilt = error_body_outcome("openai", body(etype, code))
            self.assertEqual((rebuilt["outcome"], rebuilt["halt_kind"]), ("TRANSPORT", None), code)
        self.assertEqual(error_body_outcome("openai", body("invalid_request_error", "invalid_prompt"))["outcome"],
                         "FILTERED")

        class Forbidden(RS.ScriptedProvider):
            def _exchange(self, request, stage, ref, kind):
                if request.judge == "B":
                    raw = body("request_forbidden", "unsupported_country_region_territory")
                    return Exchange(403, {}, raw, None, None)
                return super()._exchange(request, stage, ref, kind)

        led = self.ledger()
        its = items(1, "a")
        with self.stop_between_writes("B1"):
            with self.assertRaises(KeyboardInterrupt):
                self.stage("P1", its, led, provider=Forbidden("central"))
        provider = RS.ScriptedProvider()
        with self.assertRaisesRegex(RS.StageRefused, "halts without a correction note"):
            self.stage("P1", its, led, provider=provider)
        self.assertEqual(provider.sent, [])
        log = AttemptLog(self.root / "P1" / "attempt_log")
        b1 = log.attempts(its[0].trace_key, R.config_sha256(), "B", 1)
        self.assertEqual([(r["outcome"], r["halt_kind"], r["http_status"]) for r in b1], [("HALT", "permission", None)])
        self.assertEqual([(h["slot"], h["halt_kind"]) for h in log.halts()], [("B1", "permission")])
        self.assertEqual(D.capability_check(b1, R.B_MODEL)["checks"]["access"], "failed")
        held = next(e for e in led.entries if e["type"] == "hold")  # its status was lost
        self.assertEqual((led.reservations[held["id"]]["slot"], held["until_reconciled"]), ("B1", True))
        entry = self.reconcile(led)
        self.assertEqual((entry["ok"], led.entries[-1]["type"], led.entries[-1]["id"]), (True, "release", held["id"]))

    def test_a_raw_stream_that_opens_with_an_error_event_is_classified_as_the_dispatcher_would(self):
        def stream(judge, err):
            event = {"type": "error", "error": err} if judge == "A" else dict(err, type="error")
            return Exchange(200, {}, sse([("error", event)]), None, None)

        cases = (("A1", stream("A", {"type": "invalid_request_error", "message": "x"}), ("HALT", "request_error")),
                 ("B1", stream("B", {"code": "insufficient_quota", "message": "x"}), ("HALT", "spend_limit")),
                 ("A1", stream("A", {"type": "overloaded_error", "message": "x"}), ("TRANSPORT", None)))

        class Opening(RS.ScriptedProvider):
            def _exchange(self, request, stage, ref, kind):
                if request.judge == self.judge:
                    return self.opening
                return super()._exchange(request, stage, ref, kind)

        for n, (slot, exchange, expected) in enumerate(cases):
            self.root = self.tmp / f"run-{n}"
            led = self.ledger(self.tmp / f"ledgers-{n}" / "ledger-pilot.jsonl")
            its = items(1, "a")
            live = canonical_outcome("anthropic" if slot[0] == "A" else "openai", exchange)
            self.assertEqual((live["outcome"], live["halt_kind"]), expected)
            opening = Opening("central")
            opening.judge, opening.opening = slot[0], exchange
            with self.stop_between_writes(slot):
                with self.assertRaises(KeyboardInterrupt):
                    self.stage("P1", its, led, provider=opening)
            log = AttemptLog(self.root / "P1" / "attempt_log")
            if expected[0] == "HALT":
                provider = RS.ScriptedProvider()
                with self.assertRaisesRegex(RS.StageRefused, "halts without a correction note"):
                    self.stage("P1", its, led, provider=provider)
                self.assertEqual(provider.sent, [])
                self.assertEqual([(h["slot"], h["halt_kind"]) for h in log.halts()], [(slot, expected[1])])
            else:
                report, provider = self.stage("P1", its, led)
                self.assertTrue(report["complete"], report["stopped"])
                self.assertEqual([r.judge for r in provider.sent], ["A", "A", "A", "B"])
            first = log.attempts(its[0].trace_key, R.config_sha256(), slot[0], 1)[0]
            self.assertEqual((first["outcome"], first["halt_kind"], first["http_status"],
                              first["recovered"]["http_status"]), expected + (200, "inferred from the stream"))
            stopped = next(e for e in reserves(led) if e["slot"] == slot)
            self.assertEqual(led.reservations[stopped["id"]]["state"], "held")       # the stream did not complete

    def test_a_halt_attempt_without_its_halt_record_is_an_unresolved_halt(self):
        led = self.ledger()
        its = items(2, "a")
        with mock.patch.object(AttemptLog, "write_halt", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.stage("P1", its, led, events={(its[0].trace_key, "A2", 1): "http_400"})
        log = AttemptLog(self.root / "P1" / "attempt_log")
        self.assertEqual(log.halts(), [])
        provider = RS.ScriptedProvider()
        with self.assertRaisesRegex(RS.StageRefused, "halts without a correction note"):
            self.stage("P1", its, led, provider=provider)
        self.assertEqual(provider.sent, [])
        halts = log.halts()
        self.assertEqual([(h["slot"], h["attempt"], h["halt_kind"], h["recorded_at_start"]) for h in halts],
                         [("A2", 1, "request_error", True)])
        log.record_correction(halts[0]["halt_id"], "Fixed the request.", R.config_sha256(), at="now")
        report, provider = self.stage("P1", its, led)
        self.assertTrue(report["complete"])
        self.assertEqual([r.judge for r in provider.sent], ["A", "A", "B", "A", "A", "A", "B"])

    def test_a_ledger_shorter_than_the_recorded_progress_is_refused(self):
        led = self.ledger()
        its = items(2, "a")
        report, _ = self.stage("P1", its, led, events={(its[1].trace_key, "A1", 1): "http_400"})
        self.assertEqual(report["ledger"]["last_seq"], len(led.entries) - 1)
        AttemptLog(self.root / "P1" / "attempt_log").record_correction(
            report["stopped"]["halt_id"], "Noted.", R.config_sha256(), at="now")
        led.close()
        path = RS.ledger_path(self.ledgers, "pilot")
        lines = path.read_text().splitlines()
        self.assertEqual(json.loads(lines[-1])["type"], "settle")
        path.write_text("\n".join(lines[:-1]) + "\n")                # the last settlement rolled back
        provider, led = RS.ScriptedProvider(), self.ledger()
        with self.assertRaisesRegex(RS.StageRefused, "rolled back or cut"):
            self.stage("P1", its, led, provider=provider)
        self.assertEqual(provider.sent, [])
        (self.root / "ledger_marks-pilot.jsonl").unlink()            # the progress file still holds a mark
        with self.assertRaisesRegex(RS.StageRefused, "progress.json recorded entry"):
            self.stage("P1", its, led, provider=provider)
        self.assertEqual(provider.sent, [])

    def test_a_ledger_that_differs_from_a_recorded_mark_is_refused(self):
        led = self.ledger()
        report, _ = self.stage("P1", items(1, "a"), led)
        self.reconcile(led)
        led.close()
        path = RS.ledger_path(self.ledgers, "pilot")
        lines = path.read_text().splitlines()
        entry = json.loads(lines[1])
        entry["amount_usd"] = "0.01"                                  # same length, one amount edited
        path.write_text("\n".join([lines[0], json.dumps(entry, sort_keys=True)] + lines[2:]) + "\n")
        provider = RS.ScriptedProvider()
        with self.assertRaisesRegex(RS.StageRefused, "differs, up to entry"):
            self.stage("P2", items(1, "b"), self.ledger(), provider=provider)
        self.assertEqual(provider.sent, [])

    def test_an_attempt_without_its_reservation_is_refused(self):
        led = self.ledger()
        its = items(1, "a")

        class Crash(RS.ScriptedProvider):
            def send(self, request):
                if len(self.sent) == 2:
                    raise KeyboardInterrupt
                return super().send(request)

        with self.assertRaises(KeyboardInterrupt):
            self.stage("P1", its, led, provider=Crash("central"))
        self.assertFalse((self.root / "P1" / "progress.json").exists())
        led.close()
        path = RS.ledger_path(self.ledgers, "pilot")
        path.write_text(path.read_text().splitlines()[0] + "\n")      # rolled back to its header
        provider = RS.ScriptedProvider()
        with self.assertRaisesRegex(RS.StageRefused, "has no reservation"):
            self.stage("P1", its, self.ledger(), provider=provider)
        self.assertEqual(provider.sent, [])

    def test_a_run_root_stays_on_the_ledger_it_started_on(self):
        led = self.ledger()
        report, provider = self.stage("P1", items(1, "a"), led)
        self.reconcile(led, provider)
        fresh = self.ledger(self.tmp / "elsewhere" / "ledger-pilot.jsonl")
        with self.assertRaisesRegex(RS.StageRefused, "recorded pilot ledger"):
            self.stage("P2", items(1, "b"), fresh)
        (self.root / "ledger_marks-pilot.jsonl").unlink()
        (self.root / "P1" / "progress.json").unlink()
        with self.assertRaisesRegex(RS.StageRefused, "ran on pilot ledger"):     # stage.json still names it
            self.stage("P2", items(1, "b"), fresh)
        self.assertFalse((self.root / "P2").exists())


class TestEval(RunnerCase):
    def setUp(self):
        super().setUp()
        self.repo = self.tmp / "repo"
        self.repo.mkdir()
        git = ["git", "-C", str(self.repo), "-c", "user.name=t", "-c", "user.email=t@example.com"]
        (self.repo / "f.txt").write_text("frozen\n")
        for args in (["init", "-q"], ["add", "f.txt"], ["commit", "-q", "-m", "freeze"], ["tag", "freeze-1"]):
            subprocess.run(git + args, check=True, capture_output=True, env={"PATH": os.environ["PATH"]})
        pilot = self.ledger()
        for stage, tag in (("P1", "a"), ("P2", "b"), ("P3", "c")):
            report, provider = self.stage(stage, items(1, tag), pilot)
            self.assertTrue(report["complete"])
            self.reconcile(pilot, provider)
        pilot.close()
        self.panel_items = items(2, "e")
        self.write_manifest()

    def write_manifest(self, **change):
        m = {"kind": "panel-manifest", "frozen": True, "freeze_tag": "freeze-1", "config_sha256": R.config_sha256(),
             "n": 2, "tasks": [{"trace_key": i.trace_key, "pack_sha256": i.pack_sha256} for i in self.panel_items]}
        m.update(change)
        path = self.tmp / "panel.json"
        path.write_text(json.dumps(m))
        return {"manifest": str(path), "sha256": R.sha256_bytes(path.read_bytes()), "freeze_tag": "freeze-1"}

    def eval_ledger(self, clock=time.time):
        return S.Ledger(RS.ledger_path(self.ledgers, "evaluation"), "evaluation", clock, mode="fake")

    def eval(self, panel, its=None, clock=time.time, provider=None):
        with self.eval_ledger(clock) as led:
            return self.stage("EVAL", its or self.panel_items, led, panel=panel, repo=self.repo, provider=provider,
                              clock=clock)

    def test_eval_refuses_without_a_verified_frozen_panel(self):
        good = self.write_manifest()
        cases = [(None, "needs --panel-manifest"), (dict(good, sha256=None), "needs --panel-manifest"),
                 (dict(good, freeze_tag=None), "needs --panel-manifest"),
                 (dict(good, sha256="0" * 64), "recorded SHA-256"),
                 (dict(good, manifest=str(self.tmp / "none.json")), "no panel manifest")]
        for panel, message in cases:
            with self.assertRaisesRegex(RS.StageRefused, message):
                self.eval(panel)
        for change, message in (({"frozen": False}, "frozen panel"), ({"freeze_tag": "freeze-2"}, "freeze tag"),
                                ({"config_sha256": "0" * 64}, "config hash"), ({"n": 3}, "N do not agree")):
            with self.assertRaisesRegex(RS.StageRefused, message):
                self.eval(self.write_manifest(**change))
        good = self.write_manifest()
        with self.assertRaisesRegex(RS.StageRefused, "panel's tasks"):
            self.eval(good, its=list(reversed(self.panel_items)))
        with self.assertRaisesRegex(RS.StageRefused, "no freeze tag"):
            self.eval(dict(self.write_manifest(freeze_tag="freeze-9"), freeze_tag="freeze-9"))
        (self.repo / "f.txt").write_text("changed\n")
        with self.assertRaisesRegex(RS.StageRefused, "uncommitted"):
            self.eval(self.write_manifest())
        self.assertFalse((self.root / "EVAL").exists())

    def test_eval_runs_on_the_frozen_panel_and_completes_only_after_a_reconciliation(self):
        panel = self.write_manifest()
        report, provider = self.eval(panel)
        self.assertEqual((report["complete"], report["stopped"]), (False, None))
        self.assertIn("reconciliation", report["awaiting"])
        self.assertFalse((self.root / "EVAL" / "report.json").exists())
        self.assertEqual(len(provider.sent), 8)
        with self.eval_ledger() as led:
            self.reconcile(led, provider)
        report, again = self.eval(panel)
        self.assertTrue(report["complete"])
        self.assertEqual(again.sent, [])
        self.assertTrue((self.root / "EVAL" / "report.json").exists())
        spend = report["gates"]["spend"]
        self.assertEqual((spend["budget"], spend["cap_usd"], spend["reconciled_after_last_call"]),
                         ("evaluation", "285", True))
        self.assertEqual(report["panel"]["freeze_tag"], "freeze-1")
        self.assertEqual(report["settings"]["fence_profile"], "pre_join")

    def test_an_ineligible_frozen_task_halts_and_does_not_change_n(self):
        bad = ineligible_item("x")
        self.panel_items = [bad, *items(1, "e")]
        panel = self.write_manifest()
        report, provider = self.eval(panel)
        self.assertEqual(provider.sent, [])                                 # the next task is not started
        self.assertEqual((report["complete"], report["stopped"]["kind"], report["stopped"]["trace_key"]),
                         (False, "ineligible_frozen_task", bad.trace_key))
        self.assertIsNone(report["stopped"]["slot"])
        n = {"planned": 2, "given": 2, "ended": 0, "ineligible": [bad.trace_key]}
        self.assertEqual(report["cases"], n)
        self.assertFalse((self.root / "EVAL" / "report.json").exists())
        log = AttemptLog(self.root / "EVAL" / "attempt_log")
        halt = log.halts()[0]
        self.assertEqual(halt["detail"], "a task of the frozen panel is ineligible: A: 601 images, above 600")
        self.assertIn("does not drop the task", halt["resume_rule"])
        with self.assertRaisesRegex(RS.StageRefused, "halts without a correction note"):
            self.eval(panel)
        log.record_correction(halt["halt_id"], "Checked the panel.", R.config_sha256(), at="now")
        report, provider = self.eval(panel)                                 # a note does not drop the task
        self.assertEqual(provider.sent, [])
        self.assertEqual((report["complete"], report["stopped"]["kind"]), (False, "ineligible_frozen_task"))
        self.assertEqual(report["cases"], n)
        self.assertEqual(len(log.halts()), 2)

    def test_an_ineligible_frozen_task_is_not_ended_so_the_stage_never_completes_without_its_judgments(self):
        bad = ineligible_item("x")
        self.panel_items = [*items(1, "e"), bad]
        panel = self.write_manifest()
        report, provider = self.eval(panel)
        self.assertEqual(len(provider.sent), 4)
        self.assertEqual((report["stopped"]["kind"], report["cases"]["ended"]), ("ineligible_frozen_task", 1))
        with self.eval_ledger() as led:
            self.assertTrue(self.reconcile(led)["ok"])
        AttemptLog(self.root / "EVAL" / "attempt_log").record_correction(
            report["stopped"]["halt_id"], "Checked the panel.", R.config_sha256(), at="now")
        report, provider = self.eval(panel)
        self.assertEqual(provider.sent, [])
        self.assertEqual((report["complete"], report["awaiting"], report["cases"]["ended"]), (False, None, 1))
        self.assertFalse((self.root / "EVAL" / "report.json").exists())

    def test_eval_needs_the_pilot_ledger_reconciled(self):
        with S.Ledger(RS.ledger_path(self.ledgers, "pilot"), "pilot") as pilot:
            rid = pilot.reserve({"provider": "anthropic", "model": R.A_MODEL, "run": self.run_id(), "stage": "P3"},
                                Decimal("0.1"))
            pilot.close_reservation(rid, None)
        with self.assertRaisesRegex(RS.StageRefused, "reconcile the pilot ledger"):
            self.eval(self.write_manifest())

    def test_eval_halts_for_a_reconciliation_once_a_call_is_more_than_a_day_old(self):
        now = [time.time() - 3 * 86400]

        def clock():
            return now[0]

        class NextDay(RS.ScriptedProvider):
            def send(self, request):
                exchange = super().send(request)
                if request.judge == "B":
                    now[0] += 25 * 3600
                return exchange

        panel = self.write_manifest()
        report, first = self.eval(panel, clock=clock, provider=NextDay("central"))
        self.assertEqual((report["stopped"]["kind"], report["stopped"]["trace_key"], report["stopped"]["slot"]),
                         ("reconcile_due", self.panel_items[1].trace_key, "A1"))
        self.assertEqual(len(first.sent), 4)
        with self.eval_ledger(clock) as led:
            self.assertEqual(led.entries[-1]["reason"], "reconcile_due")
            self.assertTrue(self.reconcile(led)["ok"])
        AttemptLog(self.root / "EVAL" / "attempt_log").record_correction(
            report["stopped"]["halt_id"], "Reconciled both providers' usage.", R.config_sha256(), at="now")
        report, second = self.eval(panel, clock=clock)
        self.assertEqual(len(second.sent), 4)
        self.assertEqual((report["complete"], report["stopped"]), (False, None))
        with self.eval_ledger(clock) as led:
            self.reconcile(led)
        report, _ = self.eval(panel, clock=clock)
        self.assertTrue(report["complete"])

    def test_eval_will_not_start_while_a_call_more_than_a_day_old_is_unreconciled(self):
        now = [time.time() - 3 * 86400]

        def clock():
            return now[0]

        with self.eval_ledger(clock) as led:
            rid = led.reserve({"provider": "anthropic", "model": R.A_MODEL, "run": "earlier", "stage": "EVAL"},
                              Decimal("0.5"))
            led.close_reservation(rid, None)
        now[0] += 25 * 3600
        provider = RS.ScriptedProvider()
        with self.assertRaisesRegex(RS.StageRefused, "more than 23 hours old"):
            self.eval(self.write_manifest(), clock=clock, provider=provider)
        self.assertEqual(provider.sent, [])
        now[0] -= 3 * 3600                                              # 22 hours after that call
        report, provider = self.eval(self.write_manifest(), clock=clock)
        self.assertEqual((report["stopped"], len(provider.sent)), (None, 8))


class TestPaidMode(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.note = self.tmp / "approval.txt"
        self.note.write_text("Written go-ahead received on the study channel.\n")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_paid_mode_needs_all_three_conditions(self):
        empty, blank = self.tmp / "empty.txt", self.tmp / "blank.txt"
        empty.write_text("")
        blank.write_text("  \n")
        with mock.patch.object(D, "HttpsTransport") as https:
            self.assertEqual(RS.select_transport(False, False, None), ("fake", None))
            for paid in (False, True):
                for confirm in (False, True):
                    for note in (None, str(self.note), str(empty), str(blank), str(self.tmp / "missing.txt")):
                        if (paid, confirm, note) in ((False, False, None), (True, True, str(self.note))):
                            continue
                        with self.assertRaisesRegex(RS.StageRefused, "missing"):
                            RS.select_transport(paid, confirm, note, workspace="wrkspc_study",
                                                openai_project="proj_study")
            https.assert_not_called()
            with self.assertRaisesRegex(RS.StageRefused, "--workspace"):
                RS.select_transport(True, True, str(self.note), openai_project="proj_study")
            with self.assertRaisesRegex(RS.StageRefused, "--openai-project"):
                RS.select_transport(True, True, str(self.note), workspace="wrkspc_study")
            https.assert_not_called()
            mode, _ = RS.select_transport(True, True, str(self.note), workspace="wrkspc_study",
                                          openai_project="proj_study")
        self.assertEqual(mode, "paid")
        https.assert_called_once_with(confirm_paid_calls=True, openai_project="proj_study")

    def test_a_paid_stage_without_a_provider_token_counter_is_refused(self):
        led = S.Ledger(self.tmp / "ledger-pilot.jsonl", "pilot", mode="paid")
        self.addCleanup(led.close)
        with self.assertRaisesRegex(RS.StageRefused, "token-count client"):
            RS.run_stage("P1", self.tmp / "run", items(1, "a"), mode="paid", transport=StandInHttps(), ledger=led,
                         read_images=lambda e: e, workspace="wrkspc_study", openai_project="proj_study")
        self.assertFalse((self.tmp / "run").exists())
        self.assertEqual(len(led.entries), 1)

    def test_a_paid_stage_needs_the_openai_project_and_a_transport_bound_to_it(self):
        led = S.Ledger(self.tmp / "ledger-pilot.jsonl", "pilot", mode="paid")
        self.addCleanup(led.close)
        counter = RS.ScriptedProvider()
        with self.assertRaisesRegex(RS.StageRefused, "workspace and OpenAI project"):
            RS.run_stage("P1", self.tmp / "run", items(1, "a"), mode="paid", transport=StandInHttps(), ledger=led,
                         read_images=lambda e: e, workspace="wrkspc_study", counter=counter)
        with self.assertRaisesRegex(ValueError, "bound to another OpenAI project"):
            RS.run_stage("P1", self.tmp / "run", items(1, "a"), mode="paid", transport=StandInHttps("proj_other"),
                         ledger=led, read_images=lambda e: e, workspace="wrkspc_study", openai_project="proj_study",
                         counter=counter)
        self.assertFalse((self.tmp / "run").exists())
        self.assertEqual((len(led.entries), counter.sent), (1, []))

    def test_the_https_transport_itself_still_refuses_without_confirmation(self):
        with self.assertRaises(ValueError):
            D.HttpsTransport(confirm_paid_calls=False)


class ProviderHttp:
    """Stands in for ``urlopen`` and answers both providers, so the paid path runs and
    nothing leaves: a token count per judge, or (status, raw bytes) in its place, and each
    send from the fake provider's stream."""

    def __init__(self, workspace, project, a=61234, b=70123):
        self.provider, self.urls = RS.ScriptedProvider(workspace=workspace, project=project), []
        self.workspace, self.counts = workspace, {D.COUNT_ENDPOINTS["anthropic"]: a, D.COUNT_ENDPOINTS["openai"]: b}

    def _count(self, req):
        answer = self.counts[req.full_url]
        is_a = req.full_url == D.COUNT_ENDPOINTS["anthropic"]
        reply = {"input_tokens": answer} if is_a else {"object": "response.input_tokens", "input_tokens": answer}
        return 200, {"anthropic-workspace-id": self.workspace} if is_a else {}, json.dumps(reply).encode()

    def __call__(self, req, timeout):
        self.urls.append(req.full_url)
        if req.full_url in self.counts:
            given = self.counts[req.full_url]
            status, headers, raw = self._count(req) if isinstance(given, int) else (given[0], {}, given[1])
        else:
            judge, provider = ("A", "anthropic") if req.full_url == R.A_ENDPOINT else ("B", "openai")
            sent = R.PreparedRequest(judge, provider, json.loads(req.data)["model"], req.full_url, (), req.data,
                                     R.sha256_bytes(req.data))
            ex = self.provider._exchange(sent, RS.STAGES["P1"], f"ref{len(self.urls)}", "valid")
            status, headers, raw = ex.status, ex.headers, ex.raw
        if status >= 400:
            raise urllib.error.HTTPError(req.full_url, status, "error", headers, io.BytesIO(raw))
        response = io.BytesIO(raw)
        response.status, response.headers = status, headers
        return response


class TestPaidTokenCounts(unittest.TestCase):
    WORKSPACE, PROJECT = "wrkspc_study", "proj_study"
    COUNTS = [D.COUNT_ENDPOINTS["anthropic"], D.COUNT_ENDPOINTS["openai"]]
    SENDS = (R.A_ENDPOINT, R.B_ENDPOINT)
    BAD_REQUEST = (400, json.dumps({"type": "error", "error": {"type": "invalid_request_error", "message": "m"}}).encode())

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.fresh("first")
        self.its = items(1, "a")

    def fresh(self, name):
        """A new run folder, a new paid ledger, and a transport and counter bound to the study."""
        self.root = self.tmp / name / "run"
        self.led = S.Ledger(self.tmp / name / "ledger-pilot.jsonl", "pilot", mode="paid")
        self.addCleanup(self.led.close)
        self.transport = D.HttpsTransport(confirm_paid_calls=True, openai_project=self.PROJECT)
        self.counter = D.HttpsTokenCounter(self.transport, self.WORKSPACE, sleep=lambda s: None)

    def http(self, **counts):
        return ProviderHttp(self.WORKSPACE, self.PROJECT, **counts)

    def paid(self, http, counter=None):
        keys = dict.fromkeys(KEY_ENVS, "placeholder")
        with mock.patch.dict(os.environ, keys), mock.patch.object(D.urllib.request, "urlopen", http):
            return RS.run_stage("P1", self.root, self.its, mode="paid", transport=self.transport, ledger=self.led,
                                read_images=lambda e: e, workspace=self.WORKSPACE, openai_project=self.PROJECT,
                                counter=counter or self.counter)

    def test_a_paid_trajectory_is_counted_before_its_first_send_and_reserved_at_the_provider_s_count(self):
        http = self.http()
        report = self.paid(http)
        self.assertTrue(report["complete"], report["stopped"])
        self.assertEqual(http.urls, self.COUNTS + [R.A_ENDPOINT] * 3 + [R.B_ENDPOINT])
        rows = reserves(self.led)
        self.assertEqual([(r["slot"], r["input_tokens"], r["input_source"]) for r in rows],
                         [("A1", 61234, "provider_count"), ("A2", 61234, "provider_count"),
                          ("A3", 61234, "provider_count"), ("B1", 70123, "provider_count")])
        for r in rows:
            self.assertEqual(Decimal(r["amount_usd"]), S.reservation(r["model"], r["input_tokens"], R.OUTPUT_CAP))
        self.assertEqual(report["trajectories"][0]["inputs"]["B"]["input_tokens"], 70123)

    def test_a_failed_malformed_or_missing_count_halts_before_any_send(self):
        answers = {"status": self.BAD_REQUEST, "malformed": (200, b"not json"), "missing": (200, b"{}"),
                   "error_without_status": (200, b'{"error":{"type":"api_error"}}')}
        for judge in "ab":
            for name, answer in answers.items():
                self.fresh(f"{judge}-{name}")
                http = self.http(**{judge: answer})
                report = self.paid(http)
                halts = report["gates"]["halts"]
                self.assertEqual((report["complete"], report["stopped"]["kind"], len(halts)),
                                 (False, "token_count", 1), name)
                self.assertEqual((halts[0]["kind"], halts[0]["slot"]), ("token_count", None))
                self.assertFalse(set(http.urls) & set(self.SENDS), name)
                self.assertEqual(reserves(self.led), [], name)
                self.assertEqual((report["cases"]["ended"], report["cases"]["ineligible"]), (0, []))
                self.assertEqual(report["trajectories"], [{"trace_key": self.its[0].trace_key, "eligible": True}])

    def test_a_halt_on_a_count_records_the_provider_s_error_and_how_to_resume(self):
        report = self.paid(self.http(b=self.BAD_REQUEST))
        halt = AttemptLog(self.root / "P1" / "attempt_log").halts()[0]
        self.assertEqual((halt["halt_id"], halt["http_status"], halt["error"]["type"]),
                         (report["stopped"]["halt_id"], 400, "invalid_request_error"))
        self.assertIn("openai token count failed", halt["detail"])
        self.assertEqual(halt["resume_rule"], RS.RESUME_RULES["token_count"])

    def test_a_count_halt_needs_a_note_and_the_resumed_stage_counts_again(self):
        report = self.paid(self.http(a=self.BAD_REQUEST))
        http = self.http()
        with self.assertRaisesRegex(RS.StageRefused, "without a correction note"):
            self.paid(http)
        self.assertEqual(http.urls, [])
        AttemptLog(self.root / "P1" / "attempt_log").record_correction(
            report["stopped"]["halt_id"], "The count endpoint was restored.", R.config_sha256(), at="now")
        report = self.paid(http)
        self.assertTrue(report["complete"], report["stopped"])
        self.assertEqual(http.urls[:2], self.COUNTS)

    def test_the_count_feeds_the_272_000_token_halt_before_any_send(self):
        http = self.http(b=S.B_INPUT_LIMIT + 1)
        report = self.paid(http)
        self.assertEqual(report["stopped"]["kind"], "long_context")
        self.assertEqual(http.urls, self.COUNTS)
        self.fresh("at-limit")
        http = self.http(b=S.B_INPUT_LIMIT)
        self.assertTrue(self.paid(http)["complete"])
        self.assertEqual(http.urls.count(R.B_ENDPOINT), 1)

    def test_the_count_feeds_the_reservation_and_the_cap(self):
        http = self.http(a=40_000_000)
        report = self.paid(http)
        self.assertEqual(report["stopped"]["kind"], "spend_cap")
        self.assertEqual(http.urls, self.COUNTS)
        self.assertEqual(reserves(self.led), [])

    def test_a_counter_that_gives_no_count_halts_and_no_estimate_stands_in(self):
        class Silent(D.HttpsTokenCounter):
            def count(self, request):
                return None
        http = self.http()
        report = self.paid(http, Silent(self.transport, self.WORKSPACE))
        self.assertEqual(report["stopped"]["kind"], "token_count")
        self.assertEqual((http.urls, reserves(self.led)), ([], []))

    def test_paid_mode_counts_only_with_the_providers_endpoints_on_its_own_transport_and_workspace(self):
        other = D.HttpsTransport(confirm_paid_calls=True, openai_project=self.PROJECT)
        for counter, message in ((RS.ScriptedProvider(), "providers' own endpoints"),
                                 (D.HttpsTokenCounter(other, self.WORKSPACE), "another transport"),
                                 (D.HttpsTokenCounter(self.transport, "wrkspc_other"), "another transport or workspace")):
            http = self.http()
            with self.assertRaisesRegex(ValueError, message):
                self.paid(http, counter)
            self.assertEqual(http.urls, [])
        self.assertFalse(self.root.exists())

    def test_fake_mode_refuses_the_provider_counter(self):
        led = S.Ledger(self.tmp / "fake-ledger-pilot.jsonl", "pilot", mode="fake")
        self.addCleanup(led.close)
        with self.assertRaisesRegex(ValueError, "fake mode counts no tokens"):
            RS.run_stage("P1", self.root, self.its, mode="fake", transport=RS.ScriptedProvider(), ledger=led,
                         read_images=lambda e: e, workspace=RS.FAKE_WORKSPACE, counter=self.counter)

    def command(self, *flags):
        """The arguments the stage command gives ``run_stage``, with the guard, the socket
        refusal and the packs replaced, since an installed guard cannot be removed."""
        data_root, run_root = self.tmp / "data", self.tmp / "data" / "out" / "run"
        argv = ["run", "--stage", "P1", "--run-root", str(run_root), "--data-root", str(data_root), "--packs", "p",
                "--provenance", "q", *flags]
        with mock.patch.object(RS, "_guard", return_value=(data_root, run_root)), \
                mock.patch.object(RS, "_forbid_network"), mock.patch.object(RS, "load_items", return_value=[]), \
                mock.patch.object(RS, "print_report"), \
                mock.patch.object(RS, "run_stage", return_value={"complete": True}) as run_stage:
            self.assertEqual(RS.main(argv), 0)
        return run_stage.call_args.kwargs

    def test_the_paid_stage_command_counts_with_the_provider_on_its_transport_and_workspace(self):
        note = self.tmp / "approval.txt"
        note.write_text("Written go-ahead received on the study channel.\n")
        kw = self.command("--paid", "--confirm-paid-calls", "--approval-note", str(note),
                          "--workspace", self.WORKSPACE, "--openai-project", self.PROJECT)
        self.assertEqual(kw["mode"], "paid")
        self.assertIsInstance(kw["counter"], D.HttpsTokenCounter)
        self.assertIs(kw["counter"].transport, kw["transport"])
        self.assertEqual((kw["counter"].workspace, kw["transport"].openai_project), (self.WORKSPACE, self.PROJECT))

    def test_the_fake_stage_command_keeps_the_fake_counter(self):
        kw = self.command()
        self.assertEqual(kw["mode"], "fake")
        self.assertIsInstance(kw["transport"], RS.ScriptedProvider)
        self.assertIs(kw["counter"], kw["transport"])


class CommandCase(FixtureCase):
    def setUp(self):
        super().setUp()
        self.inputs = self.write_inputs(self.tmp / "dev_out")
        self.config = self.tmp / "fence_config.json"
        self.config.write_text("{}")
        r = self.command(["scripts/build_packs.py", "--data-root", str(self.tmp),
                          "--monitor-view", str(self.inputs / "monitor_view.jsonl"),
                          "--provenance", str(self.inputs / "provenance.jsonl"), "--out", str(self.tmp / "packs")])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.root = self.tmp / "out" / "run"                 # a run root sits directly under <data-root>/out
        self.ledger_root = RS.ledger_root(self.tmp)
        self.ledgers = RS.ledger_dir(self.tmp, "fake")

    def env(self):
        env = {k: v for k, v in os.environ.items() if k not in KEY_ENVS + ("SPAR_FENCE_CONFIG",)}
        env["SPAR_FENCE_CONFIG"] = str(self.config)
        return env

    def command(self, args):
        return subprocess.run([sys.executable, str(REPO / args[0])] + args[1:], capture_output=True, text=True,
                              env=self.env())

    def stage_args(self, stage, *extra, root=None):
        return ["scripts/run_stage.py", "run", "--stage", stage, "--run-root", str(root or self.root), "--data-root",
                str(self.tmp), "--packs", str(self.tmp / "packs"), "--provenance",
                str(self.inputs / "provenance.jsonl")] + list(extra)

    def reconcile(self, export, root=None):
        return self.command(["scripts/run_stage.py", "reconcile", "--run-root", str(root or self.root), "--data-root",
                             str(self.tmp), "--budget", "pilot", "--export", str(export)])


SPY = r'''
import json, os, runpy, sys
seen = []
cls = type(os.environ)
original = cls.__getitem__
def spy(self, key):
    seen.append(key)
    return original(self, key)
cls.__getitem__ = spy
script, record = sys.argv[1], sys.argv[2]
sys.argv = [script] + sys.argv[3:]
code = 0
try:
    runpy.run_path(script, run_name="__main__")
except SystemExit as e:
    code = e.code
finally:
    with open(record, "w") as f:
        json.dump(sorted(set(k for k in seen if isinstance(k, str))), f)
sys.exit(code)
'''


class TestCommands(CommandCase):
    def test_the_pilot_in_fake_mode_reads_no_key_and_installs_dev_only(self):
        record = self.tmp / "env_keys.json"
        args = self.stage_args("P1")
        r = subprocess.run([sys.executable, "-c", SPY, str(REPO / args[0]), str(record)] + args[1:],
                           capture_output=True, text=True, env=self.env())
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        seen = json.loads(record.read_text())
        self.assertIn("SPAR_FENCE_CONFIG", seen)                 # the spy works
        self.assertFalse(set(seen) & set(KEY_ENVS), seen)
        report = json.loads((self.root / "P1" / "report.json").read_text())
        self.assertEqual(report["settings"]["fence"]["profiles"], ["dev_only"])
        self.assertEqual(report["cases"], {"planned": 5, "given": 2, "ended": 2, "ineligible": []})
        self.assertIn("every socket refused", report["settings"]["transport"])
        ledger_id = report["ledger"]["id"]
        self.assertEqual(sorted(p.name for p in self.ledgers.iterdir()),
                         [f"fake_provider_export-pilot-{ledger_id}.csv", "ledger-pilot.jsonl"])
        self.assertEqual(sorted(p.name for p in self.ledger_root.iterdir()), ["budgets.jsonl", "fake", "marks-pilot.jsonl"])
        self.assertFalse(any(p.name.startswith("ledger-") for p in self.root.iterdir()))

        r = self.command(self.stage_args("P2"))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("reconcile", r.stderr)
        export = RS.export_path(self.ledgers, "pilot", ledger_id)
        r = self.reconcile(export)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn(f"{RS.FAKE_WORKSPACE} from ", r.stdout)
        r = self.command(self.stage_args("P2"))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("cache hits      not applicable", r.stdout)

        rows = list(export.read_text().splitlines())
        other = self.tmp / "other-workspace.csv"
        other.write_text("\n".join([rows[0]] + [r.replace(RS.FAKE_WORKSPACE, "wrkspc_default") for r in rows[1:]])
                         + "\n")
        r = self.reconcile(other)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("not the study's 'wrkspc_fake'", r.stderr)
        inflated = self.tmp / "inflated.csv"
        inflated.write_text("\n".join(rows + [rows[1]]) + "\n")
        r = self.reconcile(inflated)
        self.assertEqual(r.returncode, 3, r.stdout + r.stderr)
        self.assertIn("MORE THAN 1%", r.stdout)
        r = self.command(["scripts/run_stage.py", "correct", "--run-root", str(self.root), "--data-root",
                          str(self.tmp), "--budget", "pilot", "--note", "The export repeated one row."])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("settled as an adjustment", r.stdout)
        self.assertEqual(self.reconcile(export).returncode, 0)
        r = self.command(self.stage_args("P3", "--scripted-usage", "extreme"))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("frozen rubric", r.stderr)
        self.assertFalse((self.root / "P3").exists())

    def test_two_run_roots_share_one_ledger_and_the_second_is_refused_at_the_cap(self):
        with S.open_budget_ledger(self.ledger_root, "pilot", "fake", create=True) as led:
            rid = led.reserve({"provider": "anthropic", "model": R.A_MODEL, "run": "earlier", "stage": "P1"},
                              Decimal("93.75"))
            led.close_reservation(rid, None)                    # held: counted as spent for good
        first, second = self.tmp / "out" / "run-a", self.tmp / "out" / "run-b"
        r = self.command(self.stage_args("P1", root=first))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        a = json.loads((first / "P1" / "report.json").read_text())
        self.assertEqual(Decimal(a["gates"]["spend"]["budget_committed_usd"]), Decimal("94.578"))
        r = self.command(self.stage_args("P1", root=second))
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        b = json.loads((second / "P1" / "progress.json").read_text())
        self.assertEqual((b["stopped"]["kind"], b["stopped"]["slot"]), ("spend_cap", "A1"))
        self.assertEqual(b["gates"]["spend"]["budget_committed_usd"], a["gates"]["spend"]["budget_committed_usd"])
        self.assertEqual(b["ledger"]["id"], a["ledger"]["id"])
        for root in (first, second):
            self.assertFalse(any(p.name.startswith("ledger-") for p in root.iterdir()))
        with S.open_budget_ledger(self.ledger_root, "pilot", "fake") as led:
            self.assertEqual(led.totals()["committed"], Decimal("94.578"))
            ledger_id = led.ledger_id

        # Moving the ledger aside does not start a fresh cap without a new-budget note.
        RS.ledger_path(self.ledgers, "pilot").rename(self.ledgers / "ledger-pilot-moved.jsonl")
        third = self.tmp / "out" / "run-c"
        r = self.command(self.stage_args("P1", root=third))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("was created here before", r.stderr)
        r = self.command(self.stage_args("P1", "--new-budget-note", "A second pilot budget, approved in writing.",
                                         root=third))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        with S.open_budget_ledger(self.ledger_root, "pilot", "fake") as led:
            self.assertEqual((led.header["new_budget_note"], led.header["replaces"]),
                             ("A second pilot budget, approved in writing.", [ledger_id]))
            new_id = led.ledger_id
        # The new ledger's first reconciliation sees only its own calls.
        export = RS.export_path(self.ledgers, "pilot", new_id)
        r = self.reconcile(export, root=third)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("RECONCILED", r.stdout)
        self.assertTrue(RS.export_path(self.ledgers, "pilot", ledger_id).exists())

    def test_a_restored_older_ledger_copy_is_refused_for_every_run_root(self):
        path = RS.ledger_path(self.ledgers, "pilot")
        first = self.tmp / "out" / "run-a"
        r = self.command(self.stage_args("P1", "--scripted-usage", "extreme", root=first))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        full = path.read_bytes()
        path.write_bytes(full[:full.index(b"\n") + 1])                 # a copy holding only its header
        for root in (first, self.tmp / "out" / "run-b"):
            r = self.command(self.stage_args("P1", root=root))
            self.assertNotEqual(r.returncode, 0, root)
            self.assertIn("rolled back or cut", r.stderr)
        (self.ledger_root / S.marks_name("pilot")).unlink()           # the run root's marks still catch it
        r = self.command(self.stage_args("P1", root=self.tmp / "out" / "run-b"))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("ledger_marks-pilot.jsonl recorded entry", r.stderr)
        self.assertFalse((self.tmp / "out" / "run-b").exists())
        path.write_bytes(full)
        r = self.command(self.stage_args("P1", root=self.tmp / "out" / "run-b"))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_a_run_root_must_sit_directly_under_the_data_roots_out_folder(self):
        r = self.command(self.stage_args("P1", root=self.tmp / "out" / "run-a"))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        for name in ("ledgers", "Ledgers", "LEDGERS"):    # the same folder where the file system ignores case
            r = self.command(self.stage_args("P1", root=self.tmp / "out" / name))
            self.assertNotEqual(r.returncode, 0, name)
            self.assertIn("--run-root must be a folder directly under", r.stderr)
        self.assertFalse((self.ledger_root / "run.json").exists())
        shutil.rmtree(self.ledger_root)                    # every ledger, the register and the marks file
        elsewhere = (self.tmp / "run", self.tmp / "out" / "nested" / "run", self.ledger_root, self.tmp / "out")
        for root in elsewhere:
            r = self.command(self.stage_args("P1", root=root))
            self.assertNotEqual(r.returncode, 0, root)
            self.assertIn("--run-root must be a folder directly under", r.stderr)
        for command in (["reconcile", "--budget", "pilot", "--export", str(self.tmp / "export.csv")],
                        ["correct", "--budget", "pilot", "--note", "x"]):
            r = self.command(["scripts/run_stage.py", command[0], "--run-root", str(self.tmp / "run"), "--data-root",
                              str(self.tmp)] + command[1:])
            self.assertIn("--run-root must be a folder directly under", r.stderr)
        self.assertFalse((self.tmp / "run").exists())
        self.assertFalse(self.ledger_root.exists())
        # The marks of the run root that is directly under out still refuse a fresh cap.
        r = self.command(self.stage_args("P1", root=self.tmp / "out" / "run-b"))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("was created here before", r.stderr)
        self.assertFalse(self.ledger_root.exists())

    def test_paid_flags_without_all_three_and_p3_or_eval_on_a_draft_rubric_are_refused(self):
        empty = self.tmp / "empty.txt"
        empty.write_text("")
        for extra in (["--paid"], ["--confirm-paid-calls"], ["--paid", "--confirm-paid-calls"],
                      ["--paid", "--confirm-paid-calls", "--approval-note", str(empty)]):
            r = self.command(self.stage_args("P1", *extra))
            self.assertNotEqual(r.returncode, 0, extra)
            self.assertIn("missing", r.stderr, extra)
        for stage in ("P3", "EVAL"):
            r = self.command(self.stage_args(stage))
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("frozen rubric", r.stderr)
        self.assertFalse(self.root.exists())

    def test_paths_inside_the_repository_are_refused(self):
        inside = REPO / "not-a-file-in-the-repo.txt"
        r = self.command(self.stage_args("P1", "--approval-note", str(inside)))
        self.assertIn("--approval-note must be outside the repository", r.stderr)
        r = self.command(self.stage_args("EVAL", "--panel-manifest", str(inside)))
        self.assertIn("--panel-manifest must be outside the repository", r.stderr)
        r = self.command(["scripts/run_stage.py", "reconcile", "--run-root", str(self.root), "--data-root",
                          str(self.tmp), "--budget", "pilot", "--export", str(inside)])
        self.assertIn("--export must be outside the repository", r.stderr)
        self.assertFalse(inside.exists())
        self.assertFalse(self.root.exists())

    def test_the_guard_is_installed_and_the_network_refused(self):
        self.config.write_text(json.dumps({"test_output_dirs": [str(self.tmp / "packs")]}))
        r = self.command(self.stage_args("P1"))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("fence (dev_only)", r.stderr)
        probe = (f"import sys; sys.path.insert(0, {str(REPO / 'scripts')!r}); import run_stage as s; "
                 "print(s.fake_transport_line()); s._forbid_network(); print(s.fake_transport_line()); "
                 "import socket; socket.create_connection(('127.0.0.1', 9), timeout=1)")
        r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no network calls", r.stderr)
        self.assertEqual(r.stdout.splitlines(), [RS.FAKE_TRANSPORT + "; no socket guard installed",
                                                 RS.FAKE_TRANSPORT + "; every socket refused"])


if __name__ == "__main__":
    unittest.main()
