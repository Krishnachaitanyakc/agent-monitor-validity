"""Tests for the stage runner, scripts/run_stage.py.

The runner's core runs on synthetic packs through the fake transport. The commands run
in fresh processes on the data path's fixture, since an installed guard cannot be
removed. No test builds the HTTPS transport: where paid mode is checked, the class is
replaced by a stand-in that records how it was called.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
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
from judge import spend as S  # noqa: E402
from judge.storage import AttemptLog  # noqa: E402
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


class RunnerCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "run"

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def ledger(self, path=None, budget="pilot"):
        led = S.Ledger(path or RS.ledger_path(self.root, budget), budget)
        self.addCleanup(led.close)
        return led

    def stage(self, stage, its, led, profile="central", events=None, root=None, provider=None, **kw):
        provider = provider or RS.ScriptedProvider(profile, events=events)
        report = RS.run_stage(stage, root or self.root, its, mode="fake", transport=provider, ledger=led,
                              read_images=lambda e: e, workspace=RS.FAKE_WORKSPACE, counter=provider, **kw)
        return report, provider

    def pilot(self, led, sizes=(5, 20, 20), profile="central", root=None):
        reports = {}
        for stage, n in zip(("P1", "P2", "P3"), sizes):
            report, provider = self.stage(stage, items(n, stage[1]), led, profile, root=root)
            reports[stage] = report
            if not report["complete"]:
                break
            led.reconcile(provider.billed, "test")
        return reports


class TestPilot(RunnerCase):
    def test_the_cut_pilot_costs_the_addendum_figures_and_stays_under_the_cap(self):
        for profile, total in (("central", "16.35"), ("extreme", "48.25")):
            self.root = self.tmp / profile
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
        reserves = [e for e in led.entries if e["type"] == "reserve"]
        last = reserves[-1]
        self.assertEqual((last["run"], last["log_root"]), ("pilot-2", "pilot-2/P3/attempt_log"))
        sent = AttemptLog(self.tmp / last["log_root"]).attempts(last["trace_key"], last["config_sha256"],
                                                                last["judge"], last["repeat"])
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
        led.reconcile(provider.billed, "test")
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
        a2 = [e for e in led.entries if e["type"] == "reserve" and e["slot"] == "A2"]
        self.assertEqual([(e["attempt"], led.reservations[e["id"]]["state"]) for e in a2],
                         [(1, "held"), (1, "settled")])
        self.assertEqual(led.totals()["held"], r)

    def test_a_call_that_costs_more_than_its_reservation_halts(self):
        led = self.ledger()

        class Low(RS.ScriptedProvider):
            def count(self, request):
                return 10

        report, _ = self.stage("P1", items(2, "a"), led, provider=Low("extreme"))
        self.assertEqual(report["stopped"]["kind"], "over_reservation")
        self.assertTrue(report["gates"]["spend"]["over_reservation"])


class TestGating(RunnerCase):
    def test_each_stage_needs_the_previous_report_and_a_reconciliation(self):
        led = self.ledger()
        with self.assertRaisesRegex(RS.StageRefused, "P1's report"):
            self.stage("P2", items(1, "b"), led)
        _, provider = self.stage("P1", items(1, "a"), led)
        with self.assertRaisesRegex(RS.StageRefused, "reconcile"):
            self.stage("P2", items(1, "b"), led)
        led.reconcile(provider.billed, "test")
        with self.assertRaisesRegex(RS.StageRefused, "P2's report"):
            self.stage("P3", items(1, "c"), led)
        with self.assertRaisesRegex(RS.StageRefused, "already complete"):
            self.stage("P1", items(1, "a"), led)
        with self.assertRaisesRegex(RS.StageRefused, "1 to 20"):
            self.stage("P2", items(21, "b"), led)
        report, _ = self.stage("P2", items(1, "b"), led)
        self.assertTrue(report["complete"])
        self.assertEqual(report["previous_report"], str(self.root / "P1" / "report.json"))

    def test_a_run_keeps_one_mode_and_a_resumed_stage_its_inputs(self):
        led = self.ledger()
        its = items(2, "a")
        report, _ = self.stage("P1", its, led, events={(its[1].trace_key, "A1", 1): "http_400"})
        log = AttemptLog(self.root / "P1" / "attempt_log")
        log.record_correction(report["stopped"]["halt_id"], "Noted.", R.config_sha256(), at="now")
        with self.assertRaisesRegex(RS.StageRefused, "same ones"):
            self.stage("P1", its[:1], led)
        with self.assertRaisesRegex(RS.StageRefused, "same ones"):
            RS.run_stage("P1", self.root, its, mode="paid", transport=D.FakeTransport(), ledger=led,
                         read_images=lambda e: e, workspace="w")
        with self.assertRaises(ValueError):
            RS.run_stage("EVAL", self.root, its, mode="fake", transport=D.FakeTransport(), ledger=led,
                         read_images=lambda e: e, workspace="w")


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
            pilot.reconcile(provider.billed, "test")
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

    def eval(self, panel, its=None):
        with S.Ledger(RS.ledger_path(self.root, "evaluation"), "evaluation") as led:
            return self.stage("EVAL", its or self.panel_items, led, panel=panel, repo=self.repo)

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

    def test_eval_runs_on_the_frozen_panel_with_its_own_budget(self):
        report, provider = self.eval(self.write_manifest())
        self.assertTrue(report["complete"])
        spend = report["gates"]["spend"]
        self.assertEqual((spend["budget"], spend["cap_usd"]), ("evaluation", "285"))
        self.assertEqual(report["panel"]["freeze_tag"], "freeze-1")
        self.assertEqual(report["settings"]["fence_profile"], "pre_join")
        self.assertEqual(len(provider.sent), 8)

    def test_eval_needs_the_pilot_ledger_reconciled(self):
        with S.Ledger(RS.ledger_path(self.root, "pilot"), "pilot") as pilot:
            rid = pilot.reserve({"provider": "anthropic", "model": R.A_MODEL, "run": ".", "stage": "P3"},
                                Decimal("0.1"))
            pilot.close_reservation(rid, None)
        with self.assertRaisesRegex(RS.StageRefused, "reconcile the pilot ledger"):
            self.eval(self.write_manifest())


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
                            RS.select_transport(paid, confirm, note, workspace="wrkspc_study")
            https.assert_not_called()
            with self.assertRaisesRegex(RS.StageRefused, "--workspace"):
                RS.select_transport(True, True, str(self.note))
            https.assert_not_called()
            mode, _ = RS.select_transport(True, True, str(self.note), workspace="wrkspc_study")
        self.assertEqual(mode, "paid")
        https.assert_called_once_with(confirm_paid_calls=True)

    def test_the_https_transport_itself_still_refuses_without_confirmation(self):
        with self.assertRaises(ValueError):
            D.HttpsTransport(confirm_paid_calls=False)


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
        self.root = self.tmp / "run"

    def env(self):
        env = {k: v for k, v in os.environ.items() if k not in KEY_ENVS + ("SPAR_FENCE_CONFIG",)}
        env["SPAR_FENCE_CONFIG"] = str(self.config)
        return env

    def command(self, args):
        return subprocess.run([sys.executable, str(REPO / args[0])] + args[1:], capture_output=True, text=True,
                              env=self.env())

    def stage_args(self, stage, *extra):
        return ["scripts/run_stage.py", "run", "--stage", stage, "--run-root", str(self.root), "--data-root",
                str(self.tmp), "--packs", str(self.tmp / "packs"), "--provenance",
                str(self.inputs / "provenance.jsonl")] + list(extra)

    def reconcile(self, export):
        return self.command(["scripts/run_stage.py", "reconcile", "--run-root", str(self.root), "--data-root",
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

        r = self.command(self.stage_args("P2"))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("reconcile", r.stderr)
        export = self.root / "fake_provider_export-pilot.csv"
        self.assertEqual(self.reconcile(export).returncode, 0)
        r = self.command(self.stage_args("P2"))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("cache hits      not applicable", r.stdout)

        rows = list(export.read_text().splitlines())
        inflated = self.tmp / "inflated.csv"
        inflated.write_text("\n".join(rows + [rows[1]]) + "\n")
        r = self.reconcile(inflated)
        self.assertEqual(r.returncode, 3, r.stdout + r.stderr)
        self.assertIn("MORE THAN 1%", r.stdout)
        r = self.command(self.stage_args("P3"))
        self.assertIn("reconcile", r.stderr)
        r = self.command(["scripts/run_stage.py", "correct", "--run-root", str(self.root), "--data-root",
                          str(self.tmp), "--budget", "pilot", "--note", "The export repeated one row."])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.reconcile(export).returncode, 0)
        r = self.command(self.stage_args("P3", "--scripted-usage", "extreme"))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_paid_flags_without_all_three_and_eval_without_a_panel_are_refused(self):
        empty = self.tmp / "empty.txt"
        empty.write_text("")
        for extra in (["--paid"], ["--confirm-paid-calls"], ["--paid", "--confirm-paid-calls"],
                      ["--paid", "--confirm-paid-calls", "--approval-note", str(empty)]):
            r = self.command(self.stage_args("P1", *extra))
            self.assertNotEqual(r.returncode, 0, extra)
            self.assertIn("missing", r.stderr, extra)
        r = self.command(self.stage_args("EVAL"))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("needs --panel-manifest", r.stderr)
        self.assertFalse((self.root / "P1").exists())

    def test_the_guard_is_installed_and_the_network_refused(self):
        self.config.write_text(json.dumps({"test_output_dirs": [str(self.tmp / "packs")]}))
        r = self.command(self.stage_args("P1"))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("fence (dev_only)", r.stderr)
        probe = (f"import sys; sys.path.insert(0, {str(REPO / 'scripts')!r}); import run_stage as s; "
                 "s._forbid_network(); import socket; socket.create_connection(('127.0.0.1', 9), timeout=1)")
        r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no network calls", r.stderr)


if __name__ == "__main__":
    unittest.main()
