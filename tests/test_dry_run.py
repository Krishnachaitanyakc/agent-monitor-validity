"""Tests for the development dry run (scripts/dry_run.py) on the data path's fixture.

The fixture is normalized and packed as the data path and packer do, then the dry run
builds both requests per pack and dispatches them through FakeTransport. Each command
runs in a fresh process, since an installed guard cannot be removed.
"""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from judge import requests as R  # noqa: E402
from judge.storage import AttemptLog  # noqa: E402
from test_pack import FixtureCase  # noqa: E402


class TestDryRunCommand(FixtureCase):
    def setUp(self):
        super().setUp()
        self.inputs = self.write_inputs(self.tmp / "dev_out")
        self.config = self.tmp / "fence_config.json"
        self.config.write_text("{}")
        r = self.command(["scripts/build_packs.py", "--data-root", str(self.tmp),
                      "--monitor-view", str(self.inputs / "monitor_view.jsonl"),
                      "--provenance", str(self.inputs / "provenance.jsonl"), "--out", str(self.tmp / "packs")])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.keys = [json.loads(x)["trace_key"] for x in (self.tmp / "packs" / "packs.jsonl").read_text().splitlines()]

    def command(self, args, config=True):
        env = {k: v for k, v in os.environ.items() if k != "SPAR_FENCE_CONFIG"}
        if config:
            env["SPAR_FENCE_CONFIG"] = str(self.config)
        return subprocess.run([sys.executable, str(REPO / args[0])] + args[1:], capture_output=True, text=True,
                              env=env)

    def dry(self, out, packs=None, config=True):
        return self.command(["scripts/dry_run.py", "--data-root", str(self.tmp), "--packs", str(packs or self.tmp / "packs"),
                         "--provenance", str(self.inputs / "provenance.jsonl"), "--example-trace-key", self.keys[0],
                         "--out", str(self.tmp / out)], config=config)

    def test_end_to_end_and_rerun_identical(self):
        for out in ("d1", "d2"):
            r = self.dry(out)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        d1, d2 = self.tmp / "d1", self.tmp / "d2"
        self.assertEqual((d1 / "request_hashes.jsonl").read_bytes(), (d2 / "request_hashes.jsonl").read_bytes())
        for name in ("example_request_A.json", "example_request_B.json"):
            self.assertEqual((d1 / name).read_bytes(), (d2 / name).read_bytes())
        report = json.loads((d1 / "report.json").read_text())
        self.assertTrue(report["ok"])
        self.assertEqual(report["config_sha256"], R.config_sha256())
        self.assertEqual(report["settings"]["fence"]["profiles"], ["dev_only"])
        self.assertEqual((report["packs"], report["eligible"], report["deterministic"]), (2, 2, True))
        for row in report["summary"]:
            self.assertTrue(row["ok"])
            self.assertTrue(row["same_evidence_both_judges"])
            for j in "AB":
                self.assertTrue(row["requests"][j]["leak_check"]["ok"])
                self.assertEqual(row["requests"][j]["table_problems"], [])
        # Two packs run the first two scenarios: all valid, then a retry, a malformed A3 and a refusal.
        self.assertEqual(report["outcome_counts_all_attempts"]["TRANSPORT"], 1)
        self.assertEqual(report["outcome_counts_all_attempts"]["REFUSAL"], 1)
        example = json.loads((d1 / "example_request_A.json").read_text())
        self.assertTrue(example["placeholders_restore_the_exact_bytes"])
        first = json.loads((self.tmp / "packs" / "packs.jsonl").read_text().splitlines()[0])["pack"]
        self.assertEqual(example["images_replaced"], sum(s["screenshot"] is not None for s in first["steps"]))
        self.assertGreater(example["images_replaced"], 0)
        self.assertNotIn("iVBORw0KGgo", (d1 / "example_request_A.json").read_text())   # no PNG base64 left
        log = AttemptLog(d1 / "attempt_log")
        self.assertEqual(len(log.attempts(self.keys[0], report["config_sha256"], "A", 1)), 1)

    def test_guard_is_installed_and_network_refused(self):
        self.config.write_text(json.dumps({"test_output_dirs": [str(self.tmp / "packs")]}))
        r = self.dry("guarded")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("fence (dev_only)", r.stderr)
        self.assertFalse((self.tmp / "guarded").exists())
        probe = ("import sys, runpy; sys.argv = ['dry_run.py']; "
                 f"sys.path.insert(0, {str(REPO)!r}); import scripts.dry_run as d; d._forbid_network(); "
                 "import socket; socket.create_connection(('127.0.0.1', 9), timeout=1)")
        r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, cwd=str(REPO))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no network calls", r.stderr)

    def test_existing_output_or_missing_fence_config_is_refused(self):
        (self.tmp / "exists").mkdir()
        r = self.dry("exists")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("already exists", r.stderr)
        r = self.dry("nocfg", config=False)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("SPAR_FENCE_CONFIG", r.stderr)
        self.assertFalse((self.tmp / "nocfg").exists())


if __name__ == "__main__":
    unittest.main()
