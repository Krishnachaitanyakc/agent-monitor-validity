"""Tests for the stage dry run (scripts/dry_run_stages.py) on the data path's fixture.

The fixture is normalized and packed as the data path and packer do; the dry run then
runs P1, P2 and P3 through the fake transport in a fresh process.
"""

import json
import sys
import unittest
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_run_stage import CAP, CommandCase  # noqa: E402


class TestDryRunStages(CommandCase):
    def test_dry_run_on_the_fixture(self):
        out = self.tmp / "dry"
        r = self.command(["scripts/dry_run_stages.py", "--data-root", str(self.tmp), "--packs", str(self.tmp / "packs"),
                          "--provenance", str(self.inputs / "provenance.jsonl"), "--out", str(out)])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        summary = json.loads((out / "summary.json").read_text())
        self.assertTrue(summary["ok"])
        self.assertEqual(summary["settings"]["fence"]["profiles"], ["dev_only"])
        self.assertIn("every socket refused", summary["settings"]["transport"])
        self.assertEqual(summary["rubric"]["frozen_rubric_rule"], "waived for this fake-mode rehearsal")
        for profile in ("central", "extreme"):
            sc = summary["scenarios"][profile]
            self.assertTrue(sc["complete"])
            self.assertEqual(sc["ledger"], f"{profile}/ledgers/ledger-pilot.jsonl")
            self.assertTrue(all(s["export_equals_settled"] for s in sc["stages"].values()))
            self.assertTrue(all(s["reconciliation"]["coverage"]["openai"]["scope"] == "proj_fake"
                                for s in sc["stages"].values()))
        halt = summary["scenarios"]["cap-check"]["halt"]
        self.assertEqual(halt["stopped"]["kind"], "spend_cap")
        self.assertLessEqual(Decimal(halt["committed_at_halt_usd"]), CAP)
        self.assertNotIn("estimate_check", summary)
        r = self.command(["scripts/dry_run_stages.py", "--data-root", str(self.tmp), "--packs", str(self.tmp / "packs"),
                          "--provenance", str(self.inputs / "provenance.jsonl"), "--out", str(out)])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("already exists", r.stderr)


if __name__ == "__main__":
    unittest.main()
