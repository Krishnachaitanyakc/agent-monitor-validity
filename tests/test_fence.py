"""Tests for the file-open guard in datapath.fence, on a synthetic fixture.

An installed audit hook cannot be removed, so every case runs in a fresh Python process.
"""

import json
import os
import subprocess
import sys
import tempfile
import shutil
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_datapath import AGENT, build_fixture, listing_entry  # noqa: E402

PROBE = r'''
import io, json, os, sys
from unittest import mock
import urllib.request
sys.path.insert(0, sys.argv[1])
from datapath import fence, fetch, labels
from datapath.ids import Splits
p = json.loads(sys.argv[2])
for path in p["make"]:                      # fixture files, created before the guard exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("x")
fence.install(p["profile"], p["root"], p["config"])
raw = os.path.join(p["root"], "raw")
label = os.path.join(raw, "data", "annotations.csv")
splits = Splits(os.path.join(raw, "data", "splits.csv"))

def load_labels(path):                      # an impostor with the reviewed reader's name
    with open(path) as f:
        return f.read()

def redownload_label():
    good = open(label + ".copy", "rb").read()
    class R(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False
    with mock.patch.object(urllib.request, "urlopen", return_value=R(good)):
        fetch.fetch_file("data/annotations.csv", __import__("pathlib").Path(raw), p["label_entry"])

results = {}
def attempt(name, fn):
    try:
        fn()
        results[name] = "ok"
    except PermissionError:
        results[name] = "blocked"
    except Exception as e:
        results[name] = "error " + type(e).__name__
attempt("label_direct", lambda: open(label).read())
attempt("label_os_open", lambda: os.close(os.open(label, os.O_RDONLY)))
attempt("label_impostor", lambda: load_labels(label))
attempt("load_labels_dev", lambda: labels.load_labels(label, splits))
attempt("annotation_structure", lambda: labels.annotation_structure(label, splits))
attempt("fetch_hash_check", lambda: fetch.fetch_file("data/annotations.csv", __import__("pathlib").Path(raw), p["label_entry"]))
attempt("fetch_redownload", redownload_label)
attempt("part_direct", lambda: open(label + ".part", "w").write("x"))
for name, path in p["files"].items():
    attempt(name, lambda path=path: open(path, "rb").read())
print(json.dumps(results))
'''


class FenceCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.fx = build_fixture(self.tmp)
        self.raw = self.fx["raw"]
        # Created inside each probe process, never in this one: guards installed by earlier
        # in-process tests stay active here, and one of them covers any judgments folder.
        files = {name: str(self.tmp / rel) for name, rel in (
            ("judgments", "judgments/x/y.json"), ("fenced", "fenced/plan.md"),
            ("fenced_file", "notes/session_log.md"),
            ("transcript", "transcripts/s.jsonl"), ("archive", "archive/old.md"),
            ("test_output", "outputs/panel.jsonl"), ("eval_output", "eval/run.jsonl"),
            ("join", "join/joined.jsonl"))}
        self.made = list(files.values())
        files["dev_traj"] = str(self.raw / self.fx["trajs"]["webarena.1"][1])
        files["test_traj"] = str(self.raw / self.fx["trajs"]["webarena.2"][1])
        files["dev_shot"] = str(self.raw / f"screenshots/webarena/{AGENT}/webarena.1/screenshot_step_0.png")
        files["test_shot"] = str(self.raw / f"screenshots/webarena/{AGENT}/webarena.2/screenshot_step_0.png")
        files["splits"] = str(self.raw / "data/splits.csv")
        self.files = files
        self.config = {"fenced_dirs": [str(self.tmp / "fenced")],
                       "fenced_files": [str(self.tmp / "notes/session_log.md")],
                       "transcript_dirs": [str(self.tmp / "transcripts")],
                       "archive_dirs": [str(self.tmp / "archive")],
                       "test_output_dirs": [str(self.tmp / "outputs")],
                       "eval_output_dirs": [str(self.tmp / "eval")],
                       "join_dirs": [str(self.tmp / "join")]}
        label = self.raw / "data/annotations.csv"
        shutil.copy(label, str(label) + ".copy")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def probe(self, profile):
        label_entry = listing_entry(self.raw, "data/annotations.csv")
        arg = json.dumps({"profile": profile, "root": str(self.tmp), "config": self.config,
                          "files": self.files, "label_entry": label_entry, "make": self.made})
        r = subprocess.run([sys.executable, "-c", PROBE, str(REPO), arg],
                           capture_output=True, text=True, cwd=str(self.tmp))
        self.assertEqual(r.returncode, 0, r.stderr)
        return json.loads(r.stdout)


ALWAYS_BLOCKED = ("label_direct", "label_os_open", "label_impostor", "part_direct",
                  "judgments", "fenced", "fenced_file", "transcript", "archive")
READERS_ALLOWED = ("load_labels_dev", "annotation_structure", "fetch_hash_check",
                   "fetch_redownload", "dev_traj", "dev_shot", "splits")


class TestProfiles(FenceCase):
    def check(self, profile, blocked, allowed):
        got = self.probe(profile)
        for name in blocked:
            self.assertEqual(got[name], "blocked", f"{profile}: {name}")
        for name in allowed:
            self.assertEqual(got[name], "ok", f"{profile}: {name}")
        self.assertEqual(set(blocked) | set(allowed), set(got), profile)

    def test_label_free(self):
        self.check("label_free", ALWAYS_BLOCKED,
                   READERS_ALLOWED + ("test_traj", "test_shot", "test_output", "eval_output", "join"))

    def test_dev_only(self):
        self.check("dev_only", ALWAYS_BLOCKED + ("test_traj", "test_shot", "test_output"),
                   READERS_ALLOWED + ("eval_output", "join"))

    def test_pre_join(self):
        self.check("pre_join", ALWAYS_BLOCKED + ("test_traj", "test_shot", "test_output",
                                                 "eval_output", "join"),
                   READERS_ALLOWED)

    def test_unknown_profile_and_config_keys_are_refused(self):
        from datapath import fence
        with self.assertRaises(ValueError):
            fence.install("everything", self.tmp)
        cfg = self.tmp / "cfg.json"
        cfg.write_text(json.dumps({"fenced_dir": ["typo"]}))
        with self.assertRaises(ValueError):
            fence.load_config(cfg)


class TestDataPathCommand(FenceCase):
    def run_command(self, trajectories, out_name, owner=False):
        cmd = [sys.executable, str(REPO / "scripts/arb_datapath.py"), "run",
               "--data-root", str(self.tmp), "--out", str(self.tmp / out_name)]
        for t in trajectories:
            cmd += ["--trajectory", t]
        if owner:
            cmd.append("--owner-label-free")
        env = {k: v for k, v in os.environ.items() if k != "SPAR_FENCE_CONFIG"}
        return subprocess.run(cmd, capture_output=True, text=True, env=env)

    def test_default_run_cannot_open_an_evaluation_trajectory(self):
        r = self.run_command([f"webarena/{AGENT}/webarena.2"], "o_test")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("fence (dev_only)", r.stderr)
        self.assertFalse((self.tmp / "o_test").exists())

    def test_default_run_installs_both_profiles(self):
        r = self.run_command([f"webarena/{AGENT}/webarena.1"], "o_dev")
        self.assertEqual(r.returncode, 0, r.stderr)
        report = json.loads((self.tmp / "o_dev" / "report.json").read_text())
        self.assertEqual(report["settings"]["fence"]["profiles"], ["label_free", "dev_only"])

    def test_owner_run_keeps_only_the_label_profile(self):
        r = self.run_command([f"webarena/{AGENT}/webarena.2"], "o_owner", owner=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        report = json.loads((self.tmp / "o_owner" / "report.json").read_text())
        self.assertEqual(report["settings"]["fence"]["profiles"], ["label_free"])


if __name__ == "__main__":
    unittest.main()
