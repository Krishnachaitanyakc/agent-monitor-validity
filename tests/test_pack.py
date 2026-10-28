"""Tests for the pack layer (judge/) on synthetic fixtures.

Hand-built monitor-view records cover each acted-element status and the final-tree cap.
The data path's own fixture (test_datapath.build_fixture), normalized as the data path
does, covers the leak checks: its canaries sit in the reward field, the agent's prompts,
the agent and model names, the screenshot paths and hashes, and two test-split label rows.
The pack command runs in a fresh process, since an installed guard cannot be removed.
"""

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datapath import constants as C  # noqa: E402
from datapath.ids import Splits  # noqa: E402
from datapath.normalize import normalize, serialize as serialize_view  # noqa: E402
from judge import acted  # noqa: E402
from judge import pack as P  # noqa: E402
from test_datapath import (AGENT, CANARY_CHAT, CANARY_REWARD, CANARY_TEST_LABEL, MODEL,  # noqa: E402
                           build_fixture)

TREE = ("RootWebArea 'Shop', focused\n"
        "\t[12] button 'Buy', clickable, visible\n"
        "\t\t[13] textbox 'Search' value='red mug', clickable, visible, focused\n"
        "\t[47] generic, live='polite', atomic\n"
        "\t[120] link 'Item [12] details', clickable")
VISIBLE = {"visibility": 1.0, "bbox": [10, 20, 30, 40], "clickable": True, "set_of_marks": True}
OFFSCREEN = {"visibility": 0.0, "bbox": [1, -10009, 20, 20], "clickable": False, "set_of_marks": False}


def _event(step, action, axtree=TREE, elements=None, error="", reasoning=None, shot=True):
    return {
        "step": step, "action": action, "reasoning": reasoning,
        "url": f"https://wa-shopping.mcgill-nlp.org/page{step}",
        "open_pages_urls": [f"https://wa-shopping.mcgill-nlp.org/page{step}"],
        "focused_element": "13", "last_action_error": error,
        "observation": {"screenshot": ({"width": 4, "height": 3, "png_ok": True, "png_error": None}
                                       if shot else {"missing": True}),
                        "axtree": axtree, "axtree_pruned": ""},
        "elements": {"12": dict(VISIBLE)} if elements is None else elements,
    }


def _view(events, goal="Find the cheapest red mug."):
    return {"trace_key": "t0123456789abcdef", "schema_version": "0.2-arb",
            "task": {"goal": goal, "goal_image_urls": []}, "events": events}


def _two_steps(action, step0_tree=TREE, step0_elements=None, step1_tree=TREE, step1_elements=None):
    return _view([_event(0, action, step0_tree, step0_elements, reasoning="I will act."),
                  _event(1, None, step1_tree, step1_elements, error="Error: the click failed")])


class TestActed(unittest.TestCase):
    def one(self, action, axtree=TREE, elements=None):
        out = acted.describe_action(action, axtree, {"12": dict(VISIBLE)} if elements is None else elements)
        self.assertEqual(len(out), 1, out)
        return out[0]

    def test_resolved_keeps_the_line_and_properties(self):
        d = self.one("click('12')")
        self.assertEqual(d["status"], "resolved")
        self.assertEqual((d["call"], d["function"], d["parameter"], d["element_id"]), (0, "click", "bid", "12"))
        self.assertEqual(d["tree_lines"], [{"line": "\t[12] button 'Buy', clickable, visible",
                                            "role": "button", "name": "Buy",
                                            "attributes": "clickable, visible"}])
        self.assertEqual(d["properties"], VISIBLE)

    def test_resolved_without_properties(self):
        d = self.one("fill('13', 'blue mug')")
        self.assertEqual(d["status"], "resolved")
        self.assertEqual(d["tree_lines"][0]["name"], "Search")
        self.assertEqual(d["tree_lines"][0]["attributes"], "value='red mug', clickable, visible, focused")
        self.assertIsNone(d["properties"])

    def test_nonvisible_keeps_the_properties_exactly(self):
        d = self.one("click('42')", elements={"12": dict(VISIBLE), "42": dict(OFFSCREEN)})
        self.assertEqual(d["status"], "nonvisible")
        self.assertEqual(d["tree_lines"], [])
        self.assertEqual(d["properties"], OFFSCREEN)
        self.assertEqual(json.dumps(d["properties"], sort_keys=True), json.dumps(OFFSCREEN, sort_keys=True))

    def test_absent(self):
        d = self.one("click('780')")
        self.assertEqual((d["status"], d["tree_lines"], d["properties"]), ("absent", [], None))

    def test_missing_from_the_tree_and_present_but_nonvisible_stay_distinct(self):
        missing = self.one("click('42')", elements={})
        listed = self.one("click('42')", elements={"42": dict(OFFSCREEN)})
        self.assertEqual((missing["status"], listed["status"]), ("absent", "nonvisible"))

    def test_ambiguous_keeps_every_line_and_chooses_none(self):
        tree = TREE + "\n\t[12] link 'Buy again', clickable"
        d = self.one("click('12')", axtree=tree)
        self.assertEqual(d["status"], "ambiguous")
        self.assertEqual([x["line"] for x in d["tree_lines"]],
                         ["\t[12] button 'Buy', clickable, visible", "\t[12] link 'Buy again', clickable"])
        self.assertEqual(d["properties"], VISIBLE)

    def test_an_id_prefix_or_an_id_inside_a_name_is_not_a_match(self):
        self.assertEqual(self.one("click('1')")["status"], "absent")      # [12], [13], [120] only
        self.assertEqual(self.one("click('12')")["status"], "resolved")   # not the [120] line naming [12]

    def test_drag_and_drop_names_two_elements(self):
        out = acted.describe_action("drag_and_drop('12', '42')", TREE, {"12": VISIBLE, "42": OFFSCREEN})
        self.assertEqual([(d["parameter"], d["element_id"], d["status"]) for d in out],
                         [("from_bid", "12", "resolved"), ("to_bid", "42", "nonvisible")])
        out = acted.describe_action("drag_and_drop(from_bid='13', to_bid='99')", TREE, {})
        self.assertEqual([(d["parameter"], d["element_id"], d["status"]) for d in out],
                         [("from_bid", "13", "resolved"), ("to_bid", "99", "absent")])

    def test_keyword_element_id(self):
        self.assertEqual(self.one("click(bid='12', button='left')")["status"], "resolved")

    def test_multi_call_actions(self):
        out = acted.describe_action("fill('13', 'red mug')\npress('13', 'Enter')", TREE, {})
        self.assertEqual([(d["call"], d["function"], d["element_id"], d["status"]) for d in out],
                         [(0, "fill", "13", "resolved"), (1, "press", "13", "resolved")])
        out = acted.describe_action("scroll(0, 200)\nclick('42')", TREE, {})
        self.assertEqual([(d["call"], d["function"], d["status"]) for d in out], [(1, "click", "absent")])
        self.assertEqual(acted.describe_action("noop()\ngoto('https://h/')", TREE, {}),
                         [{"status": "no_target", "functions": ["noop", "goto"]}])

    def test_actions_without_an_element(self):
        for action, name in (("goto('https://wa-shopping.mcgill-nlp.org/')", "goto"),
                             ("scroll(0, 300)", "scroll"), ("send_msg_to_user('3')", "send_msg_to_user"),
                             ("noop(wait_ms=2000)", "noop"), ("tab_focus(0)", "tab_focus"),
                             ("go_back()", "go_back"), ("report_infeasible('no')", "report_infeasible"),
                             ("click()", "click")):
            self.assertEqual(acted.describe_action(action, TREE, {}),
                             [{"status": "no_target", "functions": [name]}], action)

    def test_unparseable_actions(self):
        for action in ("click('12'", "click(12)", "click(x)", "x = click('12')", "click('12') + 1",
                       "page.click('12')", "fill('12', click('13'))", "click(*ids)", "click(**kw)",
                       " click('12')", "", "# nothing", "click('12'); import os"):
            self.assertEqual(acted.describe_action(action, TREE, {}), [{"status": "unparsed"}], action)

    def test_actions_are_never_evaluated(self):
        action = "click(__import__('sys').exit(3))"
        self.assertEqual(acted.describe_action(action, TREE, {}), [{"status": "unparsed"}])
        self.assertEqual(acted.describe_action("exit(3)", TREE, {}),
                         [{"status": "no_target", "functions": ["exit"]}])

    def test_tree_line_shapes(self):
        cases = {
            "\t[47] generic, live='polite', atomic": ("generic", None, "live='polite', atomic"),
            "[9] link \"Men's shoes\", clickable": ("link", "Men's shoes", "clickable"),
            "[339] link '\\ue60a REPORTS', clickable, visible": ("link", " REPORTS", "clickable, visible"),
            "[572] image '', visible": ("image", "", "visible"),
            "[a] IframePresentational 'reCAPTCHA', visible": ("IframePresentational", "reCAPTCHA", "visible"),
            "[5] button": ("button", None, ""),
            "[5] ": (None, None, None),
            "[5] button 'unclosed": (None, None, None),
        }
        for line, (role, name, attributes) in cases.items():
            got = acted.parse_line(line)
            self.assertEqual(got, {"line": line, "role": role, "name": name, "attributes": attributes}, line)


class TestPack(unittest.TestCase):
    def test_step_fields(self):
        view = _view([_event(0, "click('12')", reasoning="I will buy it."),
                      _event(1, "send_msg_to_user('done')", error="Error: not found", shot=False),
                      _event(2, None, error="")])
        pack = P.build_pack(view)
        self.assertEqual(pack["pack_version"], P.PACK_VERSION)
        self.assertNotEqual(P.PACK_VERSION, view["schema_version"])
        self.assertEqual(pack["goal"], {"text": "Find the cheapest red mug.", "image_urls": []})
        s0, s1, s2 = pack["steps"]
        self.assertEqual((s0["step"], s0["action"], s0["reasoning_claimed_by_agent"]),
                         (0, "click('12')", "I will buy it."))
        self.assertEqual(s0["acted_elements"][0]["status"], "resolved")
        self.assertEqual(s0["url"], "https://wa-shopping.mcgill-nlp.org/page0")
        self.assertEqual(s1["error_reported_for_previous_action"], "Error: not found")
        self.assertEqual(s1["acted_elements"], [{"status": "no_target", "functions": ["send_msg_to_user"]}])
        self.assertEqual([s["screenshot"] for s in pack["steps"]], ["image-0", None, "image-2"])
        self.assertIsNone(s2["acted_elements"])
        self.assertIsNone(s2["action"])
        self.assertEqual(pack["final_tree"]["step"], 2)
        self.assertEqual(pack["final_tree"]["text"], TREE)
        for note in ("reasoning_claimed_by_agent", "error_reported_for_previous_action"):
            self.assertIn(note, pack["field_notes"])
        self.assertNotIn("trace_key", P.serialize(pack))

    def test_lookup_uses_only_the_step_that_chose_the_action(self):
        later = TREE + "\n\t[77] button 'Checkout', clickable, visible"
        cases = [
            # (step 0 elements, step 1 tree, step 1 elements, expected status at step 0)
            ({}, later, {"77": dict(VISIBLE)}, "absent"),
            ({"77": dict(OFFSCREEN)}, later, {"77": dict(VISIBLE)}, "nonvisible"),
            ({}, TREE, {"77": dict(VISIBLE)}, "absent"),
        ]
        for e0, t1, e1, status in cases:
            pack = P.build_pack(_two_steps("click('77')", step0_elements=e0, step1_tree=t1,
                                           step1_elements=e1))
            d = pack["steps"][0]["acted_elements"][0]
            self.assertEqual(d["status"], status)
            self.assertEqual(d["tree_lines"], [])
            self.assertEqual(d["properties"], e0.get("77"))
        # An earlier page does not count either.
        view = _view([_event(0, "click('12')", later), _event(1, "click('77')", TREE, {}), _event(2, None)])
        self.assertEqual(P.build_pack(view)["steps"][1]["acted_elements"][0]["status"], "absent")

    def test_deterministic_canonical_bytes(self):
        view = _two_steps("drag_and_drop('12', '42')", step0_elements={"42": dict(OFFSCREEN), "12": dict(VISIBLE)})
        a = P.serialize(P.build_pack(view))
        b = P.serialize(P.build_pack(json.loads(json.dumps(view))))
        self.assertEqual(a, b)

        def reverse(obj):
            if isinstance(obj, dict):
                return {k: reverse(obj[k]) for k in reversed(list(obj))}
            return [reverse(x) for x in obj] if isinstance(obj, list) else obj
        self.assertEqual(P.serialize(P.build_pack(reverse(view))), a)

        def sorted_pairs(pairs):
            keys = [k for k, _ in pairs]
            self.assertEqual(keys, sorted(keys))
            return dict(pairs)
        json.loads(a, object_pairs_hook=sorted_pairs)
        with self.assertRaises(ValueError):
            P.serialize({"x": float("nan")})

    def test_final_tree_cap_and_truncation_record(self):
        n = 50000
        tree = "".join(chr(ord("a") + i % 26) for i in range(n))
        pack = P.build_pack(_view([_event(0, "click('12')"), _event(1, None, tree)]))
        ft = pack["final_tree"]
        cut = ft["packer_truncation"]
        self.assertEqual(cut, {"applied": True, "original_chars": n, "kept_head_chars": 19200,
                               "kept_tail_chars": 12800, "omitted_chars": n - 32000,
                               "marker": f"\n[the packer cut {n - 32000} characters of this tree here]\n"})
        self.assertEqual(ft["text"], tree[:19200] + cut["marker"] + tree[-12800:])
        self.assertEqual(len(ft["text"]) - len(cut["marker"]), P.FINAL_TREE_CAP)
        for length, applied in ((32000, False), (32001, True), (0, False)):
            text, rec = P.cap_tree("x" * length)
            self.assertEqual(rec["applied"], applied, length)
            self.assertEqual(rec["original_chars"], length)
            if not applied:
                self.assertEqual((text, rec["kept_head_chars"], rec["marker"]), ("x" * length, length, None))
        self.assertEqual(P.cap_tree(None), (None, {"applied": False, "original_chars": 0, "kept_head_chars": 0,
                                                   "kept_tail_chars": 0, "omitted_chars": 0, "marker": None}))

    def test_truncation_is_not_recorded_as_a_run_ending(self):
        tree = "x" * 40000
        text = P.serialize(P.build_pack(_view([_event(0, "click('12')"), _event(1, None, tree)])))
        for key in ("truncated", "terminated", "ending", "cum_reward", "step_cap"):
            self.assertNotIn(f'"{key}"', text)

    def test_refuses_other_schemas_and_unordered_steps(self):
        view = _two_steps("click('12')")
        with self.assertRaises(ValueError):
            P.build_pack(dict(view, schema_version="0.1"))
        bad = json.loads(json.dumps(view))
        bad["events"][1]["step"] = 5
        with self.assertRaises(ValueError):
            P.build_pack(bad)

    def test_judge_modules_never_import_the_label_store(self):
        for rel in ("judge/acted.py", "judge/pack.py", "scripts/build_packs.py"):
            tree = ast.parse((REPO / rel).read_text(encoding="utf-8"))
            names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    names.add(node.module or "")
                    names.update(a.name for a in node.names)
                elif isinstance(node, ast.Import):
                    names.update(a.name for a in node.names)
            self.assertFalse({"labels", "datapath.labels", C.LABEL_BUNDLING_PACKAGE} & names, rel)


class FixtureCase(unittest.TestCase):
    """The data path's fixture, normalized as the data path does."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.fx = build_fixture(self.tmp)
        self.raw = self.fx["raw"]
        self.splits = Splits(self.raw / "data/splits.csv")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def norm(self, tid):
        bench, rel, t = self.fx["trajs"][tid]
        return normalize(json.loads((self.raw / rel).read_text()), self.raw, tid, self.splits.split(tid),
                         self.splits.group(tid), source_record={"path": rel, "sha256": "f" * 64})

    def write_inputs(self, folder: Path, tids=("webarena.1", "visualwebarena.resized.3")):
        folder.mkdir(parents=True, exist_ok=True)
        records = sorted((self.norm(t) for t in tids), key=lambda r: r[0]["trace_key"])
        (folder / "monitor_view.jsonl").write_text("".join(serialize_view(v) + "\n" for v, _ in records))
        (folder / "provenance.jsonl").write_text("".join(serialize_view(p) + "\n" for _, p in records))
        return folder


class TestPackLeakage(FixtureCase):
    def secrets(self, prov):
        out = [CANARY_REWARD, CANARY_CHAT, CANARY_TEST_LABEL, AGENT, MODEL, prov["trace_id"], prov["task_id"],
               prov["task_id_normalized"], prov["file"]["path"], prov["file"]["sha256"], "summary_info",
               "chat_messages", "cum_reward", "trajectory_success", "4. Completely Optimal", "2. Suboptimal",
               "screenshots/", "screenshot_step_", "sha256", "-xl-2", "annotations.csv"]
        for f in prov["screenshot_files"]:
            out += [f["path"], f["sha256"]]
        return out

    def test_no_identifier_path_hash_reward_or_label_reaches_the_pack(self):
        for tid in ("webarena.1", "visualwebarena.resized.3"):
            view, prov = self.norm(tid)
            pack, data, digest, manifest = P.build(view, prov)
            for s in self.secrets(prov):
                self.assertNotIn(s, data, f"{tid}: {s}")
            self.assertTrue(P.check_pack(pack, prov)["ok"])
            self.assertEqual(digest, P.sha256_text(data))
            self.assertEqual(len(manifest), len(prov["screenshot_files"]))
            for entry, f in zip(manifest, prov["screenshot_files"]):
                self.assertEqual((entry["step"], entry["path"], entry["sha256"], entry["bytes"]),
                                 (f["step"], f["path"], f["sha256"], f["bytes"]))
                self.assertEqual(entry["ref"], pack["steps"][f["step"]]["screenshot"])
                self.assertEqual((entry["trace_key"], entry["pack_sha256"]), (prov["trace_key"], digest))

    def test_check_pack_catches_injections(self):
        view, prov = self.norm("webarena.1")
        pack = P.build_pack(view)
        shot = prov["screenshot_files"][0]
        cases = [lambda p: p["steps"][0].update(screenshot=shot["path"]),
                 lambda p: p["steps"][0].update(screenshot=shot["sha256"]),
                 lambda p: p["steps"][0].update(image={"bytes": shot["bytes"]}),
                 lambda p: p.update(trace_key=view["trace_key"]),
                 lambda p: p["goal"].update(text=f"run by {AGENT}"),
                 lambda p: p["goal"].update(text="fake-model-2025 did this"),
                 lambda p: p["steps"][1].update(reasoning_claimed_by_agent="see webarena.1"),
                 lambda p: p["final_tree"].update(truncated=True),
                 lambda p: p["steps"][0].update(note="trajectory_success")]
        for inject in cases:
            p = json.loads(P.serialize(pack))
            inject(p)
            self.assertFalse(P.check_pack(p, prov)["ok"])

    def test_manifest_and_provenance_must_agree(self):
        view, prov = self.norm("webarena.1")
        other_view, _ = self.norm("visualwebarena.resized.3")
        with self.assertRaises(ValueError):
            P.build(other_view, prov)
        gone = json.loads(json.dumps(prov))
        gone["screenshot_files"][1].update(present=False)
        del gone["screenshot_files"][1]["sha256"]
        with self.assertRaises(ValueError):
            P.build(view, gone)
        twice = json.loads(json.dumps(prov))
        twice["screenshot_files"].append(dict(twice["screenshot_files"][0]))
        with self.assertRaises(ValueError):
            P.build(view, twice)

    def test_pack_line_round_trips_to_the_same_bytes(self):
        view, prov = self.norm("webarena.1")
        pack, data, digest, _ = P.build(view, prov)
        line = P.serialize({"trace_key": view["trace_key"], "pack_sha256": digest, "pack": pack})
        self.assertIn(data, line)
        self.assertEqual(P.serialize(json.loads(line)["pack"]), data)


class TestPackCommand(FixtureCase):
    def setUp(self):
        super().setUp()
        self.inputs = self.write_inputs(self.tmp / "dev_out")
        self.config = self.tmp / "fence_config.json"
        self.config.write_text("{}")

    def run_command(self, out_name, inputs=None, config=True):
        inputs = inputs or self.inputs
        cmd = [sys.executable, str(REPO / "scripts/build_packs.py"), "--data-root", str(self.tmp),
               "--monitor-view", str(inputs / "monitor_view.jsonl"),
               "--provenance", str(inputs / "provenance.jsonl"), "--out", str(self.tmp / out_name)]
        env = {k: v for k, v in os.environ.items() if k != "SPAR_FENCE_CONFIG"}
        if config:
            env["SPAR_FENCE_CONFIG"] = str(self.config)
        return subprocess.run(cmd, capture_output=True, text=True, env=env)

    def test_end_to_end_and_rerun_identical(self):
        for name in ("p1", "p2"):
            r = self.run_command(name)
            self.assertEqual(r.returncode, 0, r.stderr)
        for f in ("packs.jsonl", "image_manifest.jsonl"):
            self.assertEqual((self.tmp / "p1" / f).read_bytes(), (self.tmp / "p2" / f).read_bytes(), f)
        report = json.loads((self.tmp / "p1" / "report.json").read_text())
        self.assertTrue(report["ok"])
        self.assertEqual(report["settings"]["fence"]["profiles"], ["dev_only"])
        self.assertEqual(report["settings"]["pack_version"], P.PACK_VERSION)
        self.assertEqual(set(report["hashes"]), {"pinned_source", "normalized", "packs"})
        self.assertEqual(report["hashes"]["normalized"]["monitor_view.jsonl"],
                         hashlib.sha256((self.inputs / "monitor_view.jsonl").read_bytes()).hexdigest())
        # The fixture's full trees carry no element lines, and its element properties list '12'
        # only: click('0') is absent, and each click('12') is nonvisible.
        self.assertEqual(report["acted_element_status_counts"],
                         {"resolved": 0, "ambiguous": 0, "nonvisible": 2, "absent": 1, "no_target": 0,
                          "unparsed": 0})
        self.assertEqual(report["final_tree_truncation_counts"], {"cut_by_packer": 0, "kept_whole": 2})
        lines = [json.loads(x) for x in (self.tmp / "p1" / "packs.jsonl").read_text().splitlines()]
        self.assertEqual([x["trace_key"] for x in lines], list(report["hashes"]["packs"]["pack_sha256"]))
        for x in lines:
            data = P.serialize(x["pack"])
            self.assertEqual(P.sha256_text(data), x["pack_sha256"])
            self.assertEqual(report["hashes"]["packs"]["pack_sha256"][x["trace_key"]], x["pack_sha256"])
            for s in (CANARY_REWARD, CANARY_CHAT, CANARY_TEST_LABEL, AGENT, MODEL, "screenshots/", "sha256"):
                self.assertNotIn(s, data)
        manifest = (self.tmp / "p1" / "image_manifest.jsonl").read_text()
        self.assertEqual(len(manifest.splitlines()), 3 + 2)
        self.assertIn(f"screenshots/webarena/{AGENT}/webarena.1/screenshot_step_0.png", manifest)

    def test_existing_output_folder_is_refused(self):
        (self.tmp / "exists").mkdir()
        r = self.run_command("exists")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("already exists", r.stderr)
        self.assertEqual(list((self.tmp / "exists").iterdir()), [])

    def test_run_without_the_fence_config_stops(self):
        r = self.run_command("nocfg", config=False)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("SPAR_FENCE_CONFIG", r.stderr)
        self.assertFalse((self.tmp / "nocfg").exists())

    def test_guard_is_installed_before_the_inputs_are_read(self):
        guarded = self.write_inputs(self.tmp / "test_outputs")
        self.config.write_text(json.dumps({"test_output_dirs": [str(guarded)]}))
        r = self.run_command("guarded", inputs=guarded)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("fence (dev_only)", r.stderr)
        self.assertFalse((self.tmp / "guarded").exists())

    def test_evaluation_split_records_are_refused(self):
        inputs = self.write_inputs(self.tmp / "with_test", tids=("webarena.1", "webarena.2"))
        r = self.run_command("with_test_out", inputs=inputs)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("development data only", r.stderr)
        self.assertFalse((self.tmp / "with_test_out").exists())

    def test_failed_run_publishes_nothing(self):
        bad = self.write_inputs(self.tmp / "mismatch")
        lines = (bad / "provenance.jsonl").read_text().splitlines()
        (bad / "provenance.jsonl").write_text("\n".join(reversed(lines)) + "\n")
        r = self.run_command("mismatch_out", inputs=bad)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("does not match", r.stderr)
        self.assertFalse((self.tmp / "mismatch_out").exists())

    def test_refuses_an_output_folder_inside_the_repository(self):
        cmd = [sys.executable, str(REPO / "scripts/build_packs.py"), "--data-root", str(self.tmp),
               "--monitor-view", "x", "--provenance", "y", "--out", str(REPO / "out_packs")]
        r = subprocess.run(cmd, capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("outside the repository", r.stderr)
        self.assertFalse((REPO / "out_packs").exists())


if __name__ == "__main__":
    unittest.main()
