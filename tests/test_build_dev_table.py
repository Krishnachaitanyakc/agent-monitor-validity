"""Tests for the development table and the template exclusion, scripts/build_dev_table.py, on synthetic counts files."""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import build_dev_table as B  # noqa: E402
import draw_allocation as DA  # noqa: E402


def row(task_id, split, benchmark, template, image_goal=False):
    return {"task_id": task_id, "split": split, "benchmark": benchmark, "cluster_key": template, "image_goal": image_goal}


MAPPING = [
    row("webarena.1", "dev", "webarena", "w:held-once"),
    row("webarena.2", "dev", "webarena", "w:test-only"),
    row("webarena.3", "dev", "webarena", "w:dev-only"),
    row("visualwebarena.1", "dev", "visualwebarena", "v:held-twice"),
    row("visualwebarena.2", "dev", "visualwebarena", "v:held-twice", image_goal=True),
    row("assistantbench.1", "dev", "assistantbench", "a:1"),
    row("workarena.a-l1", "dev", "workarena_l1", "x:l1"),
    row("workarena.a-l2", "dev", "workarena_l2", "x:shared"),
    row("webarena.101", "test", "webarena", "w:held-once"),
    row("webarena.102", "test", "webarena", "w:test-only"),
    row("visualwebarena.101", "test", "visualwebarena", "v:held-twice"),
    row("visualwebarena.102", "test", "visualwebarena", "v:held-twice"),
    row("workarena.panel-l2", "test", "workarena_l2", "x:shared"),
]
PANEL = ["webarena.101", "visualwebarena.101", "visualwebarena.102", "workarena.panel-l2"]


class TestTable(unittest.TestCase):
    def setUp(self):
        self.table = B.build_table(MAPPING, PANEL)
        self.by_id = {r["task_id"]: r for r in self.table}

    def test_the_pool_is_the_development_split_without_assistantbench_or_image_goals(self):
        self.assertEqual(sorted(self.by_id), ["visualwebarena.1", "webarena.1", "webarena.2", "webarena.3",
                                              "workarena.a-l1", "workarena.a-l2"])

    def test_a_row_has_exactly_the_draw_s_fields_and_the_draw_can_read_it(self):
        self.assertEqual(self.table[0], {"task_id": "visualwebarena.1", "benchmark": "visualwebarena",
                                         "template_overlaps_panel": True, "family": "v:held-twice"})
        self.assertEqual(set(DA._check_table(self.table)), set(self.by_id))

    def test_rows_are_sorted_in_any_input_order(self):
        self.assertEqual(self.table, B.build_table(list(reversed(MAPPING)), list(reversed(PANEL))))
        self.assertEqual([r["task_id"] for r in self.table], sorted(self.by_id))

    def test_the_level_of_a_workarena_task_leaves_its_benchmark(self):
        self.assertEqual({self.by_id[t]["benchmark"] for t in ("workarena.a-l1", "workarena.a-l2")}, {"workarena"})

    def test_a_template_overlaps_only_when_it_holds_a_panel_task(self):
        self.assertTrue(self.by_id["webarena.1"]["template_overlaps_panel"])
        self.assertTrue(self.by_id["visualwebarena.1"]["template_overlaps_panel"])
        self.assertFalse(self.by_id["webarena.2"]["template_overlaps_panel"], "a test task outside the panel")
        self.assertFalse(self.by_id["webarena.3"]["template_overlaps_panel"], "no test task at all")

    def test_a_workarena_panel_task_does_not_make_its_template_overlap(self):
        self.assertFalse(self.by_id["workarena.a-l2"]["template_overlaps_panel"])

    def test_the_bytes_are_the_documented_json(self):
        self.assertEqual(B.json_bytes(self.table), (json.dumps(self.table, indent=2, sort_keys=True) + "\n").encode())


class TestExclusion(unittest.TestCase):
    def test_the_templates_of_p2_that_hold_panel_tasks_are_excluded_with_their_task_count(self):
        record = B.template_exclusion(["webarena.1", "webarena.2", "webarena.3", "visualwebarena.1", "workarena.a-l2"],
                                      MAPPING, PANEL)
        self.assertEqual(record, {"excluded_templates": ["v:held-twice", "w:held-once"], "n_before": 4,
                                  "excluded_tasks": 3, "n_after": 1})

    def test_a_template_counts_once_however_many_prompt_tasks_use_it(self):
        record = B.template_exclusion(["visualwebarena.1", "visualwebarena.2"], MAPPING, PANEL)
        self.assertEqual((record["excluded_tasks"], record["n_after"]), (2, 2))

    def test_no_overlap_excludes_nothing(self):
        self.assertEqual(B.template_exclusion(["webarena.2"], MAPPING, PANEL),
                         {"excluded_templates": [], "n_before": 4, "excluded_tasks": 0, "n_after": 4})

    def test_a_prompt_task_not_in_the_mapping_is_refused(self):
        with self.assertRaisesRegex(B.BuildRefused, "prompt tasks are missing"):
            B.template_exclusion(["webarena.999"], MAPPING, PANEL)


class TestRefusals(unittest.TestCase):
    def test_a_panel_task_missing_from_the_mapping_is_refused_by_count_only(self):
        with self.assertRaisesRegex(B.BuildRefused, "1 panel tasks are missing") as e:
            B.build_table(MAPPING, PANEL + ["webarena.777"])
        self.assertNotIn("webarena.777", str(e.exception))

    def test_a_panel_task_outside_the_test_split_is_refused_by_count_only(self):
        with self.assertRaisesRegex(B.BuildRefused, "1 panel tasks are not in the test split") as e:
            B.build_table(MAPPING, PANEL + ["webarena.1"])
        self.assertNotIn("webarena.1", str(e.exception))

    def test_a_task_listed_twice_in_the_mapping_is_refused(self):
        with self.assertRaisesRegex(B.BuildRefused, "1 task IDs appear twice"):
            B.build_table(MAPPING + MAPPING[:1], PANEL)


class TestCommands(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.counts = self.tmp / "counts"
        self.counts.mkdir()
        self.write_counts(MAPPING, PANEL)

    def write_counts(self, mapping, panel):
        (self.counts / B.MAPPING_FILE).write_text(json.dumps(mapping))
        (self.counts / B.ELIGIBLE_FILE).write_text(json.dumps({"eligible_task_ids_after": panel}))

    def run_tool(self, *args):
        return subprocess.run([sys.executable, str(REPO / "scripts/build_dev_table.py"), *args], capture_output=True, text=True)

    def run_table(self, out):
        return self.run_tool("table", "--counts", str(self.counts), "--out", str(out))

    def allocation(self):
        """A small allocation over the mapping's development tasks, with its own hash."""
        p1, p2_new = ["webarena.1"], ["visualwebarena.1"]
        record = {"p1": p1, "p2": sorted(p1 + p2_new), "p2_new": p2_new, "p3": [], "development": sorted(p1 + p2_new)}
        record["sha256"] = DA.content_sha256(record)
        path = self.tmp / "allocation.json"
        path.write_text(json.dumps(record))
        return path, record

    def test_the_table_is_written_as_the_build_gives_it_and_printed_as_counts(self):
        out = self.tmp / "table.json"
        r = self.run_table(out)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(out.read_bytes(), B.json_bytes(B.build_table(MAPPING, PANEL)))
        self.assertIn("rows 6", r.stdout)

    def test_a_table_is_written_once(self):
        out = self.tmp / "table.json"
        self.assertEqual(self.run_table(out).returncode, 0)
        before = out.read_bytes()
        self.write_counts(MAPPING[:-1], PANEL[:-1])
        r = self.run_table(out)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("written once", r.stderr)
        self.assertEqual(out.read_bytes(), before)

    def test_no_panel_task_id_reaches_the_terminal(self):
        table, exclusion = self.tmp / "t.json", self.tmp / "e.json"
        runs = [self.run_table(table),
                self.run_tool("exclusion", "--counts", str(self.counts), "--allocation", str(self.allocation()[0]),
                              "--out", str(exclusion))]
        self.assertEqual([r.returncode for r in runs], [0, 0], [r.stderr for r in runs])
        self.write_counts(MAPPING, PANEL + ["webarena.777"])
        runs.append(self.run_table(self.tmp / "refused.json"))
        for r in runs:
            for task in [*PANEL, "webarena.777"]:
                self.assertNotIn(task, r.stdout + r.stderr)
        self.assertFalse((self.tmp / "refused.json").exists())

    def test_the_exclusion_uses_all_of_p2_names_its_allocation_and_needs_a_verified_one(self):
        (allocation, record), out = self.allocation(), self.tmp / "exclusion.json"
        r = self.run_tool("exclusion", "--counts", str(self.counts), "--allocation", str(allocation), "--out", str(out))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("n_after 1", r.stdout)
        self.assertEqual(json.loads(out.read_text()), {
            "allocation_sha256": record["sha256"], "excluded_templates": ["v:held-twice", "w:held-once"], "n_before": 4,
            "excluded_tasks": 3, "n_after": 1})
        allocation.write_text(json.dumps(dict(record, sha256="0" * 64)))
        r = self.run_tool("exclusion", "--counts", str(self.counts), "--allocation", str(allocation),
                          "--out", str(self.tmp / "second.json"))
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse((self.tmp / "second.json").exists())

    def test_files_that_cannot_be_read_as_counts_are_refused(self):
        (self.counts / B.MAPPING_FILE).write_text("{not json")
        out = self.tmp / "t.json"
        r = self.run_table(out)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("refused", r.stderr)
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
