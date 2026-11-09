"""Tests for the development allocation draw, scripts/draw_allocation.py, on synthetic task IDs."""

import hashlib
import json
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import draw_allocation as DA  # noqa: E402

INFEASIBLE = "workarena.synthetic.infeasible"
P1 = ["webarena.0", "webarena.1", "webarena.2", "visualwebarena.0", "workarena.synthetic.0"]


def row(task_id, benchmark, overlaps=False, family=None):
    return {"task_id": task_id, "benchmark": benchmark, "template_overlaps_panel": overlaps,
            "family": family or f"{benchmark}:{task_id}"}


def table():
    """42 synthetic tasks shaped like the real pool: 22, 5 and 15 by benchmark."""
    web = [row(f"webarena.{i}", "webarena", overlaps=i % 2 == 0 and i > 2) for i in range(22)]
    visual = [row(f"visualwebarena.{i}", "visualwebarena", overlaps=i in (1, 2)) for i in range(5)]
    work = [row(f"workarena.synthetic.{i}", "workarena") for i in range(14)]
    work.append(row(INFEASIBLE, "workarena", family="workarena:l2-infeasible-navigate"))
    return web + visual + work


def overlaps(rows):
    return {r["task_id"] for r in rows if r["template_overlaps_panel"]}


class TestDraw(unittest.TestCase):
    def test_the_same_input_gives_identical_bytes_in_any_table_or_p1_order(self):
        first = DA.allocation_bytes(DA.draw(table(), P1, 169))
        self.assertEqual(first, DA.allocation_bytes(DA.draw(table(), P1, 169)))
        for shuffle in (1, 2, 3):
            shuffled = table()
            random.Random(shuffle).shuffle(shuffled)
            self.assertEqual(first, DA.allocation_bytes(DA.draw(shuffled, list(reversed(P1)), 169)))

    def test_the_draw_follows_the_rule(self):
        rows = table()
        a = DA.draw(rows, P1, 169)
        by_id = {r["task_id"]: r for r in rows}
        self.assertEqual((len(a["p1"]), len(a["p2_new"]), len(a["p2"]), len(a["p3"])), (5, 15, 20, 20))
        self.assertEqual(a["p1"], sorted(P1))
        self.assertEqual(a["p2"], sorted(set(P1) | set(a["p2_new"])))
        self.assertFalse(set(a["p2_new"]) & set(P1))
        self.assertFalse(set(a["p2_new"]) & overlaps(rows))
        self.assertFalse(set(a["p3"]) & set(a["p2"]))
        self.assertIn(INFEASIBLE, a["p2_new"])
        self.assertEqual(a["counts_by_benchmark"]["p2_new"], {"webarena": 8, "visualwebarena": 1, "workarena": 6})
        self.assertEqual(a["counts_by_benchmark"]["p1"], {"webarena": 3, "visualwebarena": 1, "workarena": 1})
        for name in DA.LISTS:
            self.assertEqual(sum(a["counts_by_benchmark"][name].values()), len(a[name]))
            counted = {b: sum(by_id[t]["benchmark"] == b for t in a[name]) for b in DA.QUOTA}
            self.assertEqual(a["counts_by_benchmark"][name], counted)
        self.assertEqual((a["seed"], a["rule"]), (169, DA.RULE))
        for name in DA.LISTS:
            self.assertEqual(a[name], sorted(a[name]), name)
        self.assertEqual(sorted(a), sorted([*DA.LISTS, DA.DEVELOPMENT, "seed", "rule", "counts_by_benchmark",
                                            "sha256"]))

    def test_the_development_list_holds_every_table_task_drawn_or_not(self):
        rows = table()
        a = DA.draw(rows, P1, 169)
        self.assertEqual(a[DA.DEVELOPMENT], sorted(r["task_id"] for r in rows))
        self.assertGreater(set(a[DA.DEVELOPMENT]), set(a["p2"]) | set(a["p3"]))

    def test_p3_comes_from_the_whole_table_after_p2_and_may_hold_overlapping_templates(self):
        rows = table()
        a = DA.draw(rows, P1, 169)
        remaining = {r["task_id"] for r in rows} - set(a["p2"])
        self.assertEqual(len(remaining), 22)
        self.assertLess(set(a["p3"]), remaining)
        self.assertTrue(set(a["p3"]) & overlaps(rows))

    def test_each_stratum_and_p3_use_the_documented_seed_of_the_seed_and_stratum(self):
        expected = int.from_bytes(hashlib.sha256(b"169:webarena").digest(), "big")
        self.assertEqual(DA.stratum_seed(169, "webarena"), expected)
        self.assertEqual(len({DA.stratum_seed(169, s) for s in (*DA.QUOTA, DA.P3_STRATUM)}), 4)
        self.assertNotEqual(DA.stratum_seed(169, "webarena"), DA.stratum_seed(170, "webarena"))
        rows, a = table(), DA.draw(table(), P1, 169)
        pool = [r["task_id"] for r in rows if r["benchmark"] == "webarena" and not r["template_overlaps_panel"]
                and r["task_id"] not in P1]
        self.assertEqual(sorted(DA.permuted(pool, 169, "webarena")[:8]),
                         sorted(t for t in a["p2_new"] if t.startswith("webarena.")))
        remaining = {r["task_id"] for r in rows} - set(a["p2"])
        self.assertEqual(sorted(DA.permuted(remaining, 169, "p3")[:20]), a["p3"])

    def test_the_permutation_sorts_then_orders_by_the_generator_s_random_values(self):
        ids = ["c", "a", "b", "d"]
        rng = random.Random(DA.stratum_seed(5, "s"))
        values = {t: rng.random() for t in ("a", "b", "c", "d")}
        self.assertEqual(DA.permuted(ids, 5, "s"), sorted(ids, key=values.__getitem__))
        self.assertEqual(DA.permuted(ids, 5, "s"), DA.permuted(sorted(ids), 5, "s"))

    def test_the_draw_for_a_known_input_never_changes(self):
        self.assertEqual(DA.permuted([f"t{i}" for i in range(8)], 169, "webarena"),
                         ["t1", "t5", "t4", "t6", "t2", "t0", "t3", "t7"])

    def test_another_seed_gives_another_draw(self):
        self.assertNotEqual(DA.draw(table(), P1, 169)["p3"], DA.draw(table(), P1, 170)["p3"])

    def test_the_rule_holds_on_every_seed(self):
        rows = table()
        for seed in range(30):
            a = DA.draw(rows, P1, seed)
            self.assertIn(INFEASIBLE, a["p2_new"], seed)
            self.assertEqual(sum(t.startswith("workarena.") for t in a["p2_new"]), 6, seed)
            self.assertFalse(set(a["p2_new"]) & set(P1), seed)
            self.assertFalse(set(a["p2_new"]) & overlaps(rows), seed)
            self.assertFalse(set(a["p3"]) & set(a["p2"]), seed)

    def test_an_infeasible_task_already_in_p1_is_not_drawn_again(self):
        p1 = [*P1[:4], INFEASIBLE]
        a = DA.draw(table(), p1, 169)
        self.assertNotIn(INFEASIBLE, a["p2_new"])
        self.assertIn(INFEASIBLE, a["p2"])
        self.assertEqual(a["counts_by_benchmark"]["p2_new"]["workarena"], 6)
        self.assertEqual(len(a["p2"]), 20)

    def test_an_infeasible_task_in_p1_may_overlap_the_panel(self):
        rows = table()
        next(r for r in rows if r["task_id"] == INFEASIBLE)["template_overlaps_panel"] = True
        self.assertEqual(len(DA.draw(rows, [*P1[:4], INFEASIBLE], 169)["p2"]), 20)

    def test_the_hash_covers_the_content_and_leaves_out_only_itself(self):
        a = DA.draw(table(), P1, 169)
        self.assertEqual(a["sha256"], DA.content_sha256(a))
        canonical = json.dumps({k: v for k, v in a.items() if k != "sha256"}, sort_keys=True, separators=(",", ":"))
        self.assertEqual(a["sha256"], hashlib.sha256(canonical.encode()).hexdigest())
        for name, change in (("p1", ["x"]), ("p2", ["x"]), ("p2_new", ["x"]), ("p3", ["x"]), (DA.DEVELOPMENT, ["x"]),
                             ("seed", 1), ("rule", "x"), ("counts_by_benchmark", {})):
            self.assertNotEqual(DA.content_sha256(dict(a, **{name: change})), a["sha256"], name)


class TestRefusals(unittest.TestCase):
    def refused(self, message, rows=None, p1=None, seed=169):
        with self.assertRaisesRegex(DA.DrawRefused, message):
            DA.draw(table() if rows is None else rows, P1 if p1 is None else p1, seed)

    def test_a_short_stratum(self):
        cases = {"webarena": 8, "visualwebarena": 1, "workarena": 6}
        for benchmark, needed in cases.items():
            rows = table()
            free = [r for r in rows if r["benchmark"] == benchmark and not r["template_overlaps_panel"]
                    and r["task_id"] not in P1 and r["task_id"] != INFEASIBLE]
            for r in free[max(needed - 2, 0):]:
                rows.remove(r)
            self.refused(f"the {benchmark} stratum holds", rows)

    def test_a_stratum_counted_with_overlapping_or_p1_tasks_is_still_short(self):
        rows = table()
        for r in rows:
            if r["benchmark"] == "visualwebarena":
                r["template_overlaps_panel"] = r["task_id"] != "visualwebarena.0"
        self.refused("the visualwebarena stratum holds", rows)

    def test_a_stratum_of_exactly_the_quota_is_enough(self):
        rows = table()
        free = [r for r in rows if r["benchmark"] == "visualwebarena" and not r["template_overlaps_panel"]
                and r["task_id"] not in P1]
        for r in free[1:]:
            rows.remove(r)
            rows.append(row(r["task_id"] + "x", "webarena", overlaps=True))
        self.assertEqual(DA.draw(rows, P1, 169)["counts_by_benchmark"]["p2_new"]["visualwebarena"], 1)

    def test_too_few_tasks_left_for_p3(self):
        rows = table()
        drawn = DA.draw(rows, P1, 169)
        spare = sorted(set(r["task_id"] for r in rows) - set(drawn["p2"]))
        self.refused("remain after P2", [r for r in rows if r["task_id"] not in spare[:3]])
        exactly = [r for r in rows if r["task_id"] not in spare[:2]]
        self.assertEqual(len(DA.draw(exactly, P1, 169)["p3"]), 20)

    def test_duplicates_in_the_table(self):
        rows = table()
        rows.append(dict(rows[10]))
        self.refused("appears twice", rows)

    def test_an_id_that_is_not_normalized(self):
        rows = table()
        rows[0]["task_id"] = "webarena.resized.0"
        self.refused("normalized", rows)
        rows = table()
        rows[1]["task_id"] = "webarena.improved.1"
        self.refused("normalized", rows)

    def test_strata_that_do_not_total_the_fifteen(self):
        with mock.patch.dict(DA.QUOTA, {"webarena": 7}):
            self.refused("the strata add up to 14")
        with mock.patch.dict(DA.QUOTA, {"webarena": 9}):
            self.refused("the strata add up to 16")

    def test_a_draw_that_does_not_total_is_refused(self):
        with mock.patch.object(DA, "permuted", lambda ids, seed, stratum: sorted(ids)[:1]):
            self.refused("the counts do not total")

    def test_totals_are_checked_list_by_list_and_by_benchmark(self):
        good = DA.draw(table(), P1, 169)
        DA.check_totals(good)
        for name in DA.LISTS:
            with self.assertRaisesRegex(DA.DrawRefused, "the counts do not total"):
                DA.check_totals(dict(good, **{name: good[name][1:]}))
        counts = dict(good["counts_by_benchmark"], p2_new={"webarena": 9, "visualwebarena": 0, "workarena": 6})
        with self.assertRaisesRegex(DA.DrawRefused, "the counts do not total"):
            DA.check_totals(dict(good, counts_by_benchmark=counts))

    def test_p1_tasks_missing_from_the_table(self):
        self.refused("missing from the table", p1=[*P1[:4], "workarena.synthetic.absent"])

    def test_p1_that_is_not_five_distinct_tasks(self):
        self.refused("5 distinct", p1=P1[:4])
        self.refused("5 distinct", p1=[*P1[:4], P1[0]])
        self.refused("5 distinct", p1=[*P1, "webarena.3"])

    def test_no_infeasible_task_or_more_than_one(self):
        rows = [r for r in table() if r["task_id"] != INFEASIBLE]
        rows.append(row("workarena.synthetic.extra", "workarena"))
        self.refused("the table has 0", rows)
        rows = table()
        rows.append(row("workarena.synthetic.second", "workarena", family="workarena:l1-infeasible"))
        self.refused("the table has 2", rows)

    def test_the_word_infeasible_outside_workarena_does_not_count(self):
        rows = table()
        rows[8]["family"] = "webarena:infeasible-looking"
        self.assertIn(INFEASIBLE, DA.draw(rows, P1, 169)["p2_new"])

    def test_a_forced_task_whose_template_overlaps_the_panel(self):
        rows = table()
        next(r for r in rows if r["task_id"] == INFEASIBLE)["template_overlaps_panel"] = True
        self.refused("overlaps the panel", rows)

    def test_a_table_of_the_wrong_shape(self):
        self.refused("non-empty JSON list", {"rows": []})
        self.refused("non-empty JSON list", [])
        for change in ({"extra": 1}, {"label": "x"}):
            rows = table()
            rows[0].update(change)
            self.refused("exactly the fields", rows)
        rows = table()
        del rows[0]["family"]
        self.refused("exactly the fields", rows)
        for field, value, message in (("task_id", 7, "normalized"), ("task_id", "", "normalized"),
                                      ("benchmark", "assistantbench", "benchmark must be"),
                                      ("benchmark", ["webarena"], "benchmark must be"),
                                      ("template_overlaps_panel", "false", "true or false"),
                                      ("template_overlaps_panel", 0, "true or false"), ("family", None, "family")):
            rows = table()
            rows[0][field] = value
            self.refused(message, rows)

    def test_a_seed_that_is_not_whole(self):
        self.refused("whole number", seed=1.5)
        self.refused("whole number", seed=True)
        self.refused("whole number", seed="169")


class TestVerifyAllocation(unittest.TestCase):
    def setUp(self):
        self.good = DA.draw(table(), P1, 169)

    def rehashed(self, **change):
        a = dict(self.good, **change)
        a["sha256"] = DA.content_sha256(a)
        return a

    def test_a_drawn_allocation_passes(self):
        self.assertEqual(DA.verify_allocation(self.good), self.good)

    def test_an_edited_file_fails_its_hash(self):
        edited = dict(self.good, p3=[*self.good["p3"][1:], "webarena.other"])
        with self.assertRaisesRegex(DA.DrawRefused, "own SHA-256"):
            DA.verify_allocation(edited)
        without = {k: v for k, v in self.good.items() if k != "sha256"}
        with self.assertRaisesRegex(DA.DrawRefused, "own SHA-256"):
            DA.verify_allocation(without)

    def test_p3_that_overlaps_p2(self):
        with self.assertRaisesRegex(DA.DrawRefused, "p3 overlaps p2"):
            DA.verify_allocation(self.rehashed(p3=[self.good["p2"][0], *self.good["p3"][1:]]))

    def test_a_development_list_that_leaves_out_a_task_of_p2_or_p3(self):
        for name in ("p1", "p2_new", "p3"):
            left_out = self.good[name][0]
            development = [t for t in self.good[DA.DEVELOPMENT] if t != left_out]
            with self.assertRaisesRegex(DA.DrawRefused, "development list leaves out", msg=name):
                DA.verify_allocation(self.rehashed(**{DA.DEVELOPMENT: development}))

    def test_p2_that_is_not_p1_plus_the_new_tasks(self):
        with self.assertRaisesRegex(DA.DrawRefused, "not p1 plus p2_new"):
            DA.verify_allocation(self.rehashed(p2=self.good["p2_new"]))
        with self.assertRaisesRegex(DA.DrawRefused, "not p1 plus p2_new"):
            DA.verify_allocation(self.rehashed(p2=[*self.good["p2"], "webarena.other"]))

    def test_lists_of_the_wrong_shape(self):
        with self.assertRaisesRegex(DA.DrawRefused, "JSON object"):
            DA.verify_allocation([])
        for name in (*DA.LISTS, DA.DEVELOPMENT):
            for value, message in (("webarena.1", "list of task IDs"), ([1], "list of task IDs"),
                                   ([""], "list of task IDs"), (["webarena.1", "webarena.1"], "twice"),
                                   (["visualwebarena.resized.1"], "not normalized")):
                with self.assertRaisesRegex(DA.DrawRefused, message, msg=name):
                    DA.verify_allocation(self.rehashed(**{name: value}))
        for name in ("p3", DA.DEVELOPMENT):
            without = {k: v for k, v in self.good.items() if k != name}
            with self.assertRaisesRegex(DA.DrawRefused, f"{name} must be"):
                DA.verify_allocation(without)


class TestCommand(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.table = self.tmp / "table.json"
        self.table.write_text(json.dumps(table()))

    def run_draw(self, out, *extra, p1=P1, seed="169", table_path=None):
        return subprocess.run([sys.executable, str(REPO / "scripts/draw_allocation.py"), "--table",
                               str(table_path or self.table), "--seed", seed, "--out", str(out), "--p1", *p1, *extra],
                              capture_output=True, text=True)

    def test_two_runs_write_identical_bytes_that_verify(self):
        a, b = self.tmp / "a.json", self.tmp / "b.json"
        for out in (a, b):
            r = self.run_draw(out)
            self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(a.read_bytes(), b.read_bytes())
        record = json.loads(a.read_bytes())
        self.assertEqual(DA.verify_allocation(record), record)
        self.assertIn(record["sha256"], r.stdout)

    def test_an_allocation_is_written_once(self):
        out = self.tmp / "a.json"
        self.assertEqual(self.run_draw(out).returncode, 0)
        before = out.read_bytes()
        r = self.run_draw(out, seed="170")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("written once", r.stderr)
        self.assertEqual(out.read_bytes(), before)

    def test_a_refusal_writes_nothing(self):
        out = self.tmp / "a.json"
        r = self.run_draw(out, p1=P1[:4])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("the draw is refused", r.stderr)
        self.assertFalse(out.exists())
        broken = self.tmp / "broken.json"
        broken.write_text("{not json")
        r = self.run_draw(out, table_path=broken)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("the draw is refused", r.stderr)
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
