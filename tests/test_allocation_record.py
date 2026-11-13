"""The allocation record names every frozen-pass task with a trace on disk and says whether it was read."""

import unittest
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"
P3_ON_DISK = ("visualwebarena.419", "webarena.430",
              "workarena.servicenow.dashboard-retrieve-incident-and-max-request-galaxy-note20-l2",
              "workarena.servicenow.filter-three-items-uniform-expenses-and-find-total-return-large-l2")


class TestKnownUseRecord(unittest.TestCase):
    def setUp(self):
        self.rows = {}
        for line in (DOCS / "ALLOCATION_RECORD.md").read_text(encoding="utf-8").splitlines():
            cells = [c.strip() for c in line.strip("|").split("|")]
            if line.startswith("| `") and len(cells) == 4:
                self.rows[cells[0].strip("`")] = cells

    def test_every_frozen_pass_task_on_disk_is_listed_in_p3(self):
        for task in P3_ON_DISK:
            self.assertIn(task, self.rows)
            self.assertEqual(self.rows[task][1], "p3", task)

    def test_each_says_whether_it_was_read_as_the_walkthrough_shows(self):
        walkthrough = (DOCS / "PACK_WALKTHROUGH.md").read_text(encoding="utf-8")
        for task in P3_ON_DISK:
            answer = "Yes." if f"development task `{task}`" in walkthrough else "No record of it being read."
            self.assertTrue(self.rows[task][2].startswith(answer), task)


if __name__ == "__main__":
    unittest.main()
