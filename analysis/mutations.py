#!/usr/bin/env python3
"""Deliberate breaks of the analysis that its unit tests must catch.

Each mutation copies the analysis folder to a temporary folder, makes one exact edit
to sap.py that reintroduces a wrong rule, runs the analysis tests there, and expects
them to fail. The unchanged copy must pass first. Run from the repository root:

    python analysis/mutations.py

Exit code 0 means the unchanged suite passed and every mutation was caught.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

MUTATIONS = [
    ("count an Unsure or missing label as reference-contradicted",
     "        e = int(st == USABLE and is_valid(a1) and a1 != verdict)\n",
     "        e = int(st != USABLE or (is_valid(a1) and a1 != verdict))\n"),
    ("use N_Y as the primary per-100 denominator",
     "    design = Design.from_labels(labels_k, s_override=s_override, N_ref=N_ref)\n",
     "    design = Design.from_labels(labels_k, s_override=s_override, N_ref=N_ref if N_ref is not None "
     "else max(1, sum(r == USABLE for r in ref_k)))\n"),
    ("drop tasks without a usable label from the panel",
     '    keep = [i for i, l in enumerate(labels) if l != "EXCL"]\n',
     '    keep = [i for i, l in enumerate(labels) if l != "EXCL" and ref[i] == USABLE]\n'),
    ("treat a halted slot as INVALID",
     "INVALID_CODES = INVALID_TERMINAL + (TRANSPORT_EXHAUSTED,)\n",
     "INVALID_CODES = INVALID_TERMINAL + (TRANSPORT_EXHAUSTED, HALT)\n"),
    ("resolve a slot that ends in a halt as exhausted",
     '        raise UnfinishedSlot("slot is unfinished: its last attempt halted the run; a halt is "\n'
     '                             "not INVALID, and the slot resumes after the cause is fixed")\n',
     "        return TRANSPORT_EXHAUSTED\n"),
    ("drop FILTERED from the terminal outcomes",
     'INVALID_TERMINAL = ("REFUSAL", "TRUNCATED", "FILTERED", "MALFORMED", "ABSTAIN")\n',
     'INVALID_TERMINAL = ("REFUSAL", "TRUNCATED", "MALFORMED", "ABSTAIN")\n'),
    ("match verdicts case-sensitively",
     "    if isinstance(outcome, str) and outcome.isascii() and outcome.lower() in VERDICTS:\n",
     "    if False:\n"),
    ("use every agreeing task in the secondary rates",
     "    nY = {j: sum(1 for l, r in zip(labels, reference) if l == j and r == USABLE) for j in STRATA}\n",
     "    nY = {j: sum(1 for l, r in zip(labels, reference) if l == j) for j in STRATA}\n"),
    ("drop the equal review count adjustment from the secondary gain",
     '    w_adj = {"U2": w["U2"], "U1": w["U1"], "S": w["S"] - B_Y / n0Y}\n',
     "    w_adj = dict(w)\n"),
    ("count Unsure labels as usable in the coverage table",
     '        return {"tasks": len(idx), "usable": st[USABLE], "unsure": st[UNSURE], "missing": st[MISSING]}\n',
     '        return {"tasks": len(idx), "usable": st[USABLE] + st[UNSURE], "unsure": st[UNSURE], '
     '"missing": st[MISSING]}\n'),
    ("make the one-task-per-template panel the default",
     'PANEL_UNIT = "task"     # the default panel',
     'PANEL_UNIT = "template"     # the default panel'),
    ("leave the plan-literal secondary unscaled by N/N_Y",
     "    ratio = Fraction(design.N_per100, den)\n",
     "    ratio = Fraction(1)\n"),
    ("print a threshold class for the usable-only secondary",
     "                       f\"for G_Y: {_g_interval_text(uz['G_L'], uz['G_U'], uz['delta_L'], uz['delta_U'])}\")\n",
     "                       f\"for G_Y: {_g_interval_text(uz['G_L'], uz['G_U'], uz['delta_L'], uz['delta_U'])}; \"\n"
     "                       f\"size: {classify_size(uz['delta_L'], uz['delta_U'], 0.5, True)}\")\n"),
    ("count flips of tasks without a usable label in the fragility",
     "    usable = {j: sum(1 for l, r in zip(labels_k, ref_k) if l == j and r == USABLE)\n",
     "    usable = {j: sum(1 for l, r in zip(labels_k, ref_k) if l == j)\n"),
    ("print zero Unsure and missing counts when statuses are not supplied",
     "    supplied = reference is not None\n",
     "    supplied = True\n"),
    ("accept the judges' words success and failure as reference labels",
     'REFERENCE_VERDICTS = {"successful": "success", "unsuccessful": "failure"}\n',
     'REFERENCE_VERDICTS = {"successful": "success", "unsuccessful": "failure", "success": "success", '
     '"failure": "failure"}\n'),
    ("report the usable-only comparison when its comparator reviews a negative number",
     "    elif implied < 0:\n",
     "    elif False:\n"),
    ("report the usable-only comparison when its comparator needs more stable tasks than exist",
     "    elif implied > n0Y:\n",
     "    elif False:\n"),
    ("move each partial-regime count only the way it moves in the coverage regime in the fragility search",
     "            if all(0 <= x[j] <= hi[j] for j in range(3)):\n",
     "            if all(0 <= x[j] <= hi[j] for j in range(3)) and min(sgn * d2, sgn * d1, -sgn * dS) >= 0:\n"),
    ("count strictly larger values only in the permutation tail of the fragility search",
     "        i = bisect.bisect_left(values, v)\n",
     "        i = bisect.bisect_right(values, v)\n"),
    ("treat every coverage-regime decision as reversible by label flips",
     "        return fisher_one_sided(n1Y, design.n1, 0, design.n0, exact=True) <= Fraction(1, 40)\n",
     "        return True\n"),
    ("treat every partial-regime decision as reversible by label flips",
     "        if tails.rejects(x):\n            return True\n    return False\n",
     "        if tails.rejects(x):\n            return True\n    return True\n"),
    ("fill the most favorable partial-regime counts in increasing order of weight",
     "    order = sorted(range(3), key=lambda j: -tails.weights[j])\n",
     "    order = sorted(range(3), key=lambda j: tails.weights[j])\n"),
    ("drop the independence assumption from the partial-regime report",
     '        out.append("  " + PARTIAL_ASSUMPTION_TEXT)\n',
     ""),
    ("leave B_Y and the implied comparator undefined in the report",
     '    if sec["n1_Y"] > 0 and sec["n0_Y"] > 0:\n',
     "    if False:\n"),
    ("name the coverage-regime gain in the partial-regime decomposition line",
     '        name = "G_hat_Y" if sec["regime"] == "COVERAGE" else "G_partial_hat_Y"\n',
     '        name = "G_hat_Y"\n'),
]


def run_tests(folder: Path) -> bool:
    result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "analysis", "-f"],
                            cwd=folder, capture_output=True, text=True)
    return result.returncode == 0


def copy_analysis(tmp: str) -> Path:
    shutil.copytree(HERE, Path(tmp) / "analysis", ignore=shutil.ignore_patterns("__pycache__"))
    return Path(tmp) / "analysis" / "sap.py"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        copy_analysis(tmp)
        if not run_tests(Path(tmp)):
            print("SETUP ERROR  the unchanged analysis tests fail")
            return 1
    print("passed  unchanged analysis tests")
    missed = 0
    for name, old, new in MUTATIONS:
        with tempfile.TemporaryDirectory() as tmp:
            target = copy_analysis(tmp)
            text = target.read_text(encoding="utf-8")
            if text.count(old) != 1:
                print(f"SETUP ERROR  {name}: expected exactly one match in sap.py")
                missed += 1
                continue
            target.write_text(text.replace(old, new), encoding="utf-8")
            caught = not run_tests(Path(tmp))
            missed += not caught
            print(f"{'caught' if caught else 'MISSED'}  {name}")
    print(f"{len(MUTATIONS) - missed} of {len(MUTATIONS)} mutations caught")
    return 0 if missed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
