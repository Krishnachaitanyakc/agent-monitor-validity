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
    ("count every task as usable in the secondary denominator",
     "    nY = {j: sum(1 for l, r in zip(labels, reference) if l == j and r == USABLE) for j in STRATA}\n",
     "    nY = {j: sum(1 for l, r in zip(labels, reference) if l == j) for j in STRATA}\n"),
    ("count Unsure labels as usable in the coverage table",
     '        return {"tasks": len(idx), "usable": st[USABLE], "unsure": st[UNSURE], "missing": st[MISSING]}\n',
     '        return {"tasks": len(idx), "usable": st[USABLE] + st[UNSURE], "unsure": st[UNSURE], '
     '"missing": st[MISSING]}\n'),
    ("make the one-task-per-template panel the default",
     'PANEL_UNIT = "task"     # the default panel',
     'PANEL_UNIT = "template"     # the default panel'),
    ("leave the secondary result unscaled by N/N_Y",
     "    ratio = Fraction(design.N_per100, den)\n",
     "    ratio = Fraction(1)\n"),
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
    ("put the usable-only comparator back in the secondary result",
     '           "regime": design.regime, "not_estimable": None, "plan_literal": None}\n',
     '           "regime": design.regime, "not_estimable": None, "plan_literal": None, "usable_only": None}\n'),
    ("make population inference the default path",
     "def analyze_panel(slots: Sequence[Mapping[str, str]], E: Optional[Sequence[int]] = None, *,\n"
     "                  path: str = PATH_DESCRIPTIVE, **kw) -> dict:\n",
     "def analyze_panel(slots: Sequence[Mapping[str, str]], E: Optional[Sequence[int]] = None, *,\n"
     "                  path: str = PATH_POPULATION, **kw) -> dict:\n"),
    ("run population inference without the explicit argument",
     "    if path != PATH_POPULATION:\n",
     "    if False:\n"),
    ("drop the opening line of the population-inference report",
     '    return "\\n".join([POPULATION_PATH_TEXT, ""] + _report_lines(res))\n',
     '    return "\\n".join(_report_lines(res))\n'),
    ("drop the opening line of the population-inference panel report",
     '    out = [POPULATION_PATH_TEXT, ""]\n',
     "    out = []\n"),
    ("head the population-inference report as the primary analysis",
     '    out.append(f"Population inference on the panel, statistical analysis plan {POPULATION_PLAN_VERSION}")\n',
     '    out.append(f"Primary analysis, statistical analysis plan {POPULATION_PLAN_VERSION}")\n'),
    ("leave the prior exposure statement out of the population-inference report",
     '        out += [f"Prior exposure statement: {prior_exposure}", ""]\n',
     "        pass\n"),
    ("omit the scope sentence from the descriptive report",
     "           SCOPE_TEXT,\n",
     ""),
    ("omit the panel wording from the descriptive report",
     '           f"Panel: {PANEL_WORDING}.",\n',
     ""),
    ("omit the prior exposure statement from the descriptive report",
     '           f"Prior exposure statement: {prior_exposure}",\n',
     ""),
    ("run the descriptive report without a prior exposure statement",
     "    if not isinstance(prior_exposure, str) or not prior_exposure.strip():\n",
     "    if False:\n"),
    ("print a p-value in the descriptive report",
     '    out += _cost_lines(d["costs"])\n',
     '    out += _cost_lines(d["costs"]) + ["p = 0.01"]\n'),
    ("state a size class instead of the observed status against the reference",
     '        status = "observed below the reference"\n',
     '        status = "the gain is below the threshold"\n'),
    ("compare the observed gain with the reference strictly",
     "    elif observed >= MATERIAL_M:\n",
     "    elif observed > MATERIAL_M:\n"),
    ("report a share of M of zero when nothing is contradicted",
     '"share_of_M": {pol: cap[pol] / M if M > 0 else None for pol in POLICY_ORDER},',
     '"share_of_M": {pol: cap[pol] / M if M > 0 else Fraction(0) for pol in POLICY_ORDER},'),
    ("report a metric with an empty denominator as zero",
     '    return {"value": Fraction(num, den) if den > 0 else None, "numerator": num, "denominator": den}\n',
     '    return {"value": Fraction(num, den) if den > 0 else Fraction(0), "numerator": num, "denominator": den}\n'),
    ("report a per-100 rate with an empty denominator as zero",
     "    return Fraction(100) * x / den if den > 0 else None\n",
     "    return Fraction(100) * x / den if den > 0 else Fraction(0)\n"),
    ("make success the positive class",
     '    tp, fp = c[("failure", "failure")], c[("failure", "success")]\n'
     '    fn, tn = c[("success", "failure")], c[("success", "success")]\n',
     '    tn, fn = c[("failure", "failure")], c[("failure", "success")]\n'
     '    fp, tp = c[("success", "failure")], c[("success", "success")]\n'),
    ("use failure recall as the balanced accuracy",
     '        balanced = (recall["value"] + success_recall["value"]) / 2\n',
     '        balanced = recall["value"]\n'),
    ("take failure recall over the failure predictions",
     "    recall, success_recall = _rate(tp, tp + fn), _rate(tn, tn + fp)\n",
     "    recall, success_recall = _rate(tp, tp + fp), _rate(tn, tn + fp)\n"),
    ("take failure precision over the failure references",
     '            "failure_precision": _rate(tp, tp + fp),\n',
     '            "failure_precision": _rate(tp, tp + fn),\n'),
    ("take valid-prediction coverage over the valid tasks",
     '            "valid_prediction_coverage": _rate(n, n_usable),\n',
     '            "valid_prediction_coverage": _rate(n, n),\n'),
    ("take prevalence over every task with a usable reference",
     '            "prevalence": _rate(tp + fn, n)}\n',
     '            "prevalence": _rate(tp + fn, n_usable)}\n'),
    ("measure the shared subset on each judge's own valid set",
     '            "shared": {k: _judge_metrics(pairs(shared, k), n) for k in ("A1", "B1")}}\n',
     '            "shared": {k: _judge_metrics(pairs(usable, k), n) for k in ("A1", "B1")}}\n'),
    ("count Unsure references in the validity denominators",
     "        if status == USABLE:\n            usable.append((sl, verdict))\n",
     "        if status != MISSING:\n            usable.append((sl, verdict))\n"),
    ("rescale the secondary gain by the panel size",
     '{"over_disagreement": _per_100(over_dis, N_Y),',
     '{"over_disagreement": _per_100(over_dis, design.N),'),
    ("leave invalid first verdicts with a usable label out of N_Y",
     '    N_Y = cov["groups"]["all"]["usable"] if statuses_given else None\n',
     '    N_Y = cov["groups"]["all"]["usable"] - cov["groups"]["invalid first verdict"]["usable"] '
     "if statuses_given else None\n"),
    ("swap the gain over disagreement-first and the gain over random",
     '    over_dis, over_rand = cap["AUG"] - cap["DIS"], cap["AUG"] - cap["RAND"]\n',
     '    over_rand, over_dis = cap["AUG"] - cap["DIS"], cap["AUG"] - cap["RAND"]\n'),
    ("count only U2 in m1 of the descriptive result",
     '"M": M, "m1": counts["U2"] + counts["U1"], "m0": counts["S"],',
     '"M": M, "m1": counts["U2"], "m0": counts["S"],'),
    ("include A1 in the extra cost of the repeats",
     '    extra = by_slot["A2"] + by_slot["A3"]\n',
     '    extra = by_slot["A1"] + by_slot["A2"] + by_slot["A3"]\n'),
    ("state a cost per extra verdict when the gain is negative",
     '"extra_per_extra_verdict": extra / gain if gain > 0 else None}\n',
     '"extra_per_extra_verdict": extra / gain if gain != 0 else None}\n'),
    ("accept slot costs that add up to more than the total",
     "    if attributed > total:\n",
     "    if False:\n"),
    ("accept a negative cost",
     "        if v < 0:\n",
     "        if False:\n"),
    ("give one policy a different review count",
     "    reviews = {pol: expected_reviewed(design, pol) for pol in POLICY_ORDER}\n",
     '    reviews = {pol: expected_reviewed(design, pol) + (pol == "RAND") for pol in POLICY_ORDER}\n'),
    ("report the review counts as unequal without checking them",
     '"K_equal": all(v == design.K for v in reviews.values()),',
     '"K_equal": False,'),
    ("count R_unobserved among agreeing tasks only",
     '"R_unobserved": int((lf["R_unobserved"] == 1).sum()),',
     '"R_unobserved": _repeat_flags(lf)["both_repeats_invalid"],'),
    ("count benchmark rows instead of tasks in the invalid table",
     "        by_slot_category[(slot, cat)] += c\n",
     "        by_slot_category[(slot, cat)] += 1\n"),
    ("hide the invalid counts by benchmark in the descriptive report",
     '    if d["invalid_counts"] is not None:\n',
     "    if False:\n"),
    ("accept E and the released labels together",
     "    if (E is None) == (Y is None):\n",
     "    if E is None and Y is None:\n"),
    ("accept statuses alongside the released labels",
     "    if Y is not None and reference is not None:\n",
     "    if False:\n"),
    ("accept a discordance other than 0 or 1",
     "        if len(E) != n or any(e not in (0, 1) for e in E):\n",
     "        if len(E) != n:\n"),
    ("let a task without a reference be contradicted on the descriptive path",
     "        _require_no_contradiction_without_reference(labels, E, ref)\n",
     "        pass\n"),
    ("treat every label as assessed when no statuses are given on the descriptive path",
     "    statuses_given = Y is not None or reference is not None\n",
     "    statuses_given = True\n"),
    ("leave the provenance out of the descriptive report",
     '    out += _provenance_lines(d["provenance"])\n',
     ""),
    ("leave the label coverage out of the descriptive report",
     "        out += _coverage_lines(d)\n",
     "        pass\n"),
    ("print no note about the regime",
     """            f"  regime: {d['regime']}; {REGIME_NOTES[d['regime']]}",\n""",
     """            f"  regime: {d['regime']}",\n"""),
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
