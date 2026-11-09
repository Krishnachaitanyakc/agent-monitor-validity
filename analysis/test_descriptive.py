"""Tests of the descriptive path in sap.py, the primary result. Synthetic data only.

Covers the exact expected captures against an enumeration of every draw, the equal
review count, the quantities committed at the prediction lock, the result sentences
and reading lines of the plan, the invalid-outcome variants, the A1 and B1 validity
metrics and contradiction rates, the cases of the recorded rule, invalid and Unsure
handling, costs, the report's fixed statements and its freedom from inferential
wording and from the words the plan does not use, and the path argument that keeps
population inference out of the primary result.

Run from the repository root: python -m unittest discover -s analysis -v
"""
import itertools
import math
import os
import re
import sys
import unittest
from collections import Counter
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sap  # noqa: E402

OTHER = {"success": "failure", "failure": "success"}
LABEL = {"success": "Successful", "failure": "Unsuccessful"}
EXPOSURE = "Seven test tasks were shown item by item before the lock and are not in the panel."
POLICIES = ("AUG", "DIS", "RAND")
REGIMES = {"DEGENERATE-NONE", "DEGENERATE-CAP", "DEGENERATE-NOSLICE", "COVERAGE", "PARTIAL"}


def slot(stratum, a1="success"):
    flip = OTHER[a1]
    return {"I": {"A1": "REFUSAL", "A2": a1, "A3": a1, "B1": a1},
            "G1": {"A1": a1, "A2": a1, "A3": a1, "B1": flip},
            "U2": {"A1": a1, "A2": flip, "A3": flip, "B1": a1},
            "U1": {"A1": a1, "A2": flip, "A3": a1, "B1": a1},
            "S": {"A1": a1, "A2": a1, "A3": a1, "B1": a1}}[stratum]


def labels_for(n_inv=0, n_D1=0, n_U2=0, n_U1=0, n_S=0):
    return ["I"] * n_inv + ["G1"] * n_D1 + ["U2"] * n_U2 + ["U1"] * n_U1 + ["S"] * n_S


def E_for(n_inv=(0, 0), G1=(0, 0), U2=(0, 0), U1=(0, 0), S=(0, 0)):
    out = []
    for n, m in (n_inv, G1, U2, U1, S):
        out += [1] * m + [0] * (n - m)
    return out


def build(labels, E):
    """Resolved slots and released labels that give these strata and this discordance."""
    slots, Y = [], []
    for i, (lab, e) in enumerate(zip(labels, E)):
        a1 = "failure" if i % 3 == 0 else "success"
        slots.append(slot(lab, a1))
        Y.append(LABEL[a1 if lab == "I" or not e else OTHER[a1]])
    return slots, Y


def describe_panel(*args, **kw):
    """sap.describe_panel with a case seed, which the real run takes from the freeze list."""
    kw.setdefault("case_seed", 0)
    return sap.describe_panel(*args, **kw)


def describe(labels, E, **kw):
    slots, Y = build(labels, E)
    return describe_panel(slots, Y=Y, **kw)


def draws(labels, policy, K):
    """Every equally likely review set of one policy, written out from the plan's rules."""
    idx = {j: [i for i, lab in enumerate(labels) if lab == j] for j in ("I", "G1", "U2", "U1", "S")}
    I, G1 = idx["I"], idx["G1"]
    if policy == "RAND":
        for c in itertools.combinations(G1 + idx["U2"] + idx["U1"] + idx["S"], K - len(I)):
            yield I + list(c)
    elif policy == "DIS":
        for c in itertools.combinations(idx["U2"] + idx["U1"] + idx["S"], K - len(I) - len(G1)):
            yield I + G1 + list(c)
    else:
        fixed, left = I + G1, K - len(I) - len(G1)
        for group in ("U2", "U1", "S"):
            if left < len(idx[group]):                   # the cutoff falls in this group
                for c in itertools.combinations(idx[group], left):
                    yield fixed + list(c)
                return
            fixed = fixed + idx[group]
            left -= len(idx[group])
        yield fixed


def brute_capture(labels, E, policy, K):
    """Mean capture over every draw; each draw reviews exactly K distinct tasks."""
    total = count = 0
    for chosen in draws(labels, policy, K):
        assert len(chosen) == K and len(set(chosen)) == K, (policy, chosen, K)
        total += sum(E[i] for i in chosen if labels[i] != "I")
        count += 1
    return Fraction(total, count)


def mixed_panel():
    """Ten tasks that hold every kind of invalid slot and reference status (see the tests)."""
    S, F = "success", "failure"
    rows = [("REFUSAL", S, S, S, "Successful", "wa"),                     # I, usable
            ("TRANSPORT_EXHAUSTED", S, S, S, "Unsure", "wa"),             # I, Unsure
            (S, S, S, "MALFORMED", "Unsuccessful", "wa"),                 # G1 by an invalid B1, discordant
            (F, F, F, S, "Unsure", "vwa"),                                # G1, Unsure
            (S, F, F, S, "Unsuccessful", "wa"),                           # U2, discordant
            (S, F, S, S, None, "vwa"),                                    # U1, missing
            (S, "TRUNCATED", "FILTERED", S, "Successful", "work"),        # S, R unobserved
            (S, "ABSTAIN", S, S, "Unsuccessful", "work"),                 # S, one invalid, no flip, discordant
            (S, "ABSTAIN", F, S, "Successful", "work"),                   # U1, one invalid, one flip
            (S, S, S, S, "Successful", "work")]                           # S
    slots = [{"A1": a1, "A2": a2, "A3": a3, "B1": b1} for a1, a2, a3, b1, _, _ in rows]
    return slots, [r[4] for r in rows], [r[5] for r in rows]


def provenance_of(slots):
    """The attempt log's provenance for resolved slots; an exhausted slot took every retry."""
    A = sap.Attempt
    log = []
    for i, s in enumerate(slots):
        for j, r in sap.SLOT_KEYS:
            outcome = s[sap.SLOT_NAMES[(j, r)]]
            if outcome == sap.TRANSPORT_EXHAUSTED:
                log += [A(str(i), j, r, "cfg-1", k, sap.TRANSPORT) for k in range(1, sap.MAX_RETRIES + 2)]
            else:
                log.append(A(str(i), j, r, "cfg-1", 1, outcome, f"{i}-{j}{r}"))
    return sap.resolve_panel(log, [str(i) for i in range(len(slots))], "cfg-1")["provenance"]


class TestExpectedCaptureByEnumeration(unittest.TestCase):
    def check(self, labels, E):
        d = describe(labels, E)
        for pol in POLICIES:
            self.assertEqual(d["captures"][pol], brute_capture(labels, E, pol, d["K"]), (labels, E, pol))
        return d

    def test_every_design_up_to_seven_tasks(self):
        rng = np.random.default_rng(20261109)
        seen = set()
        for N in range(1, 8):
            for n_inv, n_D1, n_U2, n_U1 in itertools.product(range(N + 1), repeat=4):
                n_S = N - n_inv - n_D1 - n_U2 - n_U1
                if n_S < 0:
                    continue
                labels = labels_for(n_inv, n_D1, n_U2, n_U1, n_S)
                for _ in range(3):
                    seen.add(self.check(labels, [int(rng.random() < 0.5) for _ in labels])["regime"])
        self.assertEqual(seen, REGIMES)

    def test_every_placement_of_the_discordant_verdicts(self):
        panels = [("DEGENERATE-NONE", (1, 2, 0, 0, 5)),
                  ("DEGENERATE-CAP", (0, 7, 0, 1, 0)),
                  ("DEGENERATE-NOSLICE", (0, 0, 1, 1, 1)),
                  ("COVERAGE", (1, 2, 0, 1, 5)),            # the slice ends with U1
                  ("COVERAGE", (0, 1, 0, 1, 8)),            # the slice ends inside S
                  ("PARTIAL", (0, 2, 3, 1, 3)),             # the slice ends inside U2
                  ("PARTIAL", (0, 1, 1, 3, 5))]             # the slice ends inside U1
        for regime, sizes in panels:
            labels = labels_for(*sizes)
            for E in itertools.product((0, 1), repeat=len(labels)):
                self.assertEqual(self.check(labels, list(E))["regime"], regime)

    def test_larger_panels(self):
        rng = np.random.default_rng(7)
        for sizes, want in (((2, 4, 2, 3, 9), "PARTIAL"), ((1, 3, 1, 1, 14), "COVERAGE")):
            labels = labels_for(*sizes)
            for _ in range(5):
                self.assertEqual(self.check(labels, [int(rng.random() < 0.4) for _ in labels])["regime"], want)

    def test_matches_the_population_path_captures(self):
        labels = labels_for(1, 12, 2, 4, 40)
        E = E_for(n_inv=(1, 0), G1=(12, 4), U2=(2, 1), U1=(4, 1), S=(40, 6))
        d = describe(labels, E)
        r = sap.analyze(labels, E, path=sap.PATH_POPULATION, boot_B=50, fragility=False)
        self.assertEqual(d["captures"], {p: r["captures"][p] for p in POLICIES})
        self.assertEqual(d["gain"]["over_disagreement"]["verdicts"], r["captures"]["Delta"])


class TestDesignAndReviewCount(unittest.TestCase):
    def test_K_is_equal_across_policies(self):
        rng = np.random.default_rng(1)
        for _ in range(200):
            sizes = [int(x) for x in rng.integers(0, 15, size=5)]
            labels = labels_for(*sizes)
            d = describe(labels, [0] * len(labels))
            s = min(math.floor(Fraction(3, 20) * len(labels) + Fraction(1, 2)), sizes[2] + sizes[3] + sizes[4])
            self.assertEqual((d["s"], d["K"]), (s, sizes[0] + sizes[1] + s))
            self.assertTrue(d["K_equal"])
            self.assertEqual(set(d["K_by_policy"].values()), {d["K"]})
            self.assertEqual(set(d["K_by_policy"]), set(POLICIES))

    def test_report_states_the_equal_review_count(self):
        d = describe(labels_for(0, 2, 0, 2, 16), [0] * 20)
        txt = sap.describe_report(d, EXPOSURE)
        self.assertIn("review count K = 5, K/N = 1/4 (0.2500), equal across the three policies "
                      "(Instability-first 5, Disagreement-first 5, Random 5)", txt)
        self.assertIn("NOT equal across the three policies", sap.describe_report(dict(d, K_equal=False), EXPOSURE))

    def test_counts_on_a_panel_with_every_invalid_kind(self):
        slots, Y, bench = mixed_panel()
        d = describe_panel(slots, Y=Y, benchmarks=bench)
        self.assertEqual((d["N"], d["n_inv"], d["N_V"], d["n_D1"], d["n_D0"]), (10, 2, 8, 2, 6))
        self.assertEqual((d["n_U2"], d["n_U1"], d["n_S"], d["s"], d["K"], d["regime"]), (1, 2, 3, 2, 6, "PARTIAL"))
        self.assertEqual(d["R_unobserved"], 1)
        self.assertEqual(d["repeat_flags"], {"both_repeats_invalid": 1, "one_repeat_invalid_no_flip": 1,
                                             "one_repeat_invalid_flip": 1})
        lines = sap.describe_report(d, EXPOSURE).split("\n")
        for line in ("  N = 10, invalid A1 n_inv = 2, valid N_V = 8",
                     "  disagreeing n_D1 = 2, agreeing n_D0 = 6",
                     "  agreeing groups: U2 = 1, U1 = 2, S = 3",
                     "  valid first verdicts with both repeats invalid, disagreeing tasks included (R_unobserved): 1"):
            self.assertIn(line, lines)
        start = lines.index("Repeat observation (label-free)")
        self.assertEqual(lines[start + 1:start + 4],
                         ["  agreeing items with both repeats INVALID (R set to 0): 1",
                          "  agreeing items with one repeat INVALID and no observed flip (R = 0 from one repeat): 1",
                          "  agreeing items with one repeat INVALID and an observed flip: 1"])

    def test_R_unobserved_counts_every_valid_first_verdict(self):
        both = {"A2": "REFUSAL", "A3": "MALFORMED"}
        slots = [dict(slot("S"), **both), dict(slot("G1"), **both), dict(slot("S"))]
        d = describe_panel(slots, Y=["Successful"] * 3)
        self.assertEqual(d["R_unobserved"], 2)                      # one agreeing, one disagreeing
        self.assertEqual(d["repeat_flags"]["both_repeats_invalid"], 1)

    def test_the_name_R_unobserved_has_one_value_in_the_report(self):
        both = {"A2": "REFUSAL", "A3": "MALFORMED"}
        slots = [dict(slot("S"), **both), dict(slot("G1"), **both), dict(slot("S"))]
        txt = sap.describe_report(describe_panel(slots, Y=["Successful"] * 3), EXPOSURE)
        lines = txt.split("\n")
        self.assertEqual(txt.replace("R_unobserved treated as R = 1", "").count("R_unobserved"), 1)
        self.assertIn("  agreeing groups: U2 = 0, U1 = 0, S = 2", lines)
        self.assertIn("  valid first verdicts with both repeats invalid, disagreeing tasks included (R_unobserved): 2", lines)
        self.assertIn("  agreeing items with both repeats INVALID (R set to 0): 1", lines)


class TestTemplateMultiplicities(unittest.TestCase):
    TEMPLATES = ["a"] * 3 + ["b"] * 3 + ["c"] * 2 + [f"solo{i}" for i in range(12)]

    def panel(self):
        return build(labels_for(0, 2, 0, 2, 16), [0] * 20)

    def test_N_is_reported_with_its_template_multiplicities(self):
        slots, Y = self.panel()
        d = sap.analyze_panel(slots, Y=Y, templates=self.TEMPLATES, case_seed=0)
        self.assertEqual(d["template_multiplicities"], {"n_tasks": 20, "n_templates": 15, "max_multiplicity": 3,
                                                        "tasks_sharing": 8, "histogram": {1: 12, 2: 1, 3: 2}})
        lines = sap.describe_report(d, EXPOSURE).split("\n")
        start = lines.index("  N = 20, invalid A1 n_inv = 0, valid N_V = 20")
        self.assertEqual(lines[start + 1:start + 3],
                         ["  Panel tasks: 20 in 15 templates; largest template holds 3; 8 tasks share a template.",
                          "  Templates by number of tasks: 1 task: 12, 2 tasks: 1, 3 tasks: 2"])

    def test_the_report_says_when_they_were_not_supplied(self):
        slots, Y = self.panel()
        d = describe_panel(slots, Y=Y)
        self.assertIsNone(d["template_multiplicities"])
        lines = sap.describe_report(d, EXPOSURE).split("\n")
        start = lines.index("  N = 20, invalid A1 n_inv = 0, valid N_V = 20")
        self.assertEqual(lines[start + 1], "  Template multiplicities: not supplied with the panel manifest.")

    def test_they_read_no_label(self):
        slots, Y = self.panel()
        unknown = ["Unsure"] * 20
        self.assertEqual(describe_panel(slots, Y=Y, templates=self.TEMPLATES)["template_multiplicities"],
                         describe_panel(slots, Y=unknown, templates=self.TEMPLATES)["template_multiplicities"])

    def test_one_template_id_per_task_is_required(self):
        slots, Y = self.panel()
        with self.assertRaises(ValueError):
            describe_panel(slots, Y=Y, templates=self.TEMPLATES[:-1])


class TestLockQuantities(unittest.TestCase):
    def brute(self, sizes):
        """The largest gain per 100 and the largest effect of one unstable task, over every E."""
        labels = labels_for(*sizes)
        design = sap.Design.from_labels(labels)
        K, N = design.K, len(labels)
        review = {p: [Fraction(sum(i in c for c in sets), len(sets)) for i in range(N)]
                  for p, sets in ((p, list(draws(labels, p, K))) for p in ("AUG", "DIS"))}
        agreeing = [i for i, lab in enumerate(labels) if lab in ("U2", "U1", "S")]
        effect = {i: review["AUG"][i] - review["DIS"][i] for i in agreeing}
        gains = [sum((effect[i] for i, e in zip(agreeing, bits) if e), Fraction(0))
                 for bits in itertools.product((0, 1), repeat=len(agreeing))]
        unstable = [effect[i] for i in agreeing if labels[i] in ("U2", "U1")]
        return Fraction(100, N) * max(gains), Fraction(100, N) * max(unstable, default=Fraction(0)), design

    def check(self, sizes):
        largest, one_task, design = self.brute(sizes)
        got = sap.lock_quantities(design)
        self.assertEqual(got["max_gain_per_100"], largest, sizes)
        self.assertEqual(got["reference_reachable"], largest >= 2, sizes)
        self.assertEqual(got["x_per_100"], one_task, sizes)
        return design.regime

    def test_the_maximum_is_the_brute_force_maximum_over_every_E(self):
        seen = set()
        for N in range(1, 8):
            for n_inv, n_D1, n_U2, n_U1 in itertools.product(range(N + 1), repeat=4):
                n_S = N - n_inv - n_D1 - n_U2 - n_U1
                if n_S >= 0:
                    seen.add(self.check((n_inv, n_D1, n_U2, n_U1, n_S)))
        for sizes in ((0, 0, 3, 3, 4), (0, 1, 1, 4, 5), (0, 0, 5, 2, 3), (1, 0, 0, 6, 4), (0, 2, 2, 2, 5), (0, 0, 1, 1, 8)):
            seen.add(self.check(sizes))
        self.assertEqual(seen, REGIMES)

    def test_hand_computed_coverage(self):
        # N = 200, s = 30, n_D0 = 170: an unstable task has weight 1 - 30/170 = 14/17.
        got = sap.lock_quantities(sap.Design.from_labels(labels_for(0, 30, 3, 7, 160)))
        self.assertEqual(got, {"max_gain_per_100": Fraction(70, 17), "reference_reachable": True, "x_per_100": Fraction(7, 17)})

    def test_hand_computed_partial(self):
        # N = 60, s = 9, n_D0 = 50: U2 is reviewed in full and U1 half, so the weights are
        # 1 - 9/50 = 41/50 and 1/2 - 9/50 = 8/25; S has the negative weight -9/50.
        got = sap.lock_quantities(sap.Design.from_labels(labels_for(0, 10, 4, 10, 36)))
        self.assertEqual(got, {"max_gain_per_100": Fraction(54, 5), "reference_reachable": True, "x_per_100": Fraction(41, 30)})

    def test_a_group_with_a_negative_weight_adds_nothing(self):
        # N = 100, s = 15: U2 takes the whole slice, so U1's weight is -15/50 and its tasks cannot help.
        got = sap.lock_quantities(sap.Design.from_labels(labels_for(0, 50, 20, 10, 20)))
        self.assertEqual(got["x_per_100"], Fraction(3, 4) - Fraction(3, 10))
        self.assertEqual(got["max_gain_per_100"], (Fraction(3, 4) - Fraction(3, 10)) * 20)

    def test_the_reference_is_reachable_at_exactly_two_per_100(self):
        # N = 100, s = 15, n_D0 = 30: four unstable tasks of weight 1/2 reach exactly 2.
        got = sap.lock_quantities(sap.Design.from_labels(labels_for(0, 70, 0, 4, 26)))
        self.assertEqual(got["max_gain_per_100"], 2)
        self.assertTrue(got["reference_reachable"])
        # N = 100, s = 15, n_D0 = 25: three unstable tasks of weight 2/5 reach 6/5.
        got = sap.lock_quantities(sap.Design.from_labels(labels_for(0, 75, 0, 3, 22)))
        self.assertEqual(got["max_gain_per_100"], Fraction(6, 5))
        self.assertFalse(got["reference_reachable"])

    def test_the_weights_count_only_groups_with_a_task(self):
        # No U2 task: x is U1's weight, and in COVERAGE that is (n_D0 - s)/n_D0.
        design = sap.Design.from_labels(labels_for(0, 30, 0, 7, 163))
        self.assertEqual(sap.lock_quantities(design)["x_per_100"], Fraction(100, 200) * Fraction(170 - 30, 170))

    def test_zero_in_the_degenerate_regimes(self):
        zero = {"max_gain_per_100": 0, "reference_reachable": False, "x_per_100": 0}
        for sizes, regime in (((1, 3, 0, 0, 16), "DEGENERATE-NONE"), ((0, 7, 0, 1, 0), "DEGENERATE-CAP"),
                              ((0, 0, 1, 1, 1), "DEGENERATE-NOSLICE"), ((0, 0, 0, 0, 0), "DEGENERATE-NONE")):
            design = sap.Design.from_labels(labels_for(*sizes))
            self.assertEqual(design.regime, regime)
            self.assertEqual(sap.lock_quantities(design), zero, sizes)

    def test_the_held_per_100_denominator_is_used(self):
        design = sap.Design.from_labels(labels_for(0, 30, 3, 7, 160), N_ref=400)
        self.assertEqual(sap.lock_quantities(design)["max_gain_per_100"], Fraction(35, 17))

    def test_the_result_holds_them_without_reading_a_label(self):
        labels = labels_for(0, 30, 3, 7, 160)
        slots, Y = build(labels, [0] * 200)
        other = ["Unsure" if i % 3 == 0 else ("Unsuccessful" if i % 3 == 1 else None) for i in range(200)]
        want = sap.lock_quantities(sap.Design.from_labels(labels))
        self.assertEqual(describe_panel(slots, Y=Y)["lock"], want)
        self.assertEqual(describe_panel(slots, Y=other)["lock"], want)

    def test_the_label_free_design_block_prints_them(self):
        d = describe(labels_for(0, 30, 3, 7, 160), [0] * 200)
        lines = sap.describe_report(d, EXPOSURE).split("\n")
        start = next(i for i, x in enumerate(lines) if x.startswith("  regime: COVERAGE"))
        self.assertEqual(lines[start + 1:start + 4],
                         ["  largest gain per 100 panel tasks the panel can reach: 70/17 (4.1176)",
                          "  2 per 100 reachable on this panel: yes",
                          "  change in the gain per 100 panel tasks from one unstable task, x: 7/17 (0.4118)"])
        d = describe(labels_for(0, 75, 0, 3, 22), [0] * 100)
        self.assertIn("  2 per 100 reachable on this panel: no", sap.describe_report(d, EXPOSURE).split("\n"))


CLOSING = ("This describes these tasks and these realized judge calls and carries no population confidence or "
           "significance claim.")


class TestResultSentences(unittest.TestCase):
    def test_every_case_of_the_plan_table_as_an_exact_string(self):
        F = Fraction
        cases = [
            ("DEGENERATE-NONE", F(0), F(0),
             ["No agreeing verdict was unstable on this panel, so instability added no review candidates.", CLOSING]),
            ("DEGENERATE-CAP", F(0), F(0),
             ["The budget reached every agreeing verdict, so the two policies reviewed the same tasks.", CLOSING]),
            ("DEGENERATE-NOSLICE", F(0), F(0),
             ["The budget left no added review slots, so the two policies reviewed the same tasks.", CLOSING]),
            ("COVERAGE", F(0), F(1, 2),
             ["On this panel instability-first did not select more reference-contradicted first verdicts than "
              "disagreement-first, in expectation over each policy's random choices.",
              "One unstable task changes this number by 0.5000 per 100.", CLOSING]),
            ("PARTIAL", F(-3, 2), F(7, 17),
             ["On this panel instability-first did not select more reference-contradicted first verdicts than "
              "disagreement-first, in expectation over each policy's random choices.",
              "One unstable task changes this number by 0.4118 per 100.", CLOSING]),
            ("COVERAGE", F(189, 136), F(7, 17),
             ["On this panel instability-first selected 1.3897 more reference-contradicted first verdicts per 100 "
              "panel tasks, in expectation over each policy's random choices, below the practical reference of 2 "
              "per 100.",
              "One unstable task changes this number by 0.4118 per 100.", CLOSING]),
            ("PARTIAL", F(1, 2), F(1, 3),
             ["On this panel instability-first selected 0.5000 more reference-contradicted first verdicts per 100 "
              "panel tasks, in expectation over each policy's random choices, below the practical reference of 2 "
              "per 100.",
              "One unstable task changes this number by 0.3333 per 100.", CLOSING]),
            ("COVERAGE", F(1), F(1, 3),
             ["On this panel instability-first selected 1.0000 more reference-contradicted first verdicts per 100 "
              "panel tasks, in expectation over each policy's random choices, below the practical reference of 2 "
              "per 100.",
              "One unstable task changes this number by 0.3333 per 100.", CLOSING]),
            ("PARTIAL", F(199, 100), F(1, 3),
             ["On this panel instability-first selected 1.9900 more reference-contradicted first verdicts per 100 "
              "panel tasks, in expectation over each policy's random choices, below the practical reference of 2 "
              "per 100.",
              "One unstable task changes this number by 0.3333 per 100.", CLOSING]),
            ("COVERAGE", F(2), F(4, 5),
             ["On this panel instability-first selected 2.0000 more reference-contradicted first verdicts per 100 "
              "panel tasks, in expectation over each policy's random choices, at or above the practical reference "
              "of 2 per 100.",
              "One unstable task changes this number by 0.8000 per 100.", CLOSING]),
            ("PARTIAL", F(54, 5), F(41, 30),
             ["On this panel instability-first selected 10.8000 more reference-contradicted first verdicts per 100 "
              "panel tasks, in expectation over each policy's random choices, at or above the practical reference "
              "of 2 per 100.",
              "One unstable task changes this number by 1.3667 per 100.", CLOSING]),
        ]
        for regime, g, x, want in cases:
            self.assertEqual(sap.result_sentences(regime, g, x), want, (regime, g))

    def test_a_gain_next_to_the_reference_or_to_zero_keeps_enough_digits(self):
        F = Fraction
        for g, shown, side in ((F(89375, 44688), "1.99998", "below"), (F(19999, 10000), "1.9999", "below"),
                               (F(1, 100000), "0.00001", "below"), (F(2), "2.0000", "at or above"),
                               (F(20001, 10000), "2.0001", "at or above")):
            first = sap.result_sentences("COVERAGE", g, F(1, 2))[0]
            self.assertIn(f"selected {shown} more", first, g)
            self.assertIn(f"{side} the practical reference", first, g)

    def test_the_digits_never_reach_the_reference_or_zero(self):
        for k in range(1, 40):
            self.assertLess(Fraction(sap._gain_text(2 - Fraction(1, 10 ** k))), 2, k)
            self.assertGreater(Fraction(sap._gain_text(Fraction(1, 10 ** k))), 0, k)

    def test_the_report_does_not_round_a_gain_up_to_the_reference(self):
        d, block = self.report_block((0, 0, 32, 0, 304), E_for(U2=(32, 8), S=(304, 1)))
        self.assertEqual(d["gain"]["over_disagreement"]["per_100_panel"], Fraction(89375, 44688))
        self.assertIn("selected 1.99998 more", block[0])
        self.assertIn("below the practical reference", block[0])
        self.assertIn("  gain over disagreement-first per 100 panel tasks: 89375/44688 (1.99998); observed below the reference.",
                      sap.describe_report(d, EXPOSURE).split("\n"))

    def test_a_degenerate_panel_uses_its_own_sentence_whatever_the_gain(self):
        self.assertEqual(sap.result_sentences("DEGENERATE-CAP", None, Fraction(0))[0],
                         "The budget reached every agreeing verdict, so the two policies reviewed the same tasks.")

    def report_block(self, sizes, E):
        d = describe(labels_for(*sizes), E)
        lines = sap.describe_report(d, EXPOSURE).split("\n")
        self.assertEqual(lines[4:6], ["", "RESULT"])
        end = lines.index("", 6)
        return d, lines[6:end]

    def test_the_report_opens_with_the_block_in_every_case(self):
        n = lambda t: "  " + t
        cases = (((1, 3, 0, 0, 16), [1, 0] * 10,
                  [n("No agreeing verdict was unstable on this panel, so instability added no review candidates."), n(CLOSING)]),
                 ((0, 7, 0, 1, 0), [0, 1] * 4,
                  [n("The budget reached every agreeing verdict, so the two policies reviewed the same tasks."), n(CLOSING)]),
                 ((0, 0, 1, 1, 1), [1, 0, 1],
                  [n("The budget left no added review slots, so the two policies reviewed the same tasks."), n(CLOSING)]),
                 ((0, 2, 0, 2, 16), E_for(G1=(2, 0), U1=(2, 0), S=(16, 6)),
                  [n("On this panel instability-first did not select more reference-contradicted first verdicts than "
                     "disagreement-first, in expectation over each policy's random choices."),
                   n("One unstable task changes this number by 4.1667 per 100."), n(CLOSING)]),
                 ((0, 30, 3, 7, 160), E_for(G1=(30, 11), U2=(3, 1), U1=(7, 3), S=(160, 10)),
                  [n("On this panel instability-first selected 1.3897 more reference-contradicted first verdicts per 100 "
                     "panel tasks, in expectation over each policy's random choices, below the practical reference of 2 "
                     "per 100."),
                   n("One unstable task changes this number by 0.4118 per 100."), n(CLOSING)]),
                 ((0, 25, 0, 5, 70), E_for(G1=(25, 0), U1=(5, 5), S=(70, 35)),
                  [n("On this panel instability-first selected 2.0000 more reference-contradicted first verdicts per 100 "
                     "panel tasks, in expectation over each policy's random choices, at or above the practical reference "
                     "of 2 per 100."),
                   n("One unstable task changes this number by 0.8000 per 100."), n(CLOSING)]))
        for sizes, E, want in cases:
            d, block = self.report_block(sizes, E)
            self.assertEqual(block, want, sizes)
            self.assertEqual(d["result_sentences"], [x.strip() for x in want], sizes)

    def test_the_gain_of_zero_on_a_regular_panel_is_not_more(self):
        d, block = self.report_block((0, 2, 0, 2, 16), [0] * 20)
        self.assertEqual(d["gain"]["over_disagreement"]["per_100_panel"], 0)
        self.assertIn("did not select more", block[0])

    def test_an_empty_panel_has_a_degenerate_sentence(self):
        d = describe_panel([], Y=[])
        self.assertEqual(d["result_sentences"][0], "No agreeing verdict was unstable on this panel, so instability added no review candidates.")


class TestReadingLines(unittest.TestCase):
    U2 = "  A contradicted U2 verdict has both repeats matching the released reference."
    U1 = "  A contradicted U1 verdict whose other repeat is valid has a majority that stays with A1."
    GAIN = "  Instability and the reference-contradicted outcome both use A1, so part of any gain is built in."
    INDEPENDENT = ("  With independent calls and a judge that is right more often than wrong on a task, contradicted "
                   "verdicts are relatively more common in U2 than in U1.")
    FROZEN = "  This plan freezes that direction and no numeric benchmark."

    def test_word_for_word_in_their_places_in_every_regime(self):
        seen = set()
        for d in report_panels():
            seen.add(d["regime"])
            lines = sap.describe_report(d, EXPOSURE).split("\n")
            start = lines.index("REFERENCE-CONTRADICTED FIRST VERDICTS")
            self.assertTrue(lines[start + 1].startswith("  M = "))
            self.assertEqual(lines[start + 2:start + 6], [self.U2, self.U1, self.INDEPENDENT, self.FROZEN])
            self.assertEqual(lines[start + 6], "")
            check = next(i for i, x in enumerate(lines) if x.startswith("  check, disagreement-first minus random"))
            self.assertEqual(lines[check + 1:check + 3], [self.GAIN, ""])
        self.assertEqual(seen, REGIMES)

    def test_each_is_printed_once_and_states_no_number(self):
        txt = sap.describe_report(next(report_panels()), EXPOSURE)
        for line in (self.U2, self.U1, self.GAIN, self.INDEPENDENT, self.FROZEN):
            self.assertEqual(txt.count(line), 1, line)
            self.assertIsNone(re.search(r"\d", re.sub(r"\b(U2|U1|A1)\b", "", line)), line)

    def test_what_they_say_of_a_contradicted_verdict_holds(self):
        for a1 in ("success", "failure"):
            reference = OTHER[a1]                                   # contradicted: the released label differs from A1
            two = slot("U2", a1)
            self.assertEqual((two["A2"], two["A3"]), (reference, reference))
            one = slot("U1", a1)
            self.assertEqual(sorted((one["A1"], one["A2"], one["A3"])).count(a1), 2)


class TestLimitations(unittest.TestCase):
    TEXT = ("Limitations. One dataset and two judges. Released labels are a reference, not ground truth. Runs are "
            "kept as recorded: the way a run ends could influence judge accuracy, and we have not established that "
            "it does. Tasks that share a template and WorkArena families stay in the panel.")

    def test_printed_word_for_word_right_after_the_result_in_every_regime(self):
        seen = set()
        for d in report_panels():
            seen.add(d["regime"])
            lines = sap.describe_report(d, EXPOSURE).split("\n")
            end = lines.index("", lines.index("RESULT"))
            self.assertEqual(lines[end + 1:end + 3], [self.TEXT, ""], d["regime"])
            self.assertEqual("\n".join(lines).count("Limitations."), 1)
        self.assertEqual(seen, REGIMES)

    def test_the_prior_exposure_statement_comes_with_them(self):
        lines = sap.describe_report(next(report_panels()), EXPOSURE).split("\n")
        self.assertLess(lines.index("Prior exposure statement: " + EXPOSURE), lines.index("RESULT"))


def panel_with_invalid_slots(seed, N, shape):
    """A random synthetic panel with invalid slots and its E, from a seeded generator."""
    rng = np.random.default_rng(seed)
    p_disagree, p_flip, p_invalid = shape
    slots, E = [], []
    for _ in range(N):
        a1 = str(rng.choice(sap.VERDICTS))
        outcome = lambda agree: ("MALFORMED" if rng.random() < p_invalid else (a1 if agree else OTHER[a1]))
        s = {"A1": "REFUSAL" if rng.random() < p_invalid else a1, "A2": outcome(rng.random() >= p_flip),
             "A3": outcome(rng.random() >= p_flip), "B1": outcome(rng.random() >= p_disagree)}
        slots.append(s)
        E.append(int(sap.is_valid(s["A1"]) and rng.random() < 0.4))
    return slots, E


class TestInvalidOutcomeVariants(unittest.TestCase):
    def test_hand_computed_on_the_mixed_panel(self):
        # s = 2 and N = 10 throughout. See the comments of mixed_panel for each row.
        slots, Y, _ = mixed_panel()
        d = describe_panel(slots, Y=Y)
        F = Fraction
        self.assertEqual(d["invalid_outcome_variants"], [
            {"name": "B1 INVALID treated as agreement (D = 0) instead of D = 1",
             "difference": F(1, 7), "per_100_panel": F(10, 7)},                          # 1 minus 2/7 of 3
            {"name": "R_unobserved treated as R = 1",
             "difference": F(1, 3), "per_100_panel": F(10, 3)},                          # 2 minus 1 + 1/3 of 2
            {"name": "every trajectory with any INVALID slot excluded",
             "difference": F(1, 3), "per_100_panel": F(10, 3)},                          # 1 minus 2/3 of 1
            {"name": "one INVALID repeat and no observed flip treated as R = 1, and R_unobserved treated as R = 1",
             "difference": F(7, 12), "per_100_panel": F(35, 6)}])                        # 9/4 minus 5/3

    def test_the_report_prints_each_variant(self):
        slots, Y, _ = mixed_panel()
        lines = sap.describe_report(describe_panel(slots, Y=Y), EXPOSURE).split("\n")
        start = lines.index("INVALID-OUTCOME VARIANTS: instability-first minus disagreement-first expected capture")
        self.assertEqual(lines[start + 1:start + 6], [
            "  s is held at 2, capped at each variant's n_D0. N = 10 and N_Y = 7 stay the per-100 denominators, and "
            "each line is the difference in expected capture only.",
            "  B1 INVALID treated as agreement (D = 0) instead of D = 1: 1/7 (0.1429) verdicts",
            "  R_unobserved treated as R = 1: 1/3 (0.3333) verdicts",
            "  every trajectory with any INVALID slot excluded: 1/3 (0.3333) verdicts",
            "  one INVALID repeat and no observed flip treated as R = 1, and R_unobserved treated as R = 1: 7/12 (0.5833) verdicts"])

    def test_the_heading_says_when_N_Y_was_not_assessed(self):
        slots, Y, _ = mixed_panel()
        j = sap.join_reference(slots, Y)
        lines = sap.describe_report(describe_panel(slots, j["E"]), EXPOSURE).split("\n")
        self.assertIn("  s is held at 2, capped at each variant's n_D0. N = 10 and N_Y = not assessed stay the per-100 "
                      "denominators, and each line is the difference in expected capture only.", lines)

    def test_the_last_variant_says_it_also_treats_R_unobserved_as_one(self):
        # No task has exactly one invalid repeat, and the variant still moves: the two tasks with both repeats invalid.
        both = dict(slot("S"), A2="REFUSAL", A3="MALFORMED")
        slots = [slot("U1")] * 2 + [both] * 2 + [slot("S")] * 16
        d = describe_panel(slots, [1, 0, 1, 1] + [0] * 16)
        self.assertEqual(d["repeat_flags"], {"both_repeats_invalid": 2, "one_repeat_invalid_no_flip": 0, "one_repeat_invalid_flip": 0})
        self.assertEqual(d["gain"]["over_disagreement"]["verdicts"], Fraction(119, 180))
        self.assertEqual(d["invalid_outcome_variants"][3]["difference"], Fraction(9, 5))
        self.assertIn("  one INVALID repeat and no observed flip treated as R = 1, and R_unobserved treated as R = 1: 9/5 (1.8000) verdicts",
                      sap.describe_report(d, EXPOSURE).split("\n"))

    def test_they_match_the_population_paths_own_sensitivity_gain(self):
        seen = set()
        for seed, N, shape in ((1, 20, (0.2, 0.2, 0.08)), (2, 26, (0.1, 0.5, 0.12)), (3, 16, (0.3, 0.1, 0.1)),
                               (4, 30, (0.15, 0.35, 0.05)), (5, 12, (0.4, 0.3, 0.2)), (6, 24, (0.2, 0.6, 0.15))):
            slots, E = panel_with_invalid_slots(seed, N, shape)
            d = describe_panel(slots, E)
            seen.add(d["regime"])
            runs = sap.analyze_with_sensitivities(slots, E, path=sap.PATH_POPULATION, boot_B=20, fragility=False)
            self.assertEqual(len(runs), 5)
            for (name, r), v in zip(runs[1:], d["invalid_outcome_variants"]):
                self.assertEqual(name, "sensitivity: " + v["name"], seed)
                self.assertEqual(v["difference"], r["captures"]["Delta"], (seed, name))
                self.assertAlmostEqual(float(v["per_100_panel"]), r["Delta_per100"], places=12, msg=(seed, name))
        self.assertTrue(seen & {"COVERAGE", "PARTIAL"})

    def test_each_rule_is_the_same_as_a_design_built_by_hand(self):
        for seed in range(30):
            slots, E = panel_with_invalid_slots(100 + seed, 30, (0.2, 0.3, 0.1))
            d = describe_panel(slots, E)
            for (name, opts), v in zip(sap.SENSITIVITY_VARIANTS, d["invalid_outcome_variants"]):
                labels = list(sap.derive_label_free(slots, **opts)["stratum"])
                kept = [(l, e) for l, e in zip(labels, E) if l != "EXCL"]
                n_D0 = sum(l in ("U2", "U1", "S") for l, _ in kept)
                s, n_G1 = min(d["s"], n_D0), sum(l == "G1" for l, _ in kept)
                m = {j: sum(e for l, e in kept if l == j) for j in ("U2", "U1", "S")}
                n = {j: sum(l == j for l, _ in kept) for j in ("U2", "U1", "S")}
                left, gain = s, Fraction(0)
                for j in ("U2", "U1", "S"):
                    take = min(n[j], left)
                    left -= take
                    gain += m[j] * (Fraction(take, n[j]) if n[j] else 0)
                gain -= sum(m.values()) * (Fraction(s, n_D0) if n_D0 else 0)
                self.assertEqual(v["difference"], gain, (seed, name))
                self.assertEqual(v["per_100_panel"], gain * 100 / 30, (seed, name))

    def test_s_is_capped_at_the_new_agreeing_count_and_N_stays(self):
        # Primary: N = 20, s = 3, n_D0 = 20. Excluding every task with an invalid slot leaves
        # two agreeing tasks: the cap makes both policies review them all, so the difference is 0.
        bad = dict(slot("U1"), A3="MALFORMED")
        slots = [bad] * 18 + [slot("U1"), slot("S")]
        E = [0] * 18 + [1, 0]
        d = describe_panel(slots, E)
        self.assertEqual((d["s"], d["N"]), (3, 20))
        by_name = {v["name"]: v for v in d["invalid_outcome_variants"]}
        self.assertEqual(by_name["every trajectory with any INVALID slot excluded"]["difference"], 0)
        # Six tasks kept of 20, s = 3 of 6: the per-100 denominator stays 20, not 6.
        slots = [bad] * 14 + [slot("U1")] * 2 + [slot("S")] * 4
        E = [0] * 14 + [1, 1] + [0] * 4
        d = describe_panel(slots, E)
        v = {x["name"]: x for x in d["invalid_outcome_variants"]}["every trajectory with any INVALID slot excluded"]
        self.assertEqual(v["difference"], 1)                           # 2 of 2 minus 3/6 of 2
        self.assertEqual(v["per_100_panel"], 5)

    def test_an_empty_panel_is_not_defined_per_100(self):
        d = describe_panel([], Y=[])
        self.assertTrue(all(v["per_100_panel"] is None for v in d["invalid_outcome_variants"]))
        self.assertIn("  R_unobserved treated as R = 1: 0 (0.0000) verdicts", sap.describe_report(d, EXPOSURE).split("\n"))

    def test_the_section_carries_no_test_interval_or_threshold(self):
        slots, Y, _ = mixed_panel()
        lines = sap.describe_report(describe_panel(slots, Y=Y), EXPOSURE).split("\n")
        start = lines.index("INVALID-OUTCOME VARIANTS: instability-first minus disagreement-first expected capture")
        section = "\n".join(lines[start:lines.index("", start)]).lower()
        for word in ("reference", "threshold", "test", "interval", "reachable", "p =", "2 per 100"):
            self.assertNotIn(word, section, word)


class TestGainsAndReference(unittest.TestCase):
    def test_hand_computed_panel(self):
        # K = 60. Random: 60/200 of M = 25. Disagreement-first: 11 + 30/170 of 14.
        # Instability-first: 11 + 4 + 20/160 of 10.
        labels = labels_for(0, 30, 3, 7, 160)
        d = describe(labels, E_for(G1=(30, 11), U2=(3, 1), U1=(7, 3), S=(160, 10)))
        self.assertEqual((d["regime"], d["K"], d["M"], d["m1"], d["m0"]), ("COVERAGE", 60, 25, 4, 10))
        self.assertEqual(d["captures"], {"AUG": Fraction(65, 4), "DIS": Fraction(229, 17), "RAND": Fraction(15, 2)})
        self.assertEqual(d["share_of_M"], {"AUG": Fraction(13, 20), "DIS": Fraction(229, 425), "RAND": Fraction(3, 10)})
        g = d["gain"]
        self.assertEqual(g["over_disagreement"], {"verdicts": Fraction(189, 68), "per_100_panel": Fraction(189, 136)})
        self.assertEqual(g["over_random"], {"verdicts": Fraction(35, 4), "per_100_panel": Fraction(35, 8)})
        self.assertEqual(g["disagreement_over_random"], {"verdicts": Fraction(203, 34), "per_100_panel": Fraction(203, 68)})
        self.assertEqual(d["N_Y"], 200)
        self.assertEqual(d["secondary"], {"over_disagreement": Fraction(189, 136)})
        txt = sap.describe_report(d, EXPOSURE)
        for needle in ("M = 25: disagreement m_G1 = 11; unstable agreeing m1 = 4 (m_U2 = 1, m_U1 = 3); stable agreeing m0 = 10",
                       "Instability-first | 65/4 (16.2500) | 60 | 3/10 (0.3000) | 13/20 (0.6500)",
                       "Disagreement-first | 229/17 (13.4706) | 60 | 3/10 (0.3000) | 229/425 (0.5388)",
                       "Random | 15/2 (7.5000) | 60 | 3/10 (0.3000) | 3/10 (0.3000)",
                       "over disagreement-first: 189/68 (2.7794) verdicts; 189/136 (1.3897) per 100 panel tasks (N = 200)",
                       "over random: 35/4 (8.7500) verdicts; 35/8 (4.3750) per 100 panel tasks (N = 200)",
                       "check, disagreement-first minus random: 203/34 (5.9706) verdicts; "
                       "203/68 (2.9853) per 100 panel tasks (N = 200)",
                       "(N_Y = 200 of N = 200)",
                       "gain over disagreement-first per 100 panel tasks: 189/136 (1.3897); "
                       "observed below the reference."):
            self.assertIn(needle, txt, needle)

    def test_each_policy_prints_its_review_count_and_its_share_of_the_panel(self):
        # N = 40 and N_V = 38, so K/N and K/N_V differ. s = 6 and K = 2 + 5 + 6 = 13.
        d = describe(labels_for(2, 5, 1, 2, 30), [0] * 40)
        self.assertEqual(d["K_over_N"], dict.fromkeys(POLICIES, Fraction(13, 40)))
        lines = sap.describe_report(d, EXPOSURE).split("\n")
        start = lines.index("  policy | expected capture | K | K/N | share of M")
        self.assertEqual(lines[start + 1:start + 4], ["  Instability-first | 0 (0.0000) | 13 | 13/40 (0.3250) | not defined",
                                                      "  Disagreement-first | 0 (0.0000) | 13 | 13/40 (0.3250) | not defined",
                                                      "  Random | 0 (0.0000) | 13 | 13/40 (0.3250) | not defined"])

    def test_K_over_N_is_not_defined_on_an_empty_panel(self):
        d = describe_panel([], Y=[])
        self.assertEqual(d["K_over_N"], dict.fromkeys(POLICIES))
        txt = sap.describe_report(d, EXPOSURE)
        self.assertIn("review count K = 0, K/N = not defined, equal", txt)
        self.assertIn("  Random | 0 (0.0000) | 0 | not defined | not defined", txt.split("\n"))

    def test_practical_reference_is_compared_exactly(self):
        # N = 100, s = 15. Disagreement-first: 1/5 of 40 = 8; instability-first: 5 + 1/7 of 35 = 10.
        d = describe(labels_for(0, 25, 0, 5, 70), E_for(G1=(25, 0), U1=(5, 5), S=(70, 35)))
        self.assertEqual(d["gain"]["over_disagreement"]["per_100_panel"], 2)
        self.assertEqual(d["practical_reference"], {"per_100": 2, "status": "observed at or above the reference"})
        self.assertIn("2 (2.0000); observed at or above the reference.", sap.describe_report(d, EXPOSURE))
        d = describe(labels_for(0, 25, 0, 5, 70), E_for(G1=(25, 0), U1=(5, 5), S=(70, 36)))
        self.assertEqual(d["gain"]["over_disagreement"]["per_100_panel"], Fraction(68, 35))
        self.assertEqual(d["practical_reference"]["status"], "observed below the reference")

    def test_share_of_M_is_not_defined_when_nothing_is_contradicted(self):
        d = describe(labels_for(0, 2, 0, 2, 16), [0] * 20)
        self.assertEqual(d["M"], 0)
        self.assertEqual(d["share_of_M"], dict.fromkeys(POLICIES))
        self.assertIn("Instability-first | 0 (0.0000) | 5 | 1/4 (0.2500) | not defined", sap.describe_report(d, EXPOSURE))

    def test_degenerate_panel_with_no_unstable_tasks(self):
        # N = 20, s = 3, K = 7, M = 9. Random reviews 6/19 of the valid tasks; instability-first
        # and disagreement-first both take every disagreement and 3/16 of the stable tasks.
        labels = labels_for(1, 3, 0, 0, 16)
        E = E_for(n_inv=(1, 0), G1=(3, 1), S=(16, 8))
        d = describe(labels, E, costs={"total": "8", "by_slot": {"A1": "2", "A2": "2", "A3": "2", "B1": "2"}})
        self.assertEqual((d["regime"], d["n_U2"], d["n_U1"], d["n_S"], d["s"], d["K"]),
                         ("DEGENERATE-NONE", 0, 0, 16, 3, 7))
        self.assertEqual(d["captures"]["AUG"], d["captures"]["DIS"])
        self.assertEqual(d["captures"]["AUG"], 1 + Fraction(3, 16) * 8)
        self.assertEqual(d["captures"]["RAND"], Fraction(6, 19) * 9)
        self.assertEqual(d["gain"]["over_disagreement"], {"verdicts": 0, "per_100_panel": 0})
        self.assertEqual(d["gain"]["over_random"]["verdicts"], Fraction(5, 2) - Fraction(54, 19))
        self.assertEqual(d["secondary"]["over_disagreement"], 0)
        self.assertEqual(d["practical_reference"]["status"], "observed below the reference")
        self.assertIsNone(d["costs"]["extra_per_extra_verdict"])
        txt = sap.describe_report(d, EXPOSURE)
        self.assertIn("regime: DEGENERATE-NONE; no instability among agreeing verdicts", txt)
        self.assertIn("over disagreement-first: 0 (0.0000) verdicts", txt)

    def test_secondary_uses_usable_labels_only(self):
        slots, Y, bench = mixed_panel()
        d = describe_panel(slots, Y=Y, benchmarks=bench)
        # Instability-first: 1 + 1 + 1/2 of 0; disagreement-first: 1 + 2/6 of 2; random: 4/8 of 3.
        self.assertEqual(d["captures"], {"AUG": 2, "DIS": Fraction(5, 3), "RAND": Fraction(3, 2)})
        self.assertEqual(d["N_Y"], 7)                               # the invalid A1 with a usable label counts
        self.assertEqual(d["gain"]["over_disagreement"]["per_100_panel"], Fraction(10, 3))
        self.assertEqual(d["secondary"], {"over_disagreement": Fraction(100, 21)})
        self.assertEqual(d["gain"]["disagreement_over_random"],     # 5/3 - 3/2, per 100 on the panel of 10
                         {"verdicts": Fraction(1, 6), "per_100_panel": Fraction(5, 3)})
        txt = sap.describe_report(d, EXPOSURE)
        self.assertIn("(N_Y = 7 of N = 10)", txt)
        lines = txt.split("\n")
        self.assertIn("  over disagreement-first: 100/21 (4.7619) per 100", lines)
        self.assertFalse([x for x in lines if x.startswith("  over random") and "panel tasks" not in x])

    def test_no_usable_label_at_all(self):
        labels = labels_for(0, 2, 0, 2, 16)
        slots, _ = build(labels, [0] * 20)
        d = describe_panel(slots, Y=["Unsure"] * 10 + [None] * 10)
        self.assertEqual(d["N_Y"], 0)
        self.assertEqual(d["secondary"], {"over_disagreement": None})
        self.assertEqual(d["gain"]["over_disagreement"]["per_100_panel"], Fraction(0))
        txt = sap.describe_report(d, EXPOSURE)
        self.assertIn("over disagreement-first: not defined per 100", txt)

    def secondary_block(self, d):
        lines = sap.describe_report(d, EXPOSURE).split("\n")
        start = next(i for i, x in enumerate(lines) if x.startswith("SECONDARY RESULT"))
        return lines[start:lines.index("", start)]

    def test_the_secondary_prints_in_every_regime(self):
        seen = set()
        for sizes in ((1, 3, 0, 0, 16), (0, 7, 0, 1, 0), (0, 0, 1, 1, 1), (0, 30, 3, 7, 160), (0, 10, 4, 10, 36)):
            labels = labels_for(*sizes)
            d = describe(labels, [i % 2 for i in range(len(labels))])
            seen.add(d["regime"])
            block = self.secondary_block(d)
            self.assertEqual(block[0], f"SECONDARY RESULT: the same gain per 100 tasks with a usable label "
                                       f"(N_Y = {len(labels)} of N = {len(labels)})", sizes)
            want = d["secondary"]
            self.assertEqual(block[1:], [f"  over disagreement-first: {sap._exact_text(want['over_disagreement'])} per 100"], sizes)
            self.assertEqual(want["over_disagreement"], d["gain"]["over_disagreement"]["verdicts"] * 100 / len(labels), sizes)
        self.assertEqual(seen, REGIMES)

    def test_the_secondary_is_not_defined_without_a_usable_label_in_every_regime(self):
        seen = set()
        for sizes in ((1, 3, 0, 0, 16), (0, 7, 0, 1, 0), (0, 0, 1, 1, 1), (0, 30, 3, 7, 160), (0, 10, 4, 10, 36)):
            labels = labels_for(*sizes)
            slots, _ = build(labels, [0] * len(labels))
            Y = ["Unsure" if i % 2 else None for i in range(len(labels))]
            d = describe_panel(slots, Y=Y)
            seen.add(d["regime"])
            self.assertEqual(d["N_Y"], 0, sizes)
            self.assertEqual(d["secondary"], {"over_disagreement": None}, sizes)
            self.assertEqual(self.secondary_block(d),
                             [f"SECONDARY RESULT: the same gain per 100 tasks with a usable label (N_Y = 0 of N = {len(labels)})",
                              "  over disagreement-first: not defined per 100"], sizes)
        self.assertEqual(seen, REGIMES)

    def test_the_secondary_carries_no_reference_line(self):
        for d in report_panels():
            if d["secondary"] is not None:
                block = "\n".join(self.secondary_block(d))
                self.assertNotIn("practical", block)
                self.assertNotIn("2 per 100", block)

    def test_empty_panel_rates_are_not_defined(self):
        d = describe_panel([], Y=[])
        self.assertIsNone(d["gain"]["over_disagreement"]["per_100_panel"])
        self.assertIsNone(d["practical_reference"]["status"])
        self.assertIsNone(d["gain"]["disagreement_over_random"]["per_100_panel"])
        self.assertIn("no comparison is defined", sap.describe_report(d, EXPOSURE))
        self.assertIn("check, disagreement-first minus random: 0 (0.0000) verdicts; not defined per 100 panel tasks",
                      sap.describe_report(d, EXPOSURE))


ROWS = [  # A1, B1, released label, count; failure is the positive class
    ("failure", "failure", "Unsuccessful", 2),
    ("failure", "success", "Unsuccessful", 1),
    ("failure", "failure", "Successful", 1),
    ("success", "failure", "Unsuccessful", 1),
    ("success", "success", "Unsuccessful", 1),
    ("success", "success", "Successful", 3),
    ("success", "failure", "Successful", 1),
    ("REFUSAL", "failure", "Unsuccessful", 1),
    ("REFUSAL", "success", "Successful", 1),
    ("failure", "MALFORMED", "Unsuccessful", 1),
    ("success", "TRUNCATED", "Successful", 1),
    ("failure", "failure", "Unsure", 1),
    ("success", "success", None, 1),
]


def from_rows(rows):
    slots, Y = [], []
    for a1, b1, y, count in rows:
        for _ in range(count):
            slots.append({"A1": a1, "A2": "success", "A3": "success", "B1": b1})
            Y.append(y)
    return slots, Y


class TestValidityMetrics(unittest.TestCase):
    def assert_rate(self, got, value, num, den):
        self.assertEqual((got["value"], got["numerator"], got["denominator"]), (value, num, den))

    def assert_metrics(self, m, confusion, invalid, balanced, precision, recall, success_recall, coverage, prevalence):
        self.assertEqual(m["confusion"], dict(zip(("TP", "FP", "FN", "TN"), confusion)))
        tp, fp, fn, tn = confusion
        inv_f, inv_s = invalid
        self.assertEqual(m["invalid"], {"failure_references": inv_f, "success_references": inv_s})
        failure, success = tp + fn + inv_f, tn + fp + inv_s
        self.assertEqual(m["balanced_accuracy"], {"value": balanced, "failure_references": failure,
                                                  "success_references": success})
        self.assert_rate(m["failure_precision"], precision, tp, tp + fp)
        self.assert_rate(m["failure_recall"], recall, tp, failure)
        self.assert_rate(m["success_recall"], success_recall, tn, success)
        n = tp + fp + fn + tn
        self.assert_rate(m["contradiction"], Fraction(fp + fn, n) if n else None, fp + fn, n)
        self.assertEqual(m["contradiction"]["invalid_outputs"], inv_f + inv_s)
        self.assert_rate(m["valid_coverage"], coverage[0], coverage[1], coverage[2])
        self.assert_rate(m["prevalence"], prevalence, failure, failure + success)

    def test_hand_computed_confusion_matrices(self):
        slots, Y = from_rows(ROWS)
        v = describe_panel(slots, Y=Y)["validity"]
        self.assertEqual((v["n_usable"], v["n_shared"], v["n_excluded_from_shared"]), (14, 10, 4))
        F = Fraction
        self.assert_metrics(v["all_usable"]["A1"], (4, 1, 2, 5), (1, 1), F(9, 14), F(4, 5), F(4, 7), F(5, 7), (F(6, 7), 12, 14), F(1, 2))
        self.assert_metrics(v["all_usable"]["B1"], (4, 2, 2, 4), (1, 1), F(4, 7), F(2, 3), F(4, 7), F(4, 7), (F(6, 7), 12, 14), F(1, 2))
        self.assert_metrics(v["shared"]["A1"], (3, 1, 2, 4), (0, 0), F(7, 10), F(3, 4), F(3, 5), F(4, 5), (F(5, 7), 10, 14), F(1, 2))
        self.assert_metrics(v["shared"]["B1"], (3, 2, 2, 3), (0, 0), F(3, 5), F(3, 5), F(3, 5), F(3, 5), (F(5, 7), 10, 14), F(1, 2))

    def test_a_judge_cannot_gain_by_refusing_hard_cases(self):
        # Three failure references and three success references. The judge answers the two it
        # gets right and refuses the rest: every refusal stays in the denominator as not correct.
        answers = [("failure", "failure", "Unsuccessful", 1), ("REFUSAL", "failure", "Unsuccessful", 2),
                   ("success", "success", "Successful", 1), ("REFUSAL", "success", "Successful", 2)]
        slots, Y = from_rows(answers)
        v = describe_panel(slots, Y=Y)["validity"]["all_usable"]["A1"]
        self.assertEqual(v["failure_recall"]["value"], Fraction(1, 3))
        self.assertEqual(v["success_recall"]["value"], Fraction(1, 3))
        self.assertEqual(v["balanced_accuracy"]["value"], Fraction(1, 3))
        self.assertEqual(v["failure_precision"]["value"], 1)
        self.assertEqual(v["contradiction"]["value"], 0)
        self.assertEqual(v["valid_coverage"]["value"], Fraction(1, 3))

    def test_report_gives_every_metric_with_its_denominator(self):
        slots, Y = from_rows(ROWS)
        txt = sap.describe_report(describe_panel(slots, Y=Y), EXPOSURE)
        for needle in ("Tasks with a usable reference: N_Y = 14; shared subset, where A1 and B1 are both valid: 10; "
                       "usable tasks excluded from the shared subset: 4.",
                       "A1 on all tasks with a usable reference (n = 14)",
                       "B1 on all tasks with a usable reference (n = 14)",
                       "A1 on the shared subset (n = 10)",
                       "B1 on the shared subset (n = 10)",
                       "confusion matrix of valid verdicts: TP = 4, FP = 1, FN = 2, TN = 5",
                       "invalid outputs: 1 on failure references, 1 on success references",
                       "balanced accuracy: 9/14 (0.6429), denominators 7 failure references and 7 success references, "
                       "an invalid output counted as not correct",
                       "failure precision: 4/5 (0.8000), 4 of 5 valid failure verdicts",
                       "failure recall: 4/7 (0.5714), 4 of 7 failure references, invalid outputs included",
                       "success recall: 5/7 (0.7143), 5 of 7 success references, invalid outputs included",
                       "contradiction rate: 1/4 (0.2500), 3 of 12 valid verdicts in this set; invalid outputs in this set: 2",
                       "valid coverage: 6/7 (0.8571), 12 of 14 tasks with a usable reference",
                       "prevalence of failure references: 1/2 (0.5000), 7 of 14 tasks in this set"):
            self.assertIn(needle, txt, needle)

    def test_each_heading_prints_its_own_set(self):
        slots, Y = from_rows(ROWS)
        lines = sap.describe_report(describe_panel(slots, Y=Y), EXPOSURE).split("\n")
        for heading, confusion in (("  A1 on all tasks with a usable reference (n = 14)", "TP = 4, FP = 1, FN = 2, TN = 5"),
                                   ("  B1 on all tasks with a usable reference (n = 14)", "TP = 4, FP = 2, FN = 2, TN = 4"),
                                   ("  A1 on the shared subset (n = 10)", "TP = 3, FP = 1, FN = 2, TN = 4"),
                                   ("  B1 on the shared subset (n = 10)", "TP = 3, FP = 2, FN = 2, TN = 3")):
            self.assertEqual(lines[lines.index(heading) + 1], "    confusion matrix of valid verdicts: " + confusion)
        start = lines.index("  A1 on the shared subset (n = 10)")
        self.assertEqual(lines[start + 2:start + 10],
                         ["    invalid outputs: 0 on failure references, 0 on success references",
                          "    balanced accuracy: 7/10 (0.7000), denominators 5 failure references and 5 success references, "
                          "an invalid output counted as not correct",
                          "    failure recall: 3/5 (0.6000), 3 of 5 failure references, invalid outputs included",
                          "    success recall: 4/5 (0.8000), 4 of 5 success references, invalid outputs included",
                          "    failure precision: 3/4 (0.7500), 3 of 4 valid failure verdicts",
                          "    contradiction rate: 3/10 (0.3000), 3 of 10 valid verdicts in this set; invalid outputs in this set: 0",
                          "    valid coverage: 5/7 (0.7143), 10 of 14 tasks with a usable reference",
                          "    prevalence of failure references: 1/2 (0.5000), 5 of 10 tasks in this set"])

    def test_verdicts_match_in_any_letter_case(self):
        slots, Y = from_rows(ROWS)
        want = describe_panel(slots, Y=Y)
        for spell in (str.upper, str.title):
            mixed = [{k: spell(v) if v in sap.VERDICTS else v for k, v in s.items()} for s in slots]
            d = describe_panel(mixed, Y=Y)
            self.assertEqual(d["validity"], want["validity"])
            self.assertEqual(d["contradiction"], want["contradiction"])
            self.assert_metrics(d["validity"]["all_usable"]["A1"], (4, 1, 2, 5), (1, 1), Fraction(9, 14), Fraction(4, 5),
                                Fraction(4, 7), Fraction(5, 7), (Fraction(6, 7), 12, 14), Fraction(1, 2))

    def test_empty_denominators_are_not_defined_and_never_zero(self):
        # No failure reference and no failure prediction.
        slots, Y = from_rows([("success", "success", "Successful", 4)])
        d = describe_panel(slots, Y=Y)
        a = d["validity"]["all_usable"]["A1"]
        self.assert_metrics(a, (0, 0, 0, 4), (0, 0), None, None, None, Fraction(1), (Fraction(1), 4, 4), Fraction(0))
        txt = sap.describe_report(d, EXPOSURE)
        self.assertIn("balanced accuracy: not defined, denominators 0 failure references and 4 success references", txt)
        self.assertIn("failure precision: not defined, 0 of 0 valid failure verdicts", txt)
        self.assertIn("failure recall: not defined, 0 of 0 failure references", txt)
        self.assertIn("prevalence of failure references: 0 (0.0000), 0 of 4 tasks in this set", txt)  # a true zero
        self.assertNotIn("balanced accuracy: 0", txt)
        self.assertNotIn("failure precision: 0", txt)
        self.assertNotIn("failure recall: 0", txt)
        # Failure references only, none found: recall is a true zero, balanced accuracy is not defined.
        slots, Y = from_rows([("success", "success", "Unsuccessful", 3)])
        a = describe_panel(slots, Y=Y)["validity"]["all_usable"]["A1"]
        self.assert_metrics(a, (0, 0, 3, 0), (0, 0), None, None, Fraction(0), None, (Fraction(1), 3, 3), Fraction(1))
        # No valid first verdict: every invalid output stays in the recalls as not correct.
        slots, Y = from_rows([("REFUSAL", "success", "Unsuccessful", 2), ("REFUSAL", "success", "Successful", 1)])
        d = describe_panel(slots, Y=Y)
        self.assert_metrics(d["validity"]["all_usable"]["A1"], (0, 0, 0, 0), (2, 1), Fraction(0), None, Fraction(0), Fraction(0),
                            (Fraction(0), 0, 3), Fraction(2, 3))
        self.assert_metrics(d["validity"]["shared"]["B1"], (0, 0, 0, 0), (0, 0), None, None, None, None, (Fraction(0), 0, 3), None)
        self.assertIn("prevalence of failure references: not defined, 0 of 0 tasks in this set",
                      sap.describe_report(d, EXPOSURE))
        # No usable reference: even the coverage has no denominator.
        slots, Y = from_rows([("success", "success", "Unsure", 2), ("failure", "failure", None, 1)])
        d = describe_panel(slots, Y=Y)
        self.assert_metrics(d["validity"]["all_usable"]["A1"], (0, 0, 0, 0), (0, 0), None, None, None, None, (None, 0, 0), None)
        self.assertIn("valid coverage: not defined, 0 of 0 tasks with a usable reference",
                      sap.describe_report(d, EXPOSURE))

    def test_balanced_accuracy_averages_the_two_class_recalls(self):
        slots, Y = from_rows([("failure", "failure", "Unsuccessful", 1), ("success", "success", "Unsuccessful", 3),
                              ("success", "success", "Successful", 2)])
        a = describe_panel(slots, Y=Y)["validity"]["all_usable"]["A1"]
        self.assertEqual(a["balanced_accuracy"]["value"], (Fraction(1, 4) + 1) / 2)

    def test_a_judge_word_as_a_reference_label_stops_the_analysis(self):
        slots, Y = from_rows([("success", "success", "Successful", 2)])
        with self.assertRaises(ValueError):
            describe_panel(slots, Y=["success", "Successful"])


class TestJudgeContradiction(unittest.TestCase):
    def test_each_judge_is_measured_on_its_own_valid_tasks(self):
        rows = [("failure", "failure", "Successful", 4), ("success", "success", "Successful", 2),
                ("success", "failure", "Successful", 4)]
        slots, Y = from_rows(rows)
        d = describe_panel(slots, Y=Y)
        self.assertEqual(d["contradiction"]["A1"], {"value": Fraction(2, 5), "numerator": 4, "denominator": 10, "invalid_outputs": 0})
        self.assertEqual(d["contradiction"]["B1"], {"value": Fraction(4, 5), "numerator": 8, "denominator": 10, "invalid_outputs": 0})
        lines = sap.describe_report(d, EXPOSURE).split("\n")
        self.assertIn("  A1: 2/5 (0.4000), 4 of 10 tasks with a valid A1 and a usable reference; "
                      "invalid outputs among tasks with a usable reference: 0", lines)
        self.assertIn("  B1: 4/5 (0.8000), 8 of 10 tasks with a valid B1 and a usable reference; "
                      "invalid outputs among tasks with a usable reference: 0", lines)

    def test_invalid_and_unknown_tasks_leave_the_denominator(self):
        slots, Y = from_rows(ROWS)
        d = describe_panel(slots, Y=Y)
        self.assertEqual(d["contradiction"]["A1"], {"value": Fraction(1, 4), "numerator": 3, "denominator": 12, "invalid_outputs": 2})
        self.assertEqual(d["contradiction"]["B1"], {"value": Fraction(1, 3), "numerator": 4, "denominator": 12, "invalid_outputs": 2})
        self.assertEqual(d["contradiction"]["A1"]["numerator"], d["M"])
        self.assertEqual(d["contradiction"]["A1"], d["validity"]["all_usable"]["A1"]["contradiction"])
        self.assertEqual(d["validity"]["shared"]["A1"]["contradiction"]["denominator"], 10)
        self.assertNotIn("own", d["validity"])
        self.assertEqual(d["validity"]["shared"]["B1"]["contradiction"]["numerator"], 4)

    def test_E_with_statuses_gives_A1_and_says_why_B1_is_not_assessed(self):
        slots, Y, _ = mixed_panel()
        j = sap.join_reference(slots, Y)
        d = describe_panel(slots, j["E"], reference=j["reference"])
        self.assertEqual(d["contradiction"]["A1"], {"value": Fraction(1, 2), "numerator": 3, "denominator": 6, "invalid_outputs": 1})
        self.assertIsNone(d["contradiction"]["B1"])
        lines = sap.describe_report(d, EXPOSURE).split("\n")
        self.assertIn("  A1: 1/2 (0.5000), 3 of 6 tasks with a valid A1 and a usable reference; "
                      "invalid outputs among tasks with a usable reference: 1", lines)
        self.assertIn("  B1: not assessed: the released labels were not supplied, and E alone cannot give "
                      "B1's reference where A1 is invalid.", lines)

    def test_E_without_statuses_says_every_label_was_treated_as_usable(self):
        d = describe_panel([slot("I"), slot("S"), slot("S")], [1, 1, 0])
        self.assertEqual(d["contradiction"]["A1"], {"value": Fraction(1, 2), "numerator": 1, "denominator": 2, "invalid_outputs": 1})
        self.assertIn("  A1: 1/2 (0.5000), 1 of 2 tasks with a valid A1 and a reference, every label treated as usable; "
                      "invalid outputs among tasks with a reference: 1",
                      sap.describe_report(d, EXPOSURE).split("\n"))

    def test_an_empty_set_is_not_defined(self):
        slots, Y = from_rows([("REFUSAL", "MALFORMED", "Successful", 2)])
        d = describe_panel(slots, Y=Y)
        self.assertEqual(d["contradiction"]["A1"], {"value": None, "numerator": 0, "denominator": 0, "invalid_outputs": 2})
        self.assertEqual(d["contradiction"]["B1"], {"value": None, "numerator": 0, "denominator": 0, "invalid_outputs": 2})
        lines = sap.describe_report(d, EXPOSURE).split("\n")
        self.assertIn("  A1: not defined, 0 of 0 tasks with a valid A1 and a usable reference; "
                      "invalid outputs among tasks with a usable reference: 2", lines)
        self.assertIn("  B1: not defined, 0 of 0 tasks with a valid B1 and a usable reference; "
                      "invalid outputs among tasks with a usable reference: 2", lines)
        slots, Y = from_rows([("success", "success", "Unsure", 2)])
        self.assertIsNone(describe_panel(slots, Y=Y)["contradiction"]["A1"]["value"])


SHOWS = {"A": "Reached by instability, and by disagreement-first only by chance",
         "B": "Flagged by instability, the reference agrees with A1",
         "C": "Caught by both main policies",
         "D": "Mostly missed by both main policies",
         "E": "Control matched to the first D case on benchmark and A1 verdict, falling back to A1 verdict alone",
         "F": "The unknown tasks",
         "second control": "a second control matched to the second D case"}
USABLE, UNSURE, MISSING = "usable", "unsure", "missing"


def case_panel(rows):
    """Slots, E, statuses, benchmarks and ids from (group, E, status, benchmark, A1 verdict) rows.

    The group 'G1x' is a disagreement by an invalid B1.
    """
    slots = [dict(slot("G1", r[4]), B1="MALFORMED") if r[0] == "G1x" else slot(r[0], r[4]) for r in rows]
    return (slots, [r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows],
            [f"t{i:03d}" for i in range(len(rows))])


def picks(rows, seed=0, benchmarks=True):
    slots, E, status, bench, ids = case_panel(rows)
    chosen = sap.select_cases(slots, E, status, benchmarks=bench if benchmarks else None, ids=ids, seed=seed)
    return [(c["cell"], c["index"]) for c in chosen]


def rule_written_out(rows, ids, seed, benchmarks=True):
    """Plan section 11 written out on its own, for comparison with select_cases."""
    n = len(rows)
    group, e, status, bench, a1 = zip(*rows) if rows else ((), (), (), (), ())

    def cell(index, test):
        members = sorted((i for i in range(n) if test(i)), key=lambda i: ids[i])
        rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(6, index)))
        return [members[j] for j in rng.permutation(len(members))]

    def is_u(i):
        return group[i] in ("U2", "U1")

    A = cell(0, lambda i: is_u(i) and e[i] == 1)
    B = cell(1, lambda i: is_u(i) and status[i] == USABLE and e[i] == 0)
    C = cell(2, lambda i: group[i] == "G1" and e[i] == 1)
    D = cell(3, lambda i: group[i] == "S" and e[i] == 1)
    E = cell(4, lambda i: group[i] == "S" and status[i] == USABLE and e[i] == 0)
    F = cell(5, lambda i: group[i] != "I" and status[i] != USABLE)

    def control(d, used):
        for test in ((lambda i: benchmarks and bench[i] == bench[d] and a1[i] == a1[d]), (lambda i: a1[i] == a1[d])):
            for i in E:
                if i not in used and test(i):
                    return i
        return None

    u2 = [i for i in A if group[i] == "U2"]
    u1 = [i for i in A if group[i] == "U1"]
    out = [("A", i) for i in ([u2[0], u1[0]] if u2 and u1 else A[:2])]
    out += [("B", i) for i in B[:1]] + [("C", i) for i in C[:1]] + [("D", i) for i in D[:2]]
    first = control(D[0], ()) if D else (E[0] if E else None)
    if first is not None:
        out.append(("E", first))
    if F:
        out.append(("F", F[0]))
    elif len(D) > 1:
        second = control(D[1], (first,))
        if second is not None:
            out.append(("F", second))
    return out


class TestCaseRule(unittest.TestCase):
    OK = (USABLE, "wa", "success")

    def row(self, group, e, status=USABLE, bench="wa", a1="success"):
        return (group, e, status, bench, a1)

    def test_each_cell_holds_what_the_plan_defines(self):
        r = self.row
        for one, cell in ((r("U2", 1), "A"), (r("U1", 1), "A"), (r("U2", 0), "B"), (r("U1", 0), "B"),
                          (r("G1", 1), "C"), (r("S", 1), "D"), (r("S", 0), "E"),
                          (r("U2", 0, UNSURE), "F"), (r("U1", 0, MISSING), "F"), (r("G1", 0, UNSURE), "F"),
                          (r("G1x", 0, MISSING), "F"), (r("S", 0, MISSING), "F")):
            self.assertEqual(picks([one]), [(cell, 0)], one)

    def test_no_cell_takes_these_tasks(self):
        r = self.row
        for one in (r("I", 0), r("I", 0, UNSURE), r("I", 0, MISSING), r("G1", 0), r("G1x", 1), r("G1x", 0)):
            self.assertEqual(picks([one]), [], one)

    def test_cell_A_takes_one_of_each_group_when_both_exist(self):
        rows = [self.row("U2", 1)] * 6 + [self.row("U1", 1)] * 6
        for seed in range(20):
            got = picks(rows, seed)
            self.assertEqual([c for c, _ in got], ["A", "A"], seed)
            self.assertEqual([rows[i][0] for _, i in got], ["U2", "U1"], seed)

    def test_cell_A_with_one_group_takes_up_to_two_from_it(self):
        for group in ("U2", "U1"):
            got = picks([self.row(group, 1)] * 5)
            self.assertEqual([c for c, _ in got], ["A", "A"], group)
            self.assertEqual(len({i for _, i in got}), 2, group)
            self.assertEqual(picks([self.row(group, 1)]), [("A", 0)], group)
        got = picks([self.row("U2", 1)] + [self.row("U1", 1)])       # one of each, both taken
        self.assertEqual(got, [("A", 0), ("A", 1)])

    def test_cells_take_their_numbers_and_a_short_cell_is_not_backfilled(self):
        r = self.row
        rows = [r("U2", 1)] * 3 + [r("U1", 1)] * 3 + [r("U1", 0)] * 3 + [r("G1", 1)] * 3 + [r("S", 1)] * 3 + \
               [r("S", 0)] * 3 + [r("U1", 0, UNSURE)] * 3
        self.assertEqual(Counter(c for c, _ in picks(rows)), {"A": 2, "B": 1, "C": 1, "D": 2, "E": 1, "F": 1})
        # One task in cell D: cell D gives one, and no other cell makes up for it.
        rows = [r("S", 1)] + [r("U1", 0)] * 3 + [r("S", 0)] * 3 + [r("U1", 0, UNSURE)] * 2
        self.assertEqual(Counter(c for c, _ in picks(rows)), {"B": 1, "D": 1, "E": 1, "F": 1})
        # Nothing contradicted among the agreeing tasks: cells A and D stay empty.
        rows = [r("U1", 0)] * 4 + [r("S", 0)] * 4 + [r("G1", 1)] * 4
        self.assertEqual(Counter(c for c, _ in picks(rows)), {"B": 1, "C": 1, "E": 1})

    def test_rows_come_in_cell_order_without_repeats_and_never_above_eight(self):
        rng = np.random.default_rng(404)
        groups = ("I", "G1", "G1x", "U2", "U1", "S")
        for seed in range(150):
            rows = []
            for _ in range(int(rng.integers(0, 60))):
                status = str(rng.choice([USABLE, USABLE, UNSURE, MISSING]))
                group = str(rng.choice(groups))
                e = int(group != "I" and status == USABLE and rng.random() < 0.5)
                rows.append((group, e, status, str(rng.choice(["wa", "vwa", "work"])), str(rng.choice(["success", "failure"]))))
            got = picks(rows, seed)
            self.assertLessEqual(len(got), 8)
            cells = [c for c, _ in got]
            self.assertEqual(cells, sorted(cells))
            self.assertEqual(len({i for _, i in got}), len(got))
            self.assertTrue(all(rows[i][0] != "I" for _, i in got))

    def test_the_rule_written_out_on_random_panels(self):
        rng = np.random.default_rng(1109)
        groups = ("I", "G1", "G1x", "U2", "U1", "S")
        seen = set()
        for seed in range(250):
            rows = []
            for _ in range(int(rng.integers(0, 45))):
                status = str(rng.choice([USABLE, USABLE, USABLE, UNSURE, MISSING] if seed % 2 else [USABLE]))
                group = str(rng.choice(groups))
                e = int(group != "I" and status == USABLE and rng.random() < 0.5)
                rows.append((group, e, status, str(rng.choice(["wa", "vwa", "work"])), str(rng.choice(["success", "failure"]))))
            ids = [f"task-{(7919 * i) % 1009:04d}" for i in range(len(rows))]
            slots, E, status, bench, _ = case_panel(rows)
            for with_bench in (True, False):
                got = sap.select_cases(slots, E, status, benchmarks=bench if with_bench else None, ids=ids, seed=seed)
                got = [(c["cell"], c["index"]) for c in got]
                self.assertEqual(got, rule_written_out(rows, ids, seed, with_bench), (seed, with_bench))
                seen |= {c for c, _ in got}
        self.assertEqual(seen, set("ABCDEF"))

    def test_the_control_matches_on_benchmark_and_verdict_first(self):
        r = self.row
        both = r("S", 0, USABLE, "wa", "success")
        verdict = r("S", 0, USABLE, "work", "success")
        benchmark = r("S", 0, USABLE, "wa", "failure")
        D = r("S", 1, USABLE, "wa", "success")
        for seed in range(10):
            self.assertEqual(picks([D, verdict, benchmark, both, verdict], seed), [("D", 0), ("E", 3)], seed)
            got = picks([D, verdict, benchmark], seed)
            self.assertEqual(got, [("D", 0), ("E", 1)], seed)
        # A benchmark match of the other verdict is never a control, and with no verdict match there is none.
        self.assertEqual(picks([D, benchmark]), [("D", 0)])

    def test_the_control_of_a_panel_without_benchmarks_matches_on_the_verdict_alone(self):
        rows = [self.row("S", 1, USABLE, "wa", "failure")] + [self.row("S", 0, USABLE, "work", "failure")] * 5 + \
               [self.row("S", 0, USABLE, "wa", "success")] * 5
        slots, E, status, _, ids = case_panel(rows)
        chosen = {c["cell"]: c for c in sap.select_cases(slots, E, status, ids=ids)}
        self.assertEqual(rows[chosen["E"]["index"]][4], "failure")
        self.assertEqual(chosen["E"]["matched_on"], "A1 verdict alone")

    def test_the_control_answers_to_the_first_D_case(self):
        rows = [self.row("S", 1, USABLE, "wa", "success"), self.row("S", 1, USABLE, "work", "failure")]
        rows += [self.row("S", 0, USABLE, "wa", "success")] * 3 + [self.row("S", 0, USABLE, "work", "failure")] * 3
        seen = set()
        for seed in range(30):
            got = dict(zip(("D1", "D2", "E", "F"), picks(rows, seed)))
            first = rows[got["D1"][1]]
            self.assertEqual(rows[got["E"][1]][3:], first[3:], seed)
            second = rows[got["D2"][1]]
            self.assertEqual(got["F"][0], "F", seed)
            self.assertEqual(rows[got["F"][1]][3:], second[3:], seed)      # the second control, to the second D case
            seen.add(got["D1"][1])
        self.assertEqual(seen, {0, 1})

    def test_with_no_D_case_cell_E_takes_the_first_of_its_order(self):
        rows = [self.row("S", 0, USABLE, "wa", "success")] * 7
        ids = [f"t{i:03d}" for i in range(7)]
        seen = set()
        for seed in range(25):
            rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(6, 4)))
            want = [int(i) for i in rng.permutation(7)]
            self.assertEqual(picks(rows, seed), [("E", want[0])], seed)
            seen.add(want[0])
            slots, E, status, bench, _ = case_panel(rows)
            self.assertEqual(sap.select_cases(slots, E, status, benchmarks=bench, ids=ids, seed=seed)[0]["matched_on"],
                             "no D case, so the first of the cell's order")
        self.assertGreater(len(seen), 1)

    def test_the_F_rule(self):
        r = self.row
        D = r("S", 1, USABLE, "wa", "success")
        control = r("S", 0, USABLE, "wa", "success")
        # A task with an Unsure reference fills cell F and no second control is taken.
        got = picks([D, D, control, control, r("U1", 0, UNSURE)])
        self.assertEqual([c for c, _ in got], ["D", "D", "E", "F"])
        self.assertEqual(got[-1], ("F", 4))
        # Cell F empty and two D cases: a second control that differs from the first.
        for seed in range(20):
            got = picks([D, D, control, control, control], seed)
            self.assertEqual([c for c, _ in got], ["D", "D", "E", "F"], seed)
            self.assertNotEqual(got[2][1], got[3][1], seed)
            self.assertIn(got[3][1], (2, 3, 4), seed)
        # Only one control there is: it serves the first D case and the second goes without.
        self.assertEqual([c for c, _ in picks([D, D, control])], ["D", "D", "E"])
        # One D case: no second control.
        self.assertEqual([c for c, _ in picks([D, control, control])], ["D", "E"])
        # No D case: none either.
        self.assertEqual([c for c, _ in picks([control, control])], ["E"])

    def test_an_unmatched_first_D_case_leaves_the_control_open_to_the_second(self):
        r = self.row
        rows = [r("S", 1, USABLE, "wa", "failure"), r("S", 1, USABLE, "wa", "success"), r("S", 0, USABLE, "wa", "success")]
        seen = set()
        for seed in range(30):
            got = picks(rows, seed)
            if got[0][1] == 0:                  # the first D case has no control of its verdict
                self.assertEqual(got[2:], [("F", 2)], seed)
            else:
                self.assertEqual(got[2:], [("E", 2)], seed)
            seen.add(got[0][1])
        self.assertEqual(seen, {0, 1})

    def test_each_cell_has_its_own_generator(self):
        r = self.row
        base = [r("U1", 0)] * 8 + [r("S", 0)] * 8
        grown = base + [r("S", 1)] * 4 + [r("G1", 1)] * 9
        for seed in range(10):
            a = dict(picks(base, seed))
            b = dict(picks(grown, seed))
            self.assertEqual(a["B"], b["B"], seed)

    def test_the_seed_decides_the_choice_and_is_reported(self):
        rows = [self.row("U1", 0)] * 40
        self.assertEqual(picks(rows, 3), picks(rows, 3))
        self.assertGreater(len({tuple(picks(rows, s)) for s in range(10)}), 1)

    def test_the_choice_follows_the_ids_not_the_stored_order(self):
        rng = np.random.default_rng(11)
        rows = [("U2", 1, USABLE, "wa", "success")] * 4 + [("U1", 1, USABLE, "wa", "failure")] * 4 + \
               [("S", 1, USABLE, "work", "success")] * 5 + [("S", 0, USABLE, "work", "success")] * 5 + \
               [("U1", 0, UNSURE, "vwa", "failure")] * 3 + [("G1", 1, USABLE, "wa", "failure")] * 3
        slots, E, status, bench, ids = case_panel(rows)
        order = [int(i) for i in rng.permutation(len(rows))]
        moved = lambda x: [x[i] for i in order]
        a = sap.select_cases(slots, E, status, benchmarks=bench, ids=ids, seed=5)
        b = sap.select_cases(moved(slots), moved(E), moved(status), benchmarks=moved(bench), ids=moved(ids), seed=5)
        self.assertEqual([(c["cell"], ids[c["index"]]) for c in a], [(c["cell"], moved(ids)[c["index"]]) for c in b])

    def test_input_is_checked(self):
        ok = slot("S")
        with self.assertRaises(ValueError):
            sap.select_cases([ok], [0, 0], [USABLE])
        with self.assertRaises(ValueError):
            sap.select_cases([ok], [0], [USABLE, USABLE])
        with self.assertRaises(ValueError):
            sap.select_cases([ok], [0], [USABLE], benchmarks=["a", "b"])
        with self.assertRaises(ValueError):
            sap.select_cases([ok, ok], [0, 0], [USABLE] * 2, ids=["a", "a"])


class TestCases(unittest.TestCase):
    IDS = [f"t{i}" for i in range(10)]
    # Plan section 11, word for word, with seed 9 in the place the plan leaves for it.
    RULE = ["A few cases, chosen by this recorded rule, explain what the policies caught and missed. They are a "
            "selected qualitative sample by one reviewer, not an estimate of how common anything is.",
            "Selection. At most 8 cases, on valid A1 only, chosen after the sealed join by a script with a seed in "
            "the freeze list. Seed: 9.",
            "Cell | Definition | Cases | What it shows",
            "A | Unstable agreeing, E = 1 | 2, one U2 and one U1 when both exist | Reached by instability, and by "
            "disagreement-first only by chance",
            "B | Unstable agreeing, usable label, E = 0 | 1 | Flagged by instability, the reference agrees with A1",
            "C | Disagreement with valid B1, E = 1 | 1 | Caught by both main policies",
            "D | Stable agreeing, E = 1 | 2 | Mostly missed by both main policies",
            "E | Stable agreeing, usable label, E = 0 | 1 | Control matched to the first D case on benchmark and A1 "
            "verdict, falling back to A1 verdict alone",
            "F | Unsure or missing label | 1 | The unknown tasks. If none exist, a second control matched to the "
            "second D case",
            "Within a cell, tasks are sorted by normalized task ID and permuted by a generator derived from the "
            "frozen seed and the cell. A short cell takes all it has. Nothing is backfilled except by the F rule. "
            "A recognized case is never replaced.",
            "Generator of a cell: SeedSequence(9, spawn_key=(6, cell index)), with cells A to F at index 0 to 5. "
            "A cell A with only U2 or only U1 takes up to 2 from that group. With no D case, cell E takes the first "
            "of its order. The control in cell F is distinct from the first control, and there is none without a "
            "second D case."]

    def mixed(self, **kw):
        slots, Y, _ = mixed_panel()
        return describe_panel(slots, Y=Y, ids=self.IDS, **kw)

    def test_the_rule_prints_word_for_word_with_the_seed(self):
        d = self.mixed(case_seed=9)
        self.assertEqual((d["cases"]["seed"], d["cases"]["sort_key"]), (9, "ids"))
        self.assertEqual(d["cases"]["rule"], self.RULE)
        lines = sap.describe_report(d, EXPOSURE).split("\n")
        start = lines.index("CASES: what each policy caught and missed")
        self.assertEqual(lines[start + 1:start + 1 + len(self.RULE)], ["  " + x for x in self.RULE])
        self.assertEqual(lines[start + 1 + len(self.RULE)],
                         "  E is 1 when the released reference contradicts the first verdict. Each probability is "
                         "that policy's exact chance of reviewing the task, over its own random choices. A "
                         "contradicted task is caught with that probability and missed with the rest.")
        self.assertEqual(lines[start + 2 + len(self.RULE)],
                         "  cell | task | group | A1, A2, A3, B1 | reference | released label | second annotator | E | "
                         "Instability-first | Disagreement-first | Random | what the cell shows")

    def test_no_claim_of_blindness_is_left(self):
        for d in report_panels():
            self.assertNotIn("blind", sap.describe_report(d, EXPOSURE))

    def test_each_row_gives_cell_task_slots_reference_E_probabilities_and_what_the_cell_shows(self):
        slots, Y, _ = mixed_panel()
        j = sap.join_reference(slots, Y)
        d = self.mixed()
        lines = sap.describe_report(d, EXPOSURE).split("\n")
        rows = [t for t in d["cases"]["tasks"]]
        self.assertEqual([t["cell"] for t in rows], ["A", "B", "D", "E", "F"])   # no valid B1 with E = 1 in G1
        for row in ("A | t4 | U2 | success, failure, failure, success | usable | Unsuccessful | not supplied | 1 | 1 (1.0000) | 1/3 (0.3333) | 1/2 (0.5000) | " + SHOWS["A"],
                    "B | t8 | U1 | success, ABSTAIN, failure, success | usable | Successful | not supplied | 0 | 1/2 (0.5000) | 1/3 (0.3333) | 1/2 (0.5000) | " + SHOWS["B"],
                    "D | t7 | S | success, ABSTAIN, success, success | usable | Unsuccessful | not supplied | 1 | 0 (0.0000) | 1/3 (0.3333) | 1/2 (0.5000) | " + SHOWS["D"]):
            self.assertIn("  " + row, lines, row)
        ending = {"E": "Control matched to the first D case on benchmark and A1 verdict, falling back to A1 verdict alone (matched here: A1 verdict alone)",
                  "F": "The unknown tasks"}
        for t in rows:
            i = self.IDS.index(t["task"])
            self.assertEqual((t["E"], t["reference"]), (j["E"][i], j["reference"][i]), t["task"])
            if t["cell"] in ending:
                self.assertTrue(any(x.startswith(f"  {t['cell']} | {t['task']} | ") and x.endswith(ending[t["cell"]]) for x in lines), t)

    def test_a_case_shows_its_released_label_rationales_and_second_annotator(self):
        slots, Y, _ = mixed_panel()
        rationales = [{k: f"{k} of task {i}" for k in ("A1", "A2", "A3", "B1")} for i in range(10)]
        rationales[4]["A2"] = "split over\n  two lines"
        del rationales[4]["B1"]
        rationales[4]["A3"] = "   "
        second = [f"annotator {i}" for i in range(10)]
        second[8] = None
        d = self.mixed(rationales=rationales, second_annotator=second)
        lines = sap.describe_report(d, EXPOSURE).split("\n")
        row = ("  A | t4 | U2 | success, failure, failure, success | usable | Unsuccessful | annotator 4 | 1 | "
               "1 (1.0000) | 1/3 (0.3333) | 1/2 (0.5000) | " + SHOWS["A"])
        self.assertEqual(lines[lines.index(row) + 1], "    rationales: A1: A1 of task 4; A2: split over two lines; A3: none; B1: none")
        row = ("  B | t8 | U1 | success, ABSTAIN, failure, success | usable | Successful | none | 0 | "
               "1/2 (0.5000) | 1/3 (0.3333) | 1/2 (0.5000) | " + SHOWS["B"])
        self.assertEqual(lines[lines.index(row) + 1], "    rationales: A1: A1 of task 8; A2: A2 of task 8; A3: A3 of task 8; B1: B1 of task 8")

    def test_the_released_label_of_each_status(self):
        for y, shown in (("Unsuccessful", "Unsuccessful"), ("Unsure", "Unsure"), ("unsure", "unsure"), (None, "missing"),
                         ("  ", "missing"), (float("nan"), "missing")):
            d = describe_panel([slot("S")], Y=[y], ids=["a"])
            self.assertEqual(d["cases"]["tasks"][0]["released_label"], shown, y)

    def test_inputs_that_were_not_given_are_stated(self):
        d = describe_panel([slot("S")] * 4, [0, 1, 0, 0])
        for t in d["cases"]["tasks"]:
            self.assertEqual((t["released_label"], t["second_annotator"], t["rationales"]), ("not supplied", "not supplied", None))
        lines = sap.describe_report(d, EXPOSURE).split("\n")
        self.assertEqual(lines.count("    rationales: not supplied"), len(d["cases"]["tasks"]))

    def test_the_case_inputs_need_one_entry_per_task(self):
        slots, Y, _ = mixed_panel()
        for kw in ({"rationales": [{}] * 9}, {"second_annotator": [None] * 11}):
            with self.assertRaises(ValueError, msg=repr(kw)):
                self.mixed(**kw)

    def test_a_row_says_how_a_control_was_matched(self):
        rows = [("S", 1, USABLE, "wa", "success"), ("S", 0, USABLE, "wa", "success"), ("S", 1, USABLE, "work", "failure"),
                ("S", 0, USABLE, "wa", "failure")]
        slots, E, status, bench, ids = case_panel(rows)
        Y = [LABEL[OTHER[r[4]]] if r[1] else LABEL[r[4]] for r in rows]
        d = describe_panel(slots, Y=Y, benchmarks=bench, ids=ids)
        by_cell = {t["cell"]: t for t in d["cases"]["tasks"]}
        self.assertEqual(by_cell["E"]["matched_on"], "benchmark and A1 verdict" if by_cell["E"]["task"] == "t001" else "A1 verdict alone")
        self.assertTrue(by_cell["E"]["shows"].startswith(SHOWS["E"] + " (matched here: "))
        self.assertTrue(by_cell["F"]["shows"].startswith(SHOWS["second control"] + " (matched here: "), by_cell["F"])
        self.assertEqual(by_cell["F"]["reference"], USABLE)
        d = describe_panel([slot("S")], Y=["Successful"], ids=["only"])
        self.assertEqual(d["cases"]["tasks"][0]["shows"], SHOWS["E"] + " (matched here: no D case, so the first of the cell's order)")
        for bench, how in ((("wa", "wa"), "benchmark and A1 verdict"), (("wa", "work"), "A1 verdict alone")):
            d = describe_panel([slot("S"), slot("S")], Y=["Unsuccessful", "Successful"], benchmarks=bench, ids=["d", "e"])
            self.assertEqual(d["cases"]["tasks"][1]["matched_on"], how)
            self.assertEqual(d["cases"]["tasks"][1]["shows"], f"{SHOWS['E']} (matched here: {how})")

    def test_the_case_seed_reaches_the_choice(self):
        labels = labels_for(0, 6, 6, 6, 6)
        ids = [f"t{i:02d}" for i in range(len(labels))]
        slots, Y = build(labels, [0] * len(labels))
        status = [USABLE] * len(labels)
        names = lambda d: [t["task"] for t in d["cases"]["tasks"]]
        seen = set()
        for seed in range(6):
            d = describe_panel(slots, Y=Y, ids=ids, case_seed=seed)
            want = [ids[c["index"]] for c in sap.select_cases(slots, [0] * len(labels), status, ids=ids, seed=seed)]
            self.assertEqual(names(d), want, seed)
            seen.add(tuple(want))
        self.assertGreater(len(seen), 1)

    def test_the_case_seed_is_required_and_must_be_an_integer(self):
        slots, Y, _ = mixed_panel()
        with self.assertRaises(TypeError):
            sap.describe_panel(slots, Y=Y, ids=self.IDS)
        with self.assertRaises(TypeError):
            sap.analyze_panel(slots, Y=Y, ids=self.IDS)
        for bad in (None, "9", 1.5, True):
            with self.assertRaises(ValueError, msg=repr(bad)):
                sap.describe_panel(slots, Y=Y, ids=self.IDS, case_seed=bad)
        self.assertEqual(sap.describe_panel(slots, Y=Y, ids=self.IDS, case_seed=np.int64(4))["cases"]["seed"], 4)

    def test_the_benchmarks_reach_the_choice(self):
        rows = [("S", 1, USABLE, "wa", "success")] + [("S", 0, USABLE, "work", "success")] * 6 + [("S", 0, USABLE, "wa", "success")]
        slots, _, _, bench, ids = case_panel(rows)
        Y = ["Unsuccessful"] + ["Successful"] * 7
        for seed in range(8):
            d = describe_panel(slots, Y=Y, ids=ids, benchmarks=bench, case_seed=seed)
            self.assertEqual({t["cell"]: t["task"] for t in d["cases"]["tasks"]}["E"], "t007", seed)

    def test_the_cases_depend_on_E_and_the_references(self):
        labels = labels_for(0, 2, 3, 3, 8)
        slots, Y = build(labels, [0] * len(labels))
        none = describe_panel(slots, Y=Y, ids=[f"t{i:02d}" for i in range(16)])
        self.assertEqual({t["cell"] for t in none["cases"]["tasks"]}, {"B", "E"})
        E = [1] * 16
        slots, Y = build(labels, E)
        d = describe_panel(slots, Y=Y, ids=[f"t{i:02d}" for i in range(16)])
        self.assertEqual([t["cell"] for t in d["cases"]["tasks"]], ["A", "A", "C", "D", "D"])
        self.assertEqual({t["E"] for t in d["cases"]["tasks"]}, {1})
        unknown = describe_panel(slots, Y=["Unsure"] * 16, ids=[f"t{i:02d}" for i in range(16)])
        self.assertEqual([t["cell"] for t in unknown["cases"]["tasks"]], ["F"])

    def test_review_probabilities_match_every_draw(self):
        for sizes in ((1, 2, 0, 1, 5), (0, 2, 3, 1, 3), (0, 1, 1, 3, 5), (0, 7, 0, 1, 0), (0, 0, 1, 1, 1), (1, 2, 0, 0, 5)):
            labels = labels_for(*sizes)
            d = describe(labels, [i % 2 for i in range(len(labels))])
            self.assertTrue(d["cases"]["tasks"], sizes)
            for pol in POLICIES:
                sets = list(draws(labels, pol, d["K"]))
                want = [Fraction(sum(i in s for s in sets), len(sets)) for i in range(len(labels))]
                self.assertEqual(d["review_probability"][pol], want, (sizes, pol))
                for t in d["cases"]["tasks"]:
                    self.assertEqual(t["review_probability"][pol], want[t["task"]], (sizes, pol))

    def test_the_slots_print_in_canonical_spelling(self):
        slots = [{"A1": "SUCCESS", "A2": "Failure", "A3": "success", "B1": "sUcCeSs"}]
        self.assertIn("  B | 0 | U1 | success, failure, success, success | usable | Successful | not supplied | 0 ",
                      sap.describe_report(describe_panel(slots, Y=["Successful"]), EXPOSURE))

    def test_without_ids_the_tasks_are_named_by_position(self):
        d = describe_panel([slot("S")] * 4, [0, 1, 0, 0])
        self.assertEqual(d["cases"]["sort_key"], "stored order")
        lines = sap.describe_report(d, EXPOSURE).split("\n")
        self.assertIn("  cell | position from 0 | group | A1, A2, A3, B1 | reference | released label | second annotator | E | "
                      "Instability-first | Disagreement-first | Random | what the cell shows", lines)
        self.assertIn("  Task IDs were not supplied, so the tasks are sorted by their stored position here. "
                      "The real run must supply them.", lines)
        rows = [x for x in lines if " | S | " in x]
        self.assertEqual(len(rows), 2)
        for row in rows:
            position = int(row.split(" | ")[1])
            self.assertIn(" | not assessed | ", row)                # no statuses were supplied
            self.assertIn(f" | {[0, 1, 0, 0][position]} | ", row)
        d = self.mixed()
        self.assertNotIn("Task IDs were not supplied", sap.describe_report(d, EXPOSURE))

    def test_a_panel_with_nothing_in_any_cell_has_no_case(self):
        G1_E0 = [slot("G1")] * 2
        for d in (describe_panel([slot("I")] * 3, Y=["Successful"] * 3), describe_panel([], Y=[]),
                  describe_panel(G1_E0, Y=["Successful"] * 2)):
            self.assertEqual(d["cases"]["tasks"], [])
            self.assertIn("  None: no task falls in a cell.", sap.describe_report(d, EXPOSURE).split("\n"))

    def test_bad_ids_are_refused(self):
        ok = slot("S")
        for bad in (["a"], ["a", "a"], ["a", "b", "c"]):
            with self.assertRaises(ValueError, msg=repr(bad)):
                describe_panel([ok, ok], Y=["Successful"] * 2, ids=bad)

    def test_the_keywords_reach_the_path_that_takes_them(self):
        slots, Y, _ = mixed_panel()
        out = sap.analyze_panel(slots, Y=Y, ids=self.IDS, case_seed=2)
        self.assertEqual(out["cases"]["seed"], 2)
        self.assertIn(out["cases"]["tasks"][0]["task"], self.IDS)
        with self.assertRaises(TypeError):
            sap.analyze_panel(slots, [0] * 10, case_seed=2, path=sap.PATH_POPULATION)


class TestInvalidAndUnsure(unittest.TestCase):
    def test_invalid_outputs_by_slot_and_category(self):
        slots, Y, bench = mixed_panel()
        d = describe_panel(slots, Y=Y, benchmarks=bench)
        self.assertEqual(d["invalid_by_slot"], {"A1": 2, "A2": 3, "A3": 1, "B1": 1})
        self.assertEqual(d["invalid_by_slot_category"], {
            ("A1", "REFUSAL"): 1, ("A1", "TRANSPORT_EXHAUSTED"): 1, ("A2", "TRUNCATED"): 1, ("A2", "ABSTAIN"): 2,
            ("A3", "FILTERED"): 1, ("B1", "MALFORMED"): 1})
        self.assertEqual(d["invalid_counts"], sap.invalid_counts(slots, bench))
        txt = sap.describe_report(d, EXPOSURE)
        for needle in ("A1 | 2 | REFUSAL 1, TRANSPORT_EXHAUSTED 1", "A2 | 3 | ABSTAIN 2, TRUNCATED 1",
                       "A3 | 1 | FILTERED 1", "B1 | 1 | MALFORMED 1", "A2 | ABSTAIN | work | 2"):
            self.assertIn(needle, txt, needle)
        # No benchmarks: the slot and category table stands alone.
        d = describe_panel(slots, Y=Y)
        self.assertIsNone(d["invalid_counts"])
        self.assertEqual(d["invalid_by_slot"]["A2"], 3)
        txt = sap.describe_report(d, EXPOSURE)
        self.assertNotIn("Invalid counts by judge slot, category and benchmark", txt)
        clean = sap.describe_report(describe_panel([slot("S")], Y=["Successful"]), EXPOSURE)
        self.assertIn("A1 | 0 | none", clean)

    def test_reference_status_counts_overall_by_group_and_benchmark(self):
        slots, Y, bench = mixed_panel()
        d = describe_panel(slots, Y=Y, benchmarks=bench)
        g = d["coverage"]["groups"]
        self.assertEqual(g["all"], {"tasks": 10, "usable": 7, "unsure": 2, "missing": 1, "contradicted": 3})
        self.assertEqual(g["invalid first verdict"], {"tasks": 2, "usable": 1, "unsure": 1, "missing": 0, "contradicted": 0})
        self.assertEqual(g["disagreement"], {"tasks": 2, "usable": 1, "unsure": 1, "missing": 0, "contradicted": 1})
        self.assertEqual(g["U2"], {"tasks": 1, "usable": 1, "unsure": 0, "missing": 0, "contradicted": 1})
        self.assertEqual(g["U1"], {"tasks": 2, "usable": 1, "unsure": 0, "missing": 1, "contradicted": 0})
        self.assertEqual(g["stable agreeing"], {"tasks": 3, "usable": 3, "unsure": 0, "missing": 0, "contradicted": 1})
        self.assertEqual(d["coverage"]["by_benchmark"], {
            "vwa": {"tasks": 2, "usable": 0, "unsure": 1, "missing": 1},
            "wa": {"tasks": 4, "usable": 3, "unsure": 1, "missing": 0},
            "work": {"tasks": 4, "usable": 4, "unsure": 0, "missing": 0}})
        txt = sap.describe_report(d, EXPOSURE)
        for needle in ("unknown, not correct verdicts", "U1 | 2 | 1 | 0 | 1 | 0", "wa | 4 | 3 | 1 | 0"):
            self.assertIn(needle, txt, needle)
        # Unsure then missing, in the groups all, invalid, disagreement, U2, U1, stable. K = 6, s = 2.
        lines = txt.split("\n")
        for row in ("  Random: 1.50 / 0.50; 1.00 / 0.00; 0.50 / 0.00; 0.00 / 0.00; 0.00 / 0.50; 0.00 / 0.00",
                    "  Disagreement-first: 2.00 / 0.33; 1.00 / 0.00; 1.00 / 0.00; 0.00 / 0.00; 0.00 / 0.33; 0.00 / 0.00",
                    "  Instability-first: 2.00 / 0.50; 1.00 / 0.00; 1.00 / 0.00; 0.00 / 0.00; 0.00 / 0.50; 0.00 / 0.00"):
            self.assertIn(row, lines)

    def test_each_review_group_has_its_own_row_with_its_contradicted_verdicts(self):
        slots, Y, bench = mixed_panel()
        lines = sap.describe_report(describe_panel(slots, Y=Y, benchmarks=bench), EXPOSURE).split("\n")
        start = lines.index("  group | tasks | usable | Unsure | missing | reference-contradicted")
        self.assertEqual(lines[start + 1:start + 7], ["  all | 10 | 7 | 2 | 1 | 3",
                                                      "  invalid first verdict | 2 | 1 | 1 | 0 | 0",
                                                      "  disagreement | 2 | 1 | 1 | 0 | 1",
                                                      "  U2 | 1 | 1 | 0 | 0 | 1",
                                                      "  U1 | 2 | 1 | 0 | 1 | 0",
                                                      "  stable agreeing | 3 | 3 | 0 | 0 | 1"])

    def test_the_review_group_rows_print_in_every_regime(self):
        seen = set()
        for d in report_panels():
            seen.add(d["regime"])
            if not d["reference_supplied"]:
                continue
            lines = sap.describe_report(d, EXPOSURE).split("\n")
            start = lines.index("  group | tasks | usable | Unsure | missing | reference-contradicted")
            self.assertEqual([x.split(" | ")[0].strip() for x in lines[start + 1:start + 7]],
                             ["all", "invalid first verdict", "disagreement", "U2", "U1", "stable agreeing"])
            self.assertEqual(int(lines[start + 1].split(" | ")[5]), d["M"])
        self.assertEqual(seen, REGIMES)

    def test_an_invalid_first_verdict_is_never_counted_as_contradicted(self):
        d = describe_panel([slot("I"), slot("S"), slot("S")], [1, 1, 0], reference=[sap.USABLE] * 3)
        self.assertEqual(d["M"], 1)
        self.assertEqual(d["coverage"]["groups"]["invalid first verdict"]["contradicted"], 0)
        self.assertEqual(d["coverage"]["groups"]["all"]["contradicted"], 1)

    def test_validity_ignores_unknown_references_and_invalid_slots(self):
        slots, Y, _ = mixed_panel()
        v = describe_panel(slots, Y=Y)["validity"]
        self.assertEqual((v["n_usable"], v["n_shared"]), (7, 5))
        self.assertEqual(v["n_excluded_from_shared"], 2)
        F = Fraction
        A1 = v["all_usable"]["A1"]
        self.assertEqual(A1["confusion"], {"TP": 0, "FP": 0, "FN": 3, "TN": 3})
        self.assertEqual(A1["invalid"], {"failure_references": 0, "success_references": 1})
        self.assertEqual(A1["balanced_accuracy"]["value"], F(3, 8))           # 0 of 3 and 3 of 4
        self.assertEqual(A1["valid_coverage"]["value"], F(6, 7))
        self.assertIsNone(A1["failure_precision"]["value"])
        self.assertEqual(A1["failure_recall"]["value"], 0)
        self.assertEqual(A1["success_recall"]["value"], F(3, 4))
        B1 = v["all_usable"]["B1"]
        self.assertEqual(B1["confusion"], {"TP": 0, "FP": 0, "FN": 2, "TN": 4})
        self.assertEqual(B1["invalid"], {"failure_references": 1, "success_references": 0})
        self.assertEqual(B1["prevalence"], {"value": F(3, 7), "numerator": 3, "denominator": 7})
        self.assertEqual(v["shared"]["A1"]["confusion"], {"TP": 0, "FP": 0, "FN": 2, "TN": 3})
        self.assertEqual(v["shared"]["B1"]["valid_coverage"]["value"], F(5, 7))
        lines = sap.describe_report(describe_panel(slots, Y=Y), EXPOSURE).split("\n")
        for judge, printed in (("A1", "0 on failure references, 1 on success references"),
                               ("B1", "1 on failure references, 0 on success references")):
            start = lines.index(f"  {judge} on all tasks with a usable reference (n = 7)")
            self.assertEqual(lines[start + 2], "    invalid outputs: " + printed)

    def test_E_with_statuses_matches_the_released_labels(self):
        slots, Y, bench = mixed_panel()
        j = sap.join_reference(slots, Y)
        a = describe_panel(slots, Y=Y, benchmarks=bench)
        b = describe_panel(slots, j["E"], reference=j["reference"], benchmarks=bench)
        for key in ("captures", "gain", "secondary", "coverage", "N_Y", "M"):
            self.assertEqual(a[key], b[key], key)
        self.assertIsNone(b["validity"])
        self.assertIn("Not assessed: the released labels were not supplied.", sap.describe_report(b, EXPOSURE))

    def test_E_without_statuses_assesses_no_label_coverage(self):
        d = describe_panel([slot("S")] * 4, [0, 1, 0, 0])
        self.assertFalse(d["reference_supplied"])
        self.assertIsNone(d["N_Y"])
        self.assertIsNone(d["coverage"])
        self.assertIsNone(d["secondary"])
        txt = sap.describe_report(d, EXPOSURE)
        self.assertEqual(txt.count(sap.REFERENCE_NOT_SUPPLIED_TEXT), 1)
        self.assertIn("Not assessed: the reference statuses were not supplied.", txt)
        self.assertNotIn("group | tasks", txt)

    def test_input_is_checked(self):
        ok = slot("S")
        with self.assertRaises(ValueError):
            describe_panel([ok], [0], Y=["Successful"])         # both
        with self.assertRaises(ValueError):
            describe_panel([ok])                                # neither
        with self.assertRaises(ValueError):
            describe_panel([ok], Y=["Successful"], reference=["usable"])
        with self.assertRaises(ValueError):
            describe_panel([ok], [0, 1])
        with self.assertRaises(ValueError):
            describe_panel([ok, ok, ok], [2, 0, 0])
        with self.assertRaises(ValueError):
            describe_panel([ok], Y=["Successful"], benchmarks=["a", "b"])
        with self.assertRaises(ValueError):                         # a task without a reference is never contradicted
            describe_panel([ok], [1], reference=[sap.UNSURE])
        with self.assertRaises(sap.UnfinishedSlot):
            describe_panel([dict(ok, A2=sap.HALT)], Y=["Successful"])


class TestCosts(unittest.TestCase):
    COSTS = {"total": "10.50", "by_slot": {"A1": "3.00", "A2": "2.50", "A3": "2.50", "B1": "2.00"}}

    def panel(self, **kw):
        return describe(labels_for(0, 30, 3, 7, 160), E_for(G1=(30, 11), U2=(3, 1), U1=(7, 3), S=(160, 10)), **kw)

    def test_totals_and_extra_cost(self):
        d = self.panel(costs=self.COSTS)
        c = d["costs"]
        self.assertEqual(c["total"], Fraction(21, 2))
        self.assertEqual(c["by_slot"], {"A1": 3, "A2": Fraction(5, 2), "A3": Fraction(5, 2), "B1": 2})
        self.assertEqual(c["not_attributed"], Fraction(1, 2))
        self.assertEqual(c["extra_repeats"], 5)
        self.assertEqual(c["extra_per_extra_verdict"], 5 / Fraction(189, 68))
        txt = sap.describe_report(d, EXPOSURE)
        for needle in ("total: 10.5000", "by slot: A1 3.0000, A2 2.5000, A3 2.5000, B1 2.0000",
                       "not attributed to a slot: 0.5000", "extra cost of A2 and A3: 5.0000",
                       "extra cost per extra expected verdict captured: 1.7989, "
                       "the extra cost over the gain of 189/68 (2.7794) verdicts"):
            self.assertIn(needle, txt, needle)

    def test_not_defined_unless_the_gain_is_positive(self):
        # N = 20, s = 3: instability-first spends the slice on two unstable tasks with no discordance.
        labels = labels_for(0, 2, 0, 2, 16)
        d = describe(labels, E_for(G1=(2, 0), U1=(2, 0), S=(16, 6)), costs=self.COSTS)
        self.assertEqual(d["gain"]["over_disagreement"]["verdicts"], -Fraction(5, 8))
        self.assertIsNone(d["costs"]["extra_per_extra_verdict"])
        self.assertIn("extra cost per extra expected verdict captured: not defined, since the gain over "
                      "disagreement-first review is not positive (-5/8 (-0.6250) verdicts)",
                      sap.describe_report(d, EXPOSURE))

    def test_costs_not_supplied_are_stated_never_zero(self):
        txt = sap.describe_report(self.panel(), EXPOSURE)
        self.assertIn("ACTUAL INFERENCE COSTS in USD, from the spending ledger\n  Not supplied.", txt)
        self.assertNotIn("total: 0", txt)

    def test_amount_forms(self):
        c = sap.inference_costs({"total": 9, "by_slot": {"A1": 3, "A2": 2.5, "A3": Fraction(5, 2), "B1": "1"}}, Fraction(1))
        self.assertEqual((c["total"], c["extra_repeats"]), (9, 5))

    def test_a_bad_ledger_dict_is_refused(self):
        slots = {"A1": "1", "A2": "1", "A3": "1", "B1": "1"}
        bad = [{"by_slot": slots},
               {"total": "4"},
               {"total": "4", "by_slot": {"A1": "1", "A2": "1", "A3": "1"}},
               {"total": "4", "by_slot": dict(slots, A4="0")},
               {"total": "3", "by_slot": slots},
               {"total": "-1", "by_slot": slots},
               {"total": "4", "by_slot": dict(slots, A1="-0.5")},
               {"total": "a lot", "by_slot": slots},
               {"total": None, "by_slot": slots},
               {"total": float("nan"), "by_slot": slots}]
        for costs in bad:
            with self.assertRaises(ValueError, msg=repr(costs)):
                sap.inference_costs(costs, Fraction(1))


# Forbidden in the descriptive report, once the fixed scope sentence and the one allowed
# way of stating the practical reference are set aside.
FORBIDDEN = (r"p[\s-]?values?\b", r"\bp\s*[=<>≤≥]", r"confidence", r"interval", r"signific", r"\bexceeds?\b",
             r"\bbelow\b", r"unresolved", r"fragil", r"bootstrap", r"fisher", r"permutation", r"\breject",
             r"\bH0\b", r"hypothes", r"\balpha\b", r"\bscore\b", r"\bnull\b", r"\bexistence\b", r"\bevidence\b",
             r"unreachable", r"delta_m", r"δ")


# Plan section 8, the words the report does not use for this result, in any letter case.
PLAN_UNUSED_WORDS = (r"signific", r"evidence that", r"generaliz", r"\bcorrected\b", r"blind spot",
                     r"untouched holdout", r"independent validation", r"\berrors?\b")


def plan_unused_words_in(txt):
    """Which of the words the plan does not use are in the text, set aside its two fixed sentences."""
    body = txt.replace(sap.SCOPE_TEXT, "").replace(sap.RESULT_CLOSING_TEXT, "")
    return [p for p in PLAN_UNUSED_WORDS if re.search(p, body, re.IGNORECASE)]


def report_panels():
    """A report-ready panel in every regime, with and without Unsure and invalid outcomes."""
    costs = {"total": "12", "by_slot": {"A1": "3", "A2": "3", "A3": "3", "B1": "3"}}
    for sizes, E in (((1, 3, 0, 0, 16), [1, 0] * 10), ((0, 7, 0, 1, 0), [0, 1] * 4), ((0, 0, 1, 1, 1), [1, 0, 1]),
                     ((0, 30, 3, 7, 160), E_for(G1=(30, 11), U2=(3, 1), U1=(7, 3), S=(160, 10))),
                     ((0, 10, 4, 10, 36), E_for(G1=(10, 3), U2=(4, 2), U1=(10, 3), S=(36, 3)))):
        labels = labels_for(*sizes)
        slots, Y = build(labels, E)
        bench = ["wa" if i % 2 else "work" for i in range(len(labels))]
        yield describe_panel(slots, Y=Y, benchmarks=bench, costs=costs)
        Y2 = ["Unsure" if i % 4 == 0 else (None if i % 7 == 0 else y) for i, y in enumerate(Y)]
        yield describe_panel(slots, Y=Y2)
    slots, Y, bench = mixed_panel()
    yield describe_panel(slots, Y=Y, benchmarks=bench, costs=costs, provenance=provenance_of(slots))
    yield describe_panel(slots, [0, 0, 1, 0, 1, 0, 0, 1, 0, 0])


class TestDescriptiveReport(unittest.TestCase):
    def test_no_inferential_wording_in_any_regime(self):
        seen = set()
        for d in report_panels():
            seen.add(d["regime"])
            txt = sap.describe_report(d, EXPOSURE)
            self.assertEqual(txt.count(sap.SCOPE_TEXT), 1)
            body = txt.replace(sap.SCOPE_TEXT, "").replace("observed below the reference", "")
            for sentence in d["result_sentences"]:          # the plan's fixed sentences
                self.assertEqual(body.count(sentence), 1)
                body = body.replace(sentence, "")
            for pattern in FORBIDDEN:
                self.assertIsNone(re.search(pattern, body, re.IGNORECASE), (pattern, d["regime"]))
        self.assertEqual(seen, REGIMES)

    def test_the_report_does_not_use_the_words_the_plan_leaves_out(self):
        seen = set()
        for d in report_panels():
            seen.add(d["regime"])
            txt = sap.describe_report(d, EXPOSURE)
            self.assertEqual(plan_unused_words_in(txt), [], d["regime"])
            self.assertEqual(txt.count(sap.SCOPE_TEXT), 1)
            self.assertEqual(txt.count(sap.RESULT_CLOSING_TEXT), 1)
        self.assertEqual(seen, REGIMES)

    def test_the_scan_of_the_plans_words_is_live(self):
        for phrase in ("a Significant gain", "the significance of it", "evidence that it works", "this generalizes",
                       "the corrected verdicts", "a blind spot", "an untouched holdout", "independent validation",
                       "a judge error", "two errors"):
            self.assertTrue(plan_unused_words_in(phrase), phrase)
        self.assertEqual(plan_unused_words_in(sap.SCOPE_TEXT + " " + sap.RESULT_CLOSING_TEXT), [])
        self.assertEqual(plan_unused_words_in("a reference-contradicted first verdict; not correct"), [])
        self.assertEqual(sap.RESULT_CLOSING_TEXT, CLOSING)

    def test_the_scan_catches_inferential_wording(self):
        # The patterns are live: the population-inference report trips them.
        labels = labels_for(0, 30, 3, 7, 160)
        E = E_for(G1=(30, 11), U2=(3, 1), U1=(7, 3), S=(160, 10))
        txt = sap.report(sap.analyze(labels, E, path=sap.PATH_POPULATION))
        for pattern in (r"\bp\s*[=<>≤≥]", r"interval", r"fisher", r"fragil", r"\breject"):
            self.assertIsNotNone(re.search(pattern, txt, re.IGNORECASE), pattern)

    def test_result_dict_holds_no_inferential_quantity(self):
        for d in report_panels():
            def keys(x):
                if isinstance(x, dict):
                    for k, v in x.items():
                        yield k
                        yield from keys(v)
            names = {str(k).lower() for k in keys(d)}
            for gone in ("p", "existence", "size", "fragility", "delta_l", "delta_u", "g_l", "g_u", "reject"):
                self.assertNotIn(gone, names)

    def test_fixed_statements(self):
        self.assertEqual(sap.SCOPE_TEXT, "Scope: these results apply to these tasks and these realized judge "
                                         "calls, with no population confidence or significance claim.")
        self.assertEqual(sap.PANEL_WORDING, "a retrospective evaluation panel with documented prior exposure")
        d = describe(labels_for(0, 2, 0, 2, 16), [0] * 20)
        txt = sap.describe_report(d, EXPOSURE)
        lines = txt.split("\n")
        self.assertEqual(lines[0], "Primary result: descriptive audit of the frozen panel, "
                                   "analysis plan version 4.0, Nov 8, 2026")
        self.assertEqual(lines[1], sap.SCOPE_TEXT)
        self.assertEqual(lines[2], "Panel: a retrospective evaluation panel with documented prior exposure.")
        self.assertEqual(lines[3], "Prior exposure statement: " + EXPOSURE)

    def test_only_the_population_path_refers_to_the_old_plan_version(self):
        self.assertEqual(sap.PLAN_VERSION, "4.0, Nov 8, 2026")
        self.assertEqual(sap.POPULATION_PLAN_VERSION, "v3.2")
        for d in report_panels():
            self.assertNotIn("3.2", sap.describe_report(d, EXPOSURE))

    def test_refuses_to_run_without_a_prior_exposure_statement(self):
        d = describe(labels_for(0, 2, 0, 2, 16), [0] * 20)
        for bad in (None, "", "   \n", 5):
            with self.assertRaises(ValueError, msg=repr(bad)):
                sap.describe_report(d, bad)
            with self.assertRaises(ValueError, msg=repr(bad)):
                sap.panel_report(d, bad)
        with self.assertRaises(ValueError):
            sap.describe_report(d)
        with self.assertRaises(ValueError):
            sap.panel_report(d)

    def test_provenance_is_visible(self):
        slots, Y, bench = mixed_panel()
        txt = sap.describe_report(describe_panel(slots, Y=Y, provenance=provenance_of(slots)), EXPOSURE)
        self.assertIn("under config hash cfg-1", txt)
        self.assertIn("A1 INVALID: 2 (REFUSAL 1, TRANSPORT_EXHAUSTED 1)", txt)
        self.assertIn("Not supplied: resolved slots", sap.describe_report(describe_panel(slots, Y=Y), EXPOSURE))

    def test_each_regime_is_named(self):
        for d in report_panels():
            txt = sap.describe_report(d, EXPOSURE)
            self.assertIn(f"regime: {d['regime']}; {sap.REGIME_NOTES[d['regime']]}", txt)


class TestPaths(unittest.TestCase):
    def data(self):
        labels = labels_for(0, 30, 3, 7, 160)
        E = E_for(G1=(30, 11), U2=(3, 1), U1=(7, 3), S=(160, 10))
        slots, Y = build(labels, E)
        return labels, E, slots, Y

    def test_descriptive_is_the_default(self):
        _, E, slots, Y = self.data()
        out = sap.analyze_panel(slots, Y=Y, case_seed=0)
        self.assertEqual(out["path"], sap.PATH_DESCRIPTIVE)
        self.assertEqual(sap.PATH_DESCRIPTIVE, "descriptive")
        self.assertNotIn("primary", out)
        explicit = sap.analyze_panel(slots, Y=Y, case_seed=0, path="descriptive")
        self.assertEqual(out, explicit)
        positional = sap.analyze_panel(slots, E, case_seed=0)       # E alone takes the same path
        self.assertEqual(positional["path"], sap.PATH_DESCRIPTIVE)
        self.assertEqual(positional["captures"], out["captures"])
        self.assertTrue(sap.panel_report(out, EXPOSURE).startswith("Primary result: descriptive audit"))

    def test_population_inference_needs_the_explicit_argument(self):
        labels, E, slots, Y = self.data()
        with self.assertRaises(ValueError):
            sap.analyze(labels, E)
        with self.assertRaises(ValueError):
            sap.analyze(labels, E, path="descriptive")
        with self.assertRaises(ValueError):
            sap.analyze_with_sensitivities(slots, E)
        with self.assertRaises(ValueError):
            sap.analyze_panel(slots, E, path="inference")
        with self.assertRaises(ValueError):
            sap.analyze_panel(slots, path=sap.PATH_POPULATION)      # it needs E
        with self.assertRaises(TypeError):
            sap.analyze_panel(slots, E, case_seed=0, boot_B=100)    # an option of the other path
        with self.assertRaises(TypeError):
            sap.analyze_panel(slots, E, costs={}, path=sap.PATH_POPULATION)

    def test_population_report_opens_with_the_scope_line(self):
        labels, E, slots, _ = self.data()
        line = sap.POPULATION_PATH_TEXT
        self.assertIn("outside the primary result", line)
        self.assertIn("separately scoped", line)
        res = sap.analyze(labels, E, path=sap.PATH_POPULATION)
        self.assertEqual(sap.report(res).split("\n")[0], line)
        self.assertEqual(sap.report(res).count(line), 1)
        panel = sap.analyze_panel(slots, E, path=sap.PATH_POPULATION, boot_B=200)
        self.assertEqual(panel["path"], sap.PATH_POPULATION)
        txt = sap.panel_report(panel)
        self.assertEqual(txt.split("\n")[0], line)
        self.assertEqual(txt.count(line), 1)
        self.assertNotIn("Primary analysis", txt)
        self.assertIn("Population inference on the panel, statistical analysis plan v3.2", txt)
        self.assertNotIn("Prior exposure statement", txt)
        self.assertIn("Prior exposure statement: " + EXPOSURE, sap.panel_report(panel, EXPOSURE))
        lines = txt.split("\n")
        self.assertIn("  agreeing items with both repeats INVALID (R set to 0): 0", lines)

    def test_population_report_says_primary_only_of_the_descriptive_audit(self):
        def check(txt):
            self.assertEqual(txt.count(sap.POPULATION_PATH_TEXT), 1)
            self.assertNotIn("primary", txt.replace(sap.POPULATION_PATH_TEXT, "").lower())
        seen = set()
        for sizes in ((0, 3, 0, 0, 16), (0, 7, 0, 1, 0), (0, 0, 1, 1, 1), (0, 3, 1, 1, 15), (0, 4, 3, 4, 9)):
            labels = labels_for(*sizes)
            n = len(labels)
            ref = [sap.UNSURE if i == n - 1 else sap.USABLE for i in range(n)]
            E = [int(i % 2 == 0 and i != n - 1) for i in range(n)]
            res = sap.analyze(labels, E, reference=ref, path=sap.PATH_POPULATION, boot_B=50)
            seen.add(res["label_free"]["regime"])
            check(sap.report(res))
            # Held values are printed only on a sensitivity analysis, the wording with them.
            check(sap.report(sap.analyze(labels, E, reference=ref, s_override=res["label_free"]["s"], N_ref=n + 5,
                                         N_Y_ref=n + 4, path=sap.PATH_POPULATION, boot_B=50)))
            slots, _ = build(labels, E)
            ids = [f"t{i:02d}" for i in range(n)]
            panel = sap.analyze_panel(slots, E, ids=ids, templates=[f"tpl{i // 2}" for i in range(n)],
                                      template_panel=ids[::2], reference=ref, path=sap.PATH_POPULATION, boot_B=50)
            check(sap.panel_report(panel, EXPOSURE))
        self.assertEqual(seen, REGIMES)

    def test_held_values_are_named_in_the_population_report(self):
        labels = labels_for(0, 3, 1, 1, 15)
        ref = [sap.UNSURE] + [sap.USABLE] * 19
        txt = sap.report(sap.analyze(labels, [0] * 20, reference=ref, s_override=3, N_ref=25, N_Y_ref=24,
                                     path=sap.PATH_POPULATION))
        for needle in ("held at its value in the per-100-panel-task result",
                       "per-100 denominator held at the N of the per-100-panel-task result, 25",
                       "per-100 denominator held at its value on the full panel, N_Y = 24"):
            self.assertIn(needle, txt, needle)

    def test_template_sensitivity_needs_the_explicit_argument(self):
        labels, E, tmpl = ["U1"] * 6 + ["S"] * 12, [1] * 5 + [0] * 13, [f"t{i}" for i in range(18)]
        with self.assertRaises(ValueError):
            sap.template_sensitivity(labels, E, tmpl)
        with self.assertRaises(ValueError):
            sap.template_sensitivity(labels, E, tmpl, path="descriptive")
        self.assertIn("p", sap.template_sensitivity(labels, E, tmpl, path=sap.PATH_POPULATION))

    def test_inference_code_is_still_there(self):
        labels, E, slots, _ = self.data()
        panel = sap.analyze_panel(slots, E, path=sap.PATH_POPULATION, boot_B=200)
        ex = panel["primary"]["existence"]
        self.assertEqual(ex["p"], sap.fisher_one_sided(4, 10, 10, 160))
        self.assertIn("delta_L", panel["primary"]["size"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
