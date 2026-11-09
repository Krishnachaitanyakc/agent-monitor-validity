"""Tests of the descriptive path in sap.py, the primary result. Synthetic data only.

Covers the exact expected captures against an enumeration of every draw, the equal
review count, the A1 and B1 validity metrics, invalid and Unsure handling, costs, the
report's fixed statements and its freedom from inferential wording, and the path
argument that keeps population inference out of the primary result.

Run from the repository root: python -m unittest discover -s analysis -v
"""
import itertools
import math
import os
import re
import sys
import unittest
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


def describe(labels, E, **kw):
    slots, Y = build(labels, E)
    return sap.describe_panel(slots, Y=Y, **kw)


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
        self.assertIn("review count K = 5, equal across the three policies "
                      "(Instability-first 5, Disagreement-first 5, Random 5)", txt)
        self.assertIn("NOT equal across the three policies", sap.describe_report(dict(d, K_equal=False), EXPOSURE))

    def test_counts_on_a_panel_with_every_invalid_kind(self):
        slots, Y, bench = mixed_panel()
        d = sap.describe_panel(slots, Y=Y, benchmarks=bench)
        self.assertEqual((d["N"], d["n_inv"], d["N_V"], d["n_D1"], d["n_D0"]), (10, 2, 8, 2, 6))
        self.assertEqual((d["n_U2"], d["n_U1"], d["n_S"], d["s"], d["K"], d["regime"]), (1, 2, 3, 2, 6, "PARTIAL"))
        self.assertEqual(d["R_unobserved"], 1)
        self.assertEqual(d["repeat_flags"], {"both_repeats_invalid": 1, "one_repeat_invalid_no_flip": 1,
                                             "one_repeat_invalid_flip": 1})
        txt = sap.describe_report(d, EXPOSURE)
        self.assertIn("agreeing groups: U2 = 1, U1 = 2, S = 3; R_unobserved = 1", txt)

    def test_R_unobserved_counts_every_valid_first_verdict(self):
        both = {"A2": "REFUSAL", "A3": "MALFORMED"}
        slots = [dict(slot("S"), **both), dict(slot("G1"), **both), dict(slot("S"))]
        d = sap.describe_panel(slots, Y=["Successful"] * 3)
        self.assertEqual(d["R_unobserved"], 2)                      # one agreeing, one disagreeing
        self.assertEqual(d["repeat_flags"]["both_repeats_invalid"], 1)


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
        self.assertEqual(d["N_Y"], 200)
        self.assertEqual(d["secondary"], {"over_disagreement": Fraction(189, 136), "over_random": Fraction(35, 8)})
        txt = sap.describe_report(d, EXPOSURE)
        for needle in ("M = 25: disagreement m_G1 = 11; unstable agreeing m1 = 4 (m_U2 = 1, m_U1 = 3); stable agreeing m0 = 10",
                       "Instability-first | 65/4 (16.2500) | 13/20 (0.6500)",
                       "Disagreement-first | 229/17 (13.4706) | 229/425 (0.5388)",
                       "Random | 15/2 (7.5000) | 3/10 (0.3000)",
                       "over disagreement-first: 189/68 (2.7794) verdicts; 189/136 (1.3897) per 100 panel tasks (N = 200)",
                       "over random: 35/4 (8.7500) verdicts; 35/8 (4.3750) per 100 panel tasks (N = 200)",
                       "(N_Y = 200 of N = 200)",
                       "gain over disagreement-first per 100 panel tasks: 189/136 (1.3897); "
                       "observed below the reference."):
            self.assertIn(needle, txt, needle)

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
        self.assertIn("Instability-first | 0 (0.0000) | not defined", sap.describe_report(d, EXPOSURE))

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
        d = sap.describe_panel(slots, Y=Y, benchmarks=bench)
        # Instability-first: 1 + 1 + 1/2 of 0; disagreement-first: 1 + 2/6 of 2; random: 4/8 of 3.
        self.assertEqual(d["captures"], {"AUG": 2, "DIS": Fraction(5, 3), "RAND": Fraction(3, 2)})
        self.assertEqual(d["N_Y"], 7)                               # the invalid A1 with a usable label counts
        self.assertEqual(d["gain"]["over_disagreement"]["per_100_panel"], Fraction(10, 3))
        self.assertEqual(d["secondary"], {"over_disagreement": Fraction(100, 21), "over_random": Fraction(50, 7)})
        self.assertIn("(N_Y = 7 of N = 10)", sap.describe_report(d, EXPOSURE))

    def test_no_usable_label_at_all(self):
        labels = labels_for(0, 2, 0, 2, 16)
        slots, _ = build(labels, [0] * 20)
        d = sap.describe_panel(slots, Y=["Unsure"] * 10 + [None] * 10)
        self.assertEqual(d["N_Y"], 0)
        self.assertEqual(d["secondary"], {"over_disagreement": None, "over_random": None})
        self.assertEqual(d["gain"]["over_disagreement"]["per_100_panel"], Fraction(0))
        txt = sap.describe_report(d, EXPOSURE)
        self.assertIn("over disagreement-first: not defined per 100", txt)

    def test_empty_panel_rates_are_not_defined(self):
        d = sap.describe_panel([], Y=[])
        self.assertIsNone(d["gain"]["over_disagreement"]["per_100_panel"])
        self.assertIsNone(d["practical_reference"]["status"])
        self.assertIn("no comparison is defined", sap.describe_report(d, EXPOSURE))


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

    def assert_metrics(self, m, confusion, balanced, precision, recall, coverage, prevalence):
        self.assertEqual(m["confusion"], dict(zip(("TP", "FP", "FN", "TN"), confusion)))
        tp, fp, fn, tn = confusion
        self.assertEqual(m["balanced_accuracy"], {"value": balanced, "failure_references": tp + fn,
                                                  "success_references": tn + fp})
        self.assert_rate(m["failure_precision"], precision, tp, tp + fp)
        self.assert_rate(m["failure_recall"], recall, tp, tp + fn)
        self.assert_rate(m["valid_prediction_coverage"], coverage[0], coverage[1], coverage[2])
        self.assert_rate(m["prevalence"], prevalence, tp + fn, tp + fp + fn + tn)

    def test_hand_computed_confusion_matrices(self):
        slots, Y = from_rows(ROWS)
        v = sap.describe_panel(slots, Y=Y)["validity"]
        self.assertEqual((v["n_usable"], v["n_shared"]), (14, 10))
        F = Fraction
        self.assert_metrics(v["own"]["A1"], (4, 1, 2, 5), F(3, 4), F(4, 5), F(2, 3), (F(6, 7), 12, 14), F(1, 2))
        self.assert_metrics(v["own"]["B1"], (4, 2, 2, 4), F(2, 3), F(2, 3), F(2, 3), (F(6, 7), 12, 14), F(1, 2))
        self.assert_metrics(v["shared"]["A1"], (3, 1, 2, 4), F(7, 10), F(3, 4), F(3, 5), (F(5, 7), 10, 14), F(1, 2))
        self.assert_metrics(v["shared"]["B1"], (3, 2, 2, 3), F(3, 5), F(3, 5), F(3, 5), (F(5, 7), 10, 14), F(1, 2))

    def test_report_gives_every_metric_with_its_denominator(self):
        slots, Y = from_rows(ROWS)
        txt = sap.describe_report(sap.describe_panel(slots, Y=Y), EXPOSURE)
        for needle in ("Tasks with a usable reference: 14; shared subset, where A1 and B1 are both valid: 10.",
                       "A1 on its own valid set (n = 12)",
                       "B1 on its own valid set (n = 12)",
                       "A1 on the shared subset (n = 10)",
                       "B1 on the shared subset (n = 10)",
                       "confusion matrix: TP = 4, FP = 1, FN = 2, TN = 5",
                       "balanced accuracy: 3/4 (0.7500), denominators 6 failure references and 6 success references",
                       "failure precision: 4/5 (0.8000), 4 of 5 failure predictions",
                       "failure recall: 2/3 (0.6667), 4 of 6 failure references",
                       "valid-prediction coverage: 6/7 (0.8571), 12 of 14 tasks with a usable reference",
                       "prevalence of failure references: 1/2 (0.5000), 6 of 12 tasks in this set"):
            self.assertIn(needle, txt, needle)

    def test_empty_denominators_are_not_defined_and_never_zero(self):
        # No failure reference and no failure prediction.
        slots, Y = from_rows([("success", "success", "Successful", 4)])
        d = sap.describe_panel(slots, Y=Y)
        a = d["validity"]["own"]["A1"]
        self.assert_metrics(a, (0, 0, 0, 4), None, None, None, (Fraction(1), 4, 4), Fraction(0))
        txt = sap.describe_report(d, EXPOSURE)
        self.assertIn("balanced accuracy: not defined, denominators 0 failure references and 4 success references", txt)
        self.assertIn("failure precision: not defined, 0 of 0 failure predictions", txt)
        self.assertIn("failure recall: not defined, 0 of 0 failure references", txt)
        self.assertIn("prevalence of failure references: 0 (0.0000), 0 of 4 tasks in this set", txt)  # a true zero
        self.assertNotIn("balanced accuracy: 0", txt)
        self.assertNotIn("failure precision: 0", txt)
        self.assertNotIn("failure recall: 0", txt)
        # Failure references only, none found: recall is a true zero, balanced accuracy is not defined.
        slots, Y = from_rows([("success", "success", "Unsuccessful", 3)])
        a = sap.describe_panel(slots, Y=Y)["validity"]["own"]["A1"]
        self.assert_metrics(a, (0, 0, 3, 0), None, None, Fraction(0), (Fraction(1), 3, 3), Fraction(1))
        # No valid first verdict.
        slots, Y = from_rows([("REFUSAL", "success", "Unsuccessful", 2), ("REFUSAL", "success", "Successful", 1)])
        d = sap.describe_panel(slots, Y=Y)
        self.assert_metrics(d["validity"]["own"]["A1"], (0, 0, 0, 0), None, None, None, (Fraction(0), 0, 3), None)
        self.assert_metrics(d["validity"]["shared"]["B1"], (0, 0, 0, 0), None, None, None, (Fraction(0), 0, 3), None)
        self.assertIn("prevalence of failure references: not defined, 0 of 0 tasks in this set",
                      sap.describe_report(d, EXPOSURE))
        # No usable reference: even the coverage has no denominator.
        slots, Y = from_rows([("success", "success", "Unsure", 2), ("failure", "failure", None, 1)])
        d = sap.describe_panel(slots, Y=Y)
        self.assert_metrics(d["validity"]["own"]["A1"], (0, 0, 0, 0), None, None, None, (None, 0, 0), None)
        self.assertIn("valid-prediction coverage: not defined, 0 of 0 tasks with a usable reference",
                      sap.describe_report(d, EXPOSURE))

    def test_balanced_accuracy_averages_the_two_class_recalls(self):
        slots, Y = from_rows([("failure", "failure", "Unsuccessful", 1), ("success", "success", "Unsuccessful", 3),
                              ("success", "success", "Successful", 2)])
        a = sap.describe_panel(slots, Y=Y)["validity"]["own"]["A1"]
        self.assertEqual(a["balanced_accuracy"]["value"], (Fraction(1, 4) + 1) / 2)

    def test_a_judge_word_as_a_reference_label_stops_the_analysis(self):
        slots, Y = from_rows([("success", "success", "Successful", 2)])
        with self.assertRaises(ValueError):
            sap.describe_panel(slots, Y=["success", "Successful"])


class TestInvalidAndUnsure(unittest.TestCase):
    def test_invalid_outputs_by_slot_and_category(self):
        slots, Y, bench = mixed_panel()
        d = sap.describe_panel(slots, Y=Y, benchmarks=bench)
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
        d = sap.describe_panel(slots, Y=Y)
        self.assertIsNone(d["invalid_counts"])
        self.assertEqual(d["invalid_by_slot"]["A2"], 3)
        txt = sap.describe_report(d, EXPOSURE)
        self.assertNotIn("Invalid counts by judge slot, category and benchmark", txt)
        clean = sap.describe_report(sap.describe_panel([slot("S")], Y=["Successful"]), EXPOSURE)
        self.assertIn("A1 | 0 | none", clean)

    def test_reference_status_counts_overall_by_group_and_benchmark(self):
        slots, Y, bench = mixed_panel()
        d = sap.describe_panel(slots, Y=Y, benchmarks=bench)
        g = d["coverage"]["groups"]
        self.assertEqual(g["all"], {"tasks": 10, "usable": 7, "unsure": 2, "missing": 1})
        self.assertEqual(g["invalid first verdict"], {"tasks": 2, "usable": 1, "unsure": 1, "missing": 0})
        self.assertEqual(g["disagreement"], {"tasks": 2, "usable": 1, "unsure": 1, "missing": 0})
        self.assertEqual(g["unstable agreeing"], {"tasks": 3, "usable": 2, "unsure": 0, "missing": 1})
        self.assertEqual(g["stable agreeing"], {"tasks": 3, "usable": 3, "unsure": 0, "missing": 0})
        self.assertEqual(d["coverage"]["by_benchmark"], {
            "vwa": {"tasks": 2, "usable": 0, "unsure": 1, "missing": 1},
            "wa": {"tasks": 4, "usable": 3, "unsure": 1, "missing": 0},
            "work": {"tasks": 4, "usable": 4, "unsure": 0, "missing": 0}})
        txt = sap.describe_report(d, EXPOSURE)
        for needle in ("unknown, not correct verdicts", "unstable agreeing | 3 | 2 | 0 | 1", "wa | 4 | 3 | 1 | 0"):
            self.assertIn(needle, txt, needle)

    def test_validity_ignores_unknown_references_and_invalid_slots(self):
        slots, Y, _ = mixed_panel()
        v = sap.describe_panel(slots, Y=Y)["validity"]
        self.assertEqual((v["n_usable"], v["n_shared"]), (7, 5))
        F = Fraction
        A1 = v["own"]["A1"]
        self.assertEqual(A1["confusion"], {"TP": 0, "FP": 0, "FN": 3, "TN": 3})
        self.assertEqual(A1["balanced_accuracy"]["value"], F(1, 2))
        self.assertEqual(A1["valid_prediction_coverage"]["value"], F(6, 7))
        self.assertIsNone(A1["failure_precision"]["value"])
        self.assertEqual(A1["failure_recall"]["value"], 0)
        B1 = v["own"]["B1"]
        self.assertEqual(B1["confusion"], {"TP": 0, "FP": 0, "FN": 2, "TN": 4})
        self.assertEqual(B1["prevalence"]["value"], F(1, 3))
        self.assertEqual(v["shared"]["A1"]["confusion"], {"TP": 0, "FP": 0, "FN": 2, "TN": 3})
        self.assertEqual(v["shared"]["B1"]["valid_prediction_coverage"]["value"], F(5, 7))

    def test_E_with_statuses_matches_the_released_labels(self):
        slots, Y, bench = mixed_panel()
        j = sap.join_reference(slots, Y)
        a = sap.describe_panel(slots, Y=Y, benchmarks=bench)
        b = sap.describe_panel(slots, j["E"], reference=j["reference"], benchmarks=bench)
        for key in ("captures", "gain", "secondary", "coverage", "N_Y", "M"):
            self.assertEqual(a[key], b[key], key)
        self.assertIsNone(b["validity"])
        self.assertIn("Not assessed: the released labels were not supplied.", sap.describe_report(b, EXPOSURE))

    def test_E_without_statuses_assesses_no_label_coverage(self):
        d = sap.describe_panel([slot("S")] * 4, [0, 1, 0, 0])
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
            sap.describe_panel([ok], [0], Y=["Successful"])         # both
        with self.assertRaises(ValueError):
            sap.describe_panel([ok])                                # neither
        with self.assertRaises(ValueError):
            sap.describe_panel([ok], Y=["Successful"], reference=["usable"])
        with self.assertRaises(ValueError):
            sap.describe_panel([ok], [0, 1])
        with self.assertRaises(ValueError):
            sap.describe_panel([ok, ok, ok], [2, 0, 0])
        with self.assertRaises(ValueError):
            sap.describe_panel([ok], Y=["Successful"], benchmarks=["a", "b"])
        with self.assertRaises(ValueError):                         # a task without a reference is never contradicted
            sap.describe_panel([ok], [1], reference=[sap.UNSURE])
        with self.assertRaises(sap.UnfinishedSlot):
            sap.describe_panel([dict(ok, A2=sap.HALT)], Y=["Successful"])


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


def report_panels():
    """A report-ready panel in every regime, with and without Unsure and invalid outcomes."""
    costs = {"total": "12", "by_slot": {"A1": "3", "A2": "3", "A3": "3", "B1": "3"}}
    for sizes, E in (((1, 3, 0, 0, 16), [1, 0] * 10), ((0, 7, 0, 1, 0), [0, 1] * 4), ((0, 0, 1, 1, 1), [1, 0, 1]),
                     ((0, 30, 3, 7, 160), E_for(G1=(30, 11), U2=(3, 1), U1=(7, 3), S=(160, 10))),
                     ((0, 10, 4, 10, 36), E_for(G1=(10, 3), U2=(4, 2), U1=(10, 3), S=(36, 3)))):
        labels = labels_for(*sizes)
        slots, Y = build(labels, E)
        bench = ["wa" if i % 2 else "work" for i in range(len(labels))]
        yield sap.describe_panel(slots, Y=Y, benchmarks=bench, costs=costs)
        Y2 = ["Unsure" if i % 4 == 0 else (None if i % 7 == 0 else y) for i, y in enumerate(Y)]
        yield sap.describe_panel(slots, Y=Y2)
    slots, Y, bench = mixed_panel()
    yield sap.describe_panel(slots, Y=Y, benchmarks=bench, costs=costs, provenance=provenance_of(slots))
    yield sap.describe_panel(slots, [0, 0, 1, 0, 1, 0, 0, 1, 0, 0])


class TestDescriptiveReport(unittest.TestCase):
    def test_no_inferential_wording_in_any_regime(self):
        seen = set()
        for d in report_panels():
            seen.add(d["regime"])
            txt = sap.describe_report(d, EXPOSURE)
            self.assertEqual(txt.count(sap.SCOPE_TEXT), 1)
            body = txt.replace(sap.SCOPE_TEXT, "").replace("observed below the reference", "")
            for pattern in FORBIDDEN:
                self.assertIsNone(re.search(pattern, body, re.IGNORECASE), (pattern, d["regime"]))
        self.assertEqual(seen, REGIMES)

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
        self.assertIn("descriptive", lines[0])
        self.assertIn("the primary path is descriptive", sap.PLAN_VERSION)
        self.assertIn(sap.PLAN_VERSION, lines[0])
        self.assertEqual(lines[1], sap.SCOPE_TEXT)
        self.assertEqual(lines[2], "Panel: a retrospective evaluation panel with documented prior exposure.")
        self.assertEqual(lines[3], "Prior exposure statement: " + EXPOSURE)

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
        txt = sap.describe_report(sap.describe_panel(slots, Y=Y, provenance=provenance_of(slots)), EXPOSURE)
        self.assertIn("under config hash cfg-1", txt)
        self.assertIn("A1 INVALID: 2 (REFUSAL 1, TRANSPORT_EXHAUSTED 1)", txt)
        self.assertIn("Not supplied: resolved slots", sap.describe_report(sap.describe_panel(slots, Y=Y), EXPOSURE))

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
        out = sap.analyze_panel(slots, Y=Y)
        self.assertEqual(out["path"], sap.PATH_DESCRIPTIVE)
        self.assertEqual(sap.PATH_DESCRIPTIVE, "descriptive")
        self.assertNotIn("primary", out)
        explicit = sap.analyze_panel(slots, Y=Y, path="descriptive")
        self.assertEqual(out, explicit)
        positional = sap.analyze_panel(slots, E)                    # E alone takes the same path
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
            sap.analyze_panel(slots, E, boot_B=100)                 # an option of the other path
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

    def test_inference_code_is_still_there(self):
        labels, E, slots, _ = self.data()
        panel = sap.analyze_panel(slots, E, path=sap.PATH_POPULATION, boot_B=200)
        ex = panel["primary"]["existence"]
        self.assertEqual(ex["p"], sap.fisher_one_sided(4, 10, 10, 160))
        self.assertIn("delta_L", panel["primary"]["size"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
