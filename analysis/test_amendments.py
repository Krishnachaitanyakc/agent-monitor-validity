"""Tests for the plan v3.2 amendments to sap.py. Synthetic data only.

Covers reference handling (Unsure and missing labels), label coverage, the outcome
classes (FILTERED, case-insensitive verdicts, HALT), and the panel unit.

Run from the repository root: python -m unittest discover -s analysis -v
"""
import itertools
import os
import sys
import unittest
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sap  # noqa: E402

U, UNS, MIS = sap.USABLE, sap.UNSURE, sap.MISSING


def block(stratum, statuses_and_E):
    """Labels, E and statuses for one stratum from (status, E) pairs."""
    labels = [stratum] * len(statuses_and_E)
    E = [e for _, e in statuses_and_E]
    ref = [s for s, _ in statuses_and_E]
    return labels, E, ref


def panel(*blocks):
    labels, E, ref = [], [], []
    for b in blocks:
        labels += b[0]
        E += b[1]
        ref += b[2]
    return labels, E, ref


def rows(n_usable_discordant=0, n_usable=0, n_unsure=0, n_missing=0):
    """(status, E) pairs: discordant usable first, then concordant usable, Unsure, missing."""
    return ([(U, 1)] * n_usable_discordant + [(U, 0)] * (n_usable - n_usable_discordant)
            + [(UNS, 0)] * n_unsure + [(MIS, 0)] * n_missing)


def labels_for(n_inv=0, n_D1=0, n_U2=0, n_U1=0, n_S=0):
    return ["I"] * n_inv + ["G1"] * n_D1 + ["U2"] * n_U2 + ["U1"] * n_U1 + ["S"] * n_S


def E_for(n_inv=(0, 0), G1=(0, 0), U2=(0, 0), U1=(0, 0), S=(0, 0)):
    out = []
    for n, m in (n_inv, G1, U2, U1, S):
        out += [1] * m + [0] * (n - m)
    return out


class TestOutcomeClasses(unittest.TestCase):
    def test_filtered_is_a_terminal_invalid_outcome(self):
        T = sap.TRANSPORT
        self.assertIn("FILTERED", sap.INVALID_TERMINAL)
        self.assertEqual(sap.resolve_outcomes(["FILTERED"]), "FILTERED")
        self.assertEqual(sap.resolve_outcomes([T, sap.HALT, "FILTERED"]), "FILTERED")
        with self.assertRaises(sap.ProtocolViolation):            # terminal, never retried
            sap.resolve_outcomes(["FILTERED", "success"])
        self.assertFalse(sap.is_valid("FILTERED"))
        slots = [
            {"A1": "FILTERED", "A2": "success", "A3": "success", "B1": "success"},   # I
            {"A1": "success", "A2": "FILTERED", "A3": "success", "B1": "success"},   # invalid repeat counts 0
            {"A1": "success", "A2": "success", "A3": "success", "B1": "FILTERED"},   # B1 INVALID: D = 1
            {"A1": "success", "A2": "FILTERED", "A3": "FILTERED", "B1": "success"},  # R unobserved
        ]
        lf = sap.derive_label_free(slots)
        self.assertEqual(list(lf["stratum"]), ["I", "S", "G1", "S"])
        self.assertEqual(list(lf["R_one_missing"]), [-1, 1, 0, 0])
        self.assertEqual(list(lf["R_unobserved"]), [-1, 0, 0, 1])
        counts = sap.invalid_counts(slots, ["wa"] * 4)
        self.assertEqual(counts, {("A1", "FILTERED", "wa"): 1, ("A2", "FILTERED", "wa"): 2,
                                  ("B1", "FILTERED", "wa"): 1, ("A3", "FILTERED", "wa"): 1})
        A = sap.Attempt
        log = [A("t1", j, r, "H0", 1, "success", f"t1-{j}{r}") for j, r in sap.SLOT_KEYS]
        log[0] = A("t1", "A", 1, "H0", 1, "FILTERED", "t1-A1")
        prov = sap.resolve_panel(log, ["t1"], "H0")["provenance"]["by_slot"]["A1"]
        self.assertEqual((prov["invalid"], prov["invalid:FILTERED"], prov["response_ids"]), (1, 1, 1))

    def test_verdicts_match_case_insensitively(self):
        T = sap.TRANSPORT
        self.assertEqual(sap.resolve_outcomes(["Success"]), "success")
        self.assertEqual(sap.resolve_outcomes([T, "FAILURE"]), "failure")
        self.assertTrue(sap.is_valid("SUCCESS"))
        slots = [
            {"A1": "Success", "A2": "success", "A3": "SUCCESS", "B1": "success"},    # one verdict: S
            {"A1": "Failure", "A2": "failure", "A3": "Success", "B1": "FAILURE"},    # one real flip: U1
            {"A1": "success", "A2": "Success", "A3": "success", "B1": "Failure"},    # disagreement: G1
        ]
        lf = sap.derive_label_free(slots)
        self.assertEqual(list(lf["stratum"]), ["S", "U1", "G1"])
        self.assertEqual(list(lf["R"]), [0, 1, 0])
        A = sap.Attempt
        log = [A("t1", j, r, "H0", 1, v, f"t1-{j}{r}")
               for (j, r), v in zip(sap.SLOT_KEYS, ("Success", "SUCCESS", "success", "Failure"))]
        self.assertEqual(sap.resolve_panel(log, ["t1"], "H0")["slots"][0],
                         {"A1": "success", "A2": "success", "A3": "success", "B1": "failure"})
        # Only ASCII spellings are verdicts, and outcome codes still match exactly.
        for bad in ("ſuccess", "succes", "success ", "refusal", "Halt"):
            with self.assertRaises(ValueError, msg=bad):
                sap.is_valid(bad)
        self.assertEqual(sap.canonical_outcome("REFUSAL"), "REFUSAL")

    def test_halted_slot_is_unfinished_never_invalid(self):
        T, H = sap.TRANSPORT, sap.HALT
        self.assertNotIn(H, sap.INVALID_CODES)
        self.assertTrue(issubclass(sap.UnfinishedSlot, ValueError))
        self.assertFalse(issubclass(sap.UnfinishedSlot, sap.ProtocolViolation))
        for outcomes in ([H], [T, H], [T, T, H], [H, T], [T]):
            with self.assertRaises(sap.UnfinishedSlot, msg=outcomes):
                sap.resolve_outcomes(outcomes)
        self.assertEqual(sap.resolve_outcomes([H, "success"]), "success")       # resumed after the fix
        self.assertEqual(sap.resolve_outcomes([T, H, "REFUSAL"]), "REFUSAL")
        for outcome in (H, T):
            with self.assertRaises(sap.UnfinishedSlot):
                sap.is_valid(outcome)
        ok = {"A1": "success", "A2": "success", "A3": "success", "B1": "success"}
        for k in ("A1", "A2", "A3", "B1"):
            halted = dict(ok, **{k: H})
            with self.assertRaises(sap.UnfinishedSlot, msg=k):
                sap.derive_label_free([ok, halted])
            with self.assertRaises(sap.UnfinishedSlot, msg=k):
                sap.invalid_counts([halted])
            with self.assertRaises(sap.UnfinishedSlot, msg=k):
                sap.analyze_panel([ok, halted], [0, 0])
        with self.assertRaises(sap.UnfinishedSlot):
            sap.join_reference([dict(ok, A1=H)], ["Successful"])
        A = sap.Attempt
        log = [A(t, j, r, "H0", 1, "success", f"{t}-{j}{r}") for t in ("t1", "t2") for j, r in sap.SLOT_KEYS]
        log[-1] = A("t2", "B", 1, "H0", 1, H)                    # a request error halted B1 of t2
        with self.assertRaises(sap.UnfinishedSlot):
            sap.resolve_panel(log, ["t1", "t2"], "H0")
        log.append(A("t2", "B", 1, "H0", 2, "failure", "t2-B1"))   # resumed after the correction
        out = sap.resolve_panel(log, ["t1", "t2"], "H0")
        self.assertEqual(out["slots"][1]["B1"], "failure")
        self.assertEqual((out["provenance"]["by_slot"]["B1"]["halts"],
                          out["provenance"]["by_slot"]["B1"]["invalid"]), (1, 0))


class TestReferenceJoin(unittest.TestCase):
    def test_reference_status(self):
        cases = {"Successful": (U, "success"), "unsuccessful": (U, "failure"), "SUCCESS": (U, "success"),
                 "failure": (U, "failure"), "Unsure": (UNS, None), "UNSURE": (UNS, None),
                 None: (MIS, None), "": (MIS, None), "  ": (MIS, None), float("nan"): (MIS, None)}
        for y, want in cases.items():
            self.assertEqual(sap.reference_status(y), want, y)
        for bad in ("maybe", "Successful ", "ſuccessful", 1, 0.0):
            with self.assertRaises(ValueError, msg=repr(bad)):
                sap.reference_status(bad)

    def test_join_reference(self):
        sl = lambda a1: {"A1": a1, "A2": "success", "A3": "success", "B1": "success"}
        slots = [sl("success"), sl("success"), sl("FAILURE"), sl("success"), sl("failure"),
                 sl("failure"), sl("REFUSAL"), sl("REFUSAL")]
        Y = ["Successful", "Unsuccessful", "failure", "Unsure", None, "", "Successful", "Unsure"]
        out = sap.join_reference(slots, Y)
        self.assertEqual(out["E"], [0, 1, 0, 0, 0, 0, 0, 0])
        self.assertEqual(out["reference"], [U, U, U, UNS, MIS, MIS, U, UNS])
        with self.assertRaises(ValueError):
            sap.join_reference(slots, Y[:-1])

    def test_a_task_without_a_reference_is_never_discordant(self):
        labels = labels_for(0, 10, 0, 5, 40)
        E = [0] * 55
        ref = [U] * 55
        E[12], ref[12] = 1, UNS                                  # an Unsure task marked discordant
        with self.assertRaises(ValueError):
            sap.analyze(labels, E, reference=ref)
        ref[12] = MIS
        with self.assertRaises(ValueError):
            sap.analyze(labels, E, reference=ref)
        E2, ref2 = [1] + [0] * 54, [UNS] + [U] * 54             # an invalid A1: E is ignored, as before
        sap.analyze(["I"] + labels[1:], E2, reference=ref2)
        with self.assertRaises(ValueError):
            sap.analyze(labels, [0] * 55, reference=["sure"] * 55)


GOLDEN_V31 = {
    # Plan v3.1 results of the original implementation on four synthetic panels.
    "coverage": dict(
        panel=(labels_for(0, 30, 3, 7, 160), E_for(G1=(30, 11), U2=(3, 1), U1=(7, 3), S=(160, 10)), {}),
        captures={"RAND": Fraction(15, 2), "DIS": Fraction(229, 17), "AUG": Fraction(65, 4),
                  "Delta": Fraction(189, 68)},
        Delta_per100=1.3897058823529411, p=0.004643666349805531, reject=True, fragility=1,
        size={"delta_L": 0.06, "delta_U": 0.66, "G_L": 0.24705882352941172, "G_U": 2.717647058823529,
              "G_hat": 1.3897058823529411, "class": "unresolved"}),
    "coverage with invalid first verdicts": dict(
        panel=(labels_for(5, 20, 2, 3, 100),
               E_for(n_inv=(5, 0), G1=(20, 6), U2=(2, 1), U1=(3, 1), S=(100, 4)), {}),
        captures={"RAND": Fraction(96, 25), "DIS": Fraction(50, 7), "AUG": Fraction(43, 5),
                  "Delta": Fraction(51, 35)},
        Delta_per100=1.1208791208791207, p=0.025385559247397743, reject=False, fragility=1,
        size={"delta_L": 0.0, "delta_U": 0.8, "G_L": 0.0, "G_U": 2.490842490842491,
              "G_hat": 1.1208791208791211, "class": "unresolved"}),
    "Berger-Boos example": dict(
        panel=(["G1"] * 15 + ["U1"] * 5 + ["S"] * 160, [0] * 15 + [1, 1, 1, 1, 0] + [1] * 24 + [0] * 136, {}),
        captures={"RAND": Fraction(98, 15), "DIS": Fraction(252, 55), "AUG": Fraction(73, 10),
                  "Delta": Fraction(299, 110)},
        Delta_per100=1.5101010101010102, p=0.0030284827206282683, reject=True, fragility=1,
        size={"delta_L": 0.16, "delta_U": 0.86, "G_L": 0.3717171717171717, "G_U": 1.9979797979797977,
              "G_hat": 1.5101010101010102, "class": "below"}),
    "partial": dict(
        panel=(labels_for(0, 10, 4, 10, 36), E_for(G1=(10, 3), U2=(4, 2), U1=(10, 3), S=(36, 3)),
               {"boot_B": 2000}),
        captures={"RAND": Fraction(209, 60), "DIS": Fraction(111, 25), "AUG": Fraction(13, 2),
                  "Delta": Fraction(103, 50)},
        Delta_per100=3.433333333333333, p=0.020113763138094615, reject=True, fragility=1,
        partial=(3.4333333333333327, 0.10000000000000009, 6.466666666666668)),
}


class TestNoUnsureEqualsV31(unittest.TestCase):
    def test_results_equal_v31_exactly(self):
        for name, g in GOLDEN_V31.items():
            labels, E, kw = g["panel"]
            for ref in (None, [U] * len(labels)):
                with self.subTest(panel=name, reference=ref is not None):
                    r = sap.analyze(labels, E, reference=ref, **kw)
                    for k, v in g["captures"].items():
                        self.assertEqual(r["captures"][k], v)
                    self.assertEqual(r["Delta_per100"], g["Delta_per100"])
                    ex = r["existence"]
                    self.assertEqual((ex["p"], ex["reject"], ex["fragility"]), (g["p"], g["reject"], g["fragility"]))
                    if "size" in g:
                        for k, v in g["size"].items():
                            self.assertEqual(r["size"][k], v, k)
                    else:
                        b = r["size"]["partial"]
                        self.assertEqual((b["G_partial_hat"], b["lower"], b["upper"]), g["partial"])
                    # Secondary equals primary when every label is usable.
                    sec = r["secondary"]
                    self.assertEqual((sec["N_Y"], sec["B_Y"], sec["not_estimable"]), (len(labels), 0, None))
                    self.assertEqual(sec["existence"]["p"], ex["p"])
                    self.assertEqual(sec["Delta_per100_usable"], r["Delta_per100"])
                    if "size" in g:
                        for k in ("delta_hat", "delta_L", "delta_U", "G_L", "G_U", "G_hat", "class"):
                            self.assertEqual(sec["size"][k], r["size"][k], k)
                        self.assertEqual(sec["L_Y"], r["label_free"]["L"])
                        self.assertEqual(sec["coverage_term"], 0)
                    else:
                        self.assertEqual(sec["size"]["partial"], r["size"]["partial"])
                    cov = r["coverage"]["groups"]["all"]
                    self.assertEqual((cov["unsure"], cov["missing"], cov["usable"]), (0, 0, len(labels)))

    def test_reference_all_usable_changes_nothing_in_the_panel_analysis(self):
        rng = np.random.default_rng(31)
        slots, E = [], []
        for i in range(150):
            a1 = "success" if rng.random() < 0.6 else "failure"
            flip = "failure" if a1 == "success" else "success"
            sl = {"A1": a1, "A2": flip if rng.random() < 0.1 else a1,
                  "A3": flip if rng.random() < 0.1 else a1, "B1": flip if rng.random() < 0.15 else a1}
            if i % 31 == 0:
                sl["B1"] = "FILTERED"
            slots.append(sl)
            E.append(int(rng.random() < 0.2))
        ids = [f"t{i:03d}" for i in range(150)]
        a = sap.analyze_panel(slots, E, ids=ids, seed=7, boot_B=500)
        b = sap.analyze_panel(slots, E, ids=ids, seed=7, boot_B=500, reference=[U] * 150)
        for (_, ra), (_, rb) in zip([("p", a["primary"])] + a["sensitivities"],
                                    [("p", b["primary"])] + b["sensitivities"]):
            for key in ("label_free", "m_counts", "captures", "Delta_per100", "existence", "size", "secondary"):
                self.assertEqual(repr(ra[key]), repr(rb[key]), key)


class TestReferenceHandling(unittest.TestCase):
    def test_unsure_tasks_stay_in_the_panel_and_review_count(self):
        # 80 of 160 stable agreeing tasks are Unsure.
        labels, E, ref = panel(block("G1", rows(5, 20)), block("U1", rows(2, 10)), block("S", rows(16, 80, 80)))
        r = sap.analyze(labels, E, reference=ref)
        bare = sap.analyze(labels, E)                            # same E, statuses not supplied
        lf = r["label_free"]
        self.assertEqual(lf, bare["label_free"])                  # design, s, K and regime fixed on N
        self.assertEqual((lf["N"], lf["N_per100"], lf["n0"], lf["s"], lf["K"], lf["regime"]),
                         (190, 190, 160, 29, 49, "COVERAGE"))
        # Primary: per 100 panel tasks, Unsure counted as not reference-contradicted.
        L = lf["L"]
        self.assertEqual(L, Fraction(10 * (170 - 29), 170))
        self.assertEqual(r["captures"]["Delta"], L * (Fraction(2, 10) - Fraction(16, 160)))
        self.assertEqual(r["Delta_per100"], float(r["captures"]["Delta"]) * 100 / 190)
        self.assertEqual(r["existence"]["p"], sap.fisher_one_sided(2, 10, 16, 160))
        sz = r["size"]
        self.assertAlmostEqual(sz["G_hat"], 100 * float(L) / 190 * 0.1, places=12)
        self.assertEqual(sz["G_L"], 100.0 * float(L) / 190 * sz["delta_L"])
        self.assertEqual(r["captures"], bare["captures"])
        self.assertEqual(r["existence"]["p"], bare["existence"]["p"])

    def test_unsure_share_differs_between_unstable_and_stable(self):
        # Stable agreeing tasks are half Unsure; usable discordance is 20% in both U and S.
        labels, E, ref = panel(block("G1", rows(5, 20)), block("U1", rows(2, 10)), block("S", rows(16, 80, 80)))
        r = sap.analyze(labels, E, reference=ref)
        sec, lf = r["secondary"], r["label_free"]
        L, Delta = lf["L"], r["captures"]["Delta"]
        self.assertEqual((sec["N_Y"], sec["n1_Y"], sec["n0_Y"]), (110, 10, 80))
        self.assertEqual(sec["L_Y"], L)                          # every unstable task has a usable label
        self.assertEqual(sec["G_hat_exact"], 0)                  # equal usable rates: no gain
        self.assertEqual(sec["size"]["delta_hat"], 0.0)
        self.assertEqual(sec["existence"]["p"], sap.fisher_one_sided(2, 10, 16, 80))
        self.assertFalse(sec["existence"]["reject"])
        self.assertLess(sec["size"]["delta_L"], 0.0)
        self.assertGreater(sec["size"]["delta_U"], 0.0)
        # The primary shows a gain that comes only from the missing references among S.
        self.assertAlmostEqual(r["size"]["delta_hat"], 0.1, places=15)
        self.assertLess(r["existence"]["p"], sec["existence"]["p"])
        # Identity: 100 Delta / N_Y = G_hat_Y + (100 L / N_Y)(n1Y/n1 - n0Y/n0)(m0/n0Y).
        self.assertEqual(sec["coverage_term"], Fraction(100) * L / 110 * (1 - Fraction(1, 2)) * Fraction(16, 80))
        self.assertEqual(sec["G_hat_exact"] + sec["coverage_term"], 100 * Delta / 110)
        self.assertEqual(sec["Delta_per100_usable"], float(Delta) * 100 / 110)
        self.assertEqual(sec["B_Y"], L * (1 - Fraction(1, 2)))

        # Reverse: half of the unstable tasks are Unsure.
        labels, E, ref = panel(block("G1", rows(5, 20)), block("U1", rows(2, 5, 5)), block("S", rows(16, 160)))
        r = sap.analyze(labels, E, reference=ref)
        sec, lf = r["secondary"], r["label_free"]
        L, Delta = lf["L"], r["captures"]["Delta"]
        self.assertEqual((sec["N_Y"], sec["n1_Y"], sec["n0_Y"]), (185, 5, 160))
        self.assertEqual(sec["L_Y"], L * Fraction(5, 10))
        dY = Fraction(2, 5) - Fraction(16, 160)
        self.assertEqual(sec["G_hat_exact"], Fraction(100) * sec["L_Y"] * dY / 185)
        self.assertAlmostEqual(sec["size"]["G_hat"], float(sec["G_hat_exact"]), places=12)
        self.assertEqual(sec["delta_m_Y"], Fraction(2) * 185 / (100 * sec["L_Y"]))
        self.assertEqual(sec["G_hat_exact"] + sec["coverage_term"], 100 * Delta / 185)
        self.assertLess(sec["coverage_term"], 0)
        self.assertEqual(sec["existence"]["p"], sap.fisher_one_sided(2, 5, 16, 160))
        # The coverage-regime secondary p equals the permutation p of its own statistic.
        nY = {"U2": 0, "U1": 5, "S": 160}
        perm = sap.permutation_p(sap.Design.from_labels(labels), r["m_counts"], sizes=nY, weights=sec["weights"])
        self.assertEqual(perm, sap.fisher_one_sided(2, 5, 16, 160, exact=True))
        # The secondary weights are budget-balanced on the tasks with a usable label.
        self.assertEqual(sum(sec["weights"][j] * nY[j] for j in nY), 0)
        txt = sap.report(r)
        self.assertIn("SECONDARY RESULT: gain per 100 tasks with a usable label (N_Y = 185 of N = 190)", txt)
        self.assertLess(txt.index("EXISTENCE QUESTION"), txt.index("SECONDARY RESULT"))

    def test_all_labels_unsure_in_one_stratum(self):
        # All stable agreeing tasks Unsure.
        labels, E, ref = panel(block("I", rows(0, 2)), block("G1", rows(5, 20)), block("U1", rows(3, 10)),
                               block("S", rows(0, 0, 160)))
        r = sap.analyze(labels, E, reference=ref)
        self.assertEqual(r["label_free"]["regime"], "COVERAGE")
        self.assertEqual(r["existence"]["p"], sap.fisher_one_sided(3, 10, 0, 160))
        self.assertIsNotNone(r["size"])
        sec = r["secondary"]
        self.assertEqual(sec["not_estimable"], "no stable agreeing task has a usable label")
        self.assertIsNone(sec["existence"])
        self.assertIsNone(sec["size"])
        self.assertEqual(sec["N_Y"], 32)
        self.assertEqual(r["coverage"]["groups"]["stable agreeing"],
                         {"tasks": 160, "usable": 0, "unsure": 160, "missing": 0})
        self.assertIn("Not estimable: no stable agreeing task has a usable label", sap.report(r))
        # All unstable agreeing tasks Unsure: m1 is 0 by construction.
        labels, E, ref = panel(block("G1", rows(5, 20)), block("U2", rows(0, 0, 4)), block("U1", rows(0, 0, 6)),
                               block("S", rows(16, 160)))
        r = sap.analyze(labels, E, reference=ref)
        self.assertEqual((r["m1"], r["existence"]["p"]), (0, 1.0))
        self.assertEqual(r["secondary"]["not_estimable"], "no unstable agreeing task has a usable label")
        # Every label Unsure or missing.
        labels = labels_for(0, 20, 0, 10, 160)
        ref = [UNS] * 100 + [MIS] * 90
        r = sap.analyze(labels, [0] * 190, reference=ref)
        self.assertEqual(r["secondary"]["N_Y"], 0)
        self.assertIsNone(r["secondary"]["Delta_per100_usable"])
        self.assertEqual(r["secondary"]["not_estimable"], "no panel task has a usable label")
        self.assertEqual(r["captures"]["Delta"], 0)
        self.assertIn("Not estimable: no panel task has a usable label", sap.report(r))
        # All invalid-first-verdict tasks Unsure: only N_Y moves.
        labels, E, ref = panel(block("I", rows(0, 0, 6)), block("G1", rows(5, 20)), block("U1", rows(2, 10)),
                               block("S", rows(16, 160)))
        r = sap.analyze(labels, E, reference=ref)
        sec = r["secondary"]
        self.assertEqual((sec["N_Y"], sec["n1_Y"], sec["n0_Y"], sec["B_Y"]), (190, 10, 160, 0))
        self.assertEqual(sec["existence"]["p"], r["existence"]["p"])
        self.assertEqual(sec["size"]["delta_L"], r["size"]["delta_L"])
        self.assertEqual(sec["G_hat_exact"], Fraction(100) * r["label_free"]["L"]
                         * (Fraction(2, 10) - Fraction(16, 160)) / 190)

    def test_secondary_in_the_partial_regime(self):
        # N = 26, s = 4 < n1 = 9: partial regime, with Unsure tasks in every agreeing stratum.
        labels, E, ref = panel(block("G1", rows(2, 5)), block("U2", rows(1, 2, 1)), block("U1", rows(1, 4, 2)),
                               block("S", rows(2, 8, 4)))
        r = sap.analyze(labels, E, reference=ref, boot_B=2000)
        self.assertEqual(r["label_free"]["regime"], "PARTIAL")
        sec = r["secondary"]
        nY = {"U2": 2, "U1": 4, "S": 8}
        self.assertEqual({j: sec["n_Y"][j] for j in nY}, nY)
        w = sap.delta_weights(sap.Design.from_labels(labels))
        B_Y = sum(w[j] * nY[j] for j in nY)
        self.assertEqual(sec["B_Y"], B_Y)
        self.assertEqual(sec["weights"], {"U2": w["U2"], "U1": w["U1"], "S": w["S"] - B_Y / 8})
        # Brute force: every way to place the 4 discordant verdicts among the 14 usable agreeing tasks.
        strata = ["U2"] * 2 + ["U1"] * 4 + ["S"] * 8
        ws = [sec["weights"][j] for j in strata]
        obs = sec["weights"]["U2"] * 1 + sec["weights"]["U1"] * 1 + sec["weights"]["S"] * 2
        hits = total = 0
        for chosen in itertools.combinations(range(14), 4):
            total += 1
            hits += sum(ws[i] for i in chosen) >= obs
        self.assertEqual(sap.permutation_p(sap.Design.from_labels(labels), r["m_counts"], sizes=nY,
                                           weights=sec["weights"]), Fraction(hits, total))
        self.assertEqual(sec["existence"]["p"], hits / total)
        self.assertEqual(sum(sec["weights"][j] * nY[j] for j in nY), 0)
        b = sec["size"]["partial"]
        self.assertAlmostEqual(b["G_partial_hat"], float(sec["G_hat_exact"]), places=12)
        self.assertEqual(sec["G_hat_exact"], Fraction(100, 19) * sum(sec["weights"][j] * r["m_counts"][j] for j in nY))
        self.assertLessEqual(b["lower"], b["upper"])
        self.assertIn("no threshold claim", sap.report(r))

    def test_sensitivities_hold_both_denominators(self):
        rng = np.random.default_rng(5)
        slots, Y = [], []
        for i in range(120):
            a1 = "success" if rng.random() < 0.6 else "failure"
            flip = "failure" if a1 == "success" else "success"
            sl = {"A1": a1, "A2": flip if rng.random() < 0.12 else a1, "A3": a1,
                  "B1": flip if rng.random() < 0.15 else a1}
            if i % 13 == 0:
                sl["B1"] = "MALFORMED"
            slots.append(sl)
            Y.append("Unsure" if i % 9 == 0 else (None if i % 17 == 0 else
                                                   ("Successful" if rng.random() < 0.5 else "Unsuccessful")))
        j = sap.join_reference(slots, Y)
        outs = sap.analyze_with_sensitivities(slots, j["E"], reference=j["reference"], boot_B=300)
        NY0 = outs[0][1]["secondary"]["N_Y"]
        self.assertEqual(NY0, sum(s == U for s in j["reference"]))
        for _, r in outs[1:]:
            self.assertEqual(r["secondary"]["N_Y_per100"], NY0)
            self.assertEqual(r["label_free"]["N_per100"], 120)


class TestCoverage(unittest.TestCase):
    def _panel(self):
        labels, E, ref = panel(block("I", [(UNS, 0), (MIS, 0), (U, 0), (U, 0)]),
                               block("G1", rows(4, 16, 3, 1)),
                               block("U2", rows(1, 1, 0, 1)),
                               block("U1", rows(1, 4, 2)),
                               block("S", rows(9, 153, 10, 5)))
        bench = ["webarena" if i % 3 else "workarena" for i in range(len(labels))]
        return labels, E, ref, bench

    def test_coverage_counts_by_group_and_benchmark(self):
        labels, E, ref, bench = self._panel()
        r = sap.analyze(labels, E, reference=ref, benchmarks=bench, seed=11)
        lf = r["label_free"]
        self.assertEqual((lf["N"], lf["s"], lf["K"], lf["n1"], lf["regime"]), (200, 30, 54, 8, "COVERAGE"))
        g = r["coverage"]["groups"]
        self.assertEqual(g["all"], {"tasks": 200, "usable": 176, "unsure": 16, "missing": 8})
        self.assertEqual(g["invalid first verdict"], {"tasks": 4, "usable": 2, "unsure": 1, "missing": 1})
        self.assertEqual(g["disagreement"], {"tasks": 20, "usable": 16, "unsure": 3, "missing": 1})
        self.assertEqual(g["unstable agreeing"], {"tasks": 8, "usable": 5, "unsure": 2, "missing": 1})
        self.assertEqual(g["stable agreeing"], {"tasks": 168, "usable": 153, "unsure": 10, "missing": 5})
        bb = r["coverage"]["by_benchmark"]
        for b in ("webarena", "workarena"):
            idx = [i for i in range(200) if bench[i] == b]
            self.assertEqual(bb[b], {"tasks": len(idx), "usable": sum(ref[i] == U for i in idx),
                                     "unsure": sum(ref[i] == UNS for i in idx),
                                     "missing": sum(ref[i] == MIS for i in idx)})

    def test_policy_selections_of_unknown_labels(self):
        labels, E, ref, bench = self._panel()
        r = sap.analyze(labels, E, reference=ref, seed=11)
        sel = r["coverage"]["selected_expected"]
        f_dis, f_aug_S, f_rand = Fraction(30, 176), Fraction(30 - 8, 168), Fraction(54 - 4, 196)
        want = {
            "AUG": {"invalid first verdict": (1, 1), "disagreement": (3, 1), "unstable agreeing": (2, 1),
                    "stable agreeing": (10 * f_aug_S, 5 * f_aug_S)},
            "DIS": {"invalid first verdict": (1, 1), "disagreement": (3, 1),
                    "unstable agreeing": (2 * f_dis, f_dis), "stable agreeing": (10 * f_dis, 5 * f_dis)},
            "RAND": {"invalid first verdict": (1, 1), "disagreement": (3 * f_rand, f_rand),
                     "unstable agreeing": (2 * f_rand, f_rand), "stable agreeing": (10 * f_rand, 5 * f_rand)},
        }
        for pol, groups in want.items():
            for name, (u, m) in groups.items():
                self.assertEqual((sel[pol][name][UNS], sel[pol][name][MIS]), (u, m), (pol, name))
            self.assertEqual(sel[pol]["all"][UNS], sum(u for u, _ in groups.values()))
            self.assertEqual(sel[pol]["all"]["tasks"], 54)               # every policy reviews K
        probs = sap.selection_probabilities(sap.Design.from_labels(labels), labels)
        for pol in ("RAND", "DIS", "AUG"):
            self.assertEqual(sum(probs[pol]), 54)
        # The seeded draw's own counts.
        dr = r["coverage"]["selected_draw"]
        for pol in ("RAND", "DIS", "AUG"):
            chosen = set(int(i) for i in r["draw"]["reviewed"][pol])
            self.assertEqual(dr[pol]["all"][UNS], sum(1 for i in chosen if ref[i] == UNS))
            self.assertEqual(dr[pol]["all"][MIS], sum(1 for i in chosen if ref[i] == MIS))
            self.assertEqual((dr[pol]["invalid first verdict"][UNS], dr[pol]["invalid first verdict"][MIS]), (1, 1))
        self.assertEqual((dr["AUG"]["unstable agreeing"][UNS], dr["AUG"]["unstable agreeing"][MIS]), (2, 1))

    def test_coverage_is_reported_as_unknown(self):
        labels, E, ref, bench = self._panel()
        slots = []
        for lab in labels:
            a1 = "success"
            slots.append({"I": {"A1": "REFUSAL", "A2": a1, "A3": a1, "B1": a1},
                          "G1": {"A1": a1, "A2": a1, "A3": a1, "B1": "failure"},
                          "U2": {"A1": a1, "A2": "failure", "A3": "failure", "B1": a1},
                          "U1": {"A1": a1, "A2": "failure", "A3": a1, "B1": a1},
                          "S": {"A1": a1, "A2": a1, "A3": a1, "B1": a1}}[lab])
        ids = [f"t{i:03d}" for i in range(200)]
        p = sap.analyze_panel(slots, E, ids=ids, benchmarks=bench, reference=ref, seed=3, boot_B=300)
        self.assertEqual(list(p["primary"]["label_free"].values()),
                         list(sap.analyze_panel(slots, E, ids=ids, seed=3, boot_B=300)["primary"]["label_free"].values()))
        txt = sap.panel_report(p)
        for needle in ("REFERENCE COVERAGE: Unsure and missing labels", "unknown, not correct verdicts",
                       "count as not reference-contradicted", "stable agreeing | 168 | 153 | 10 | 5",
                       "Instability-first:", "One operator's draw", "workarena |",
                       "secondary (per 100 tasks with a usable label, N_Y = 176)"):
            self.assertIn(needle, txt, needle)
        self.assertNotIn("counted as correct", txt)


class TestPanelUnit(unittest.TestCase):
    RUNS = [
        {"run_id": "a1", "task_id": "a", "template_id": "T1", "passes": True},
        {"run_id": "a2", "task_id": "a", "template_id": "T1", "passes": True},
        {"run_id": "a3", "task_id": "a", "template_id": "T1", "passes": False},
        {"run_id": "b1", "task_id": "b", "template_id": "T1", "passes": False},
        {"run_id": "b2", "task_id": "b", "template_id": "T1", "passes": True},
        {"run_id": "c1", "task_id": "c", "template_id": "T1", "passes": True},
        {"run_id": "d1", "task_id": "d", "template_id": "T2", "passes": True},
        {"run_id": "d2", "task_id": "d", "template_id": "T2", "passes": True},
        {"run_id": "f1", "task_id": "f", "template_id": "T2", "passes": False},
        {"run_id": "e1", "task_id": "e", "template_id": "T3", "passes": False},
    ]

    def test_default_panel_is_one_trajectory_per_task(self):
        self.assertEqual(sap.PANEL_UNIT, "task")
        passing = {r["run_id"] for r in self.RUNS if r["passes"]}
        task_of = {r["run_id"]: r["task_id"] for r in self.RUNS}
        seen_a = set()
        for seed in range(25):
            pan = sap.select_panel(self.RUNS, seed)
            self.assertEqual(pan, sap.select_panel(self.RUNS, seed, unit="task"))
            self.assertEqual([task_of[x] for x in pan], ["a", "b", "c", "d"])  # one per task; e and f have no passing run
            self.assertTrue(set(pan) <= passing)
            self.assertIn("b2", pan)
            seen_a.add(pan[0])
            shuffled = [self.RUNS[i] for i in np.random.default_rng(seed).permutation(len(self.RUNS))]
            self.assertEqual(sap.select_panel(shuffled, seed), pan)            # input order does not matter
        self.assertEqual(seen_a, {"a1", "a2"})                                 # seeded, never a3

    def test_task_rule_is_the_first_passing_run_of_a_seeded_permutation(self):
        seed = 20261030
        rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(4,)))
        want = []
        for t in ("a", "b", "c", "d", "e", "f"):
            members = sorted((r for r in self.RUNS if r["task_id"] == t), key=lambda r: r["run_id"])
            hit = [members[int(k)]["run_id"] for k in rng.permutation(len(members)) if members[int(k)]["passes"]]
            if hit:
                want.append(hit[0])
        self.assertEqual(sap.select_panel(self.RUNS, seed), want)

    def test_template_panel_is_a_labelled_sensitivity_subset(self):
        task_of = {r["run_id"]: r["task_id"] for r in self.RUNS}
        tmpl_of = {r["run_id"]: r["template_id"] for r in self.RUNS}
        for seed in range(25):
            full = sap.select_panel(self.RUNS, seed)
            tp = sap.select_panel(self.RUNS, seed, unit="template")
            self.assertTrue(set(tp) <= set(full))
            self.assertEqual(sorted(tmpl_of[x] for x in tp), ["T1", "T2"])     # T3 has no passing run
            self.assertIn("d", [task_of[x] for x in tp])                        # f never qualifies
        bad = [dict(self.RUNS[0]), dict(self.RUNS[0])]
        with self.assertRaises(ValueError):
            sap.select_panel(bad, 0)                                            # duplicate run ids
        with self.assertRaises(ValueError):
            sap.select_panel([self.RUNS[0], dict(self.RUNS[1], template_id="T9")], 0)
        with self.assertRaises(ValueError):
            sap.select_panel([dict(self.RUNS[0], passes=1)], 0)
        with self.assertRaises(ValueError):
            sap.select_panel(self.RUNS, 0, unit="trajectory")

    def test_template_panel_analysis_in_the_report(self):
        rng = np.random.default_rng(12)
        slots, E, ids, tmpl = [], [], [], []
        for i in range(200):
            a1 = "success" if rng.random() < 0.55 else "failure"
            flip = "failure" if a1 == "success" else "success"
            r = rng.random()
            a2, a3 = (flip, a1) if r < 0.06 else ((flip, flip) if r < 0.08 else (a1, a1))
            slots.append({"A1": a1, "A2": a2, "A3": a3, "B1": flip if rng.random() < 0.15 else a1})
            E.append(int(rng.random() < (0.45 if a2 != a1 or a3 != a1 else 0.08)))
            ids.append(f"task-{i:04d}")
            tmpl.append(f"tpl-{i // 2}")
        runs = [{"run_id": t, "task_id": t, "template_id": g, "passes": True} for t, g in zip(ids, tmpl)]
        self.assertEqual(sap.select_panel(runs, 4), ids)
        tp = sap.select_panel(runs, 4, unit="template")
        self.assertEqual(len(tp), 100)
        out = sap.analyze_panel(slots, E, ids=ids, templates=tmpl, template_panel=tp, seed=4, boot_B=300)
        self.assertEqual(out["primary"]["label_free"]["N"], 200)
        self.assertEqual(out["template_panel"]["n"], 100)
        self.assertEqual(out["template_panel"]["result"]["label_free"]["N"], 100)
        self.assertEqual(out["template_multiplicities"],
                         {"n_tasks": 200, "n_templates": 100, "max_multiplicity": 2, "tasks_sharing": 200,
                          "histogram": {2: 100}})
        txt = sap.panel_report(out)
        for needle in ("The primary panel keeps one trajectory per task (N = 200)",
                       "One-task-per-template panel (chosen label-free before evaluation",
                       "- one task per template: N = 100", "Panel tasks: 200 in 100 templates",
                       "One agreeing item per template"):
            self.assertIn(needle, txt, needle)
        with self.assertRaises(ValueError):
            sap.analyze_panel(slots, E, ids=ids, template_panel=tp + ["task-9999"])
        with self.assertRaises(ValueError):
            sap.analyze_panel(slots, E, template_panel=tp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
