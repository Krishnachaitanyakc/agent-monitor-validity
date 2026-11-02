"""Tests of the plan's label, outcome and panel rules in sap.py. Synthetic data only.

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
        # The released vocabulary: Successful, Unsuccessful and Unsure, in any letter case.
        cases = {"Successful": (U, "success"), "SUCCESSFUL": (U, "success"), "successful": (U, "success"),
                 "Unsuccessful": (U, "failure"), "unsuccessful": (U, "failure"),
                 "UNSUCCESSFUL": (U, "failure"), "Unsure": (UNS, None), "UNSURE": (UNS, None),
                 "unSure": (UNS, None), None: (MIS, None), "": (MIS, None), "  ": (MIS, None),
                 float("nan"): (MIS, None)}
        for y, want in cases.items():
            self.assertEqual(sap.reference_status(y), want, y)
        # Anything else stops the analysis, the judges' verdict words included.
        for bad in ("success", "failure", "Success", "FAILURE", "maybe", "Successful ", "ſuccessful",
                    "Unsuccessfull", 1, 0.0):
            with self.assertRaises(ValueError, msg=repr(bad)):
                sap.reference_status(bad)

    def test_join_reference(self):
        sl = lambda a1: {"A1": a1, "A2": "success", "A3": "success", "B1": "success"}
        slots = [sl("success"), sl("success"), sl("FAILURE"), sl("success"), sl("failure"),
                 sl("failure"), sl("REFUSAL"), sl("REFUSAL")]
        Y = ["Successful", "Unsuccessful", "UNSUCCESSFUL", "Unsure", None, "", "Successful", "Unsure"]
        out = sap.join_reference(slots, Y)
        self.assertEqual(out["E"], [0, 1, 0, 0, 0, 0, 0, 0])
        self.assertEqual(out["reference"], [U, U, U, UNS, MIS, MIS, U, UNS])
        with self.assertRaises(ValueError):
            sap.join_reference(slots, Y[:-1])
        for word in ("success", "failure"):
            with self.assertRaises(ValueError, msg=word):
                sap.join_reference(slots[:1], [word])

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
        E2, ref2 = [1] + [0] * 54, [UNS] + [U] * 54             # an invalid A1 has no verdict: E is ignored
        sap.analyze(["I"] + labels[1:], E2, reference=ref2)
        with self.assertRaises(ValueError):
            sap.analyze(labels, [0] * 55, reference=["sure"] * 55)


GOLDEN_ALL_USABLE = {
    # Results with every label usable, on four synthetic panels.
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


class TestAllLabelsUsable(unittest.TestCase):
    def test_results_match_golden_values(self):
        for name, g in GOLDEN_ALL_USABLE.items():
            labels, E, kw = g["panel"]
            for ref in (None, [U] * len(labels)):
                with self.subTest(panel=name, reference=ref is not None):
                    r = sap.analyze(labels, E, reference=ref, **kw)
                    for k, v in g["captures"].items():
                        self.assertEqual(r["captures"][k], v)
                    self.assertEqual(r["Delta_per100"], g["Delta_per100"])
                    ex = r["existence"]
                    self.assertEqual((ex["p"], ex["reject"], ex["fragility"]), (g["p"], g["reject"], g["fragility"]))
                    self.assertTrue(ex["fragility_defined"])
                    if "size" in g:
                        for k, v in g["size"].items():
                            self.assertEqual(r["size"][k], v, k)
                    else:
                        b = r["size"]["partial"]
                        self.assertEqual((b["G_partial_hat"], b["lower"], b["upper"]), g["partial"])
                    self.assertEqual(r["reference_supplied"], ref is not None)
                    if ref is None:
                        # Statuses not supplied: nothing about Unsure or missing labels is assessed.
                        self.assertIsNone(r["secondary"])
                        self.assertIsNone(r["coverage"])
                        continue
                    # With every label usable, (a) is the primary result itself, and (b) has the
                    # primary's point estimate and test.
                    sec = r["secondary"]
                    N, Delta = len(labels), r["captures"]["Delta"]
                    self.assertEqual((sec["N_Y"], sec["B_Y"], sec["not_estimable"], sec["usable_only_omitted"]),
                                     (N, 0, None, None))
                    pl, uo = sec["plan_literal"], sec["usable_only"]
                    self.assertEqual(pl["rescale"], 1)
                    self.assertEqual(pl["G_hat_exact"], 100 * Delta / N)
                    self.assertEqual(pl["existence"]["p"], ex["p"])
                    self.assertEqual(uo["G_hat_exact"], 100 * Delta / N)
                    self.assertEqual(uo["existence"]["p"], ex["p"])
                    self.assertEqual(uo["usable_share_term"], 0)
                    if "size" in g:
                        for k in ("delta_L", "delta_U", "G_L", "G_U"):
                            self.assertEqual(pl[k], r["size"][k], k)
                        self.assertEqual(uo["L_Y"], r["label_free"]["L"])
                        self.assertEqual(sec["implied_stable_reviews"],
                                         r["label_free"]["s"] - r["label_free"]["n1"])
                        # (b) carries no threshold, so its grid is k/100 plus 0 only.
                        m1, m0, n1, n0 = r["m1"], r["m0"], r["label_free"]["n1"], r["label_free"]["n0"]
                        ci = sap.ExactScoreDiff(n1, n0).interval(m1, m0, extra_grid=[0.0])
                        self.assertEqual((uo["size"]["delta_L"], uo["size"]["delta_U"]),
                                         (ci["delta_L"], ci["delta_U"]))
                        self.assertEqual(uo["size"]["delta_hat"], r["size"]["delta_hat"])
                    else:
                        self.assertEqual(pl["partial"], r["size"]["partial"])
                        self.assertEqual(uo["size"]["partial"], r["size"]["partial"])
                        self.assertEqual(sec["implied_stable_reviews"], 0)
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
            for key in ("label_free", "m_counts", "captures", "Delta_per100", "existence", "size"):
                self.assertEqual(repr(ra[key]), repr(rb[key]), key)
            self.assertIsNone(ra["secondary"])
            self.assertEqual(rb["secondary"]["N_Y_per100"], 150)                # held at the primary N_Y
            if rb["secondary"]["plan_literal"] is not None:
                self.assertEqual(rb["secondary"]["plan_literal"]["G_hat_exact"],
                                 100 * rb["captures"]["Delta"] / 150)
        self.assertNotIn("N_Y =", sap.panel_report(a))
        self.assertIn("per-100 denominator N = 150 are held", sap.panel_report(a))
        self.assertIn("per-100 denominators N = 150 and N_Y = 150 are held", sap.panel_report(b))


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
        self.assertEqual((sec["N_Y"], sec["n1_Y"], sec["n0_Y"], sec["n_D0_Y"]), (110, 10, 80, 90))
        # (a) The first number is the primary rescaled by N/N_Y, with the primary's test.
        pl = sec["plan_literal"]
        self.assertEqual(pl["rescale"], Fraction(190, 110))
        self.assertEqual(pl["G_hat_exact"], 100 * Delta / 110)
        self.assertEqual(pl["existence"], {"test": "same test as the primary", "p": r["existence"]["p"],
                                           "reject": r["existence"]["reject"]})
        self.assertEqual((pl["delta_L"], pl["delta_U"]), (r["size"]["delta_L"], r["size"]["delta_U"]))
        self.assertAlmostEqual(pl["G_L"], 100 * float(L) / 110 * r["size"]["delta_L"], places=12)
        self.assertAlmostEqual(pl["G_U"], 100 * float(L) / 110 * r["size"]["delta_U"], places=12)
        # (b) The usable-only comparison.
        uo = sec["usable_only"]
        self.assertEqual(uo["L_Y"], L)                           # every unstable task has a usable label
        self.assertEqual(uo["G_hat_exact"], 0)                   # equal usable rates: no gain
        self.assertEqual(uo["size"]["delta_hat"], 0.0)
        self.assertEqual(uo["existence"]["p"], sap.fisher_one_sided(2, 10, 16, 80))
        self.assertFalse(uo["existence"]["reject"])
        self.assertLess(uo["size"]["delta_L"], 0.0)
        self.assertGreater(uo["size"]["delta_U"], 0.0)
        self.assertNotIn("class", uo["size"])                    # no threshold claim for the secondary
        self.assertNotIn("delta_m_Y", uo)
        # The primary shows a gain that comes only from the missing references among S.
        self.assertAlmostEqual(r["size"]["delta_hat"], 0.1, places=15)
        self.assertLess(r["existence"]["p"], uo["existence"]["p"])
        # Identity: 100 Delta / N_Y = G_hat_Y + (100 L / N_Y)(n1Y/n1 - n0Y/n0)(m0/n0Y).
        self.assertEqual(uo["usable_share_term"],
                         Fraction(100) * L / 110 * (1 - Fraction(1, 2)) * Fraction(16, 80))
        self.assertEqual(uo["G_hat_exact"] + uo["usable_share_term"], pl["G_hat_exact"])
        self.assertEqual(sec["B_Y"], L * (1 - Fraction(1, 2)))
        self.assertEqual(sec["implied_stable_reviews"], Fraction(29 * 90, 170) - 10)

        # Reverse: half of the unstable tasks are Unsure.
        labels, E, ref = panel(block("G1", rows(5, 20)), block("U1", rows(2, 5, 5)), block("S", rows(16, 160)))
        r = sap.analyze(labels, E, reference=ref)
        sec, lf = r["secondary"], r["label_free"]
        L, Delta = lf["L"], r["captures"]["Delta"]
        uo = sec["usable_only"]
        self.assertEqual((sec["N_Y"], sec["n1_Y"], sec["n0_Y"]), (185, 5, 160))
        self.assertEqual(sec["plan_literal"]["G_hat_exact"], 100 * Delta / 185)
        self.assertEqual(uo["L_Y"], L * Fraction(5, 10))
        dY = Fraction(2, 5) - Fraction(16, 160)
        self.assertEqual(uo["G_hat_exact"], Fraction(100) * uo["L_Y"] * dY / 185)
        self.assertAlmostEqual(uo["size"]["G_hat"], float(uo["G_hat_exact"]), places=12)
        self.assertEqual(uo["G_hat_exact"] + uo["usable_share_term"], 100 * Delta / 185)
        self.assertLess(uo["usable_share_term"], 0)
        self.assertEqual(uo["existence"]["p"], sap.fisher_one_sided(2, 5, 16, 160))
        # The coverage-regime (b) p equals the permutation p of its own statistic.
        nY = {"U2": 0, "U1": 5, "S": 160}
        perm = sap.permutation_p(sap.Design.from_labels(labels), r["m_counts"], sizes=nY, weights=uo["weights"])
        self.assertEqual(perm, sap.fisher_one_sided(2, 5, 16, 160, exact=True))
        # The (b) weights are budget-balanced on the tasks with a usable label.
        self.assertEqual(sum(uo["weights"][j] * nY[j] for j in nY), 0)
        txt = sap.report(r)
        self.assertIn("SECONDARY RESULT: gain per 100 tasks with a usable label (N_Y = 185 of N = 190)", txt)
        self.assertLess(txt.index("EXISTENCE QUESTION"), txt.index("SECONDARY RESULT"))
        self.assertIn("(a) = (b) - ", txt)

    def test_all_labels_unsure_in_one_stratum(self):
        # All stable agreeing tasks Unsure.
        labels, E, ref = panel(block("I", rows(0, 2)), block("G1", rows(5, 20)), block("U1", rows(3, 10)),
                               block("S", rows(0, 0, 160)))
        r = sap.analyze(labels, E, reference=ref)
        self.assertEqual(r["label_free"]["regime"], "COVERAGE")
        self.assertEqual(r["existence"]["p"], sap.fisher_one_sided(3, 10, 0, 160))
        self.assertIsNotNone(r["size"])
        sec = r["secondary"]
        self.assertIsNone(sec["not_estimable"])                  # (a) needs only N_Y > 0
        self.assertEqual(sec["N_Y"], 32)
        self.assertEqual(sec["plan_literal"]["G_hat_exact"], 100 * r["captures"]["Delta"] / 32)
        self.assertEqual(sec["usable_only_omitted"], "no stable agreeing task has a usable label")
        self.assertIsNone(sec["usable_only"])
        self.assertEqual(r["coverage"]["groups"]["stable agreeing"],
                         {"tasks": 160, "usable": 0, "unsure": 160, "missing": 0})
        txt = sap.report(r)
        self.assertIn("Not reported: no stable agreeing task has a usable label", txt)
        self.assertIn("(a) First number", txt)
        # All unstable agreeing tasks Unsure: m1 is 0 by construction.
        labels, E, ref = panel(block("G1", rows(5, 20)), block("U2", rows(0, 0, 4)), block("U1", rows(0, 0, 6)),
                               block("S", rows(16, 160)))
        r = sap.analyze(labels, E, reference=ref)
        self.assertEqual((r["m1"], r["existence"]["p"]), (0, 1.0))
        self.assertEqual(r["secondary"]["usable_only_omitted"], "no unstable agreeing task has a usable label")
        # Every label Unsure or missing.
        labels = labels_for(0, 20, 0, 10, 160)
        ref = [UNS] * 100 + [MIS] * 90
        r = sap.analyze(labels, [0] * 190, reference=ref)
        self.assertEqual(r["secondary"]["N_Y"], 0)
        self.assertIsNone(r["secondary"]["plan_literal"])
        self.assertIsNone(r["secondary"]["usable_only"])
        self.assertEqual(r["secondary"]["not_estimable"], "no panel task has a usable label")
        self.assertEqual(r["captures"]["Delta"], 0)
        self.assertIn("Not estimable: no panel task has a usable label", sap.report(r))
        # All invalid-first-verdict tasks Unsure: only N_Y moves.
        labels, E, ref = panel(block("I", rows(0, 0, 6)), block("G1", rows(5, 20)), block("U1", rows(2, 10)),
                               block("S", rows(16, 160)))
        r = sap.analyze(labels, E, reference=ref)
        sec = r["secondary"]
        uo = sec["usable_only"]
        self.assertEqual((sec["N_Y"], sec["n1_Y"], sec["n0_Y"], sec["B_Y"]), (190, 10, 160, 0))
        self.assertEqual(uo["existence"]["p"], r["existence"]["p"])
        self.assertEqual(sec["plan_literal"]["G_hat_exact"], 100 * r["captures"]["Delta"] / 190)
        self.assertEqual(uo["G_hat_exact"], Fraction(100) * r["label_free"]["L"]
                         * (Fraction(2, 10) - Fraction(16, 160)) / 190)

    def test_secondary_in_the_partial_regime(self):
        # N = 26, s = 4 < n1 = 9: partial regime, with Unsure tasks in every agreeing stratum.
        labels, E, ref = panel(block("G1", rows(2, 5)), block("U2", rows(1, 2, 1)), block("U1", rows(1, 4, 2)),
                               block("S", rows(2, 8, 4)))
        r = sap.analyze(labels, E, reference=ref, boot_B=2000)
        self.assertEqual(r["label_free"]["regime"], "PARTIAL")
        sec = r["secondary"]
        self.assertEqual(sec["N_Y"], 19)
        nY = {"U2": 2, "U1": 4, "S": 8}
        self.assertEqual({j: sec["n_Y"][j] for j in nY}, nY)
        # (a) The primary's partial-regime result rescaled by N/N_Y = 26/19.
        pl, pb = sec["plan_literal"], r["size"]["partial"]
        self.assertEqual(pl["rescale"], Fraction(26, 19))
        for k in ("G_partial_hat", "lower", "upper"):
            self.assertEqual(pl["partial"][k], float(Fraction(26, 19)) * pb[k], k)
        self.assertEqual(pl["G_hat_exact"], 100 * r["captures"]["Delta"] / 19)
        self.assertAlmostEqual(pl["partial"]["G_partial_hat"], pl["G_hat"], places=12)
        self.assertEqual(pl["existence"]["p"], r["existence"]["p"])
        # (b) Here B_Y = 0: instability-first reviews as many usable agreeing tasks as
        # disagreement-first does, so the implied comparator reviews no stable usable task.
        w = sap.delta_weights(sap.Design.from_labels(labels))
        B_Y = sum(w[j] * nY[j] for j in nY)
        self.assertEqual(sec["B_Y"], B_Y)
        self.assertEqual(B_Y, 0)
        self.assertEqual(sec["implied_stable_reviews"], 0)
        uo = sec["usable_only"]
        self.assertEqual(uo["weights"], {"U2": w["U2"], "U1": w["U1"], "S": w["S"] - B_Y / 8})
        # Brute force: every way to place the 4 discordant verdicts among the 14 usable agreeing tasks.
        strata = ["U2"] * 2 + ["U1"] * 4 + ["S"] * 8
        ws = [uo["weights"][j] for j in strata]
        obs = uo["weights"]["U2"] * 1 + uo["weights"]["U1"] * 1 + uo["weights"]["S"] * 2
        hits = total = 0
        for chosen in itertools.combinations(range(14), 4):
            total += 1
            hits += sum(ws[i] for i in chosen) >= obs
        self.assertEqual(sap.permutation_p(sap.Design.from_labels(labels), r["m_counts"], sizes=nY,
                                           weights=uo["weights"]), Fraction(hits, total))
        self.assertEqual(uo["existence"]["p"], hits / total)
        self.assertEqual(sum(uo["weights"][j] * nY[j] for j in nY), 0)
        b = uo["size"]["partial"]
        self.assertAlmostEqual(b["G_partial_hat"], float(uo["G_hat_exact"]), places=12)
        self.assertEqual(uo["G_hat_exact"], Fraction(100, 19) * sum(uo["weights"][j] * r["m_counts"][j] for j in nY))
        self.assertEqual(uo["G_hat_exact"] + uo["usable_share_term"], pl["G_hat_exact"])
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
        for needle in ("REFERENCE LABEL COVERAGE: Unsure and missing labels", "unknown, not correct verdicts",
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

    def test_template_panel_is_a_labeled_sensitivity_subset(self):
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


def random_slots(n, seed):
    """Synthetic resolved slots, discordance and ids for an n-task panel."""
    rng = np.random.default_rng(seed)
    slots, E, ids, tmpl = [], [], [], []
    for i in range(n):
        a1 = "success" if rng.random() < 0.55 else "failure"
        flip = "failure" if a1 == "success" else "success"
        r = rng.random()
        a2, a3 = (flip, a1) if r < 0.06 else ((flip, flip) if r < 0.08 else (a1, a1))
        slots.append({"A1": a1, "A2": a2, "A3": a3, "B1": flip if rng.random() < 0.15 else a1})
        E.append(int(rng.random() < (0.45 if a2 != a1 or a3 != a1 else 0.08)))
        ids.append(f"task-{i:04d}")
        tmpl.append(f"tpl-{i // 2}")
    return slots, E, ids, tmpl


class TestReferenceNotSupplied(unittest.TestCase):
    def test_report_says_statuses_were_not_supplied(self):
        labels, E, ref = panel(block("G1", rows(5, 20)), block("U1", rows(2, 10)), block("S", rows(16, 80, 80)))
        bare = sap.analyze(labels, E)
        self.assertFalse(bare["reference_supplied"])
        self.assertIsNone(bare["secondary"])
        self.assertIsNone(bare["coverage"])
        self.assertEqual(sap.REFERENCE_NOT_SUPPLIED_TEXT,
                         "Reference statuses not supplied: every label treated as usable; "
                         "Unsure and missing coverage not assessed")
        txt = sap.report(bare)
        self.assertEqual(txt.count(sap.REFERENCE_NOT_SUPPLIED_TEXT), 1)
        for absent in ("SECONDARY RESULT", "REFERENCE LABEL COVERAGE", "Unsure 0", "missing 0", "N_Y",
                       "count as not reference-contradicted", "group | tasks"):
            self.assertNotIn(absent, txt, absent)
        full = sap.report(sap.analyze(labels, E, reference=ref))
        self.assertNotIn(sap.REFERENCE_NOT_SUPPLIED_TEXT, full)
        for present in ("SECONDARY RESULT", "REFERENCE LABEL COVERAGE", "(Unsure 80, missing 0)"):
            self.assertIn(present, full, present)

    def test_panel_and_template_panel_without_statuses(self):
        slots, E, ids, tmpl = random_slots(120, 8)
        runs = [{"run_id": t, "task_id": t, "template_id": g, "passes": True} for t, g in zip(ids, tmpl)]
        tp = sap.select_panel(runs, 2, unit="template")
        out = sap.analyze_panel(slots, E, ids=ids, templates=tmpl, template_panel=tp, seed=2, boot_B=200)
        self.assertIsNone(out["primary"]["secondary"])
        self.assertIsNone(out["template_panel"]["result"]["secondary"])
        self.assertTrue(all(r["secondary"] is None for _, r in out["sensitivities"]))
        txt = sap.panel_report(out)
        self.assertEqual(txt.count(sap.REFERENCE_NOT_SUPPLIED_TEXT), 1)
        for absent in ("N_Y", "SECONDARY RESULT", "REFERENCE LABEL COVERAGE", "secondary (per 100"):
            self.assertNotIn(absent, txt, absent)
        ref = [U] * 120
        out = sap.analyze_panel(slots, E, ids=ids, templates=tmpl, template_panel=tp, seed=2, boot_B=200,
                                reference=ref)
        self.assertEqual(out["template_panel"]["result"]["secondary"]["N_Y"], len(tp))
        self.assertIn("N_Y = 120", sap.panel_report(out))

    def test_by_benchmark_not_supplied(self):
        labels, E, ref = panel(block("G1", rows(5, 20)), block("U1", rows(2, 10)), block("S", rows(16, 80, 80)))
        txt = sap.report(sap.analyze(labels, E, reference=ref))
        self.assertIn("  By benchmark: not supplied", txt)
        bench = ["webarena" if i % 3 else "workarena" for i in range(len(labels))]
        txt = sap.report(sap.analyze(labels, E, reference=ref, benchmarks=bench))
        self.assertNotIn("By benchmark: not supplied", txt)
        self.assertIn("benchmark | tasks | usable | Unsure | missing", txt)


class TestSecondaryResult(unittest.TestCase):
    @staticmethod
    def section(txt):
        return txt[txt.index("SECONDARY RESULT"):txt.index("REFERENCE LABEL COVERAGE")]

    def check_layout(self, r):
        txt = sap.report(r)
        sec = self.section(txt)
        self.assertLess(sec.index("(a) First number: 100 Delta / N_Y, the primary realized gain "
                                  "rescaled by N/N_Y"),
                        sec.index("(b) Usable-only comparison: a proposed analysis, pending confirmation"))
        self.assertIn("Existence: same test as the primary", sec)
        self.assertEqual(txt.count("The threshold of 2 per 100 refers to the primary result only"), 1)
        self.assertIn("The threshold of 2 per 100 refers to the primary result only", sec)
        for word in ("exceeds", "below the threshold", "unresolved", "size:", "δ_m", "delta_m",
                     "reachable", "class"):
            self.assertNotIn(word, sec, word)
        return sec

    def test_plan_literal_first_and_no_threshold_claim(self):
        labels, E, ref = panel(block("G1", rows(5, 20)), block("U1", rows(2, 5, 5)), block("S", rows(16, 160)))
        r = sap.analyze(labels, E, reference=ref)
        sec = self.check_layout(r)
        self.assertIn("G_hat_Y =", sec)
        labels, E, ref = panel(block("G1", rows(2, 5)), block("U2", rows(1, 2, 1)), block("U1", rows(1, 4, 2)),
                               block("S", rows(2, 8, 4)))
        r = sap.analyze(labels, E, reference=ref, boot_B=500)
        self.assertEqual(r["label_free"]["regime"], "PARTIAL")
        sec = self.check_layout(r)
        self.assertIn("G_partial_hat_Y =", sec)

    def test_report_defines_the_comparator_and_names_each_regime_gain(self):
        definition = ("It compares disagreement-first review with an implied comparator, which reviews "
                      "unstable agreeing tasks with a usable label as instability-first review does, then "
                      "stable ones until it has reviewed as many agreeing tasks with a usable label as "
                      "disagreement-first review, s n_D0Y / n_D0 = {}. B_Y = {} is how many more agreeing "
                      "tasks with a usable label instability-first review reviews than disagreement-first "
                      "review (exact, tie-averaged; negative means fewer).")
        # Coverage, reported: s = 29, n_D0 = 170, n_D0Y = 170 - 5 Unsure.
        labels, E, ref = panel(block("G1", rows(5, 20)), block("U1", rows(2, 5, 5)), block("S", rows(16, 160)))
        r = sap.analyze(labels, E, reference=ref)
        sec, s2 = r["secondary"], self.section(sap.report(r))
        self.assertIsNotNone(sec["usable_only"])
        self.assertIn(definition.format(sap._f(Fraction(29 * 165, 170)), sap._f(sec["B_Y"])), s2)
        self.assertLess(s2.index("It compares disagreement-first review"), s2.index("The implied comparator reviews"))
        self.assertIn("which G_hat_Y removes", s2)
        self.assertNotIn("G_partial_hat_Y", s2)
        # Partial regime, reported: the decomposition names the partial-regime gain.
        labels, E, ref = panel(block("G1", rows(2, 5)), block("U2", rows(1, 1, 2)), block("U1", rows(1, 6)),
                               block("S", rows(2, 12)))
        r = sap.analyze(labels, E, reference=ref, boot_B=500)
        s2 = self.section(sap.report(r))
        self.assertIsNotNone(r["secondary"]["usable_only"])
        self.assertIn("which G_partial_hat_Y removes", s2)
        self.assertNotIn("which G_hat_Y removes", s2)
        # Partial regime, omitted because B_Y > 0: B_Y is defined before the reason uses it.
        labels, E, ref = panel(block("G1", rows(2, 5)), block("U2", rows(1, 2, 1)), block("U1", rows(1, 4, 2)),
                               block("S", rows(1, 2, 10)))
        r = sap.analyze(labels, E, reference=ref, boot_B=500)
        s2 = self.section(sap.report(r))
        self.assertIsNone(r["secondary"]["usable_only"])
        self.assertLess(s2.index("B_Y = 1.1429 is how many more"), s2.index("Not reported: "))
        # No usable unstable label: nothing to define.
        labels, E, ref = panel(block("G1", rows(0, 10)), block("U1", rows(0, 0, 8)), block("S", rows(5, 80)))
        s2 = self.section(sap.report(sap.analyze(labels, E, reference=ref)))
        self.assertIn("Not reported: no unstable agreeing task has a usable label.", s2)
        self.assertNotIn("implied comparator", s2)

    def test_usable_only_omitted_when_its_comparator_is_negative_in_coverage(self):
        # N = 200, n_D0 = 150, s = 30, n1Y = 10: the implied comparator reviews
        # s n_D0Y / n_D0 - n1Y = n_D0Y / 5 - 10 stable usable tasks.
        def make(n0Y):
            return panel(block("G1", rows(10, 50)), block("U1", rows(4, 10)), block("S", rows(5, n0Y, 140 - n0Y)))
        labels, E, ref = make(40)                                 # exactly zero: reported
        r = sap.analyze(labels, E, reference=ref)
        sec = r["secondary"]
        self.assertEqual((r["label_free"]["s"], r["label_free"]["n_D0"], r["label_free"]["regime"]),
                         (30, 150, "COVERAGE"))
        self.assertEqual(sec["implied_stable_reviews"], 0)
        self.assertIsNone(sec["usable_only_omitted"])
        self.assertIsNotNone(sec["usable_only"])
        labels, E, ref = make(39)                                 # 9.8 - 10 < 0: not reported
        r = sap.analyze(labels, E, reference=ref)
        sec = r["secondary"]
        self.assertEqual(sec["implied_stable_reviews"], Fraction(30 * 49, 150) - 10)
        self.assertIsNone(sec["usable_only"])
        self.assertIsNotNone(sec["plan_literal"])                 # (a) is still reported
        why = ("its implied comparator would review -0.2000 stable agreeing tasks with a usable label, "
               "a negative number (n1Y = 10 > s n_D0Y / n_D0 = 9.8000)")
        self.assertEqual(sec["usable_only_omitted"], why)
        txt = sap.report(r)
        self.assertIn("Not reported: " + why, txt)
        self.assertNotIn("G_hat_Y =", txt)
        self.assertIn("(a) First number", txt)
        self.assertTrue(any("(b) proposed usable-only comparison: not reported, " + why in x
                            for x in sap._brief("x", r)))

    def test_usable_only_in_the_partial_regime(self):
        # N = 26, s = 4: instability-first reviews the 3 U2 tasks and 1 of the 6 U1 tasks.
        def run(u2, u1, s_rows):
            labels, E, ref = panel(block("G1", rows(2, 5)), block("U2", rows(*u2)), block("U1", rows(*u1)),
                                   block("S", rows(*s_rows)))
            r = sap.analyze(labels, E, reference=ref, boot_B=500)
            self.assertEqual(r["label_free"]["regime"], "PARTIAL")
            return r, labels
        # B_Y > 0: most stable tasks are Unsure; the comparator would review -B_Y < 0 of them.
        r, _ = run((1, 2, 1), (1, 4, 2), (1, 2, 10))
        sec = r["secondary"]
        self.assertEqual(sec["B_Y"], Fraction(8, 7))
        self.assertEqual(sec["implied_stable_reviews"], -Fraction(8, 7))
        self.assertIsNone(sec["usable_only"])
        self.assertEqual(sec["usable_only_omitted"], "its implied comparator would review -1.1429 stable "
                         "agreeing tasks with a usable label, a negative number (B_Y = 1.1429 > 0)")
        self.assertEqual(sec["plan_literal"]["G_hat_exact"], 100 * r["captures"]["Delta"] / 13)
        self.assertNotIn("G_partial_hat_Y", sap.report(r))
        # B_Y < 0: the comparator reviews 34/21 of the 12 stable usable tasks; reported.
        r, labels = run((1, 1, 2), (1, 6), (2, 12))
        sec = r["secondary"]
        self.assertEqual(sec["B_Y"], -Fraction(34, 21))
        self.assertEqual(sec["implied_stable_reviews"], Fraction(34, 21))
        uo = sec["usable_only"]
        nY = {"U2": 1, "U1": 6, "S": 12}
        self.assertEqual(sum(uo["weights"][j] * nY[j] for j in nY), 0)
        strata = ["U2"] * 1 + ["U1"] * 6 + ["S"] * 12
        ws = [uo["weights"][j] for j in strata]
        obs = sum(uo["weights"][j] * r["m_counts"][j] for j in nY)
        combos = list(itertools.combinations(range(19), sum(r["m_counts"][j] for j in nY)))
        hits = sum(sum(ws[i] for i in c) >= obs for c in combos)
        self.assertEqual(uo["existence"]["p"], hits / len(combos))
        self.assertEqual(uo["G_hat_exact"] + uo["usable_share_term"], sec["plan_literal"]["G_hat_exact"])
        self.assertIn("G_partial_hat_Y =", sap.report(r))
        # More stable usable tasks than exist: N = 60, s = 9, all 8 U2 tasks Unsure, 1 of 26 S usable.
        labels, E, ref = panel(block("G1", rows(1, 6)), block("U2", rows(0, 0, 8)), block("U1", rows(2, 20)),
                               block("S", rows(0, 1, 25)))
        r = sap.analyze(labels, E, reference=ref, boot_B=500)
        sec = r["secondary"]
        self.assertEqual((r["label_free"]["s"], r["label_free"]["regime"]), (9, "PARTIAL"))
        self.assertEqual(sec["implied_stable_reviews"], Fraction(5, 2))
        self.assertIsNone(sec["usable_only"])
        self.assertEqual(sec["usable_only_omitted"], "its implied comparator would review 2.5000 stable "
                         "agreeing tasks with a usable label, more than the n0Y = 1 there are")

    def test_implied_comparator_algebra(self):
        # The usable-only gain is the implied comparator's capture minus disagreement-first's,
        # per 100 tasks with a usable label, at equal review counts among usable agreeing tasks.
        rng = np.random.default_rng(41)
        done = reported = 0
        while done < 150:
            d = sap.Design(int(rng.integers(0, 3)), int(rng.integers(0, 15)), int(rng.integers(0, 6)),
                           int(rng.integers(0, 10)), int(rng.integers(1, 40)))
            if d.regime not in ("COVERAGE", "PARTIAL"):
                continue
            labels = labels_for(d.n_inv, d.n_D1, d.n_U2, d.n_U1, d.n_S)
            ref = [U if rng.random() < 0.7 else UNS for _ in labels]
            E = [int(s == U and l != "I" and rng.random() < 0.25) for l, s in zip(labels, ref)]
            res = sap.analyze(labels, E, reference=ref, boot_B=50, fragility=False)
            sec = res["secondary"]
            if sec["not_estimable"] is not None:
                continue
            nY, m = sec["n_Y"], res["m_counts"]
            pi = dict(zip(sap.STRATA, sap.review_fractions([d.n_inv, d.n_D1, d.n_U2, d.n_U1, d.n_S], d.K)))
            dis = Fraction(d.s, d.n_D0)
            implied = dis * sec["n_D0_Y"] - pi["U2"] * nY["U2"] - pi["U1"] * nY["U1"]
            self.assertEqual(sec["implied_stable_reviews"], implied)
            if d.regime == "COVERAGE":
                self.assertEqual(implied, Fraction(d.s * sec["n_D0_Y"], d.n_D0) - sec["n1_Y"])
                self.assertLessEqual(implied, sec["n0_Y"])
            else:
                self.assertEqual(implied, -sec["B_Y"])
            feasible = sec["n1_Y"] > 0 and sec["n0_Y"] > 0 and 0 <= implied <= sec["n0_Y"]
            self.assertEqual(sec["usable_only"] is not None, feasible)
            if feasible:
                f_S = implied / sec["n0_Y"]
                diff = (pi["U2"] - dis) * m["U2"] + (pi["U1"] - dis) * m["U1"] + (f_S - dis) * m["S"]
                self.assertEqual(sec["usable_only"]["G_hat_exact"], 100 * diff / sec["N_Y"])
                reviewed = pi["U2"] * nY["U2"] + pi["U1"] * nY["U1"] + implied
                self.assertEqual(reviewed, dis * sec["n_D0_Y"])
                reported += 1
            done += 1
        self.assertGreater(reported, 50)
        self.assertLess(reported, 150)


NOT_DEFINED = ("Fragility: not defined; no set of usable reference-label flips among agreeing tasks "
               "reverses this decision.")


def full_search(d, m, nY):
    """Fewest flips (None above FRAGILITY_MAX) and whether any reverses, over every usable count."""
    def p(x2, x1, xS):
        if d.regime == "COVERAGE":
            return sap.fisher_one_sided(x2 + x1, d.n1, xS, d.n0, exact=True)
        return sap.permutation_p(d, {"U2": x2, "U1": x1, "S": xS})
    reject = p(m["U2"], m["U1"], m["S"]) <= Fraction(1, 40)
    best = None
    for x in itertools.product(range(nY["U2"] + 1), range(nY["U1"] + 1), range(nY["S"] + 1)):
        if (p(*x) <= Fraction(1, 40)) != reject:
            k = abs(x[0] - m["U2"]) + abs(x[1] - m["U1"]) + abs(x[2] - m["S"])
            best = k if best is None else min(best, k)
    return reject, (best if best is not None and best <= sap.FRAGILITY_MAX else None), best is not None


class TestFragilityUsableLabels(unittest.TestCase):
    def test_no_usable_label_to_flip(self):
        labels = labels_for(0, 20, 0, 10, 160)
        r = sap.analyze(labels, [0] * 190, reference=[UNS] * 100 + [MIS] * 90)
        ex = r["existence"]
        self.assertFalse(ex["reject"])
        self.assertIsNone(ex["fragility"])
        self.assertFalse(ex["fragility_defined"])
        txt = sap.report(r)
        self.assertIn(NOT_DEFINED, txt)
        self.assertNotIn("would reverse this decision", txt)
        # The same discordance with every label usable: two flips in U reverse the decision.
        bare = sap.analyze(labels, [0] * 190)
        self.assertEqual((bare["existence"]["fragility"], bare["existence"]["fragility_defined"]), (2, True))
        self.assertIn("Fragility: 2 usable reference-label flips among agreeing tasks", sap.report(bare))

    def test_flips_stay_within_usable_labels(self):
        # U1: one discordant and one concordant usable label, 8 Unsure; S: 6 of 160 discordant.
        labels, E, ref = panel(block("G1", rows(5, 20)), block("U1", rows(1, 2, 8)), block("S", rows(6, 160)))
        r = sap.analyze(labels, E, reference=ref)
        lf, ex = r["label_free"], r["existence"]
        self.assertEqual((lf["regime"], lf["n1"], lf["n0"]), ("COVERAGE", 10, 160))
        self.assertFalse(ex["reject"])

        def brute(n1Y):
            for k in range(1, sap.FRAGILITY_MAX + 1):
                for fu in range(k + 1):
                    a, c = 1 + fu, 6 - (k - fu)
                    if a <= n1Y and c >= 0 and sap.fisher_one_sided(a, 10, c, 160, exact=True) <= Fraction(1, 40):
                        return k
            return None
        self.assertEqual((ex["fragility"], brute(2)), (5, 5))     # only one usable label in U can flip
        self.assertTrue(ex["fragility_defined"])
        everyone = sap.existence_fragility(sap.Design.from_labels(labels), r["m_counts"], False)
        self.assertEqual((everyone, brute(10)), (2, 2))           # flipping Unsure tasks would give 2
        self.assertIn("Fragility: 5 usable reference-label flips", sap.report(r))

    def test_partial_regime_flips_stay_within_usable_labels(self):
        labels, E, ref = panel(block("G1", rows(2, 5)), block("U2", rows(0, 1, 2)), block("U1", rows(0, 2, 4)),
                               block("S", rows(3, 12)))
        r = sap.analyze(labels, E, reference=ref, boot_B=200)
        self.assertEqual(r["label_free"]["regime"], "PARTIAL")
        ex, d = r["existence"], sap.Design.from_labels(labels)
        # One usable U2 label and two usable U1 labels, all concordant, cannot carry a rejection.
        self.assertFalse(ex["reject"])
        self.assertEqual(full_search(d, r["m_counts"], {"U2": 1, "U1": 2, "S": 12}), (False, None, False))
        self.assertEqual((ex["fragility"], ex["fragility_defined"]), (None, False))
        self.assertIn(NOT_DEFINED, sap.report(r))
        # Flipping the Unsure tasks as well would give 3.
        everyone = {"U2": 3, "U1": 6, "S": 12}
        self.assertEqual(full_search(d, r["m_counts"], everyone), (False, 3, True))
        self.assertEqual(sap.existence_fragility(d, r["m_counts"], False), 3)

    def test_partial_regime_search_tries_every_direction(self):
        # N = 20, s = 3: instability-first reviews 3 of the 5 U2 tasks, so the weights of
        # U2, U1 and S are 2/5, -1/5 and -1/5. Rejection needs fewer discordant U1 tasks.
        labels = labels_for(0, 5, 5, 7, 3)
        E = E_for(G1=(5, 0), U2=(5, 1), U1=(7, 4), S=(3, 1))
        r = sap.analyze(labels, E, boot_B=50)
        d, ex = sap.Design.from_labels(labels), r["existence"]
        self.assertEqual((d.regime, d.s, d.n_D0), ("PARTIAL", 3, 15))
        self.assertEqual(sap.delta_weights(d), {"U2": Fraction(2, 5), "U1": Fraction(-1, 5), "S": Fraction(-1, 5)})
        self.assertFalse(ex["reject"])
        nY = {"U2": 5, "U1": 7, "S": 3}
        self.assertEqual(full_search(d, r["m_counts"], nY), (False, 6, True))
        self.assertLessEqual(sap.permutation_p(d, {"U2": 5, "U1": 2, "S": 1}), Fraction(1, 40))
        self.assertEqual(ex["fragility"], 6)
        self.assertIn("Fragility: 6 usable reference-label flips among agreeing tasks would reverse "
                      "this decision.", sap.report(r))
        # Moving every count only the way it moves in the coverage regime finds nothing.
        one_way = [k for k in range(1, sap.FRAGILITY_MAX + 1) for f2 in range(k + 1) for f1 in range(k - f2 + 1)
                   if 1 + f2 <= 5 and 4 + f1 <= 7 and 0 <= 1 - (k - f2 - f1)
                   and sap.permutation_p(d, {"U2": 1 + f2, "U1": 4 + f1, "S": 1 - (k - f2 - f1)}) <= Fraction(1, 40)]
        self.assertEqual(one_way, [])

    def test_not_defined_when_no_set_of_flips_reverses(self):
        # Coverage, every unstable label Unsure, 5 of 80 usable stable labels discordant:
        # m1 stays 0 and p stays 1 whatever usable label is flipped.
        labels, E, ref = panel(block("G1", rows(0, 10)), block("U2", rows(0, 0, 3)), block("U1", rows(0, 0, 5)),
                               block("S", rows(5, 80)))
        r = sap.analyze(labels, E, reference=ref, boot_B=50)
        ex = r["existence"]
        self.assertEqual((r["label_free"]["regime"], r["m1"], r["m0"]), ("COVERAGE", 0, 5))
        self.assertEqual((ex["p"], ex["reject"], ex["fragility"], ex["fragility_defined"]), (1.0, False, None, False))
        txt = sap.report(r)
        self.assertIn(NOT_DEFINED, txt)
        self.assertNotIn("more than 15", txt)
        # One usable unstable label: the best reachable table, m1 = 1 of 8 and m0 = 0 of 80,
        # gives p = 1/11, so no set of flips reverses the decision.
        labels, E, ref = panel(block("G1", rows(0, 10)), block("U1", rows(0, 1, 7)), block("S", rows(0, 80)))
        r = sap.analyze(labels, E, reference=ref, boot_B=50)
        ex = r["existence"]
        self.assertEqual(sap.fisher_one_sided(1, 8, 0, 80, exact=True), Fraction(1, 11))
        self.assertEqual((ex["fragility"], ex["fragility_defined"]), (None, False))
        self.assertIn(NOT_DEFINED, sap.report(r))
        # Every label usable: one unstable task among 31 agreeing tasks gives at best p = 1/31.
        r = sap.analyze(labels_for(0, 4, 0, 1, 30), [0] * 35)
        ex = r["existence"]
        self.assertEqual(r["label_free"]["regime"], "COVERAGE")
        self.assertEqual((ex["fragility"], ex["fragility_defined"]), (None, False))
        self.assertIn(NOT_DEFINED, sap.report(r))
        # Partial regime: every unstable label Unsure, 3 of 10 usable stable labels discordant.
        labels, E, ref = panel(block("G1", rows(0, 13)), block("U2", rows(0, 0, 10)), block("S", rows(3, 10)))
        r = sap.analyze(labels, E, reference=ref, boot_B=50)
        ex = r["existence"]
        self.assertEqual(r["label_free"]["regime"], "PARTIAL")
        self.assertEqual((ex["reject"], ex["fragility"], ex["fragility_defined"]), (False, None, False))
        self.assertIn(NOT_DEFINED, sap.report(r))

    def test_more_than_the_search_depth_only_when_reversible(self):
        # All 20 unstable tasks discordant, no stable one: more than 15 flips are needed.
        labels = labels_for(0, 10, 0, 20, 150)
        r = sap.analyze(labels, E_for(U1=(20, 20), S=(150, 0), G1=(10, 0)), boot_B=50)
        ex = r["existence"]
        self.assertEqual(r["label_free"]["regime"], "COVERAGE")
        self.assertEqual((ex["reject"], ex["fragility"], ex["fragility_defined"]), (True, None, True))
        self.assertIn("Fragility: more than 15 usable reference-label flips among agreeing tasks would "
                      "reverse this decision.", sap.report(r))

    def test_count_and_reversibility_match_a_full_search(self):
        rng = np.random.default_rng(2718)
        seen = {"COVERAGE": 0, "PARTIAL": 0, "not reversible": 0, "beyond the depth": 0}
        done = 0
        while done < 150:
            d = sap.Design(int(rng.integers(0, 3)), int(rng.integers(0, 8)), int(rng.integers(0, 6)),
                           int(rng.integers(0, 8)), int(rng.integers(0, 22)))
            if d.regime not in ("COVERAGE", "PARTIAL"):
                continue
            full = {"U2": d.n_U2, "U1": d.n_U1, "S": d.n_S}
            nY = dict(full) if rng.random() < 0.3 else {j: int(rng.integers(0, n + 1)) for j, n in full.items()}
            m = {j: int(rng.integers(0, nY[j] + 1)) for j in full}
            reject, want, reversible = full_search(d, m, nY)
            counts = dict(m, G1=0)
            self.assertEqual(sap.existence_fragility(d, counts, reject, usable=nY), want)
            self.assertEqual(sap.fragility_reversible(d, counts, reject, usable=nY), reversible)
            if d.regime == "PARTIAL":
                tails = sap._PermutationTails(d)
                for x in itertools.product(range(full["U2"] + 1), range(full["U1"] + 1), range(full["S"] + 1)):
                    want_x = sap.permutation_p(d, dict(zip(("U2", "U1", "S"), x))) <= Fraction(1, 40)
                    self.assertEqual(tails.rejects(x), want_x)
            seen[d.regime] += 1
            seen["not reversible"] += not reversible
            seen["beyond the depth"] += reversible and want is None
            done += 1
        self.assertTrue(all(v > 0 for v in seen.values()), seen)


if __name__ == "__main__":
    unittest.main(verbosity=2)
