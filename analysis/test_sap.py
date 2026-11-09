"""Unit tests for sap.py. Synthetic data only.

Run from the repository root: python -m unittest discover -s analysis -v
"""
import itertools
import math
import os
import sys
import unittest
from fractions import Fraction

import numpy as np
from scipy import optimize, stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sap  # noqa: E402


# The population-inference path runs only when asked for explicitly.
def analyze(*args, **kw):
    return sap.analyze(*args, path=sap.PATH_POPULATION, **kw)


def analyze_panel(*args, **kw):
    return sap.analyze_panel(*args, path=sap.PATH_POPULATION, **kw)


def analyze_with_sensitivities(*args, **kw):
    return sap.analyze_with_sensitivities(*args, path=sap.PATH_POPULATION, **kw)


def labels_for(n_inv=0, n_D1=0, n_U2=0, n_U1=0, n_S=0):
    return ["I"] * n_inv + ["G1"] * n_D1 + ["U2"] * n_U2 + ["U1"] * n_U1 + ["S"] * n_S


def E_for(n_inv=(0, 0), G1=(0, 0), U2=(0, 0), U1=(0, 0), S=(0, 0)):
    """Each argument is (size, discordant count); discordant items come first."""
    out = []
    for n, m in (n_inv, G1, U2, U1, S):
        out += [1] * m + [0] * (n - m)
    return out


def mn_closed_form_p0(x1, n1, x0, n0, d):
    """Farrington-Manning (1990) closed-form restricted MLE, group 1 = (x1, n1)."""
    th = n0 / n1
    a = 1 + th
    p1h, p0h = x1 / n1, x0 / n0
    b = -(1 + th + p1h + th * p0h + d * (th + 2))
    c = d * d + d * (2 * p1h + th + 1) + p1h + th * p0h
    dd = -p1h * d * (1 + d)
    v = b ** 3 / (27 * a ** 3) - b * c / (6 * a ** 2) + dd / (2 * a)
    u = math.copysign(math.sqrt(b ** 2 / (9 * a ** 2) - c / (3 * a)), v)
    w = (math.pi + math.acos(max(-1.0, min(1.0, v / u ** 3)))) / 3
    p1 = 2 * u * math.cos(w) - b / (3 * a)
    return p1 - d


def loglik_p0_numeric(x1, n1, x0, n0, d):
    lo, hi = max(0.0, -d), min(1.0, 1.0 - d)

    def nll(p0):
        p1 = p0 + d
        t = 0.0
        for x, n, p in ((x1, n1, p1), (x0, n0, p0)):
            if x > 0:
                t -= x * math.log(max(p, 1e-300))
            if n - x > 0:
                t -= (n - x) * math.log(max(1 - p, 1e-300))
        return t
    r = optimize.minimize_scalar(nll, bounds=(lo, hi), method="bounded", options={"xatol": 1e-13})
    cands = [lo, hi, r.x]
    return min(cands, key=nll)


class TestBudgetAndRegimes(unittest.TestCase):
    def test_round_half_up(self):
        # 0.15 N = x.5 cases: round half up, never banker's rounding.
        cases = {10: 2, 30: 5, 50: 8, 70: 11, 190: 29, 210: 32, 230: 35, 250: 38, 270: 41}
        for N, s in cases.items():
            self.assertEqual(Fraction(3, 20) * N % 1, Fraction(1, 2))
            self.assertEqual(sap.slice_nominal(N), s)
        self.assertEqual(round(34.5), 34)            # Python's round would be wrong here
        self.assertEqual(sap.slice_nominal(230), 35)
        self.assertEqual(sap.slice_nominal(180), 27)
        self.assertEqual(sap.slice_nominal(200), 30)

    def test_degenerate_cap(self):
        # N = 40, n_D1 = 35, n_D0 = 5; s_nom = floor(6.5) = 6 -> s = min(6, 5) = 5 = n_D0.
        d = sap.Design(n_inv=0, n_D1=35, n_U2=1, n_U1=1, n_S=3)
        self.assertEqual(d.N, 40)
        self.assertEqual((d.s_nom, d.s, d.K), (6, 5, 40))
        self.assertTrue(d.cap_binds)
        self.assertEqual(d.regime, "DEGENERATE-CAP")
        for mU2, mU1, mS in itertools.product(range(2), range(2), range(4)):
            cap = sap.expected_captures(d, {"G1": 7, "U2": mU2, "U1": mU1, "S": mS})
            self.assertEqual(cap["Delta"], 0)
            self.assertEqual(cap["AUG"], 7 + mU2 + mU1 + mS)
        res = analyze(labels_for(0, 35, 1, 1, 3), [0] * 35 + [1, 0, 1, 0, 0])
        self.assertIsNone(res["existence"])
        self.assertIn("cap binds", sap.report(res))

    def test_n1_zero(self):
        d = sap.Design.simple(N=200, n_D1=30, n1=0)
        self.assertEqual(d.regime, "DEGENERATE-NONE")
        self.assertEqual(d.L, 0)
        self.assertIsNone(d.delta_threshold())
        self.assertFalse(d.threshold_reachable())
        cap = sap.expected_captures(d, {"G1": 5, "U2": 0, "U1": 0, "S": 12})
        self.assertEqual(cap["Delta"], 0)
        res = analyze(labels_for(0, 30, 0, 0, 170), E_for(G1=(30, 5), S=(170, 12)))
        self.assertIsNone(res["existence"])
        self.assertIn("no instability among agreeing verdicts", sap.report(res))

    def test_regimes(self):
        self.assertEqual(sap.assign_regime(0, 5, 5), "DEGENERATE-NONE")
        self.assertEqual(sap.assign_regime(3, 5, 5), "DEGENERATE-CAP")
        self.assertEqual(sap.assign_regime(5, 5, 6), "COVERAGE")
        self.assertEqual(sap.assign_regime(6, 5, 7), "PARTIAL")
        self.assertEqual(sap.Design.simple(20, 5, 5).regime, "PARTIAL")   # s = 3
        self.assertEqual(sap.assign_regime(1, 0, 3), "DEGENERATE-NOSLICE")
        d = sap.Design.simple(3, 0, 1)                                     # N = 3: s_nom = 0
        self.assertEqual((d.s, d.regime), (0, "DEGENERATE-NOSLICE"))
        res = analyze(["U1", "S", "S"], [1, 0, 0])
        self.assertIsNone(res["existence"])
        self.assertIn("s = 0", sap.report(res))

    def test_K_identical_across_policies(self):
        rng = np.random.default_rng(1)
        for _ in range(200):
            n = rng.integers(0, 40, size=5)
            d = sap.Design(*[int(x) for x in n])
            for pol in ("RAND", "DIS", "AUG"):
                self.assertEqual(sap.expected_reviewed(d, pol), d.K)


class TestHandCheck(unittest.TestCase):
    def test_hand_check(self):
        # N = 200, n_D1 = 30, s = 30, n1 = 10 with m1 = 4, n0 = 160 with m0 = 10.
        d = sap.Design(n_inv=0, n_D1=30, n_U2=3, n_U1=7, n_S=160)
        self.assertEqual((d.N, d.s, d.K, d.regime), (200, 30, 60, "COVERAGE"))
        cap = sap.expected_captures(d, {"G1": 11, "U2": 1, "U1": 3, "S": 10})
        self.assertEqual(cap["AUG"], 11 + 4 + Fraction(20 * 10, 160))
        self.assertEqual(cap["DIS"], 11 + Fraction(30 * 14, 170))
        self.assertEqual(cap["Delta"], Fraction(189, 68))
        self.assertAlmostEqual(float(cap["Delta"]), 2.78, places=2)
        self.assertEqual(d.L, Fraction(140, 17))
        self.assertAlmostEqual(float(d.L), 8.235294117647058, places=12)
        self.assertEqual(Fraction(4, 10) - Fraction(10, 160), Fraction(27, 80))   # RD0 = 0.3375
        self.assertEqual(sap.identity_delta(d, 4, 10), cap["Delta"])

    def test_hand_check_analysis_and_report(self):
        labels = labels_for(0, 30, 3, 7, 160)
        E = E_for(G1=(30, 11), U2=(3, 1), U1=(7, 3), S=(160, 10))
        res = analyze(labels, E, seed=20260929)
        self.assertAlmostEqual(float(res["captures"]["Delta"]), 189 / 68, places=12)
        p = sap.fisher_one_sided(4, 10, 10, 160)
        self.assertAlmostEqual(res["existence"]["p"], p, places=15)
        sz = res["size"]
        self.assertLessEqual(sz["delta_L"], sz["delta_hat"])
        self.assertGreaterEqual(sz["delta_U"], sz["delta_hat"])
        txt = sap.report(res)
        self.assertIn(sap.COVERS_TEXT, txt)
        self.assertIn("EXISTENCE QUESTION", txt)
        self.assertIn("SIZE QUESTION", txt)
        self.assertLess(txt.index("EXISTENCE QUESTION"), txt.index("SIZE QUESTION"))
        self.assertEqual(res["draw"]["K"], 60)
        for pol in ("RAND", "DIS", "AUG"):
            self.assertEqual(len(res["draw"]["reviewed"][pol]), 60)
            self.assertEqual(len(set(res["draw"]["reviewed"][pol].tolist())), 60)
        again = analyze(labels, E, seed=20260929)
        self.assertEqual(res["draw"]["Delta_draw"], again["draw"]["Delta_draw"])


class TestInvalidHandling(unittest.TestCase):
    def test_resolve_outcomes(self):
        T, H = sap.TRANSPORT, sap.HALT
        self.assertEqual(sap.resolve_outcomes(["success"]), "success")
        self.assertEqual(sap.resolve_outcomes([T, T, "failure"]), "failure")
        self.assertEqual(sap.resolve_outcomes([T, "MALFORMED"]), "MALFORMED")
        self.assertEqual(sap.resolve_outcomes([T, T, T, T]), sap.TRANSPORT_EXHAUSTED)
        # HALT carries no model output and is not a retry
        self.assertEqual(sap.resolve_outcomes([T, H, T, H, T, "success"]), "success")
        self.assertEqual(sap.resolve_outcomes([H, T, T, T, T]), sap.TRANSPORT_EXHAUSTED)
        with self.assertRaises(sap.ProtocolViolation):
            sap.resolve_outcomes(["MALFORMED", "success"])      # terminal outcomes are never retried
        with self.assertRaises(sap.ProtocolViolation):
            sap.resolve_outcomes(["REFUSAL", T])
        with self.assertRaises(sap.ProtocolViolation):
            sap.resolve_outcomes([T, T, T, T, "success"])      # more than 3 retries
        with self.assertRaises(sap.ProtocolViolation):
            sap.resolve_outcomes([T, T, T, T, H])               # nothing after exhaustion
        with self.assertRaises(ValueError):
            sap.resolve_outcomes([T, T])                        # still open
        with self.assertRaises(ValueError):
            sap.resolve_outcomes([T, H])                        # halted, still open

    def test_resolve_slot_records(self):
        A = sap.Attempt
        slot = ("t1", "A", 1)
        ok = [A("t1", "A", 1, "H0", 1, sap.TRANSPORT), A("t1", "A", 1, "H0", 2, "success", "r1")]
        res = sap.resolve_slot(list(reversed(ok)), slot, "H0")        # stored order does not matter
        self.assertEqual((res.outcome, res.resolving_attempt, res.response_id), ("success", 2, "r1"))
        with self.assertRaises(sap.ProtocolViolation):                 # attempt under another hash
            sap.resolve_slot([A("t1", "A", 1, "H1", 1, "success")], slot, "H0")
        with self.assertRaises(sap.ProtocolViolation):                 # late response logged
            sap.resolve_slot([A("t1", "A", 1, "H0", 1, sap.TRANSPORT),
                              A("t1", "A", 1, "H0", 1, "failure")], slot, "H0")
        with self.assertRaises(sap.ProtocolViolation):                 # gap in attempt indices
            sap.resolve_slot([A("t1", "A", 1, "H0", 2, "success")], slot, "H0")
        with self.assertRaises(sap.ProtocolViolation):                 # record of another slot
            sap.resolve_slot([A("t1", "A", 2, "H0", 1, "success")], slot, "H0")
        with self.assertRaises(TypeError):
            sap.resolve_slot(["success"], slot, "H0")

    def test_resolve_panel(self):
        A = sap.Attempt
        log = []
        for t, v in (("t1", "success"), ("t2", "failure")):
            for j, r in sap.SLOT_KEYS:
                log.append(A(t, j, r, "H0", 1, v, f"{t}-{j}{r}"))
        log[0] = A("t1", "A", 1, "H0", 1, sap.TRANSPORT)
        log.append(A("t1", "A", 1, "H0", 2, "success", "t1-A1"))
        out = sap.resolve_panel(log, ["t1", "t2"], "H0")
        self.assertEqual(out["slots"][0], {"A1": "success", "A2": "success", "A3": "success", "B1": "success"})
        a1 = out["provenance"]["by_slot"]["A1"]
        self.assertEqual((a1["slots"], a1["transport_retries"], a1["slots_with_retries"]), (2, 1, 1))
        # A payload-keyed response cache returns A1's response for A2: same response id.
        cached = [x if not (x.trajectory == "t2" and x.repeat == 2) else
                  A("t2", "A", 2, "H0", 1, "failure", "t2-A1") for x in log]
        with self.assertRaises(sap.ProtocolViolation):
            sap.resolve_panel(cached, ["t1", "t2"], "H0")
        with self.assertRaises(ValueError):                             # a slot never dispatched
            sap.resolve_panel(log[1:], ["t1", "t2"], "H0")
        with self.assertRaises(sap.ProtocolViolation):
            sap.resolve_panel(log + [A("t9", "A", 1, "H0", 1, "success")], ["t1", "t2"], "H0")

    def test_invalid_A1(self):
        slots = [
            {"A1": "MALFORMED", "A2": "success", "A3": "success", "B1": "success"},  # never replaced
            {"A1": "success", "A2": "success", "A3": "success", "B1": "success"},    # S
            {"A1": "success", "A2": "failure", "A3": "success", "B1": "success"},    # U1
            {"A1": "failure", "A2": "success", "A3": "success", "B1": "failure"},    # U2
            {"A1": "success", "A2": "success", "A3": "success", "B1": "failure"},    # G1
            {"A1": "TRANSPORT_EXHAUSTED", "A2": "failure", "A3": "failure", "B1": "failure"},
        ]
        lf = sap.derive_label_free(slots)
        self.assertEqual(list(lf["stratum"]), ["I", "S", "U1", "U2", "G1", "I"])
        self.assertEqual(lf["D"][0], -1)
        self.assertEqual(lf["R"][0], -1)
        self.assertEqual(lf["in_I"].sum(), 2)
        # A panel with invalid A1: reviewed first, zero capture, excluded from M.
        d = sap.Design(n_inv=5, n_D1=20, n_U2=2, n_U1=3, n_S=100)
        self.assertEqual(d.N, 130)
        self.assertEqual(d.s_nom, sap.slice_nominal(130))        # full N, invalid items included
        self.assertEqual(d.K, 5 + 20 + d.s)
        m = {"G1": 6, "U2": 1, "U1": 1, "S": 4}
        cap = sap.expected_captures(d, m)
        M = 6 + 1 + 1 + 4
        self.assertEqual(cap["RAND"], Fraction(d.K - 5, 125) * M)
        labels = labels_for(5, 20, 2, 3, 100)
        E0 = E_for(n_inv=(5, 0), G1=(20, 6), U2=(2, 1), U1=(3, 1), S=(100, 4))
        E1 = E_for(n_inv=(5, 5), G1=(20, 6), U2=(2, 1), U1=(3, 1), S=(100, 4))
        r0, r1 = analyze(labels, E0), analyze(labels, E1)
        self.assertEqual(r0["M"], M)
        self.assertEqual(r1["M"], M)
        self.assertEqual(r0["captures"], r1["captures"])
        self.assertEqual(r0["existence"]["p"], r1["existence"]["p"])
        self.assertEqual(r0["size"]["delta_L"], r1["size"]["delta_L"])
        self.assertEqual(r0["label_free"]["n1"], 5)

    def test_B1_invalid(self):
        slots = [{"A1": "success", "A2": "success", "A3": "success", "B1": "REFUSAL"},
                 {"A1": "failure", "A2": "failure", "A3": "failure", "B1": "failure"}]
        lf = sap.derive_label_free(slots)
        self.assertEqual(lf["D"][0], 1)
        self.assertEqual(lf["stratum"][0], "G1")
        self.assertEqual(lf["stratum"][1], "S")
        lf2 = sap.derive_label_free(slots, b1_invalid="exclude")
        self.assertEqual(lf2["stratum"][0], "EXCL")
        self.assertFalse(lf2["kept"][0])
        lf3 = sap.derive_label_free(slots, exclude_any_invalid=True)
        self.assertEqual(list(lf3["stratum"]), ["EXCL", "S"])

    def test_R_rules(self):
        slots = [
            {"A1": "success", "A2": "ABSTAIN", "A3": "TRUNCATED", "B1": "success"},   # both invalid
            {"A1": "success", "A2": "failure", "A3": "MALFORMED", "B1": "success"},   # R = 1
            {"A1": "success", "A2": "REFUSAL", "A3": "success", "B1": "success"},     # R = 0
        ]
        lf = sap.derive_label_free(slots)
        self.assertEqual(list(lf["R"]), [0, 1, 0])
        self.assertEqual(list(lf["R_unobserved"]), [1, 0, 0])
        self.assertEqual(list(lf["stratum"]), ["S", "U1", "S"])
        lf1 = sap.derive_label_free(slots, r_unobserved="one")
        self.assertEqual(list(lf1["stratum"]), ["U1", "U1", "S"])
        counts = sap.invalid_counts(slots, ["wa", "wa", "wl"])
        self.assertEqual(counts[("A2", "ABSTAIN", "wa")], 1)
        self.assertEqual(sum(counts.values()), 4)

    def test_sensitivity_runner(self):
        rng = np.random.default_rng(3)
        slots, E = [], []
        for i in range(60):
            a1 = "success" if rng.random() < 0.6 else "failure"
            other = lambda: a1 if rng.random() < 0.85 else ("failure" if a1 == "success" else "success")
            sl = {"A1": a1, "A2": other(), "A3": other(), "B1": other()}
            if i % 17 == 0:
                sl["B1"] = "MALFORMED"
            if i % 23 == 0:
                sl["A2"] = sl["A3"] = "REFUSAL"
            slots.append(sl)
            E.append(int(rng.random() < 0.15))
        outs = analyze_with_sensitivities(slots, E)
        self.assertEqual(len(outs), 5)
        self.assertTrue(all(name.startswith("sensitivity") for name, _ in outs[1:]))
        self.assertEqual(outs[1][1]["label_free"]["N"], outs[0][1]["label_free"]["N"])   # B1 -> D = 0 keeps every item
        self.assertLess(outs[3][1]["label_free"]["N"], outs[0][1]["label_free"]["N"])    # "any INVALID excluded" drops items
        s0 = outs[0][1]["label_free"]["s"]
        for _, r in outs[1:]:                                   # s and N held
            self.assertEqual(r["label_free"]["s"], min(s0, r["label_free"]["n_D0"]))
            self.assertEqual(r["label_free"]["N_per100"], outs[0][1]["label_free"]["N"])

    def test_sensitivity_holds_budget(self):
        # Example: N = 200 with 12 B1-INVALID trajectories and n1 = 29.
        sl = lambda a1, a2, a3, b1: {"A1": a1, "A2": a2, "A3": a3, "B1": b1}
        slots = ([sl("success", "success", "success", "MALFORMED")] * 12 +
                 [sl("success", "success", "success", "failure")] * 20 +
                 [sl("success", "failure", "success", "success")] * 29 +
                 [sl("success", "success", "success", "success")] * 139)
        E = [0] * 12 + [1] * 6 + [0] * 14 + [1] * 9 + [0] * 20 + [1] * 10 + [0] * 129
        outs = dict(analyze_with_sensitivities(slots, E))
        prim = outs["primary"]
        b1 = outs["sensitivity: B1 INVALID treated as agreement (D = 0) instead of D = 1"]
        self.assertEqual(prim["label_free"]["regime"], "COVERAGE")
        self.assertEqual(b1["label_free"]["regime"], "COVERAGE")
        self.assertEqual(b1["label_free"]["s"], prim["label_free"]["s"])        # s held
        self.assertEqual(b1["label_free"]["N_per100"], prim["label_free"]["N"])
        # The 12 B1-INVALID items (R = 0) move from G1 into S: K drops by 12 and n0 rises by 12.
        self.assertEqual(b1["label_free"]["K"], prim["label_free"]["K"] - 12)
        self.assertEqual(b1["label_free"]["n0"], prim["label_free"]["n0"] + 12)

    def test_fisher_matches_scipy(self):
        rng = np.random.default_rng(2)
        for _ in range(100):
            n1, n0 = int(rng.integers(1, 30)), int(rng.integers(1, 200))
            m1, m0 = int(rng.integers(0, n1 + 1)), int(rng.integers(0, n0 + 1))
            ref = stats.fisher_exact([[m1, n1 - m1], [m0, n0 - m0]], alternative="greater")[1]
            self.assertAlmostEqual(sap.fisher_one_sided(m1, n1, m0, n0), ref, places=10)
        self.assertEqual(sap.fisher_one_sided(0, 5, 0, 50), 1.0)
        tab = sap.fisher_pvalue_table(4, 30)
        for x1 in range(5):
            for x0 in range(31):
                self.assertAlmostEqual(tab[x1, x0], sap.fisher_one_sided(x1, 4, x0, 30), places=14)

    def test_permutation_equals_fisher_in_coverage(self):
        rng = np.random.default_rng(4)
        n_checked = 0
        while n_checked < 150:
            N = int(rng.integers(60, 280))
            n_D1 = int(rng.integers(0, N // 3))
            n1 = int(rng.integers(1, 25))
            n_U2 = int(rng.integers(0, n1 + 1))
            if N - n_D1 - n1 < 1:
                continue
            d = sap.Design.simple(N, n_D1, n1, n_U2=n_U2)
            if d.regime != "COVERAGE":
                continue
            mU2 = int(rng.integers(0, d.n_U2 + 1))
            mU1 = int(rng.integers(0, d.n_U1 + 1))
            mS = int(rng.integers(0, min(d.n_S, 30) + 1))
            p_perm = sap.permutation_p(d, {"U2": mU2, "U1": mU1, "S": mS})
            p_f = sap.fisher_one_sided(mU2 + mU1, d.n1, mS, d.n0, exact=True)
            self.assertEqual(p_perm, p_f)
            n_checked += 1

    def test_partial_permutation_against_monte_carlo(self):
        d = sap.Design.simple(N=60, n_D1=10, n1=14, n_U2=4)    # s = 9 < n1 = 14: PARTIAL
        self.assertEqual(d.regime, "PARTIAL")
        m = {"G1": 3, "U2": 2, "U1": 3, "S": 3}
        p = float(sap.permutation_p(d, m))
        rng = np.random.default_rng(5)
        E0 = np.array([1] * 2 + [0] * 2 + [1] * 3 + [0] * 7 + [1] * 3 + [0] * 33)
        R0 = np.array(["U2"] * 4 + ["U1"] * 10 + ["S"] * 36)
        w = {k: float(v) for k, v in sap.delta_weights(d).items()}
        obs = sum(w[k] * E0[R0 == k].sum() for k in w)
        B, hits = 20000, 0
        for _ in range(B):
            Rp = rng.permutation(R0)
            hits += sum(w[k] * E0[Rp == k].sum() for k in w) >= obs - 1e-12
        se = math.sqrt(p * (1 - p) / B)
        self.assertLess(abs(hits / B - p), 4 * se + 1e-9)
        res = analyze(labels_for(0, 10, 4, 10, 36),
                          E_for(G1=(10, 3), U2=(4, 2), U1=(10, 3), S=(36, 3)), boot_B=2000)
        self.assertIn("partial", res["size"])
        self.assertEqual(res["size"]["partial"]["label"], "approximate")
        self.assertIn("No threshold claim", sap.report(res))

    def test_identity_random_panels(self):
        rng = np.random.default_rng(6)
        done = 0
        while done < 300:
            d = sap.Design(int(rng.integers(0, 5)), int(rng.integers(0, 60)), int(rng.integers(0, 8)),
                           int(rng.integers(0, 15)), int(rng.integers(1, 250)))
            if d.regime != "COVERAGE":
                continue
            m = {"G1": int(rng.integers(0, d.n_D1 + 1)), "U2": int(rng.integers(0, d.n_U2 + 1)),
                 "U1": int(rng.integers(0, d.n_U1 + 1)), "S": int(rng.integers(0, d.n_S + 1))}
            cap = sap.expected_captures(d, m)
            self.assertEqual(cap["Delta"], sap.identity_delta(d, m["U2"] + m["U1"], m["S"]))
            done += 1


class TestScoreInterval(unittest.TestCase):
    def test_restricted_mle(self):
        rng = np.random.default_rng(7)
        for _ in range(300):
            n1, n0 = int(rng.integers(1, 31)), int(rng.integers(1, 270))
            x1, x0 = int(rng.integers(0, n1 + 1)), int(rng.integers(0, n0 + 1))
            d = float(rng.uniform(-0.95, 0.95))
            p0 = float(sap.restricted_mle_p0(x1, n1, x0, n0, d))
            ref = loglik_p0_numeric(x1, n1, x0, n0, d)
            self.assertAlmostEqual(p0, ref, places=6)
            lo, hi = max(0.0, -d), min(1.0, 1.0 - d)
            if 0 < x1 < n1 and 0 < x0 < n0:
                cf = mn_closed_form_p0(x1, n1, x0, n0, d)
                if lo < cf < hi:
                    self.assertAlmostEqual(p0, cf, places=9)

    def test_binom_pmf(self):
        for n in (1, 7, 30, 264):
            p = np.array([0.0, 1e-9, 0.03, 0.5, 0.97, 1 - 1e-9, 1.0])
            ours = sap.binom_pmf_rows(n, p)
            ref = stats.binom.pmf(np.arange(n + 1)[None, :], n, p[:, None])
            self.assertLess(np.abs(ours - ref).max(), 1e-12)

    def test_d0_matches_scipy_barnard(self):
        # At d = 0 the restricted MLE is the pooled proportion, so Z(0) is Barnard's
        # pooled score statistic and the upper-tail p-value is Barnard's one-sided p.
        for n1, n0, x1, x0 in [(10, 40, 4, 3), (5, 60, 2, 3), (12, 30, 6, 5), (3, 25, 2, 2),
                               (8, 50, 1, 9)]:
            T = sap.ExactScoreDiff(n1, n0, method="full")
            ours = T.pvalue(x1, x0, 0.0, "upper", refine="always")
            ref = stats.barnard_exact([[x1, x0], [n1 - x1, n0 - x0]], alternative="greater",
                                      pooled=True, n=256).pvalue
            self.assertAlmostEqual(ours, ref, delta=1e-9, msg=(n1, n0, x1, x0))

    def test_bruteforce_pvalue(self):
        n1, n0 = 3, 6
        T = sap.ExactScoreDiff(n1, n0, method="full")
        dense = np.linspace(0, 1, 200001)
        for d in (-0.35, 0.0, 0.15, 0.6):
            lo, hi = max(0.0, -d), min(1.0, 1.0 - d)
            p0s = lo + (hi - lo) * dense
            b1 = stats.binom.pmf(np.arange(n1 + 1)[None, :], n1, np.clip(p0s + d, 0, 1)[:, None])
            b0 = stats.binom.pmf(np.arange(n0 + 1)[None, :], n0, p0s[:, None])
            Z = {}
            for x1 in range(n1 + 1):
                for x0 in range(n0 + 1):
                    p0 = loglik_p0_numeric(x1, n1, x0, n0, d)
                    p1 = min(max(p0 + d, 0), 1)
                    v = p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0
                    num = x1 / n1 - x0 / n0 - d
                    Z[(x1, x0)] = 0.0 if abs(num) < 1e-15 else num / math.sqrt(v)
            for (x1, x0), z in Z.items():
                up = np.zeros(len(p0s))
                for (a, b), zz in Z.items():
                    if zz >= z - 1e-7 * max(1, abs(z)):
                        up += b1[:, a] * b0[:, b]
                ours = T.pvalue(x1, x0, d, "upper", refine="always")
                self.assertAlmostEqual(ours, up.max(), delta=1e-7, msg=(d, x1, x0))

    def test_sawtooth_is_genuine(self):
        # Exact unconditional p-values are not monotone in d. For m1 = 0 of 2 and
        # m0 = 146 of 163, the lower-tail p-value is below 0.025 at d = -0.06 but above it
        # at d = -0.02. An independent brute-force computation must agree.
        n1, n0, x1o, x0o = 2, 163, 0, 146
        T = sap.ExactScoreDiff(n1, n0, method="full")
        vals = {}
        for d in (-0.06, -0.02):
            Z = np.empty((n1 + 1, n0 + 1))
            for a in range(n1 + 1):
                for b in range(n0 + 1):
                    p0 = loglik_p0_numeric(a, n1, b, n0, d)
                    p1 = min(max(p0 + d, 0), 1)
                    v = p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0
                    num = a / n1 - b / n0 - d
                    Z[a, b] = 0.0 if abs(num) < 1e-15 else num / math.sqrt(v)
            zo = Z[x1o, x0o]
            mask = (Z <= zo + 1e-9 * max(1, abs(zo))).astype(float)
            p0s = np.linspace(max(0, -d), min(1, 1 - d), 200001)
            b1 = stats.binom.pmf(np.arange(n1 + 1)[None, :], n1, np.clip(p0s + d, 0, 1)[:, None])
            b0 = stats.binom.pmf(np.arange(n0 + 1)[None, :], n0, p0s[:, None])
            brute = float(((b1 @ mask) * b0).sum(1).max())
            ours = T.pvalue(x1o, x0o, d, "lower", refine="always")
            self.assertAlmostEqual(ours, brute, delta=1e-6)
            vals[d] = ours
        self.assertLess(vals[-0.06], sap.ALPHA)
        self.assertGreater(vals[-0.02], sap.ALPHA)

    def test_batch_equals_single(self):
        T = sap.ExactScoreDiff(5, 40)
        for d in (-0.3, 0.0, 0.17, 0.55):
            pu, pl = T.pvalues_all(d)
            for x1 in range(6):
                for x0 in range(0, 41, 3):
                    o = T.index(x1, x0)
                    su = T.pvalue(x1, x0, d, "upper")
                    sl = T.pvalue(x1, x0, d, "lower")
                    self.assertAlmostEqual(pu[o], su, delta=1e-12)
                    self.assertAlmostEqual(pl[o], sl, delta=1e-12)
                    self.assertEqual(pu[o] > sap.ALPHA, su > sap.ALPHA)
                    self.assertEqual(pl[o] > sap.ALPHA, sl > sap.ALPHA)

    def test_interval_properties(self):
        T = sap.ExactScoreDiff(10, 160)
        ci = T.interval(4, 10, extra_grid=[0.0, 0.2])
        g, iL, iU = ci["grid"], ci["grid_index_L"], ci["grid_index_U"]
        self.assertLess(ci["delta_L"], 0.3375)
        self.assertGreater(ci["delta_U"], 0.3375)
        # Outward rounding: the reported ends are the rejected grid points
        # next to the first and last accepted ones.
        self.assertEqual(ci["delta_L"], g[iL - 1])
        self.assertEqual(ci["delta_U"], g[iU + 1])
        self.assertLessEqual(T.pvalue(4, 10, ci["delta_L"], "upper"), sap.ALPHA)
        self.assertGreater(T.pvalue(4, 10, g[iL], "upper"), sap.ALPHA)
        self.assertLessEqual(T.pvalue(4, 10, ci["delta_U"], "lower"), sap.ALPHA)
        self.assertGreater(T.pvalue(4, 10, g[iU], "lower"), sap.ALPHA)
        for i in range(iL):
            self.assertLessEqual(T.pvalue(4, 10, float(g[i]), "upper"), sap.ALPHA)
        # Zero discordance: the interval still exists and contains 0.
        T2 = sap.ExactScoreDiff(2, 30)
        ci2 = T2.interval(0, 0)
        self.assertLess(ci2["delta_L"], 0.0)
        self.assertGreater(ci2["delta_U"], 0.0)

    def test_offgrid_hole_example(self):
        # 8 of 10 versus 20 of 220. The lower-tail test accepts d = 0.879 inside the
        # cell (0.87, 0.88) although it rejects 0.8785. A bisection between grid points can
        # stop near 0.87845, below an accepted point, so the upper end stays on the grid: 0.88.
        T = sap.ExactScoreDiff(10, 220, method="full")
        dm = float(sap.Design.simple(270, 40, 10).delta_threshold())
        self.assertGreater(T.pvalue(8, 20, 0.879, "lower", refine="always"), sap.ALPHA)
        self.assertLessEqual(T.pvalue(8, 20, 0.8785, "lower", refine="always"), sap.ALPHA)
        ci = T.interval(8, 20, extra_grid=[0.0, dm])
        self.assertEqual(ci["delta_U"], 0.88)
        self.assertGreater(ci["delta_U"], 0.879)

    def test_classify_size(self):
        # Open interval with rejected ends: an end equal to delta_m means delta_m is rejected.
        self.assertEqual(sap.classify_size(0.3, 0.6, 0.2, True), "exceeds")
        self.assertEqual(sap.classify_size(0.2, 0.6, 0.2, True), "exceeds")
        self.assertEqual(sap.classify_size(-0.1, 0.15, 0.2, True), "below")
        self.assertEqual(sap.classify_size(-0.1, 0.2, 0.2, True), "below")
        self.assertEqual(sap.classify_size(0.1, 0.3, 0.2, True), "unresolved")
        self.assertEqual(sap.classify_size(0.1, 0.3, 1.2, False), "unreachable-on-panel")
        d = sap.Design.simple(N=180, n_D1=15, n1=3)            # delta_m > 1
        self.assertGreaterEqual(d.delta_threshold(), 1)
        self.assertFalse(d.threshold_reachable())


def brute_bb_pvalue(n1, n0, x1o, x0o, d, tail, gamma=sap.BB_GAMMA, dense=200001):
    """Berger-Boos p-value by brute force: numeric restricted MLE, dense grid over the CP set."""
    Z = np.empty((n1 + 1, n0 + 1))
    for a in range(n1 + 1):
        for b in range(n0 + 1):
            p0 = loglik_p0_numeric(a, n1, b, n0, d)
            p1 = min(max(p0 + d, 0), 1)
            v = p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0
            num = a / n1 - b / n0 - d
            Z[a, b] = 0.0 if abs(num) < 1e-15 else num / math.sqrt(v)
    zo = Z[x1o, x0o]
    tol = 1e-7 * max(1, abs(zo))
    mask = (Z >= zo - tol) if tail == "upper" else (Z <= zo + tol)
    lo = 0.0 if x0o == 0 else stats.beta.ppf(gamma / 2, x0o, n0 - x0o + 1)
    hi = 1.0 if x0o == n0 else stats.beta.ppf(1 - gamma / 2, x0o + 1, n0 - x0o)
    a, b = max(lo, max(0.0, -d)), min(hi, min(1.0, 1.0 - d))
    if a > b:
        return gamma
    p0s = np.linspace(a, b, dense)
    b1 = stats.binom.pmf(np.arange(n1 + 1)[None, :], n1, np.clip(p0s + d, 0, 1)[:, None])
    b0 = stats.binom.pmf(np.arange(n0 + 1)[None, :], n0, p0s[:, None])
    return float(((b1 @ mask.astype(float)) * b0).sum(1).max()) + gamma


class TestBergerBoos(unittest.TestCase):
    def test_bruteforce(self):
        T = sap.ExactScoreDiff(3, 20, method="berger-boos")
        for d in (-0.3, 0.0, 0.25, 0.6):
            for x1, x0 in ((0, 0), (1, 3), (3, 2), (2, 12), (0, 20), (3, 0)):
                for tail in ("upper", "lower"):
                    ours = T.pvalue(x1, x0, d, tail, refine="always")
                    ref = brute_bb_pvalue(3, 20, x1, x0, d, tail)
                    self.assertAlmostEqual(ours, ref, delta=2e-7, msg=(d, x1, x0, tail))
                    self.assertGreaterEqual(ours, sap.BB_GAMMA)

    def test_batch_equals_single(self):
        T = sap.ExactScoreDiff(4, 35, method="berger-boos")
        for d in (-0.2, 0.0, 0.3):
            pu, pl = T.pvalues_all(d)
            for x1 in range(5):
                for x0 in range(0, 36, 5):
                    o = T.index(x1, x0)
                    self.assertAlmostEqual(pu[o], T.pvalue(x1, x0, d, "upper"), delta=1e-12)
                    self.assertAlmostEqual(pl[o], T.pvalue(x1, x0, d, "lower"), delta=1e-12)

    def test_interval_excludes_zero_when_fisher_rejects(self):
        # 4 of 5 unstable versus 24 of 160 stable agreeing verdicts are discordant.
        labels = ["G1"] * 15 + ["U1"] * 5 + ["S"] * 160
        E = [0] * 15 + [1, 1, 1, 1, 0] + [1] * 24 + [0] * 136
        full = analyze(labels, E, method="full")
        bb = analyze(labels, E, method="berger-boos")
        self.assertTrue(full["existence"]["reject"])
        self.assertLess(full["size"]["delta_L"], 0.0)           # the full-range interval keeps 0
        self.assertTrue(full["size"]["disagrees_with_existence"])
        self.assertIn("Reconciliation", sap.report(full))
        self.assertGreater(bb["size"]["delta_L"], 0.0)          # the Berger-Boos interval does not
        self.assertFalse(bb["size"]["disagrees_with_existence"])
        self.assertNotIn("Reconciliation", sap.report(bb))


class TestReporting(unittest.TestCase):
    def _panel(self):
        rng = np.random.default_rng(12)
        slots, E, ids, bench, tmpl = [], [], [], [], []
        for i in range(200):
            a1 = "success" if rng.random() < 0.55 else "failure"
            flip = "failure" if a1 == "success" else "success"
            r = rng.random()
            a2, a3 = (flip, a1) if r < 0.06 else ((flip, flip) if r < 0.08 else (a1, a1))
            b1 = flip if rng.random() < 0.15 else a1
            sl = {"A1": a1, "A2": a2, "A3": a3, "B1": b1}
            if i in (3, 77):
                sl["A1"] = "REFUSAL"
            if i in (10, 11):
                sl["B1"] = "MALFORMED"
            if i == 50:
                sl["A2"] = "TRUNCATED"
            slots.append(sl)
            unstable = a2 != a1 or a3 != a1
            E.append(int(rng.random() < (0.45 if unstable else 0.08)))
            ids.append(f"task-{i:04d}")
            bench.append("webarena" if i % 3 else "workarena")
            tmpl.append(f"tpl-{i // 2}")
        return slots, E, ids, bench, tmpl

    def test_panel_report_contents(self):
        slots, E, ids, bench, tmpl = self._panel()
        A = sap.Attempt
        log = [A(t, j, r, "cfg-7f3a", 1, slots[i][sap.SLOT_NAMES[(j, r)]], f"{t}-{j}{r}")
               for i, t in enumerate(ids) for j, r in sap.SLOT_KEYS]
        res = sap.resolve_panel(log, ids, "cfg-7f3a")
        self.assertEqual(res["slots"], slots)
        panel = analyze_panel(res["slots"], E, ids=ids, benchmarks=bench, templates=tmpl,
                                  provenance=res["provenance"], seed=20260929)
        txt = sap.panel_report(panel)
        for needle in ("A1 is the first complete response of slot (i, A, 1) that passes the per-response checks, under config hash cfg-7f3a",
                       "H0: p1 = p0", "Statistic: m1", "P(M1 >= m1 | n1, n0, m_D0)",
                       "H0_thr: δ ≤ δ_m (G ≤ m)", "Statistic: Z(d)", "Fragility:",
                       "What the interval covers", "Poisson-binomial",
                       "Sensitivity analyses (labeled; s = ", "B1 INVALID treated as agreement",
                       "Invalid counts by judge slot, category and benchmark", "A1 | REFUSAL",
                       "Template sensitivity", "One agreeing item per template",
                       "items ordered by ids", "Repeat observation"):
            self.assertIn(needle, txt, needle)
        self.assertNotIn("a positive gain exists", txt)

    def test_partial_report(self):
        res = analyze(labels_for(0, 10, 4, 10, 36),
                          E_for(G1=(10, 3), U2=(4, 2), U1=(10, 3), S=(36, 3)), boot_B=2000)
        txt = sap.report(res)
        self.assertNotIn("delta_m =", txt)                                   # coverage-only quantities
        self.assertNotIn("L = ", txt)
        self.assertIn("exchangeable within D = 0", txt)
        self.assertIn("What it covers, approximately", txt)
        # The partial regime states the independence assumption too.
        self.assertIn("  " + sap.PARTIAL_ASSUMPTION_TEXT, txt)
        self.assertIn("assume independent tasks", sap.PARTIAL_ASSUMPTION_TEXT)
        self.assertIn("are still to come and are required before the freeze", sap.PARTIAL_ASSUMPTION_TEXT)

    def test_fragility(self):
        d = sap.Design(0, 30, 3, 7, 160)
        c = {"G1": 11, "U2": 1, "U1": 3, "S": 10}
        k = sap.existence_fragility(d, c, reject=True)
        self.assertIsNotNone(k)
        # k flips reverse it; k - 1 flips never do.
        m1, m0 = 4, 10
        ok_k = any(sap.fisher_one_sided(m1 - f, 10, m0 + (k - f), 160) > sap.ALPHA for f in range(k + 1))
        ok_k1 = any(sap.fisher_one_sided(m1 - f, 10, m0 + (k - 1 - f), 160) > sap.ALPHA for f in range(k))
        self.assertTrue(ok_k)
        self.assertFalse(ok_k1)
        # n1 = 1, one discordant, 4 discordant among 165 agreeing.
        d1 = sap.Design(0, 15, 0, 1, 164)
        self.assertEqual(sap.existence_fragility(d1, {"G1": 0, "U2": 0, "U1": 1, "S": 3}, reject=True), 1)

    def test_draw_uses_sort_key(self):
        # With ids, the draw does not depend on stored order.
        labels = labels_for(2, 10, 3, 5, 40)
        E = [0, 0] + [1] * 4 + [0] * 6 + [1, 0, 0] + [1, 1, 0, 0, 0] + [1] * 5 + [0] * 35
        ids = [f"id{i:03d}" for i in range(len(labels))]
        perm = np.random.default_rng(3).permutation(len(labels))
        a = sap.one_operators_draw(labels, E, 9, seed=5, ids=ids)
        b = sap.one_operators_draw([labels[i] for i in perm], [E[i] for i in perm], 9, seed=5,
                                   ids=[ids[i] for i in perm])
        for pol in ("RAND", "DIS", "AUG"):
            got_a = sorted(ids[j] for j in a["reviewed"][pol])
            got_b = sorted(ids[perm[j]] for j in b["reviewed"][pol])
            self.assertEqual(got_a, got_b)
        self.assertEqual(a["Delta_draw"], b["Delta_draw"])

    def test_template_subsample(self):
        labels = ["G1"] * 4 + ["U1"] * 6 + ["S"] * 20
        tmpl = [f"t{i // 3}" for i in range(30)]
        ch, info = sap.template_subsample(labels, tmpl, seed=1)
        self.assertEqual(len(ch), info["n_templates"])
        self.assertEqual(len({tmpl[i] for i in ch}), len(ch))
        self.assertTrue(all(labels[i] != "G1" for i in ch))
        ch2, _ = sap.template_subsample(labels, tmpl, seed=1)
        self.assertEqual(ch, ch2)                                             # reproducible
        # Label-free: E is not an input, so the choice cannot depend on it.
        E1 = [0] * 30
        E2 = [1] * 30
        r1 = sap.template_sensitivity(labels, E1, tmpl, seed=1, path=sap.PATH_POPULATION)
        r2 = sap.template_sensitivity(labels, E2, tmpl, seed=1, path=sap.PATH_POPULATION)
        self.assertEqual((r1["n1"], r1["n0"]), (r2["n1"], r2["n0"]))


class TestB1AgreeSensitivity(unittest.TestCase):
    def test_b1_invalid_agree_routes_by_R(self):
        slots = [
            {"A1": "success", "A2": "failure", "A3": "success", "B1": "REFUSAL"},
            {"A1": "failure", "A2": "failure", "A3": "failure", "B1": "MALFORMED"},
        ]
        prim = sap.derive_label_free(slots)
        self.assertEqual(list(prim["D"]), [1, 1])
        agree = sap.derive_label_free(slots, b1_invalid="agree")
        self.assertEqual(list(agree["D"]), [0, 0])
        self.assertEqual(list(agree["stratum"]), ["U1", "S"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
