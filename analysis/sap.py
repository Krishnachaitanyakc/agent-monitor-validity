"""Statistical analysis of the instability-first review study: reference implementation.

This module implements the study's statistical analysis plan (v3.1). Comments
name the plan's sections by topic: units, budget, policies, regimes, the
coverage regime, invalid responses and the partial regime. Where the code has
to fill a gap in the plan, the comment says "Implementation choice".

Everything here is a pure function of label-free counts, plus the reference
discordance E only where the plan marks a quantity as needing labels. Nothing
here reads data from disk.

Conventions
-----------
A panel is described by one stratum label per trajectory:
    'I'  : A1 invalid (the set I)
    'G1' : A1 valid and D = 1
    'U2' : A1 valid, D = 0, R = 2
    'U1' : A1 valid, D = 0, R = 1
    'S'  : A1 valid, D = 0, R = 0
derive_label_free() produces these labels from resolved slot outcomes,
and resolve_panel() produces the resolved outcomes from the attempt log.
Discordance E_i = 1(A1_i != Y_i) is 0 or 1 per trajectory; it is ignored for 'I'.

Exact interval, in one paragraph
--------------------------------
For a candidate difference d, Z(d) is the Farrington-Manning / Miettinen-
Nurminen score statistic with the restricted-MLE variance. The restricted MLE
is computed by bisection on the derivative of the concave restricted
log-likelihood (80 halvings, exact to machine precision); tests check it
against the Miettinen-Nurminen closed form. The Miettinen-Nurminen factor
(n1+n0)/(n1+n0-1) is a constant for fixed d, so it cannot change the ordering
of outcomes and therefore cannot change any exact p-value; it is omitted. The
one-sided exact unconditional p-value maximizes the tail probability of the
ordering over the nuisance p0 on a 1001-point grid that is dense near both ends
of the admissible range, then refines every grid-local maximum within
REFINE_WINDOW of the grid maximum with bounded scalar optimization. Refinement
can only raise a p-value; it is applied whenever the grid p-value lies in
(ALPHA - REFINE_WINDOW, ALPHA], which is every case where it could change an
accept/reject decision, provided the grid discretization error is below
REFINE_WINDOW. pvalues_all(refine=False) returns the grid maxima, which is how
that error is measured.

Two nuisance sets are implemented. 'full': the supremum
runs over the whole admissible range of p0. 'berger-boos': the supremum runs over a 99.9% Clopper-Pearson set for p0 computed
from m0 of n0, and BB_GAMMA = 0.001 is added (Berger and Boos 1994).
INTERVAL_METHOD selects the frozen one; the plan freezes 'berger-boos'.

Interval endpoints. Tests are run on the d grid k/100 plus the
label-free points 0 and delta_m. delta_L is the grid point just below the first
accepted grid point of the upper-tail test, and delta_U the grid point just
above the last accepted grid point of the lower-tail test. Both are rejected
points, and the reported interval is the OPEN interval (delta_L, delta_U), with
-1 and 1 as closed ends when nothing is rejected on that side. Because every
outcome uses the same grid, the one-sided miss probability at any delta, on or
off the grid, is bounded by the size of one composite one-sided test, so
coverage holds everywhere, not only at grid points; the cost is at most one
grid step (0.01) per side. The earlier bisection between grid points could land
inside a sub-grid hole of the saw-toothed p-value and was removed.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize, special, stats

# ---------------------------------------------------------------------------
# Frozen constants
# ---------------------------------------------------------------------------
ALPHA = 0.025                      # one-sided level for every test and bound
SLICE_FRACTION = Fraction(3, 20)   # 0.15 of N
MATERIAL_M = 2                     # threshold, additional discordances per 100
D_GRID_DEN = 100                   # d grid: k/100, k = -99..99, plus label-free points
NUISANCE_POINTS = 1001             # the plan asks for at least 1000
REFINE_WINDOW = 2e-3               # see module docstring
TIE_RTOL = 1e-10                   # score values this close are treated as tied
STRATA = ("I", "G1", "U2", "U1", "S")
INTERVAL_METHODS = ("full", "berger-boos")
INTERVAL_METHOD = "berger-boos"    # the frozen interval method; "full" is kept for comparison
BB_GAMMA = 0.001                   # Berger-Boos: 1 - 0.999 Clopper-Pearson set for p0
FRAGILITY_MAX = 15                 # search depth for the fragility count

NUISANCE_TEXT = {
    "full": "supremum over the whole admissible range of p0",
    "berger-boos": ("supremum over a 99.9% Clopper-Pearson set for p0 from m0 of n0, "
                    "plus 0.001 (Berger and Boos 1994; the frozen method)"),
}

# Wording of what the interval covers.
COVERS_TEXT = (
    "What the interval covers:\n"
    "- delta = p1 - p0, where p1 is the discordance rate of agreeing first verdicts whose "
    "repeats changed (U2 and U1 pooled) and p0 that of stable agreeing first verdicts. The "
    "interval for delta covers delta with probability at least 95%, at most 2.5% on each "
    "side. delta does not depend on the budget.\n"
    "- The interval for G covers G, defined above, with the same probability. The "
    "probability is over repeated draws of the discordance indicators E, independent "
    "Bernoulli(p1) in U and Bernoulli(p0) in S, with the label-free design held fixed.\n"
    "- U2 and U1 share one rate p1. If their rates differ, m1 is Poisson-binomial rather "
    "than binomial and p1 is their size-weighted average; the exact guarantee is proved "
    "only for the binomial case, and the synthetic checks measure unequal rates.\n"
    "- G_hat equals the realized Delta per 100 exactly, as an identity. The interval is "
    "for G, the expected gain for this design, not for the realized Delta of this panel, "
    "which is known exactly once labels are joined.\n"
    "- It does not cover the gain at another budget, because L changes.\n"
    "- It does not cover any gain in true errors as opposed to reference discordance, "
    "or gains for other judges, views or benchmarks.\n"
    "- The odds ratio is not used for any interval."
)
G_DEFINITION_TEXT = (
    "G = (100/N)·L·δ is the expected number of additional "
    "reference-discordant first verdicts that instability-first review captures, per "
    "100 panel trajectories, for a panel with this panel's label-free design (the same "
    "N, n_D0, n1 and s)."
)

# ---------------------------------------------------------------------------
# Invalid responses, abstentions, failures, retries
# ---------------------------------------------------------------------------
VERDICTS = ("success", "failure")
INVALID_TERMINAL = ("REFUSAL", "TRUNCATED", "MALFORMED", "ABSTAIN")
TRANSPORT = "TRANSPORT"                       # no complete model response; retryable
HALT = "HALT"                                 # run stopped; not a retry
TRANSPORT_EXHAUSTED = "TRANSPORT_EXHAUSTED"   # no complete response after the retries: INVALID
MAX_RETRIES = 3
INVALID_CODES = INVALID_TERMINAL + (TRANSPORT_EXHAUSTED,)
SLOT_KEYS = (("A", 1), ("A", 2), ("A", 3), ("B", 1))
SLOT_NAMES = {("A", 1): "A1", ("A", 2): "A2", ("A", 3): "A3", ("B", 1): "B1"}


class ProtocolViolation(ValueError):
    """An attempt log that the frozen retry rules forbid."""


@dataclass(frozen=True)
class Attempt:
    """One logged attempt, written once.

    trajectory, judge ('A' or 'B') and repeat (1..3 for A, 1 for B) are the slot,
    fixed at dispatch. config_hash is the evaluation configuration the attempt was
    sent under. attempt is the 1-based dispatch index within the slot. outcome is
    TRANSPORT, HALT, 'success' or 'failure' (VALID), or one of INVALID_TERMINAL.
    response_id is the provider's id for a complete response, when the runner has it.

    Runner rules this record encodes:
      - TRANSPORT: no complete model response. Connection error, timeout, HTTP 408,
        429 (rate limit), 5xx, overloaded, any other non-2xx without model output
        that is not a HALT class, and a stream that ends without a final stop
        reason. Partial stream content and any response arriving after the attempt
        was logged TRANSPORT are discarded unparsed and never logged as an outcome.
      - HALT: the run stops. HTTP 401, 403, 404, any billing, quota or credit error
        (whatever its status code), the spending cap, or an operator stop. It is
        logged, is not a retry, and the slot resumes under the same frozen hash.
    """
    trajectory: str
    judge: str
    repeat: int
    config_hash: str
    attempt: int
    outcome: str
    response_id: Optional[str] = None

    @property
    def slot(self) -> Tuple[str, str, int]:
        return (self.trajectory, self.judge, self.repeat)


@dataclass(frozen=True)
class SlotResolution:
    outcome: str
    n_records: int
    n_transport: int
    n_halt: int
    resolving_attempt: Optional[int]      # attempt index of the complete response
    response_id: Optional[str]
    config_hash: Optional[str]


def resolve_outcomes(outcomes: Sequence[str]) -> str:
    """The frozen retry rule on a slot's outcomes in dispatch order.

    Only TRANSPORT is retried, at most MAX_RETRIES times, so at most 1 + MAX_RETRIES
    dispatches can fail in transport. HALT records carry no model output
    and do not count as retries; the slot stays open after them. Any complete
    response ends the slot and is never retried. A slot whose 1 + MAX_RETRIES
    transport failures are exhausted resolves to TRANSPORT_EXHAUSTED, which is
    INVALID: the plan makes a slot with no complete response after the retries
    INVALID. Nothing may follow exhaustion.
    """
    if len(outcomes) == 0:
        raise ValueError("slot has no attempts; it was never dispatched")
    n_tr = 0
    for k, a in enumerate(outcomes):
        if n_tr >= 1 + MAX_RETRIES:
            raise ProtocolViolation(f"a record follows {1 + MAX_RETRIES} transport failures; "
                                    f"at most {MAX_RETRIES} retries are allowed")
        if a == HALT:
            continue
        if a == TRANSPORT:
            n_tr += 1
            continue
        if a in VERDICTS or a in INVALID_TERMINAL:
            if k != len(outcomes) - 1:
                raise ProtocolViolation("an attempt follows a complete model response; "
                                        "terminal outcomes are never retried")
            return a
        raise ValueError(f"unknown attempt outcome {a!r}")
    if n_tr >= 1 + MAX_RETRIES:
        return TRANSPORT_EXHAUSTED
    raise ValueError("slot is still open: transport failures or a halt with retries left")


def resolve_slot(attempts: Sequence[Attempt], slot: Tuple[str, str, int],
                 frozen_config_hash: str) -> SlotResolution:
    """Resolve one slot from its logged attempts, asserting the log's integrity.

    Raises ProtocolViolation when a record belongs to another slot, was sent under
    a config hash other than the frozen one, or when attempt indices are duplicated
    (a late or duplicate response was logged) or have gaps.
    """
    if len(attempts) == 0:
        raise ValueError(f"slot {slot} has no attempts; it was never dispatched")
    for a in attempts:
        if not isinstance(a, Attempt):
            raise TypeError("resolve_slot takes Attempt records; use resolve_outcomes for bare outcomes")
        if a.slot != tuple(slot):
            raise ProtocolViolation(f"record for slot {a.slot} passed to slot {tuple(slot)}")
        if a.config_hash != frozen_config_hash:
            raise ProtocolViolation(f"attempt {a.attempt} of slot {a.slot} was sent under config hash "
                                    f"{a.config_hash!r}; the frozen hash is {frozen_config_hash!r}")
    idx = [a.attempt for a in attempts]
    if len(set(idx)) != len(idx):
        raise ProtocolViolation(f"slot {tuple(slot)}: two records share an attempt index "
                                "(a late or duplicate response was logged)")
    if sorted(idx) != list(range(1, len(idx) + 1)):
        raise ProtocolViolation(f"slot {tuple(slot)}: attempt indices {sorted(idx)} are not 1..k")
    recs = sorted(attempts, key=lambda a: a.attempt)
    outcome = resolve_outcomes([a.outcome for a in recs])
    last = recs[-1]
    complete = outcome in VERDICTS or outcome in INVALID_TERMINAL
    return SlotResolution(outcome=outcome, n_records=len(recs),
                          n_transport=sum(a.outcome == TRANSPORT for a in recs),
                          n_halt=sum(a.outcome == HALT for a in recs),
                          resolving_attempt=last.attempt if complete else None,
                          response_id=last.response_id if complete else None,
                          config_hash=frozen_config_hash)


def resolve_panel(log: Iterable[Attempt], trajectories: Sequence[str],
                  frozen_config_hash: str) -> dict:
    """Resolve every slot of the panel from the attempt log.

    Returns {'slots': [ {'A1','A2','A3','B1'} -> outcome ] in trajectory order,
    'provenance': {...}}. Raises ProtocolViolation for records of trajectories
    outside the panel, for any record under another config hash, and when two
    complete responses share a response id, which is what a response cache
    keyed on the payload would produce for the byte-identical repeats.
    """
    trajectories = list(trajectories)
    if len(set(trajectories)) != len(trajectories):
        raise ValueError("trajectory ids must be unique")
    known = set(trajectories)
    groups: Dict[Tuple[str, str, int], List[Attempt]] = defaultdict(list)
    for a in log:
        if a.trajectory not in known:
            raise ProtocolViolation(f"record for trajectory {a.trajectory!r} outside the panel")
        if (a.judge, a.repeat) not in SLOT_KEYS:
            raise ProtocolViolation(f"unknown slot ({a.judge}, {a.repeat})")
        groups[a.slot].append(a)
    slots, seen_ids = [], {}
    prov = {"config_hash": frozen_config_hash, "n_trajectories": len(trajectories),
            "by_slot": {name: Counter() for name in SLOT_NAMES.values()}}
    for t in trajectories:
        row = {}
        for key in SLOT_KEYS:
            name = SLOT_NAMES[key]
            res = resolve_slot(groups.get((t,) + key, []), (t,) + key, frozen_config_hash)
            row[name] = res.outcome
            c = prov["by_slot"][name]
            c["slots"] += 1
            # every transport failure is followed by a retry, except the last one of an exhausted slot
            c["transport_retries"] += res.n_transport - int(res.outcome == TRANSPORT_EXHAUSTED)
            c["slots_with_retries"] += int(res.n_transport > 0)
            c["halts"] += res.n_halt
            c["exhausted"] += int(res.outcome == TRANSPORT_EXHAUSTED)
            c["invalid"] += int(not is_valid(res.outcome))
            if not is_valid(res.outcome):
                c["invalid:" + res.outcome] += 1
            if res.response_id is not None:
                c["response_ids"] += 1
                if res.response_id in seen_ids:
                    raise ProtocolViolation(
                        f"response id {res.response_id!r} resolves both {seen_ids[res.response_id]} and "
                        f"{(t, name)}: a response cache is active or a response was reused")
                seen_ids[res.response_id] = (t, name)
        slots.append(row)
    prov["by_slot"] = {k: dict(v) for k, v in prov["by_slot"].items()}
    return {"slots": slots, "provenance": prov}


def is_valid(outcome: str) -> bool:
    if outcome in VERDICTS:
        return True
    if outcome in INVALID_CODES:
        return False
    raise ValueError(f"not a resolved slot outcome: {outcome!r}")


def derive_label_free(slots: Sequence[Mapping[str, str]], b1_invalid: str = "disagree",
                      r_unobserved: str = "zero", exclude_any_invalid: bool = False) -> dict:
    """Resolved slot outcomes -> (I, V, D, R, R_unobserved).

    slots[i] maps 'A1', 'A2', 'A3', 'B1' to a resolved outcome ('success',
    'failure', or an INVALID code). Primary rules are the defaults:
      - A1 INVALID -> i in I. A1 is never replaced by a repeat or a later call.
      - B1 INVALID -> D = 1 (b1_invalid='disagree'). Labelled sensitivity:
        'agree' sets D = 0 so the item is routed by A's own R. 'exclude' is kept for
        completeness; with s held it provably cannot move Delta, p or the interval.
      - R counts valid repeats whose verdict differs from A1; invalid repeats
        count 0; both repeats invalid -> R = 0 and R_unobserved = 1
        (r_unobserved='zero'); sensitivity: 'one' sets R = 1 instead.
        Labelled sensitivity 'one_any' also sets R = 1 when exactly one repeat is
        INVALID and the valid one agrees with A1.
      - exclude_any_invalid=True is the third sensitivity analysis: drop every
        trajectory with any INVALID slot.
    Excluded trajectories leave the panel. The sensitivity runner holds s and the
    per-100 denominator N at their primary values.

    Returns a dict of numpy arrays over the input order:
      stratum ('I','G1','U2','U1','S', or 'EXCL'), in_I, in_V, D, R
      (-1 where undefined), R_unobserved (0/1, -1 where undefined),
      R_one_missing (0/1, -1 where undefined), kept.
    """
    if b1_invalid not in ("disagree", "exclude", "agree"):
        raise ValueError("b1_invalid must be 'disagree', 'exclude' or 'agree'")
    if r_unobserved not in ("zero", "one", "one_any"):
        raise ValueError("r_unobserved must be 'zero', 'one' or 'one_any'")
    n = len(slots)
    stratum = np.empty(n, dtype=object)
    D = np.full(n, -1)
    R = np.full(n, -1)
    Ru = np.full(n, -1)
    R1m = np.full(n, -1)
    for i, sl in enumerate(slots):
        a1, a2, a3, b1 = sl["A1"], sl["A2"], sl["A3"], sl["B1"]
        valid = {k: is_valid(v) for k, v in (("A1", a1), ("A2", a2), ("A3", a3), ("B1", b1))}
        if exclude_any_invalid and not all(valid.values()):
            stratum[i] = "EXCL"
            continue
        if not valid["A1"]:
            stratum[i] = "I"
            continue
        if not valid["B1"]:
            if b1_invalid == "exclude":
                stratum[i] = "EXCL"
                continue
            d = 0 if b1_invalid == "agree" else 1
        else:
            d = int(b1 != a1)
        reps = [v for k, v in (("A2", a2), ("A3", a3)) if valid[k]]
        r = sum(int(v != a1) for v in reps)
        ru = int(len(reps) == 0)
        r1m = int(len(reps) == 1)
        if ru and r_unobserved in ("one", "one_any"):
            r = 1
        if r1m and r == 0 and r_unobserved == "one_any":
            r = 1
        D[i], R[i], Ru[i], R1m[i] = d, r, ru, r1m
        if d == 1:
            stratum[i] = "G1"
        else:
            stratum[i] = {2: "U2", 1: "U1", 0: "S"}[r]
    kept = stratum != "EXCL"
    return {"stratum": stratum, "in_I": stratum == "I",
            "in_V": kept & (stratum != "I"), "D": D, "R": R,
            "R_unobserved": Ru, "R_one_missing": R1m, "kept": kept}


def invalid_counts(slots: Sequence[Mapping[str, str]],
                   benchmarks: Optional[Sequence[str]] = None) -> Dict[Tuple[str, str, str], int]:
    """INVALID counts keyed by (slot, category, benchmark), for the report."""
    c: Counter = Counter()
    for i, sl in enumerate(slots):
        bench = benchmarks[i] if benchmarks is not None else "all"
        for k in ("A1", "A2", "A3", "B1"):
            if not is_valid(sl[k]):
                c[(k, sl[k], bench)] += 1
    return dict(c)


# ---------------------------------------------------------------------------
# Budget, rounding, cap, regimes
# ---------------------------------------------------------------------------
def slice_nominal(N: int, frac: Fraction = SLICE_FRACTION) -> int:
    """s_nom = floor(frac*N + 1/2), round half up, in exact rational arithmetic."""
    return math.floor(Fraction(frac) * N + Fraction(1, 2))


def assign_regime(n1: int, s: int, n_D0: int) -> str:
    """Regime table, first match wins. s = 0 (N <= 3) is DEGENERATE-NOSLICE."""
    if n1 == 0:
        return "DEGENERATE-NONE"
    if s == n_D0:
        return "DEGENERATE-CAP"
    if s == 0:
        return "DEGENERATE-NOSLICE"
    if n1 <= s:
        return "COVERAGE"
    return "PARTIAL"


def leverage(n1: int, n_D0: int, s: int) -> Fraction:
    """L = n1 (n_D0 - s) / n_D0, label-free; L := 0 when n_D0 = 0."""
    if n_D0 == 0:
        return Fraction(0)
    return Fraction(n1 * (n_D0 - s), n_D0)


@dataclass(frozen=True)
class Design:
    """Label-free panel design, committed before labels are joined.

    s_override replaces s_nom (invariance checks, and the sensitivity analyses,
    which hold s at its primary value; s is still capped at n_D0). N_ref replaces
    N as the per-100 denominator of G and delta_m (sensitivity analyses hold it
    at the primary N).
    """
    n_inv: int
    n_D1: int
    n_U2: int
    n_U1: int
    n_S: int
    s_override: Optional[int] = None
    frac: Fraction = SLICE_FRACTION
    N_ref: Optional[int] = None

    def __post_init__(self):
        for name in ("n_inv", "n_D1", "n_U2", "n_U1", "n_S"):
            v = getattr(self, name)
            if int(v) != v or v < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    @classmethod
    def from_labels(cls, labels: Iterable[str], **kw) -> "Design":
        c = Counter(x for x in labels if x != "EXCL")
        bad = set(c) - set(STRATA)
        if bad:
            raise ValueError(f"unknown stratum labels {bad}")
        return cls(c["I"], c["G1"], c["U2"], c["U1"], c["S"], **kw)

    @classmethod
    def simple(cls, N: int, n_D1: int, n1: int, n_inv: int = 0, n_U2: int = 0, **kw) -> "Design":
        """Design with the given totals; n1 split as n_U2 and n1 - n_U2."""
        n_S = N - n_inv - n_D1 - n1
        if n_S < 0:
            raise ValueError("N too small for the requested counts")
        return cls(n_inv, n_D1, n_U2, n1 - n_U2, n_S, **kw)

    @property
    def N(self) -> int:
        return self.n_inv + self.n_D1 + self.n_U2 + self.n_U1 + self.n_S

    @property
    def N_per100(self) -> int:
        return self.N_ref if self.N_ref is not None else self.N

    @property
    def N_V(self) -> int:
        return self.N - self.n_inv

    @property
    def n_D0(self) -> int:
        return self.n_U2 + self.n_U1 + self.n_S

    @property
    def n1(self) -> int:
        return self.n_U2 + self.n_U1

    @property
    def n0(self) -> int:
        return self.n_S

    @property
    def s_nom(self) -> int:
        return self.s_override if self.s_override is not None else slice_nominal(self.N, self.frac)

    @property
    def s(self) -> int:
        return min(self.s_nom, self.n_D0)

    @property
    def K(self) -> int:
        return self.n_inv + self.n_D1 + self.s

    @property
    def cap_binds(self) -> bool:
        return self.s == self.n_D0

    @property
    def regime(self) -> str:
        return assign_regime(self.n1, self.s, self.n_D0)

    @property
    def L(self) -> Fraction:
        return leverage(self.n1, self.n_D0, self.s)

    def delta_threshold(self, m: float = MATERIAL_M) -> Optional[Fraction]:
        """delta_m = m N / (100 L); None when L = 0 (no leverage at all)."""
        if self.L == 0:
            return None
        return Fraction(m) * self.N_per100 / (100 * self.L)

    def threshold_reachable(self, m: float = MATERIAL_M) -> bool:
        """Label-free: the threshold is unreachable when delta_m >= 1."""
        dm = self.delta_threshold(m)
        return dm is not None and dm < 1


# ---------------------------------------------------------------------------
# Policies and exact tie-averaged expected capture
# ---------------------------------------------------------------------------
def review_fractions(sizes: Sequence[int], k: int) -> List[Fraction]:
    """pi_j = min(n_j, max(0, k - N_<j)) / n_j for strata in priority order."""
    out, above = [], 0
    for n in sizes:
        take = min(n, max(0, k - above))
        out.append(Fraction(take, n) if n > 0 else Fraction(0))
        above += n
    return out


def expected_capture(sizes: Sequence[int], discordant: Sequence[int], k: int) -> Fraction:
    """E_U[C] = sum_j m_j min(n_j, max(0, k - N_<j)) / n_j; E[C_Rand] := 0 when N_V = 0."""
    return sum((f * m for f, m in zip(review_fractions(sizes, k), discordant)), Fraction(0))


def _layout(design: Design, policy: str) -> List[Tuple[str, int]]:
    d = design
    if policy == "RAND":
        return [("I", d.n_inv), ("V", d.N_V)]
    if policy == "DIS":
        return [("I", d.n_inv), ("G1", d.n_D1), ("G0", d.n_D0)]
    if policy == "AUG":
        return [("I", d.n_inv), ("G1", d.n_D1), ("U2", d.n_U2), ("U1", d.n_U1), ("S", d.n_S)]
    raise ValueError(policy)


def _stratum_discordance(m: Mapping[str, int]) -> Dict[str, int]:
    g0 = m["U2"] + m["U1"] + m["S"]
    return {"I": 0, "G1": m["G1"], "U2": m["U2"], "U1": m["U1"], "S": m["S"],
            "G0": g0, "V": m["G1"] + g0}


def expected_captures(design: Design, m: Mapping[str, int]) -> Dict[str, Fraction]:
    """Exact expected captures (Fractions) for Random, DIS and AUG at budget K.

    m: discordant counts per stratum, keys 'G1','U2','U1','S' [labels].
    Invalid-A1 items have no verdict and contribute zero capture.
    """
    for key, n in (("G1", design.n_D1), ("U2", design.n_U2), ("U1", design.n_U1), ("S", design.n_S)):
        if not 0 <= m[key] <= n:
            raise ValueError(f"m[{key}] = {m[key]} outside 0..{n}")
    mm = _stratum_discordance(m)
    out: Dict[str, Fraction] = {}
    for pol in ("RAND", "DIS", "AUG"):
        lay = _layout(design, pol)
        out[pol] = expected_capture([n for _, n in lay], [mm[k] for k, _ in lay], design.K)
    out["Delta"] = out["AUG"] - out["DIS"]
    out["Delta_DR"] = out["DIS"] - out["RAND"]
    out["K"] = Fraction(design.K)
    return out


def expected_reviewed(design: Design, policy: str) -> Fraction:
    """Expected number of reviewed items; equals K for every policy."""
    lay = _layout(design, policy)
    fr = review_fractions([n for _, n in lay], design.K)
    return sum((f * n for f, (_, n) in zip(fr, lay)), Fraction(0))


def delta_weights(design: Design) -> Dict[str, Fraction]:
    """w_j = pi_AUG(j) - pi_DIS(j) for the agreeing strata; Delta = sum_j w_j m_j."""
    aug = dict(zip(["I", "G1", "U2", "U1", "S"],
                   review_fractions([n for _, n in _layout(design, "AUG")], design.K)))
    dis = review_fractions([n for _, n in _layout(design, "DIS")], design.K)[2]
    return {j: aug[j] - dis for j in ("U2", "U1", "S")}


def identity_delta(design: Design, m1: int, m0: int) -> Fraction:
    """Coverage identity: Delta = L (m1/n1 - m0/n0). Holds in COVERAGE only."""
    return design.L * (Fraction(m1, design.n1) - Fraction(m0, design.n0))


def _sorted_by_key(indices: Sequence[int], keys) -> np.ndarray:
    return np.array(sorted((int(i) for i in indices), key=lambda j: keys[j]), dtype=int)


def one_operators_draw(labels: Sequence[str], E: Sequence[int], s: int, seed: int,
                       ids: Optional[Sequence[str]] = None) -> dict:
    """One seeded realization of each policy, labelled "one operator's draw" [labels].

    Items are ordered by the frozen sort key before permuting, so the
    draw is reproducible from the manifest alone. The sort key is the normalized
    task id (ids); without ids it falls back to stored order, which the real run
    must not use. Permutations use child generators of SeedSequence(seed): one per
    policy, in the order RAND, DIS, AUG.
    """
    labels = np.asarray(labels, dtype=object)
    E = np.asarray(E, dtype=int)
    n = len(labels)
    if ids is None:
        keys = list(range(n))
    else:
        keys = list(ids)
        if len(keys) != n or len(set(keys)) != n:
            raise ValueError("ids must be unique and one per trajectory")
    idx = {k: _sorted_by_key(np.flatnonzero(labels == k), keys) for k in STRATA}
    rs = [np.random.default_rng(c) for c in np.random.SeedSequence(seed).spawn(3)]
    I, G1 = idx["I"], idx["G1"]
    V = _sorted_by_key(np.concatenate([idx["G1"], idx["U2"], idx["U1"], idx["S"]]), keys)
    G0 = _sorted_by_key(np.concatenate([idx["U2"], idx["U1"], idx["S"]]), keys)
    K = len(I) + len(G1) + s
    rand = np.concatenate([I, rs[0].permutation(V)[: K - len(I)]]).astype(int)
    dis = np.concatenate([I, G1, rs[1].permutation(G0)[:s]]).astype(int)
    aug_order = np.concatenate([rs[2].permutation(idx["U2"]), rs[2].permutation(idx["U1"]),
                                rs[2].permutation(idx["S"])])
    aug = np.concatenate([I, G1, aug_order[:s]]).astype(int)
    cap = lambda sel: int(sum(E[j] for j in sel if labels[j] != "I"))
    return {"seed": seed, "K": K, "reviewed": {"RAND": rand, "DIS": dis, "AUG": aug},
            "sort_key": "ids" if ids is not None else "stored order",
            "C_RAND": cap(rand), "C_DIS": cap(dis), "C_AUG": cap(aug),
            "Delta_draw": cap(aug) - cap(dis)}


# ---------------------------------------------------------------------------
# Coverage regime: existence test, one-sided Fisher exact
# ---------------------------------------------------------------------------
def fisher_one_sided(m1: int, n1: int, m0: int, n0: int, exact: bool = False):
    """P(M1 >= m1 | n1, n0, m_D0) under the hypergeometric null; 1 if m_D0 = 0."""
    t = m1 + m0
    if t == 0:
        return Fraction(1) if exact else 1.0
    hi = min(n1, t)
    num = sum(math.comb(t, x) * math.comb(n1 + n0 - t, n1 - x) for x in range(m1, hi + 1))
    den = math.comb(n1 + n0, n1)
    return Fraction(num, den) if exact else num / den


def fisher_pvalue_table(n1: int, n0: int) -> np.ndarray:
    """One-sided Fisher p for every outcome (x1, x0): array of shape (n1+1, n0+1)."""
    out = np.empty((n1 + 1, n0 + 1))
    c1 = [math.comb(n1, a) for a in range(n1 + 1)]
    c0 = [math.comb(n0, b) for b in range(n0 + 1)]
    for t in range(n1 + n0 + 1):
        amin, amax = max(0, t - n0), min(n1, t)
        w = [c1[a] * c0[t - a] for a in range(amin, amax + 1)]
        tot = sum(w)
        tail = 0
        for j in range(len(w) - 1, -1, -1):
            tail += w[j]
            a = amin + j
            out[a, t - a] = tail / tot
    return out


# ---------------------------------------------------------------------------
# Coverage regime: exact unconditional score interval for delta = p1 - p0
# ---------------------------------------------------------------------------
def _log_comb(n: int) -> np.ndarray:
    k = np.arange(n + 1)
    return special.gammaln(n + 1) - special.gammaln(k + 1) - special.gammaln(n - k + 1)


def binom_pmf_rows(n: int, p: np.ndarray, lgc: Optional[np.ndarray] = None) -> np.ndarray:
    """Binomial pmf, shape (len(p), n+1); exact point masses at p = 0 and p = 1."""
    p = np.atleast_1d(np.asarray(p, dtype=float))
    if lgc is None:
        lgc = _log_comb(n)
    k = np.arange(n + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.exp(lgc[None, :] + k[None, :] * np.log(p)[:, None]
                     + (n - k)[None, :] * np.log1p(-p)[:, None])
    z = p <= 0.0
    if z.any():
        out[z] = 0.0
        out[z, 0] = 1.0
    o = p >= 1.0
    if o.any():
        out[o] = 0.0
        out[o, n] = 1.0
    return out


def restricted_mle_p0(x1, n1: int, x0, n0: int, d: float, iters: int = 80) -> np.ndarray:
    """Restricted MLE of p0 under p1 - p0 = d, vectorized over outcomes.

    Maximizes the concave log-likelihood x1 log p1 + (n1-x1) log(1-p1) +
    x0 log p0 + (n0-x0) log(1-p0) with p1 = p0 + d over
    p0 in [max(0,-d), min(1,1-d)] by bisection on its (decreasing) derivative.
    """
    x1 = np.asarray(x1, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    lo, hi = max(0.0, -d), min(1.0, 1.0 - d)
    a = np.full(np.broadcast(x1, x0).shape, lo)
    b = np.full_like(a, hi)
    with np.errstate(divide="ignore", invalid="ignore"):
        for _ in range(iters):
            mid = 0.5 * (a + b)
            p1 = mid + d
            q1 = (1.0 - d) - mid
            g = x1 / p1 - (n1 - x1) / q1 + x0 / mid - (n0 - x0) / (1.0 - mid)
            up = g > 0
            a = np.where(up, mid, a)
            b = np.where(up, b, mid)
    return 0.5 * (a + b)


def score_z(x1, n1: int, x0, n0: int, d: float) -> np.ndarray:
    """Z(d) = (p1hat - p0hat - d) / sqrt(V~(d)), restricted-MLE variance."""
    x1 = np.asarray(x1, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    p0 = restricted_mle_p0(x1, n1, x0, n0, d)
    p1 = np.clip(p0 + d, 0.0, 1.0)
    v = p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0
    num = x1 / n1 - x0 / n0 - d
    with np.errstate(divide="ignore", invalid="ignore"):
        z = num / np.sqrt(v)
        z = np.where(num == 0, 0.0, z)
        z = np.where((v <= 0) & (num > 0), np.inf, z)
        z = np.where((v <= 0) & (num < 0), -np.inf, z)
    return z


def _sorted_groups(z: np.ndarray):
    """Sort descending; return order and, per sorted position, its tie group's first/last position."""
    order = np.argsort(-z, kind="stable")
    zs = z[order]
    if len(zs) == 1:
        return order, np.array([0]), np.array([0])
    scale = np.maximum(1.0, np.maximum(np.abs(zs[:-1]), np.abs(zs[1:])))
    with np.errstate(invalid="ignore"):
        gap = np.abs(np.diff(zs)) > TIE_RTOL * scale
    ends = np.flatnonzero(np.r_[gap, True])
    starts = np.r_[0, ends[:-1] + 1]
    gid = np.r_[0, np.cumsum(gap)]
    return order, starts[gid], ends[gid]


def nuisance_grid(d: float, G: int = NUISANCE_POINTS) -> np.ndarray:
    """p0 grid over [max(0,-d), min(1,1-d)], dense near both ends (sin^2 spacing)."""
    lo, hi = max(0.0, -d), min(1.0, 1.0 - d)
    t = np.linspace(0.0, 1.0, G)
    p0 = lo + (hi - lo) * np.sin(0.5 * np.pi * t) ** 2
    p0[0], p0[-1] = lo, hi
    return p0


def clopper_pearson(n: int, gamma: float = BB_GAMMA) -> Tuple[np.ndarray, np.ndarray]:
    """Two-sided (1 - gamma) Clopper-Pearson bounds for x = 0..n successes of n."""
    x = np.arange(n + 1)
    lo = np.where(x == 0, 0.0, stats.beta.ppf(gamma / 2, np.maximum(x, 1), n - x + 1))
    hi = np.where(x == n, 1.0, stats.beta.ppf(1 - gamma / 2, x + 1, np.maximum(n - x, 1)))
    return lo, hi


class ExactScoreDiff:
    """Exact unconditional score tests and interval for delta = p1 - p0.

    Outcomes (x1, x0) are enumerated in x1-major order: index = x1 (n0+1) + x0.
    method='full' or 'berger-boos' (the frozen method).
    """

    def __init__(self, n1: int, n0: int, G: int = NUISANCE_POINTS,
                 method: Optional[str] = None, gamma: float = BB_GAMMA):
        if n1 < 1 or n0 < 1:
            raise ValueError("need n1 >= 1 and n0 >= 1")
        method = INTERVAL_METHOD if method is None else method
        if method not in INTERVAL_METHODS:
            raise ValueError(f"method must be one of {INTERVAL_METHODS}")
        self.n1, self.n0, self.G = n1, n0, G
        self.method = method
        self.gamma = gamma if method == "berger-boos" else 0.0
        x1, x0 = np.meshgrid(np.arange(n1 + 1), np.arange(n0 + 1), indexing="ij")
        self.x1 = x1.ravel()
        self.x0 = x0.ravel()
        self.lgc1 = _log_comb(n1)
        self.lgc0 = _log_comb(n0)
        self.n_out = len(self.x1)
        if method == "berger-boos":
            self.cp_lo, self.cp_hi = clopper_pearson(n0, gamma)
        self.refinements = 0          # diagnostic counter
        self.max_refine_gain = 0.0    # largest refined - grid difference seen

    def index(self, x1: int, x0: int) -> int:
        return x1 * (self.n0 + 1) + x0

    # -- building blocks ---------------------------------------------------
    def _set_bounds(self, d: float) -> Tuple[np.ndarray, np.ndarray]:
        """Per x0: the nuisance set [a, b] for p0 at d (empty when a > b)."""
        lo, hi = max(0.0, -d), min(1.0, 1.0 - d)
        if self.method == "full":
            return np.full(self.n0 + 1, lo), np.full(self.n0 + 1, hi)
        return np.maximum(self.cp_lo, lo), np.minimum(self.cp_hi, hi)

    def _setup(self, d: float):
        z = score_z(self.x1, self.n1, self.x0, self.n0, d)
        order, gfirst, glast = _sorted_groups(z)
        p0g = nuisance_grid(d, self.G)
        if self.method == "berger-boos":
            a, b = self._set_bounds(d)
            ok = a <= b
            p0g = np.unique(np.concatenate([p0g, a[ok], b[ok]]))
        p1g = np.clip(p0g + d, 0.0, 1.0)
        B1 = binom_pmf_rows(self.n1, p1g, self.lgc1)
        B0 = binom_pmf_rows(self.n0, p0g, self.lgc0)
        return z, order, gfirst, glast, p0g, B1, B0

    def _rows(self, p0g: np.ndarray, d: float, x0: int) -> np.ndarray:
        a, b = self._set_bounds(d)
        if a[x0] > b[x0]:
            return np.empty(0, dtype=int)
        return np.flatnonzero((p0g >= a[x0]) & (p0g <= b[x0]))

    def _mask(self, order: np.ndarray, positions: slice) -> np.ndarray:
        mask = np.zeros((self.n1 + 1, self.n0 + 1))
        sel = order[positions]
        mask[self.x1[sel], self.x0[sel]] = 1.0
        return mask

    def _tail(self, d: float, mask: np.ndarray, p0: float) -> float:
        p1 = min(max(p0 + d, 0.0), 1.0)
        b1 = binom_pmf_rows(self.n1, np.array([p1]), self.lgc1)[0]
        b0 = binom_pmf_rows(self.n0, np.array([p0]), self.lgc0)[0]
        return float(b1 @ mask @ b0)

    def _refine(self, d: float, mask: np.ndarray, p0g: np.ndarray, vals: np.ndarray) -> float:
        """Local refinement of the nuisance maximum around grid-local maxima.

        p0g is the (sub-)grid the supremum runs over; the bracket of each candidate
        stays between its neighbours on that grid, so it never leaves the nuisance set.
        """
        vmax = float(vals.max())
        left = np.r_[-np.inf, vals[:-1]]
        right = np.r_[vals[1:], -np.inf]
        cand = np.flatnonzero((vals >= left) & (vals >= right) & (vals >= vmax - REFINE_WINDOW))
        cand = cand[np.argsort(-vals[cand], kind="stable")][:5]
        best = vmax
        f = lambda p: -self._tail(d, mask, p)
        for k in cand:
            a, b = p0g[max(k - 1, 0)], p0g[min(k + 1, len(p0g) - 1)]
            if b <= a:
                continue
            res = optimize.minimize_scalar(f, bounds=(a, b), method="bounded",
                                           options={"xatol": 1e-12})
            best = max(best, -float(res.fun))
        self.refinements += 1
        self.max_refine_gain = max(self.max_refine_gain, best - vmax)
        return best

    @staticmethod
    def _needs_refine(v: float, alpha: float) -> bool:
        return alpha - REFINE_WINDOW < v <= alpha

    # -- single outcome ----------------------------------------------------
    def pvalue(self, x1: int, x0: int, d: float, tail: str, alpha: float = ALPHA,
               refine: str = "decision") -> float:
        """One-sided exact unconditional p-value at d for one outcome.

        tail='upper': sup_p0 P(Z(d) >= z_obs), used for the lower bound.
        tail='lower': sup_p0 P(Z(d) <= z_obs), used for the upper bound.
        For 'berger-boos' the supremum runs over the Clopper-Pearson set of x0 and
        gamma is added; an empty set gives p = gamma.
        refine='decision' refines only when it can change the decision at alpha;
        refine='always' always refines; refine='never' returns the grid maximum.
        """
        z, order, gfirst, glast, p0g, B1, B0 = self._setup(d)
        pos = int(np.flatnonzero(order == self.index(x1, x0))[0])
        if tail == "upper":
            region = slice(0, glast[pos] + 1)
        elif tail == "lower":
            region = slice(gfirst[pos], self.n_out)
        else:
            raise ValueError(tail)
        mask = self._mask(order, region)
        vals = ((B1 @ mask) * B0).sum(axis=1)
        if self.method == "berger-boos":
            rows = self._rows(p0g, d, x0)
            if len(rows) == 0:
                return self.gamma
            p0g, vals = p0g[rows], vals[rows]
        v = float(vals.max()) + self.gamma
        if refine == "always" or (refine == "decision" and self._needs_refine(v, alpha)):
            v = self._refine(d, mask, p0g, vals) + self.gamma
        return v

    # -- all outcomes at once -----------------------------------------------
    def pvalues_all(self, d: float, alpha: float = ALPHA,
                    refine: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """(p_upper, p_lower) for every outcome at d, in outcome order.

        Sort outcomes by Z(d), cumulative-sum outcome probabilities along the
        sorted order for every nuisance value, then maximize over the nuisance.
        Refinement follows the same 'decision' rule as pvalue(); refine=False
        returns grid maxima only (used to measure discretization error).
        """
        z, order, gfirst, glast, p0g, B1, B0 = self._setup(d)
        x1s, x0s = self.x1[order], self.x0[order]
        P = B1[:, x1s] * B0[:, x0s]
        np.cumsum(P, axis=1, out=P)                 # P[:, j] = P(sorted position <= j)
        if self.method == "berger-boos":
            return self._pvalues_all_bb(d, alpha, refine, order, gfirst, glast, p0g, P)
        up_col = P.max(axis=0)
        for j in np.unique(glast):
            if refine and self._needs_refine(up_col[j], alpha):
                mask = self._mask(order, slice(0, j + 1))
                up_col[j] = self._refine(d, mask, p0g, P[:, j].copy())
        tot = P[:, -1].copy()
        np.subtract(tot[:, None], P, out=P)         # P[:, j] = P(sorted position > j)
        lo_col = P.max(axis=0)                      # lo_col[j] = sup P(position >= j+1)
        lo_start = np.empty(self.n_out)
        for s in np.unique(gfirst):
            if s == 0:
                val = float(tot.max())
                if refine and self._needs_refine(val, alpha):
                    val = self._refine(d, self._mask(order, slice(0, self.n_out)), p0g, tot.copy())
            else:
                val = float(lo_col[s - 1])
                if refine and self._needs_refine(val, alpha):
                    val = self._refine(d, self._mask(order, slice(s, self.n_out)), p0g,
                                       P[:, s - 1].copy())
            lo_start[s] = val
        p_up = np.empty(self.n_out)
        p_lo = np.empty(self.n_out)
        p_up[order] = up_col[glast]
        p_lo[order] = lo_start[gfirst]
        return p_up, p_lo

    def _pvalues_all_bb(self, d, alpha, refine, order, gfirst, glast, p0g, P):
        """Berger-Boos p-values: per x0, the supremum runs over that x0's nuisance set."""
        g = self.gamma
        tot = P[:, -1].copy()
        a, b = self._set_bounds(d)
        pos = np.empty(self.n_out, dtype=int)
        pos[order] = np.arange(self.n_out)
        p_up = np.full(self.n_out, g)
        p_lo = np.full(self.n_out, g)
        for x0 in range(self.n0 + 1):
            if a[x0] > b[x0]:
                continue
            rows = np.flatnonzero((p0g >= a[x0]) & (p0g <= b[x0]))
            outs = np.arange(self.n1 + 1) * (self.n0 + 1) + x0
            cu = glast[pos[outs]]
            cs = gfirst[pos[outs]]
            Vu = P[np.ix_(rows, cu)]
            sub = P[np.ix_(rows, np.maximum(cs - 1, 0))]
            sub[:, cs == 0] = 0.0
            Vl = tot[rows][:, None] - sub
            vu = Vu.max(axis=0) + g
            vl = Vl.max(axis=0) + g
            if refine:
                for k in range(len(outs)):
                    if self._needs_refine(vu[k], alpha):
                        mask = self._mask(order, slice(0, cu[k] + 1))
                        vu[k] = self._refine(d, mask, p0g[rows], Vu[:, k].copy()) + g
                    if self._needs_refine(vl[k], alpha):
                        mask = self._mask(order, slice(cs[k], self.n_out))
                        vl[k] = self._refine(d, mask, p0g[rows], Vl[:, k].copy()) + g
            p_up[outs] = vu
            p_lo[outs] = vl
        return p_up, p_lo

    def decisions(self, grid: np.ndarray, alpha: float = ALPHA):
        """Accept matrices on a d grid: acc_up[i, o] = p_upper(grid[i]; o) > alpha, etc."""
        acc_up = np.zeros((len(grid), self.n_out), dtype=bool)
        acc_lo = np.zeros((len(grid), self.n_out), dtype=bool)
        for i, d in enumerate(grid):
            pu, pl = self.pvalues_all(float(d), alpha)
            acc_up[i] = pu > alpha
            acc_lo[i] = pl > alpha
        return acc_up, acc_lo

    # -- the interval ------------------------------------------------------
    def interval(self, x1: int, x0: int, extra_grid: Sequence[float] = (),
                 alpha: float = ALPHA) -> dict:
        """Two one-sided (1 - alpha) bounds for delta, outward-rounded to the grid.

        iL is the first grid point the upper-tail test accepts and iU the last one
        the lower-tail test accepts (the hull of the accepted set). delta_L =
        grid[iL - 1] and delta_U = grid[iU + 1] are rejected grid points (or -1 and
        1), and the reported interval is the open interval (delta_L, delta_U).
        """
        grid = d_grid(extra_grid)
        iL = next((i for i, d in enumerate(grid)
                   if self.pvalue(x1, x0, float(d), "upper", alpha) > alpha), None)
        iU = next((i for i in range(len(grid) - 1, -1, -1)
                   if self.pvalue(x1, x0, float(grid[i]), "lower", alpha) > alpha), None)
        if iL is None or iU is None:
            raise RuntimeError("no accepted grid point; the grid is too coarse")
        dL, dU = outward_bounds(grid, iL, iU)
        return {"delta_L": dL, "delta_U": dU, "grid_index_L": iL, "grid_index_U": iU,
                "first_accepted": float(grid[iL]), "last_accepted": float(grid[iU]),
                "grid": grid, "method": self.method}


def d_grid(extra: Sequence[float] = ()) -> np.ndarray:
    """Base grid k/100 (k = -99..99) plus label-free extra points in (-1, 1)."""
    pts = {k / D_GRID_DEN for k in range(-D_GRID_DEN + 1, D_GRID_DEN)}
    pts |= {float(e) for e in extra if -1.0 < float(e) < 1.0}
    return np.array(sorted(pts))


def outward_bounds(grid: np.ndarray, iL, iU):
    """delta_L = grid[iL - 1] (or -1), delta_U = grid[iU + 1] (or 1); works on arrays too."""
    iL = np.asarray(iL)
    iU = np.asarray(iU)
    dL = np.where(iL == 0, -1.0, grid[np.maximum(iL - 1, 0)])
    dU = np.where(iU == len(grid) - 1, 1.0, grid[np.minimum(iU + 1, len(grid) - 1)])
    if dL.ndim == 0:
        return float(dL), float(dU)
    return dL, dU


def interval_indices(acc_up: np.ndarray, acc_lo: np.ndarray):
    """Per outcome: index of the first accepted upper-test grid point and the last
    accepted lower-test grid point (the grid-level interval endpoints)."""
    if not acc_up.any(axis=0).all() or not acc_lo.any(axis=0).all():
        raise RuntimeError("some outcome has no accepted grid point")
    iL = acc_up.argmax(axis=0)
    iU = acc_lo.shape[0] - 1 - acc_lo[::-1].argmax(axis=0)
    return iL, iU


def classify_size(delta_L: float, delta_U: float, delta_m: Optional[float], reachable: bool) -> str:
    """Size classification, decided only by the unconditional interval.

    The interval is open with rejected grid endpoints and delta_m is a grid point,
    so 'exceeds' (delta_L >= delta_m) holds exactly when the upper-tail test rejects
    delta_m and every grid point below it, and 'below' is the mirror image.
    """
    if not reachable:
        return "unreachable-on-panel"
    if delta_L >= delta_m:
        return "exceeds"
    if delta_U <= delta_m:
        return "below"
    return "unresolved"


# ---------------------------------------------------------------------------
# Partial regime
# ---------------------------------------------------------------------------
def permutation_p(design: Design, m: Mapping[str, int]) -> Fraction:
    """Exact within-D permutation p-value of Delta, as a Fraction.

    Holding E fixed and shuffling R within D = 0, (m_U2, m_U1, m_S) is
    multivariate hypergeometric given (n_U2, n_U1, n_S, m_D0). Delta is linear in
    these counts with exact weights delta_weights(design), so the enumeration
    is exact. Valid in every regime; in coverage it equals Fisher's p.
    """
    w = delta_weights(design)
    nU2, nU1, nS = design.n_U2, design.n_U1, design.n_S
    t = m["U2"] + m["U1"] + m["S"]
    obs = w["U2"] * m["U2"] + w["U1"] * m["U1"] + w["S"] * m["S"]
    acc = 0
    for a in range(max(0, t - nU1 - nS), min(nU2, t) + 1):
        for b in range(max(0, t - a - nS), min(nU1, t - a) + 1):
            c = t - a - b
            if w["U2"] * a + w["U1"] * b + w["S"] * c >= obs:
                acc += math.comb(nU2, a) * math.comb(nU1, b) * math.comb(nS, c)
    return Fraction(acc, math.comb(design.n_D0, t))


def partial_bootstrap(design: Design, m: Mapping[str, int], B: int = 10000, seed: int = 0,
                      level: float = 0.95) -> dict:
    """APPROXIMATE Jeffreys-smoothed parametric bootstrap interval for G_partial.

    G_partial = (100/N) sum_j w_j n_j p_j over j in {U2, U1, S}, w = delta_weights.
    Implementation choice: smoothed plug-in p~_j = (m_j + 1/2)/(n_j + 1); B draws of
    m*_j ~ Binomial(n_j, p~_j), independent across strata; percentile interval of
    (100/N) sum_j w_j m*_j. No threshold claim is ever made from it.
    """
    w = {k: float(v) for k, v in delta_weights(design).items()}
    n = {"U2": design.n_U2, "U1": design.n_U1, "S": design.n_S}
    rng = np.random.default_rng(seed)
    tot = np.zeros(B)
    for j in ("U2", "U1", "S"):
        if n[j] == 0:
            continue
        pt = (m[j] + 0.5) / (n[j] + 1.0)
        tot += w[j] * rng.binomial(n[j], pt, size=B)
    tot *= 100.0 / design.N_per100
    q = (1 - level) / 2
    point = 100.0 / design.N_per100 * sum(w[j] * m[j] for j in ("U2", "U1", "S"))
    return {"G_partial_hat": point, "lower": float(np.quantile(tot, q)),
            "upper": float(np.quantile(tot, 1 - q)), "B": B, "seed": seed,
            "label": "approximate"}


# ---------------------------------------------------------------------------
# Fragility of the existence decision
# ---------------------------------------------------------------------------
def existence_fragility(design: Design, counts: Mapping[str, int], reject: bool,
                        k_max: int = FRAGILITY_MAX) -> Optional[int]:
    """Smallest number of E flips among agreeing items that reverses the existence decision.

    A flip changes one agreeing item's discordance. When the test rejects, flips
    move toward no gain (a discordant U item becomes concordant, or a concordant S
    item discordant); otherwise the other way. Coverage uses Fisher's p on (m1, m0);
    the partial regime uses the permutation p on (m_U2, m_U1, m_S). Returns None when
    more than k_max flips are needed.
    """
    thr = Fraction(1, 40)
    sgn = -1 if reject else 1
    mU2, mU1, mS = counts["U2"], counts["U1"], counts["S"]
    if design.regime == "COVERAGE":
        m1, n1, n0 = mU2 + mU1, design.n1, design.n0
        for k in range(1, k_max + 1):
            for fu in range(k + 1):
                a, c = m1 + sgn * fu, mS - sgn * (k - fu)
                if 0 <= a <= n1 and 0 <= c <= n0:
                    if (fisher_one_sided(a, n1, c, n0, exact=True) <= thr) != reject:
                        return k
        return None
    for k in range(1, k_max + 1):
        for f2 in range(k + 1):
            for f1 in range(k - f2 + 1):
                a2, a1, c = mU2 + sgn * f2, mU1 + sgn * f1, mS - sgn * (k - f2 - f1)
                if 0 <= a2 <= design.n_U2 and 0 <= a1 <= design.n_U1 and 0 <= c <= design.n_S:
                    if (permutation_p(design, {"U2": a2, "U1": a1, "S": c}) <= thr) != reject:
                        return k
    return None


# ---------------------------------------------------------------------------
# Template sensitivity
# ---------------------------------------------------------------------------
def template_subsample(labels: Sequence[str], templates: Sequence[str],
                       ids: Optional[Sequence[str]] = None, seed: int = 0) -> Tuple[List[int], dict]:
    """One agreeing item per template, chosen label-free.

    Agreeing items (U2, U1, S) are grouped by template id; within a template, items
    are ordered by the sort key (ids, else stored order) and one is drawn uniformly
    with a generator from SeedSequence(seed, spawn_key=(3,)), independent of the
    operator's-draw streams. The choice never looks at R or E, so the subsample's
    (R, E) pairs are an unbiased, template-independent sample under the model.
    """
    labels = list(labels)
    keys = list(ids) if ids is not None else list(range(len(labels)))
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, (l, t) in enumerate(zip(labels, templates)):
        if l in ("U2", "U1", "S"):
            groups[str(t)].append(i)
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(3,)))
    chosen = []
    for t in sorted(groups):
        members = sorted(groups[t], key=lambda i: keys[i])
        chosen.append(members[int(rng.integers(len(members)))])
    mult = [len(v) for v in groups.values()]
    info = {"n_G0": sum(mult), "n_templates": len(groups), "max_multiplicity": max(mult, default=0),
            "items_sharing": sum(x for x in mult if x > 1)}
    return sorted(chosen), info


def template_sensitivity(labels: Sequence[str], E: Sequence[int], templates: Sequence[str],
                         ids: Optional[Sequence[str]] = None, seed: int = 0,
                         method: Optional[str] = None) -> dict:
    """Existence p and delta interval on one agreeing item per template (labelled sensitivity)."""
    chosen, info = template_subsample(labels, templates, ids, seed)
    lab = [labels[i] for i in chosen]
    e = [int(E[i]) for i in chosen]
    n1 = sum(l in ("U2", "U1") for l in lab)
    n0 = sum(l == "S" for l in lab)
    m1 = sum(x for l, x in zip(lab, e) if l in ("U2", "U1"))
    m0 = sum(x for l, x in zip(lab, e) if l == "S")
    out = dict(info, n1=n1, n0=n0, m1=m1, m0=m0, seed=seed)
    if n1 >= 1 and n0 >= 1:
        p = fisher_one_sided(m1, n1, m0, n0)
        ci = ExactScoreDiff(n1, n0, method=method).interval(m1, m0, extra_grid=[0.0])
        out.update(p=p, reject=p <= ALPHA, delta_hat=m1 / n1 - m0 / n0,
                   delta_L=ci["delta_L"], delta_U=ci["delta_U"])
    return out


# ---------------------------------------------------------------------------
# Analysis and report
# ---------------------------------------------------------------------------
def label_free_summary(design: Design, m: float = MATERIAL_M) -> dict:
    """Stage-0 quantities, all label-free, committed before the join."""
    dm = design.delta_threshold(m)
    return {"N": design.N, "N_per100": design.N_per100, "n_inv": design.n_inv, "N_V": design.N_V,
            "n_D1": design.n_D1, "n_D0": design.n_D0, "n_U2": design.n_U2, "n_U1": design.n_U1,
            "n1": design.n1, "n0": design.n0, "s_nom": design.s_nom, "s": design.s, "K": design.K,
            "s_held": design.s_override is not None,
            "cap_binds": design.cap_binds, "regime": design.regime, "L": design.L,
            "delta_m": dm, "reachable": design.threshold_reachable(m), "m": m}


def analyze(labels: Sequence[str], E: Sequence[int], seed: Optional[int] = None,
            m: float = MATERIAL_M, boot_B: int = 10000, boot_seed: int = 0,
            s_override: Optional[int] = None, N_ref: Optional[int] = None,
            ids: Optional[Sequence[str]] = None, method: Optional[str] = None,
            fragility: bool = True) -> dict:
    """Primary analysis of one panel [labels]. Returns a result dict for report()."""
    method = INTERVAL_METHOD if method is None else method
    labels = list(labels)
    E = [int(e) for e in E]
    if len(labels) != len(E):
        raise ValueError("labels and E differ in length")
    keep = [i for i, l in enumerate(labels) if l != "EXCL"]
    labels_k = [labels[i] for i in keep]
    E_k = [E[i] for i in keep]
    ids_k = [ids[i] for i in keep] if ids is not None else None
    design = Design.from_labels(labels_k, s_override=s_override, N_ref=N_ref)
    counts = {j: sum(e for l, e in zip(labels_k, E_k) if l == j) for j in ("G1", "U2", "U1", "S")}
    lf = label_free_summary(design, m)
    cap = expected_captures(design, counts)
    res = {"label_free": lf, "m_counts": counts, "M": sum(counts.values()), "method": method,
           "captures": cap, "Delta_per100": float(cap["Delta"]) * 100 / design.N_per100}
    if seed is not None:
        res["draw"] = one_operators_draw(labels_k, E_k, design.s, seed, ids=ids_k)
    m1, m0 = counts["U2"] + counts["U1"], counts["S"]
    res["m1"], res["m0"] = m1, m0
    reg = design.regime
    if reg == "COVERAGE":
        p = fisher_one_sided(m1, design.n1, m0, design.n0)
        res["existence"] = {"test": "one-sided Fisher exact (equals the exact within-D "
                                    "permutation p-value of Delta)", "p": p,
                            "reject": p <= ALPHA}
        dm = lf["delta_m"]
        extra = [0.0] + ([float(dm)] if lf["reachable"] else [])
        ci = ExactScoreDiff(design.n1, design.n0, method=method).interval(m1, m0, extra_grid=extra)
        scale = 100.0 * float(design.L) / design.N_per100
        dL, dU = ci["delta_L"], ci["delta_U"]
        res["size"] = {"delta_hat": m1 / design.n1 - m0 / design.n0,
                       "delta_L": dL, "delta_U": dU, "G_L": scale * dL, "G_U": scale * dU,
                       "G_hat": scale * (m1 / design.n1 - m0 / design.n0),
                       "grid_index_L": ci["grid_index_L"], "grid_index_U": ci["grid_index_U"],
                       "first_accepted": ci["first_accepted"], "last_accepted": ci["last_accepted"],
                       "class": classify_size(dL, dU, None if dm is None else float(dm), lf["reachable"])}
        # Flag when the two questions point different ways.
        ex_rej = res["existence"]["reject"]
        res["size"]["disagrees_with_existence"] = (ex_rej and dL < 0.0) or ((not ex_rej) and dL >= 0.0)
    elif reg == "PARTIAL":
        p = permutation_p(design, counts)
        res["existence"] = {"test": "exact within-D permutation p-value of Delta "
                                    "(multivariate hypergeometric enumeration)",
                            "p": float(p), "reject": p <= Fraction(1, 40)}
        res["size"] = {"partial": partial_bootstrap(design, counts, boot_B, boot_seed),
                       "class": "no threshold claim (partial regime)"}
        res["weights"] = delta_weights(design)
    else:
        res["existence"] = None
        res["size"] = None
    if fragility and res["existence"] is not None:
        res["existence"]["fragility"] = existence_fragility(design, counts, res["existence"]["reject"])
    return res


def _f(x, nd=4):
    return f"{float(x):.{nd}f}"


def _interval_text(lo: float, hi: float, nd: int = 4) -> str:
    left = "[" if lo <= -1.0 else "("
    right = "]" if hi >= 1.0 else ")"
    return f"{left}{_f(lo, nd)}, {_f(hi, nd)}{right}"


def _g_interval_text(lo: float, hi: float, dlo: float, dhi: float) -> str:
    left = "[" if dlo <= -1.0 else "("
    right = "]" if dhi >= 1.0 else ")"
    return f"{left}{_f(lo)}, {_f(hi)}{right}"


def report(res: dict) -> str:
    """Human-readable report of one analysis. Existence and size are printed separately."""
    lf, cap = res["label_free"], res["captures"]
    reg = lf["regime"]
    out = []
    out.append("Primary analysis, statistical analysis plan v3.1")
    out.append("")
    out.append("Label-free design (committed before the join)")
    out.append(f"  N = {lf['N']}, invalid A1 n_inv = {lf['n_inv']}, valid N_V = {lf['N_V']}")
    out.append(f"  disagreeing n_D1 = {lf['n_D1']}, agreeing n_D0 = {lf['n_D0']}: "
               f"unstable n1 = {lf['n1']} (U2 = {lf['n_U2']}, U1 = {lf['n_U1']}), stable n0 = {lf['n0']}")
    s_src = "held at the primary value" if lf.get("s_held") else f"s_nom = {lf['s_nom']}"
    out.append(f"  slice s = {lf['s']} ({s_src}, cap binds: "
               f"{'yes' if lf['cap_binds'] else 'no'}); review count K = {lf['K']} "
               f"({100 * lf['K'] / lf['N']:.1f}% of N), identical for all three policies")
    if lf.get("N_per100", lf["N"]) != lf["N"]:
        out.append(f"  per-100 denominator held at the primary N = {lf['N_per100']}")
    if reg == "COVERAGE":
        out.append(f"  regime: {reg}; L = {_f(lf['L'])}")
        if lf["delta_m"] is not None:
            out.append(f"  threshold m = {lf['m']} per 100 -> delta_m = {_f(lf['delta_m'])}; "
                       f"reachable on this panel: {'yes' if lf['reachable'] else 'no'}")
    else:
        # L and delta_m are coverage-regime quantities.
        out.append(f"  regime: {reg}")
    out.append("")
    out.append("Expected captures (exact, tie-averaged)")
    out.append(f"  E[C_Rand] = {_f(cap['RAND'])}, E[C_DIS] = {_f(cap['DIS'])}, E[C_AUG] = {_f(cap['AUG'])}")
    out.append(f"  Delta = E[C_AUG] - E[C_DIS] = {_f(cap['Delta'])} ({_f(res['Delta_per100'])} per 100)")
    out.append(f"  Delta_DR = E[C_DIS] - E[C_Rand] = {_f(cap['Delta_DR'])} (sanity check)")
    if "draw" in res:
        dr = res["draw"]
        out.append(f"  one operator's draw (seed {dr['seed']}, items ordered by {dr['sort_key']}): "
                   f"C_Rand = {dr['C_RAND']}, C_DIS = {dr['C_DIS']}, C_AUG = {dr['C_AUG']}, "
                   f"difference = {dr['Delta_draw']}")
    out.append("")
    if reg == "DEGENERATE-NONE":
        out.append("Existence question: not run. n1 = 0, so Delta = 0 identically.")
        out.append("  Finding: no instability among agreeing verdicts.")
        out.append("Size question: not applicable.")
        return "\n".join(out)
    if reg == "DEGENERATE-CAP":
        out.append("Existence question: not run. The cap binds (s = n_D0): both policies "
                   "review all agreeing items, so Delta = 0 identically.")
        out.append("Size question: not applicable.")
        return "\n".join(out)
    if reg == "DEGENERATE-NOSLICE":
        out.append("Existence question: not run. s = 0: neither policy reviews any agreeing "
                   "item, so Delta = 0 identically.")
        out.append("Size question: not applicable.")
        return "\n".join(out)
    ex = res["existence"]
    out.append("EXISTENCE QUESTION: is there any positive gain?")
    if reg == "COVERAGE":
        out.append("  H0: p1 = p0, equivalently discordance is exchangeable between U and S given "
                   "the margins; H1: p1 > p0.")
        out.append(f"  Statistic: m1, the number of reference-discordant first verdicts in U. "
                   f"p = P(M1 >= m1 | n1, n0, m_D0), one-sided Fisher exact test, which equals "
                   f"the exact within-D permutation p-value of Delta.")
        out.append(f"  m1 = {res['m1']} of n1 = {lf['n1']}; m0 = {res['m0']} of n0 = {lf['n0']}; "
                   f"m_D0 = {res['m1'] + res['m0']}")
        yes, no = "evidence at one-sided 0.025 that p1 > p0", "no evidence at one-sided 0.025 that p1 > p0"
    else:
        w = res["weights"]
        out.append("  H0: given E, the instability labels R are exchangeable within D = 0, so "
                   "p_U2 = p_U1 = p_S; H1: discordance concentrates where instability-first "
                   "review looks first.")
        out.append(f"  Statistic: Delta = w_U2 m_U2 + w_U1 m_U1 + w_S m_S with w_U2 = {_f(w['U2'])}, "
                   f"w_U1 = {_f(w['U1'])}, w_S = {_f(w['S'])}; p = P(Delta* >= Delta) under the "
                   f"within-D shuffle of R, exact by multivariate hypergeometric enumeration.")
        mc = res["m_counts"]
        out.append(f"  m_U2 = {mc['U2']} of {lf['n_U2']}; m_U1 = {mc['U1']} of {lf['n_U1']}; "
                   f"m_S = {mc['S']} of {lf['n0']}")
        yes = "evidence at one-sided 0.025 against exchangeability, toward a positive Delta"
        no = "no evidence at one-sided 0.025 against exchangeability"
    out.append(f"  p = {ex['p']:.6g}; alpha = {ALPHA}: "
               f"{('REJECT H0: ' + yes) if ex['reject'] else ('do not reject H0: ' + no)}")
    fr = ex.get("fragility")
    if "fragility" in ex:
        out.append(f"  Fragility: {fr if fr is not None else 'more than ' + str(FRAGILITY_MAX)} "
                   f"reference-label flip{'s' if fr != 1 else ''} among agreeing items would "
                   f"reverse this decision.")
    out.append("")
    sz = res["size"]
    out.append(f"SIZE QUESTION: does the gain exceed m = {lf['m']} per 100?")
    if reg == "COVERAGE":
        meth = res.get("method", INTERVAL_METHOD)
        if lf["reachable"]:
            out.append(f"  H0_thr: δ ≤ δ_m (G ≤ m) against δ > δ_m, with δ_m = {_f(lf['delta_m'])}.")
        else:
            out.append("  H0_thr: δ ≤ δ_m (G ≤ m); here δ_m ≥ 1, so G ≤ m for every δ and "
                       "'exceeds' is impossible.")
        out.append("  Statistic: Z(d), the Farrington-Manning score statistic with restricted-MLE "
                   "variance; exact unconditional one-sided p-value, the "
                   f"{NUISANCE_TEXT[meth]}.")
        out.append("  'Exceeds' iff the upper-tail test rejects δ_m and every grid point below it; "
                   "'below' iff the lower-tail test rejects δ_m and every grid point above it.")
        out.append("  Interval: two one-sided 97.5% bounds (Chan-Zhang inversion), grid k/100 plus "
                   "0 and δ_m, outward-rounded: both ends are rejected grid points, so the "
                   "interval is open.")
        out.append(f"  delta_hat = {_f(sz['delta_hat'])}; interval for δ: "
                   f"{_interval_text(sz['delta_L'], sz['delta_U'])}")
        out.append("  " + G_DEFINITION_TEXT)
        out.append(f"  G_hat = {_f(sz['G_hat'])} per 100 (the realized Delta per 100); interval for G: "
                   f"{_g_interval_text(sz['G_L'], sz['G_U'], sz['delta_L'], sz['delta_U'])} per 100")
        cls = sz["class"]
        msg = {"exceeds": "the gain exceeds the threshold (δ_L ≥ δ_m: δ_m is rejected)",
               "below": "the gain is below the threshold (δ_U ≤ δ_m: δ_m is rejected)",
               "unresolved": "size unresolved (the interval contains δ_m)",
               "unreachable-on-panel": "the threshold is unreachable on this panel "
                                       "(δ_m ≥ 1, decided label-free)"}[cls]
        out.append(f"  Size result: {cls}: {msg}")
        if sz.get("disagrees_with_existence"):
            if ex["reject"]:
                why = (f"with n1 = {lf['n1']} that worst case can sit at p0 values the observed "
                       f"m0 = {res['m0']} of {lf['n0']} rules out" if meth == "full" else
                       "even over the plausible range of p0 the worst case keeps zero gain in the interval")
                out.append(
                    "  Reconciliation: the existence test rejects, yet the interval for G contains 0. "
                    "The two use different tests on purpose. Fisher's test conditions on the number "
                    "of discordant agreeing verdicts; the unconditional interval takes the worst case "
                    f"over the unknown stable rate p0, and {why}. Existence is decided by Fisher's p "
                    "alone; the interval only sizes the gain.")
            else:
                out.append(
                    "  Reconciliation: the existence test does not reject, yet the interval for G "
                    "excludes 0 on the left. The two use different tests on purpose; existence is "
                    "decided by Fisher's p alone, and the interval only sizes the gain.")
        out.append("")
        out.append(COVERS_TEXT)
    else:
        b = sz["partial"]
        out.append(f"  Partial regime (n1 > s): AUG reviews only s = {lf['s']} of the n1 = {lf['n1']} "
                   f"unstable items, so the coverage identity, L and δ_m do not apply.")
        out.append(f"  Delta = {_f(cap['Delta'])}; G_partial_hat = {_f(b['G_partial_hat'])} per 100")
        out.append(f"  APPROXIMATE Jeffreys-smoothed parametric bootstrap interval for G_partial: "
                   f"[{_f(b['lower'])}, {_f(b['upper'])}] per 100 (B = {b['B']}, seed {b['seed']})")
        out.append("  What it covers, approximately: G_partial = (100/N) Σ_j w_j n_j p_j, the "
                   "expected Delta per 100 under independent binomial discordance in U2, U1 and S "
                   "for this design. Its coverage is not guaranteed; the synthetic checks give "
                   "informational Monte Carlo coverage.")
        out.append("  No threshold claim is made in either direction in this regime.")
    out.append("")
    out.append("The two questions are answered separately and are not merged into one verdict.")
    return "\n".join(out)


def print_report(res: dict) -> str:
    """Print the report (existence and size separately) and return its text."""
    txt = report(res)
    print(txt)
    return txt


SENSITIVITY_VARIANTS = [
    ("sensitivity: B1 INVALID treated as agreement (D = 0) instead of D = 1", {"b1_invalid": "agree"}),
    ("sensitivity: R_unobserved treated as R = 1", {"r_unobserved": "one"}),
    ("sensitivity: every trajectory with any INVALID slot excluded", {"exclude_any_invalid": True}),
    ("sensitivity: one INVALID repeat and no observed flip treated as R = 1",
     {"r_unobserved": "one_any"}),
]


def analyze_with_sensitivities(slots: Sequence[Mapping[str, str]], E: Sequence[int],
                               **kw) -> List[Tuple[str, dict]]:
    """Primary analysis plus the labelled sensitivity analyses for invalid responses.

    Every sensitivity analysis holds s at its primary value (capped at
    the new n_D0) and the per-100 denominator N at the primary N, so K changes only
    through reclassified items and the analysis measures the invalid rule, not a
    budget change.
    """
    kw = dict(kw)
    kw.pop("s_override", None)
    kw.pop("N_ref", None)
    lf0 = derive_label_free(slots)
    prim = analyze(lf0["stratum"], E, **kw)
    s0, N0 = prim["label_free"]["s"], prim["label_free"]["N"]
    out = [("primary", prim)]
    skw = {k: v for k, v in kw.items() if k not in ("seed",)}
    for name, opts in SENSITIVITY_VARIANTS:
        lf = derive_label_free(slots, **opts)
        out.append((name, analyze(lf["stratum"], E, s_override=s0, N_ref=N0, **skw)))
    return out


def analyze_panel(slots: Sequence[Mapping[str, str]], E: Sequence[int], *,
                  ids: Optional[Sequence[str]] = None,
                  benchmarks: Optional[Sequence[str]] = None,
                  templates: Optional[Sequence[str]] = None,
                  provenance: Optional[dict] = None,
                  seed: Optional[int] = None, m: float = MATERIAL_M,
                  method: Optional[str] = None, boot_B: int = 10000, boot_seed: int = 0) -> dict:
    """Everything the frozen report needs: primary analysis, labelled
    sensitivity analyses, invalid counts, repeat-observation counts, template
    sensitivity and A1 provenance."""
    kw = dict(m=m, method=method, boot_B=boot_B, boot_seed=boot_seed)
    sens = analyze_with_sensitivities(slots, E, seed=seed, ids=ids, **kw)
    lf = derive_label_free(slots)
    g0 = np.isin(lf["stratum"], ["U2", "U1", "S"])
    rep = {"both_repeats_invalid": int(((lf["R_unobserved"] == 1) & g0).sum()),
           "one_repeat_invalid_no_flip": int(((lf["R_one_missing"] == 1) & (lf["R"] == 0) & g0).sum()),
           "one_repeat_invalid_flip": int(((lf["R_one_missing"] == 1) & (lf["R"] >= 1) & g0).sum())}
    out = {"primary": sens[0][1], "sensitivities": sens[1:], "invalid_counts": invalid_counts(slots, benchmarks),
           "repeat_flags": rep, "provenance": provenance, "templates": None,
           "n_trajectories": len(slots)}
    if templates is not None:
        out["templates"] = template_sensitivity(list(lf["stratum"]), E, templates, ids,
                                                seed=0 if seed is None else seed, method=method)
    return out


def _brief(name: str, r: dict) -> List[str]:
    lf = r["label_free"]
    head = (f"- {name}: N = {lf['N']}, s = {lf['s']}, K = {lf['K']}, n1 = {lf['n1']}, "
            f"n0 = {lf['n0']}, regime {lf['regime']}; Delta = {_f(r['Delta_per100'])} per 100")
    lines = [head]
    if r["existence"] is not None:
        ex = r["existence"]
        lines.append(f"    existence p = {ex['p']:.6g} ({'reject' if ex['reject'] else 'do not reject'})")
    if r["size"] is not None and "delta_L" in r["size"]:
        sz = r["size"]
        lines.append(f"    interval for δ {_interval_text(sz['delta_L'], sz['delta_U'])}; for G "
                     f"{_g_interval_text(sz['G_L'], sz['G_U'], sz['delta_L'], sz['delta_U'])} per 100; "
                     f"size: {sz['class']}")
    elif r["size"] is not None:
        lines.append(f"    size: {r['size']['class']}")
    return lines


def panel_report(panel: dict) -> str:
    """The frozen report: provenance, primary analysis, sensitivities, invalid counts, templates."""
    prim = panel["primary"]
    lf = prim["label_free"]
    out = []
    prov = panel.get("provenance")
    out.append("A1 provenance (frozen rule)")
    if prov is not None:
        a1 = prov["by_slot"]["A1"]
        cats = ", ".join(f"{k.split(':', 1)[1]} {v}" for k, v in sorted(a1.items()) if k.startswith("invalid:"))
        out.append(f"  A1 is the first complete response of slot (i, A, 1) under config hash "
                   f"{prov['config_hash']}; it is never replaced by a repeat or a later call.")
        out.append(f"  A1 slots: {a1.get('slots', 0)}; slots that needed transport retries: "
                   f"{a1.get('slots_with_retries', 0)} ({a1.get('transport_retries', 0)} retries in total); "
                   f"halts logged: {a1.get('halts', 0)}; A1 INVALID: {a1.get('invalid', 0)}"
                   f"{' (' + cats + ')' if cats else ''}.")
        tot_retry = sum(v.get("transport_retries", 0) for v in prov["by_slot"].values())
        n_ids = sum(v.get("response_ids", 0) for v in prov["by_slot"].values())
        n_complete = sum(v.get("slots", 0) - v.get("exhausted", 0) for v in prov["by_slot"].values())
        out.append(f"  All slots: {tot_retry} transport retries, "
                   f"{sum(v.get('halts', 0) for v in prov['by_slot'].values())} halts, "
                   f"{sum(v.get('exhausted', 0) for v in prov['by_slot'].values())} slots exhausted; "
                   f"response ids checked on {n_ids} of {n_complete} complete responses, none repeated "
                   f"(a payload-keyed response cache would repeat them).")
    else:
        out.append("  Not supplied: resolved slots were passed without the attempt log. The frozen rule "
                   "is that A1 is the first complete response of slot (i, A, 1) and is never replaced.")
    out.append("")
    out.append(report(prim))
    out.append("")
    rf = panel["repeat_flags"]
    out.append("Repeat observation (label-free)")
    out.append(f"  agreeing items with both repeats INVALID (R_unobserved, R set to 0): {rf['both_repeats_invalid']}")
    out.append(f"  agreeing items with one repeat INVALID and no observed flip (R = 0 from one repeat): "
               f"{rf['one_repeat_invalid_no_flip']}")
    out.append(f"  agreeing items with one repeat INVALID and an observed flip: {rf['one_repeat_invalid_flip']}")
    out.append("")
    out.append(f"Sensitivity analyses (labelled; s = {lf['s']} and the per-100 denominator N = {lf['N']} "
               f"are held at their primary values, s capped at n_D0)")
    for name, r in panel["sensitivities"]:
        out.extend(_brief(name, r))
    out.append("")
    out.append("Invalid counts by judge slot, category and benchmark")
    inv = panel["invalid_counts"]
    if not inv:
        out.append("  none")
    else:
        out.append("  slot | category | benchmark | count")
        for (slot, cat, bench), c in sorted(inv.items()):
            out.append(f"  {slot} | {cat} | {bench} | {c}")
    tm = panel.get("templates")
    out.append("")
    out.append("Template sensitivity (labelled)")
    if tm is None:
        out.append("  Not run: no template ids were supplied with the panel manifest.")
    else:
        out.append(f"  Agreeing items: {tm['n_G0']} in {tm['n_templates']} templates; largest template "
                   f"holds {tm['max_multiplicity']}; {tm['items_sharing']} items share a template.")
        if "p" in tm:
            out.append(f"  One agreeing item per template (seeded, label-free): m1 = {tm['m1']} of "
                       f"n1 = {tm['n1']}, m0 = {tm['m0']} of n0 = {tm['n0']}; existence p = {tm['p']:.6g} "
                       f"({'reject' if tm['reject'] else 'do not reject'}); interval for δ "
                       f"{_interval_text(tm['delta_L'], tm['delta_U'])}")
        else:
            out.append(f"  One agreeing item per template leaves n1 = {tm['n1']}, n0 = {tm['n0']}; no test.")
    return "\n".join(out)
