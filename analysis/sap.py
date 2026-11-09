"""Statistical analysis of the instability-first review study: reference implementation.

The primary path is descriptive and implements version 4.0 of the study's analysis
plan, dated Nov 8, 2026 (PLAN_VERSION). describe_panel() audits the frozen panel: the
exact expected capture of each policy over its own random choices, the full A1 and
B1 validity metrics, invalid outputs, Unsure and missing references and actual
inference costs. describe_report() prints it. The conclusion applies to these tasks
and these realized judge calls, with no population claim. analyze_panel() runs this
path unless told otherwise.

The population inference of the study's statistical analysis plan, version 3.2, is
outside the primary result and separately scoped. It stays in this module, and its
entry points analyze(), analyze_with_sensitivities(), analyze_panel() and
template_sensitivity() run it only on the explicit argument
path="population_inference". The numeric building blocks they call, such as
fisher_one_sided(), take no path. Everything below this paragraph describes that
path unless it says otherwise, and "primary" there means the plan's result per 100
panel tasks, which that path's report calls the per-100-panel-task result. One
piece of it is still to come. The
inference that handles the template dependence retained on the
one-trajectory-per-task panel, and its synthetic validation, do not exist. Until
they do, the path's test and interval assume independent tasks, and its report says so.
The method is frozen only at the study freeze; until then "frozen" in this file
names the rule the plan fixes, not a completed freeze. Comments name the plan's
sections by topic: units, budget, policies, regimes, the coverage regime, invalid
responses, the partial regime and the panel. Where the code has to fill a gap in
the plan, the comment says "Implementation choice".

Results depend only on the inputs and seeds: label-free counts, plus the
reference discordance E and the reference statuses only where the plan marks a
quantity as needing labels. Nothing here reads or writes files.

Main rules
----------
- Outcomes. FILTERED is a terminal INVALID outcome. Verdicts match
  case-insensitively, as the plan's parser does.
- Fragility. The fragility count is the fewest label flips that reverse the
  existence decision. In the partial regime the search tries every direction,
  since a reversal can need, for example, fewer discordant U1 tasks when the
  weight of U1 is negative. When no set of label flips can reverse the decision,
  the count is reported as not defined.
- Request and configuration halts. HTTP 400 that is not a spend or usage limit
  or a provider content-policy error, HTTP 413 and the provider's equivalent
  invalid-request errors halt the run, as the other halt causes do. A halt is never INVALID; a slot that ends in HALT
  is unfinished, and the analysis refuses it (UnfinishedSlot).
- Unsure and missing labels. A task whose released primary label is Unsure or
  missing stays in the panel, in the selections and in the review count K. The
  primary result is per 100 panel tasks (N) and counts such a task as not
  reference-contradicted (E = 0); its fragility count flips usable labels only.
  The secondary result is the same gain rescaled per 100 tasks with a usable label
  (N_Y), 100 Delta / N_Y. Coverage of Unsure and missing labels is reported
  separately; these tasks are unknown, never correct. See join_reference(), analyze(), secondary_analysis() and
  reference_coverage().
- Panel unit. The default panel is one trajectory per task; the
  one-task-per-template panel is a labeled sensitivity analysis, and the wider
  template exclusion is the default panel with the tasks of every
  development-split template dropped. See select_panel(). The inference for the retained dependence is not yet
  implemented (see above).

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

Two nuisance sets are implemented. 'full': the supremum runs over the whole
admissible range of p0. 'berger-boos': the supremum runs over a 99.9%
Clopper-Pearson set for p0 computed from m0 of n0, and BB_GAMMA = 0.001 is
added (Berger and Boos 1994). INTERVAL_METHOD selects the frozen one; the plan
freezes 'berger-boos'.

Interval endpoints. Tests are run on the d grid k/100 plus the
label-free points 0 and delta_m. delta_L is the grid point just below the first
accepted grid point of the upper-tail test, and delta_U the grid point just
above the last accepted grid point of the lower-tail test. Both are rejected
points, and the reported interval is the OPEN interval (delta_L, delta_U), with
-1 and 1 as closed ends when nothing is rejected on that side. Because every
outcome uses the same grid, the one-sided miss probability at any delta, on or
off the grid, is bounded by the size of one composite one-sided test, so
coverage holds everywhere, not only at grid points; the cost is at most one
grid step (0.01) per side. Endpoints are not refined between grid points by
bisection, because such a search can land inside a sub-grid hole of the
saw-toothed p-value.
"""
from __future__ import annotations

import bisect
import math
import numbers
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
NUISANCE_POINTS = 1001             # grid size for the maximum over p0
REFINE_WINDOW = 2e-3               # see module docstring
TIE_RTOL = 1e-10                   # score values this close are treated as tied
STRATA = ("I", "G1", "U2", "U1", "S")
INTERVAL_METHODS = ("full", "berger-boos")
INTERVAL_METHOD = "berger-boos"    # the frozen interval method; "full" is kept for comparison
BB_GAMMA = 0.001                   # Berger-Boos: 1 - 0.999 Clopper-Pearson set for p0
FRAGILITY_MAX = 15                 # search depth for the fragility count
PLAN_VERSION = "4.0, Nov 8, 2026"   # the analysis plan the descriptive path implements
POPULATION_PLAN_VERSION = "v3.2"   # the plan the population-inference path implements
PATH_DESCRIPTIVE = "descriptive"   # the primary path and the default
PATH_POPULATION = "population_inference"   # outside the primary result, separately scoped

NUISANCE_TEXT = {
    "full": "supremum over the whole admissible range of p0",
    "berger-boos": ("supremum over a 99.9% Clopper-Pearson set for p0 from m0 of n0, "
                    "plus 0.001 (Berger and Boos 1994; the plan's method)"),
}

# What the interval covers. The plan requires this fixed statement in every
# coverage-regime report.
COVERS_BULLETS = (
    "- p1 and p0 are the rates of reference-contradicted first verdicts among unstable "
    "and stable agreeing tasks, with a task that has no usable reference label counted "
    "as not contradicted.",
    "- The interval for δ covers δ with probability at least 95%, at most 2.5% on each "
    "side. δ does not depend on the budget.",
    "- The interval for G covers G with the same probability, over repeated draws of E: "
    "independent Bernoulli(p1) in U and Bernoulli(p0) in S, with "
    "the label-free design held fixed.",
    "- These guarantees assume independent tasks. On a panel with more than one task per "
    "template they hold only once the synthetic validation under retained template "
    "dependence has passed.",
    "- U2 and U1 share one rate p1. If their rates differ, m1 is Poisson-binomial and p1 "
    "is their size-weighted average. Exactness is proved for the binomial case, and "
    "checked for unequal rates.",
    "- Ĝ equals the realized Δ per 100 exactly. The interval is for G, not for the "
    "realized Δ.",
    "- It does not cover the gain at another budget, because L changes. It does not cover "
    "gains in true errors as opposed to reference contradiction, or gains for other judges, "
    "views or benchmarks.",
    "- No odds ratio is used for any interval.",
)
COVERS_TEXT = "What the interval covers:\n" + "\n".join(COVERS_BULLETS)
# Explanations kept apart from the fixed statement above.
COVERS_NOTES_TEXT = (
    "Notes on these statements: U pools U2 and U1, the unstable agreeing tasks, and S "
    "holds the stable agreeing tasks; G is defined above, and Ĝ is G_hat. "
    "The inference that handles the "
    "template dependence retained on a one-trajectory-per-task panel, and its synthetic "
    "validation, are still to come and are required before the freeze."
)
# The statement above belongs to the coverage regime. The partial regime has no interval
# for delta or G, so it states its own assumption.
PARTIAL_ASSUMPTION_TEXT = (
    "Assumptions: the exact permutation p-value and the approximate bootstrap interval "
    "assume independent tasks. On a panel with more than one task per template they hold "
    "only once the synthetic validation under retained template dependence has passed. "
    "The inference that handles "
    "the template dependence retained on a one-trajectory-per-task panel, and its synthetic "
    "validation, are still to come and are required before the freeze."
)
# The two statements the descriptive report prints word for word.
SCOPE_TEXT = ("Scope: these results apply to these tasks and these realized judge calls, with no "
              "population confidence or significance claim.")
PANEL_WORDING = "a retrospective evaluation panel with documented prior exposure"
# The first line of every population-inference report.
POPULATION_PATH_TEXT = ("Population inference: this report is outside the primary result and "
                        "separately scoped. The primary result is the descriptive audit.")
REFERENCE_NOT_SUPPLIED_TEXT = ("Reference statuses not supplied: every label treated as usable; "
                               "Unsure and missing coverage not assessed")
G_DEFINITION_TEXT = (
    "G = (100/N)·L·δ is the expected number of additional "
    "reference-contradicted first verdicts that instability-first review captures, per "
    "100 panel trajectories, for a panel with this panel's label-free design (the same "
    "N, n_D0, n1 and s)."
)

# ---------------------------------------------------------------------------
# Invalid responses, abstentions, failures, retries
# ---------------------------------------------------------------------------
VERDICTS = ("success", "failure")
INVALID_TERMINAL = ("REFUSAL", "TRUNCATED", "FILTERED", "MALFORMED", "ABSTAIN")
TRANSPORT = "TRANSPORT"                       # no complete model response; retryable
HALT = "HALT"                                 # run stopped; not a retry and never INVALID
TRANSPORT_EXHAUSTED = "TRANSPORT_EXHAUSTED"   # no complete response after the retries: INVALID
MAX_RETRIES = 3
INVALID_CODES = INVALID_TERMINAL + (TRANSPORT_EXHAUSTED,)
UNFINISHED = (TRANSPORT, HALT)                # attempt outcomes that leave a slot open
SLOT_KEYS = (("A", 1), ("A", 2), ("A", 3), ("B", 1))
SLOT_NAMES = {("A", 1): "A1", ("A", 2): "A2", ("A", 3): "A3", ("B", 1): "B1"}


class ProtocolViolation(ValueError):
    """An attempt log that the frozen retry rules forbid."""


class UnfinishedSlot(ValueError):
    """A slot without a terminal outcome: open after transport failures, or halted.

    A halted slot is not INVALID. The run resumes it after the cause is fixed, so
    the analysis refuses a panel that still holds one.
    """


def canonical_outcome(outcome: str) -> str:
    """The canonical spelling of one attempt or slot outcome.

    Verdicts match case-insensitively, as the parser does: 'Success', 'SUCCESS'
    and 'success' are the verdict 'success'. Only ASCII spellings qualify, so a
    look-alike character never becomes a verdict. Outcome codes (REFUSAL, HALT and
    so on) are returned unchanged and must match exactly.
    """
    if isinstance(outcome, str) and outcome.isascii() and outcome.lower() in VERDICTS:
        return outcome.lower()
    return outcome


@dataclass(frozen=True)
class Attempt:
    """One logged attempt, written once.

    trajectory, judge ('A' or 'B') and repeat (1..3 for A, 1 for B) are the slot,
    fixed at dispatch. config_hash is the evaluation configuration the attempt was
    sent under. attempt is the 1-based dispatch index within the slot. outcome is
    TRANSPORT, HALT, 'success' or 'failure' (VALID, matched case-insensitively), or
    one of INVALID_TERMINAL.
    response_id is the provider's id for a complete response, when the runner has it.

    Runner rules this record encodes:
      - TRANSPORT: no complete model response. Connection error, timeout, HTTP 408,
        429 (rate limit), 5xx, overloaded, any other non-2xx without model output
        that is not a HALT class, and a stream that ends without a final stop
        reason. Partial stream content and any response arriving after the attempt
        was logged TRANSPORT are never parsed, used or logged as an outcome.
      - HALT: the run stops. HTTP 401, 403, 404, any billing, quota, credit or
        spend-limit error (whatever its status code), request or configuration
        errors (HTTP 400 that is not a spend or usage limit or a provider
        content-policy error, HTTP 413, and the provider's equivalent invalid-request
        errors), a failed per-response check, the spending cap, or an operator stop. It is logged, is not a retry and is never INVALID. The cause
        and the correction are recorded in the run log before the slot resumes
        under the same frozen hash.
      - A correction that changes the config hash: every slot completed under the
        old hash is dispatched again under the new one, the caller sets the
        voided records aside, and the change is disclosed in the run log.
        resolve_panel() refuses records under any other hash by design, so it is
        called with the new hash and only the records sent under it.
      - Terminal outcomes end the slot and are never retried: VALID, REFUSAL,
        TRUNCATED, FILTERED (content filter), MALFORMED and ABSTAIN. Every
        non-VALID terminal outcome is INVALID for its slot.
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
    dispatches can fail in transport. HALT records give the analysis no verdict
    and do not count as retries; the slot stays open after them. Any complete
    response that passes the per-response checks ends the slot and is never retried. A slot whose 1 + MAX_RETRIES
    transport failures are exhausted resolves to TRANSPORT_EXHAUSTED, which is
    INVALID: the plan makes a slot with no complete response after the retries
    INVALID. Nothing may follow exhaustion. Verdicts match case-insensitively
    and are returned in canonical spelling.

    A slot whose last record is HALT, or TRANSPORT with retries left, has no
    terminal outcome yet and raises UnfinishedSlot. A halt is never INVALID.
    """
    if len(outcomes) == 0:
        raise ValueError("slot has no attempts; it was never dispatched")
    n_tr = 0
    for k, a in enumerate(outcomes):
        a = canonical_outcome(a)
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
    if canonical_outcome(outcomes[-1]) == HALT:
        raise UnfinishedSlot("slot is unfinished: its last attempt halted the run; a halt is "
                             "not INVALID, and the slot resumes after the cause is fixed")
    raise UnfinishedSlot("slot is still open: transport failures with retries left")


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
    outside the panel, for any record under another config hash (see Attempt for
    a correction that changes the hash), and when two complete responses share a
    response id, which is what a response cache keyed on the payload would produce
    for the byte-identical repeats.
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
    """True for a verdict, False for an INVALID code; refuses anything unfinished.

    A slot marked HALT or TRANSPORT has no terminal outcome. It raises
    UnfinishedSlot, so a halted slot is never counted, as INVALID or otherwise.
    """
    outcome = canonical_outcome(outcome)
    if outcome in VERDICTS:
        return True
    if outcome in INVALID_CODES:
        return False
    if outcome in UNFINISHED:
        raise UnfinishedSlot(f"slot outcome {outcome!r} is unfinished: the slot has no terminal "
                             "outcome yet and cannot enter the analysis")
    raise ValueError(f"not a resolved slot outcome: {outcome!r}")


def derive_label_free(slots: Sequence[Mapping[str, str]], b1_invalid: str = "disagree",
                      r_unobserved: str = "zero", exclude_any_invalid: bool = False) -> dict:
    """Resolved slot outcomes -> (I, V, D, R, R_unobserved).

    slots[i] maps 'A1', 'A2', 'A3', 'B1' to a resolved outcome ('success' or
    'failure' in any letter case, or an INVALID code, FILTERED included). A slot
    marked HALT or TRANSPORT is unfinished and raises UnfinishedSlot; it is never
    counted as INVALID. Primary rules are the defaults:
      - A1 INVALID -> i in I. A1 is never replaced by a repeat or a later call.
      - B1 INVALID -> D = 1 (b1_invalid='disagree'). Labeled sensitivity:
        'agree' sets D = 0 so the item is routed by A's own R. 'exclude' is kept for
        completeness; with s held it provably cannot move Delta, p or the interval.
      - R counts valid repeats whose verdict differs from A1; invalid repeats
        count 0; both repeats invalid -> R = 0 and R_unobserved = 1
        (r_unobserved='zero'); sensitivity: 'one' sets R = 1 instead.
        Labeled sensitivity 'one_any' also sets R = 1 when exactly one repeat is
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
        a1, a2, a3, b1 = (canonical_outcome(sl[k]) for k in ("A1", "A2", "A3", "B1"))
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
                c[(k, canonical_outcome(sl[k]), bench)] += 1
    return dict(c)


# ---------------------------------------------------------------------------
# Reference labels at the join
# ---------------------------------------------------------------------------
USABLE, UNSURE, MISSING = "usable", "unsure", "missing"
REFERENCE_STATUSES = (USABLE, UNSURE, MISSING)
# The released primary label vocabulary: Successful and Unsuccessful are usable and
# map to the judges' verdicts; Unsure is not a reference. Matched in any letter case.
REFERENCE_VERDICTS = {"successful": "success", "unsuccessful": "failure"}
UNSURE_VALUES = ("unsure",)


def reference_status(y) -> Tuple[str, Optional[str]]:
    """(status, verdict) for one released primary label [labels].

    ('usable', 'success') for Successful and ('usable', 'failure') for Unsuccessful,
    in any letter case (ASCII only); ('unsure', None) for Unsure in any letter case;
    ('missing', None) for None, NaN, or an empty or blank string. Any other value
    raises ValueError and stops the analysis, the judges' words 'success' and
    'failure' included, so an unknown label is never guessed.
    """
    if y is None or (isinstance(y, float) and math.isnan(y)):
        return MISSING, None
    if not isinstance(y, str):
        raise ValueError(f"reference label {y!r} is not a string")
    if y.strip() == "":
        return MISSING, None
    if y.isascii():
        t = y.lower()
        if t in REFERENCE_VERDICTS:
            return USABLE, REFERENCE_VERDICTS[t]
        if t in UNSURE_VALUES:
            return UNSURE, None
    raise ValueError(f"unknown reference label {y!r}")


def join_reference(slots: Sequence[Mapping[str, str]], Y: Sequence[Optional[str]]) -> dict:
    """Discordance E and reference status per trajectory, at the join [labels].

    For trajectory i with released primary label Y_i:
      status_i = 'usable', 'unsure' or 'missing' (reference_status);
      E_i = 1 if status_i = 'usable', A1_i is a verdict and A1_i != Y_i, else 0.
    A task without a usable label has no reference. It stays in the panel and is
    not reference-contradicted (E_i = 0), which is how the primary result counts it;
    the secondary result leaves it out through its status. For an invalid A1, E_i = 0
    and is ignored, since there is no verdict to contradict. Nothing label-free
    depends on Y: the panel, D, R, the selections and K are fixed before the join.

    Returns {'E': list of 0 or 1, 'reference': list of statuses}, in panel order.
    """
    if len(Y) != len(slots):
        raise ValueError("one reference label per trajectory is required")
    E, status = [], []
    for sl, y in zip(slots, Y):
        st, verdict = reference_status(y)
        a1 = canonical_outcome(sl["A1"])
        e = int(st == USABLE and is_valid(a1) and a1 != verdict)
        E.append(e)
        status.append(st)
    return {"E": E, "reference": status}


def _require_population_path(path: str, name: str) -> None:
    """Refuses to run population inference unless it was asked for explicitly."""
    if path != PATH_POPULATION:
        raise ValueError(f"{name} runs population inference, which is outside the primary result; "
                         f"pass path={PATH_POPULATION!r} to run it")


def _require_no_contradiction_without_reference(labels: Sequence[str], E: Sequence[int],
                                                reference: Sequence[str]) -> None:
    """A task without a usable label cannot be reference-contradicted; invalid A1 is ignored."""
    for i, (l, e, r) in enumerate(zip(labels, E, reference)):
        if r != USABLE and l not in ("I", "EXCL") and e != 0:
            raise ValueError(f"trajectory {i} has no usable reference label ({r}) but E = {e}; "
                             "a task without a reference is never reference-contradicted")


def _reference_list(reference: Optional[Sequence[str]], n: int) -> List[str]:
    """Reference statuses for n trajectories; None (not supplied) treats every label as usable."""
    if reference is None:
        return [USABLE] * n
    ref = list(reference)
    if len(ref) != n:
        raise ValueError("reference must give one status per trajectory")
    bad = set(ref) - set(REFERENCE_STATUSES)
    if bad:
        raise ValueError(f"unknown reference statuses {bad}; use {REFERENCE_STATUSES}")
    return ref


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
    """One seeded realization of each policy, labeled "one operator's draw" [labels].

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
        stays between its neighbors on that grid, so it never leaves the nuisance set.
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
    return Fraction(acc, math.comb(nU2 + nU1 + nS, t))


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
    den = design.N_per100
    rng = np.random.default_rng(seed)
    tot = np.zeros(B)
    for j in ("U2", "U1", "S"):
        if n[j] == 0:
            continue
        pt = (m[j] + 0.5) / (n[j] + 1.0)
        tot += w[j] * rng.binomial(n[j], pt, size=B)
    tot *= 100.0 / den
    q = (1 - level) / 2
    point = 100.0 / den * sum(w[j] * m[j] for j in ("U2", "U1", "S"))
    return {"G_partial_hat": point, "lower": float(np.quantile(tot, q)),
            "upper": float(np.quantile(tot, 1 - q)), "B": B, "seed": seed,
            "label": "approximate"}


# ---------------------------------------------------------------------------
# Fragility of the existence decision
# ---------------------------------------------------------------------------
def _usable_sizes(design: Design, usable: Optional[Mapping[str, int]]) -> Dict[str, int]:
    """Tasks with a usable reference label in U2, U1 and S; None means all of them."""
    if usable is None:
        return {"U2": design.n_U2, "U1": design.n_U1, "S": design.n_S}
    full = {"U2": design.n_U2, "U1": design.n_U1, "S": design.n_S}
    out = {j: int(usable[j]) for j in ("U2", "U1", "S")}
    if any(not 0 <= out[j] <= full[j] for j in out):
        raise ValueError("usable counts must lie between 0 and the stratum sizes")
    return out


class _PermutationTails:
    """Exact rejection check of the partial-regime permutation test at any discordance counts.

    For a total t of discordant agreeing tasks, the within-D shuffle makes
    (M_U2, M_U1, M_S) multivariate hypergeometric given the full-panel stratum
    sizes, so the null distribution of Delta = sum_j w_j M_j depends on t only. It
    is tabulated once per t, with the weights scaled to integers, and the p-value
    of counts x is the tail at sum_j w_j x_j: exactly permutation_p(design, x).
    """

    def __init__(self, design: Design):
        w = delta_weights(design)
        den = math.lcm(*(w[j].denominator for j in ("U2", "U1", "S")))
        self.weights = tuple(int(w[j] * den) for j in ("U2", "U1", "S"))
        self.sizes = (design.n_U2, design.n_U1, design.n_S)
        self._tables: Dict[int, Tuple[List[int], List[int], int]] = {}

    def _table(self, t: int) -> Tuple[List[int], List[int], int]:
        if t not in self._tables:
            (nU2, nU1, nS), (wU2, wU1, wS) = self.sizes, self.weights
            mass: Dict[int, int] = defaultdict(int)
            for a in range(max(0, t - nU1 - nS), min(nU2, t) + 1):
                for b in range(max(0, t - a - nS), min(nU1, t - a) + 1):
                    c = t - a - b
                    ways = math.comb(nU2, a) * math.comb(nU1, b) * math.comb(nS, c)
                    mass[wU2 * a + wU1 * b + wS * c] += ways
            values = sorted(mass)
            tails = [0] * len(values)
            acc = 0
            for i in range(len(values) - 1, -1, -1):
                acc += mass[values[i]]
                tails[i] = acc
            self._tables[t] = (values, tails, math.comb(nU2 + nU1 + nS, t))
        return self._tables[t]

    def rejects(self, x: Sequence[int]) -> bool:
        """Whether the permutation p-value at counts x = (x_U2, x_U1, x_S) is at most 1/40."""
        values, tails, total = self._table(sum(x))
        v = sum(wj * xj for wj, xj in zip(self.weights, x))
        i = bisect.bisect_left(values, v)
        return Fraction(tails[i] if i < len(values) else 0, total) <= Fraction(1, 40)


def _l1_moves(k: int):
    """Every change (d_U2, d_U1, d_S) of the three discordance counts with |d|_1 = k."""
    for d2 in range(-k, k + 1):
        for d1 in range(abs(d2) - k, k - abs(d2) + 1):
            r = k - abs(d2) - abs(d1)
            yield d2, d1, r
            if r:
                yield d2, d1, -r


def existence_fragility(design: Design, counts: Mapping[str, int], reject: bool,
                        k_max: int = FRAGILITY_MAX,
                        usable: Optional[Mapping[str, int]] = None) -> Optional[int]:
    """Smallest number of usable reference-label flips among agreeing tasks that
    reverses the existence decision.

    A flip changes the discordance of one agreeing task that has a usable reference
    label. A task without a usable label has no reference, so it is never flipped
    and never counted as a flip. usable gives the number of tasks with a usable label
    in U2, U1 and S (None: every task has one); the flipped counts stay within them,
    while the test keeps its full-panel margins. Returns None when no set of at most
    k_max flips reverses the decision; fragility_reversible() tells whether any set
    of flips does.

    Coverage: Fisher's p on (m1, m0). It falls as m1 rises with m0 fixed, and as m0
    falls with m1 fixed, so the fewest reversing flips move m1 down and m0 up when
    the test rejects, and the other way otherwise; only those moves are searched.
    Partial regime: the permutation p on (m_U2, m_U1, m_S). The weight of U1 can
    have either sign, and the p-value is not monotone in each count, so every
    change of the three counts of L1 length k is tried, k = 1, 2, ...
    """
    thr = Fraction(1, 40)
    sgn = -1 if reject else 1
    mU2, mU1, mS = counts["U2"], counts["U1"], counts["S"]
    nY = _usable_sizes(design, usable)
    if design.regime == "COVERAGE":
        m1, n1, n0 = mU2 + mU1, design.n1, design.n0
        n1Y, n0Y = nY["U2"] + nY["U1"], nY["S"]
        for k in range(1, k_max + 1):
            for fu in range(k + 1):
                a, c = m1 + sgn * fu, mS - sgn * (k - fu)
                if 0 <= a <= n1Y and 0 <= c <= n0Y:
                    if (fisher_one_sided(a, n1, c, n0, exact=True) <= thr) != reject:
                        return k
        return None
    tails = _PermutationTails(design)
    hi = (nY["U2"], nY["U1"], nY["S"])
    for k in range(1, k_max + 1):
        for d2, d1, dS in _l1_moves(k):
            x = (mU2 + d2, mU1 + d1, mS + dS)
            if all(0 <= x[j] <= hi[j] for j in range(3)):
                if tails.rejects(x) != reject:
                    return k
    return None


def fragility_reversible(design: Design, counts: Mapping[str, int], reject: bool,
                         usable: Optional[Mapping[str, int]] = None) -> bool:
    """Whether any set of usable reference-label flips among agreeing tasks reverses
    the existence decision. When none does, the fragility count is not defined.

    Only tasks with a usable label can be discordant, so a rejection is always
    reversible: making every discordant agreeing task concordant gives p = 1.
    Otherwise the most favorable reachable counts decide. Coverage: m1 = n1Y and
    m0 = 0, by the monotonicity of Fisher's p (see existence_fragility). Partial
    regime: for each total t of discordant agreeing tasks the null distribution is
    fixed and the p-value falls as Delta rises, so the counts that maximize Delta at
    that total (filling the strata with usable labels in decreasing order of weight)
    decide.
    """
    if reject:
        return True
    nY = _usable_sizes(design, usable)
    if design.regime == "COVERAGE":
        n1Y = nY["U2"] + nY["U1"]
        return fisher_one_sided(n1Y, design.n1, 0, design.n0, exact=True) <= Fraction(1, 40)
    tails = _PermutationTails(design)
    order = sorted(range(3), key=lambda j: -tails.weights[j])
    hi = (nY["U2"], nY["U1"], nY["S"])
    for t in range(sum(hi) + 1):
        x, left = [0, 0, 0], t
        for j in order:
            x[j] = min(hi[j], left)
            left -= x[j]
        if tails.rejects(x):
            return True
    return False


# ---------------------------------------------------------------------------
# Panel: one trajectory per task by default, one task per template as sensitivity
# ---------------------------------------------------------------------------
PANEL_UNITS = ("task", "template")
PANEL_UNIT = "task"     # the default panel; "template" gives the labeled sensitivity panel


def select_panel(runs: Sequence[Mapping[str, object]], seed: int, unit: str = PANEL_UNIT) -> List[str]:
    """Label-free choice of the panel from the eligible agent runs.

    runs: one mapping per eligible agent run, with 'run_id', 'task_id' (the
    normalized task id), 'template_id' and 'passes' (True when the run passes the
    evidence completeness and fit checks). Eligibility (split, benchmark and
    template exclusions) is settled before this call, and no label is read. The
    wider-exclusion sensitivity panel is not a new call: it is the default panel's
    runs minus every task whose template appears in the development split. Calling
    this function again on fewer runs would shift the generator and choose other
    runs. When a
    benchmark's template ids cannot be recovered, pass each task id as its own
    template id.

    unit='task', the default panel: for each task in task-id order, its runs are
    sorted by run id and permuted with the generator of
    SeedSequence(seed, spawn_key=(4,)); the first run that passes is taken. A task
    with no passing run is not in the panel. Tasks that share a template all stay.

    unit='template', the labeled sensitivity panel: the task panel is built
    first; then, for each template in template-id order, its eligible tasks are
    sorted by task id and permuted with the generator of
    SeedSequence(seed, spawn_key=(5,)), and the first task that has a panel run is
    taken with that run. For the same seed the template panel is therefore a
    subset of the task panel.

    Both streams are independent of the operator's draw and of the
    one-agreeing-item template check. Returns the chosen run ids in task-id order.
    """
    if unit not in PANEL_UNITS:
        raise ValueError(f"unit must be one of {PANEL_UNITS}")
    run_ids = [str(r["run_id"]) for r in runs]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("run ids must be unique")
    by_task: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    template_of: Dict[str, str] = {}
    for r in runs:
        t, tpl = str(r["task_id"]), str(r["template_id"])
        if template_of.setdefault(t, tpl) != tpl:
            raise ValueError(f"task {t!r} is listed under two templates")
        if not isinstance(r["passes"], bool):
            raise ValueError("'passes' must be True or False")
        by_task[t].append(r)
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(4,)))
    chosen: Dict[str, str] = {}
    for t in sorted(by_task):
        members = sorted(by_task[t], key=lambda r: str(r["run_id"]))
        for k in rng.permutation(len(members)):
            if members[int(k)]["passes"]:
                chosen[t] = str(members[int(k)]["run_id"])
                break
    if unit == "task":
        return [chosen[t] for t in sorted(chosen)]
    tasks_of: Dict[str, List[str]] = defaultdict(list)
    for t, tpl in template_of.items():
        tasks_of[tpl].append(t)
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(5,)))
    picked = []
    for tpl in sorted(tasks_of):
        members = sorted(tasks_of[tpl])
        for k in rng.permutation(len(members)):
            if members[int(k)] in chosen:
                picked.append(members[int(k)])
                break
    return [chosen[t] for t in sorted(picked)]


def template_multiplicities(templates: Sequence[str]) -> dict:
    """Label-free: how many panel tasks each template holds, for the panel manifest."""
    c = Counter(str(t) for t in templates)
    mult = sorted(c.values(), reverse=True)
    return {"n_tasks": sum(mult), "n_templates": len(c), "max_multiplicity": mult[0] if mult else 0,
            "tasks_sharing": sum(x for x in mult if x > 1),
            "histogram": dict(sorted(Counter(mult).items()))}


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
                         method: Optional[str] = None, path: str = PATH_DESCRIPTIVE) -> dict:
    """Existence p and delta interval on one agreeing item per template (labeled sensitivity).

    Runs only with path="population_inference", as analyze() does.
    """
    _require_population_path(path, "template_sensitivity")
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
            fragility: bool = True, reference: Optional[Sequence[str]] = None,
            N_Y_ref: Optional[int] = None,
            benchmarks: Optional[Sequence[str]] = None, path: str = PATH_DESCRIPTIVE) -> dict:
    """Population inference on one panel [labels]. Returns a result dict for report().

    Runs only with path="population_inference"; any other value is refused, the
    default included, since this analysis is outside the primary result.

    reference gives each trajectory's reference status ('usable', 'unsure' or
    'missing', as join_reference returns it). None means the statuses were not
    supplied: every label is treated as usable, and the secondary result and the
    label coverage are not assessed (res['secondary'] and res['coverage'] are None, and the report says
    so). E must be 0 for every task without a usable label (except invalid A1,
    where E is ignored), since such a task cannot be reference-discordant.

    Primary result, per 100 panel tasks, with E_i = 0 for a task without a usable
    label (not reference-contradicted):
      m_j = sum of E over stratum j; m1 = m_U2 + m_U1, m0 = m_S; n1, n0, N, L as
      in the label-free design;
      G_hat = 100 Delta / N; in coverage G_hat = (100/N) L (m1/n1 - m0/n0);
      G = (100/N) L delta with delta = p1 - p0, p1 and p0 the rates of
      reference-contradicted first verdicts in U and S;
      existence: one-sided Fisher exact p = P(M1 >= m1 | n1, n0, m1 + m0);
      interval for delta from (m1 of n1, m0 of n0); delta_m = m N / (100 L);
      fragility: the fewest flips of usable reference labels among agreeing tasks
      that reverse the existence decision (not defined when no set of such flips
      reverses it, for example when no agreeing task has a usable label).
    The secondary result, per 100 tasks with a usable label, is in res['secondary']
    (secondary_analysis), and the label coverage in res['coverage']
    (reference_coverage). N_Y_ref holds the secondary per-100 denominator, as N_ref
    does for the primary; benchmarks only feed the coverage table.
    """
    _require_population_path(path, "analyze")
    method = INTERVAL_METHOD if method is None else method
    labels = list(labels)
    E = [int(e) for e in E]
    if len(labels) != len(E):
        raise ValueError("labels and E differ in length")
    ref = _reference_list(reference, len(labels))
    _require_no_contradiction_without_reference(labels, E, ref)
    if benchmarks is not None and len(benchmarks) != len(labels):
        raise ValueError("benchmarks must give one entry per trajectory")
    keep = [i for i, l in enumerate(labels) if l != "EXCL"]
    labels_k = [labels[i] for i in keep]
    E_k = [E[i] for i in keep]
    ref_k = [ref[i] for i in keep]
    ids_k = [ids[i] for i in keep] if ids is not None else None
    bench_k = [benchmarks[i] for i in keep] if benchmarks is not None else None
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
    # Agreeing tasks with a usable reference label, per stratum: only these can be flipped.
    usable = {j: sum(1 for l, r in zip(labels_k, ref_k) if l == j and r == USABLE)
              for j in ("U2", "U1", "S")}
    if fragility and res["existence"] is not None:
        ex = res["existence"]
        ex["fragility"] = existence_fragility(design, counts, ex["reject"], usable=usable)
        # Not defined when no set of usable flips reverses the decision, so that the report
        # never says "more than FRAGILITY_MAX" of a reversal that cannot happen.
        ex["fragility_defined"] = (ex["fragility"] is not None
                                   or fragility_reversible(design, counts, ex["reject"], usable))
    # Without reference statuses, Unsure and missing labels are not assessed at all:
    # no secondary result and no coverage table, rather than counts of zero.
    supplied = reference is not None
    res["reference_supplied"] = supplied
    res["secondary"] = None
    res["coverage"] = None
    if supplied:
        res["secondary"] = secondary_analysis(design, labels_k, ref_k, counts, res, N_Y_ref=N_Y_ref)
        res["coverage"] = reference_coverage(labels_k, ref_k, design, benchmarks=bench_k,
                                             draw=res.get("draw"))
    return res


def secondary_analysis(design: Design, labels: Sequence[str], reference: Sequence[str],
                       counts: Mapping[str, int], primary: Mapping[str, object],
                       N_Y_ref: Optional[int] = None) -> dict:
    """Secondary result: the gain per 100 tasks with a usable label [labels].

    The label-free design, the regime and the selections stay those of the full
    panel. primary is the primary result of analyze() on the same panel. N_Y counts
    the panel tasks with a usable label, in every stratum, I included, and n_jY those
    of stratum j.

    The result is 100 Delta / N_Y. A task without a usable label has E = 0, so this
    is the primary realized gain rescaled by N/N_Y, and it carries the primary's
    test. Coverage regime: the interval for G rescaled, (100 L / N_Y) (delta_L,
    delta_U) from the primary interval for delta; existence by the primary's Fisher
    p. Partial regime: the primary's G_partial_hat and its approximate bootstrap
    interval, each times N/N_Y. Not estimable when N_Y = 0.

    No threshold claim is made: the threshold of 2 per 100 refers to the primary
    result only. In the degenerate regimes Delta = 0 identically and nothing is
    tested. With every label usable, the result equals the primary result.
    """
    nY = {j: sum(1 for l, r in zip(labels, reference) if l == j and r == USABLE) for j in STRATA}
    N_Y = sum(nY.values())
    den = N_Y if N_Y_ref is None else N_Y_ref
    Delta = expected_captures(design, counts)["Delta"]
    out = {"N_Y": N_Y, "N_Y_per100": den, "n_Y": nY, "n1_Y": nY["U2"] + nY["U1"], "n0_Y": nY["S"],
           "regime": design.regime, "not_estimable": None, "plan_literal": None}
    reg = design.regime
    if reg not in ("COVERAGE", "PARTIAL"):
        return out
    if N_Y == 0 or den == 0:
        out["not_estimable"] = "no panel task has a usable label"
        return out

    ratio = Fraction(design.N_per100, den)
    ex, sz = primary["existence"], primary["size"]
    pl = {"rescale": ratio,
          "G_hat_exact": ratio * Fraction(100) * Delta / design.N_per100,
          "existence": {"test": "same test as the primary", "p": ex["p"], "reject": ex["reject"]}}
    pl["G_hat"] = float(pl["G_hat_exact"])
    r = float(ratio)
    if reg == "COVERAGE":
        pl.update(delta_L=sz["delta_L"], delta_U=sz["delta_U"],
                  G_L=r * sz["G_L"], G_U=r * sz["G_U"])
    else:
        b = sz["partial"]
        pl["partial"] = dict(b, G_partial_hat=r * b["G_partial_hat"], lower=r * b["lower"],
                             upper=r * b["upper"])
    out["plan_literal"] = pl
    return out


REVIEW_GROUPS = (("invalid first verdict", ("I",)), ("disagreement", ("G1",)), ("U2", ("U2",)), ("U1", ("U1",)),
                 ("stable agreeing", ("S",)))
POLICY_NAMES = {"RAND": "Random", "DIS": "Disagreement-first", "AUG": "Instability-first"}


def selection_probabilities(design: Design, labels: Sequence[str]) -> Dict[str, List[Fraction]]:
    """Each task's exact probability of review at budget K, per policy (label-free).

    The probability of a task in layout stratum j is min(n_j, max(0, K - N_<j)) / n_j,
    the uniform tie-breaking of the plan; the probabilities of each policy sum to K.
    """
    key = {"RAND": lambda l: "I" if l == "I" else "V",
           "DIS": lambda l: l if l in ("I", "G1") else "G0",
           "AUG": lambda l: l}
    out = {}
    for pol in ("RAND", "DIS", "AUG"):
        lay = _layout(design, pol)
        frac = dict(zip([k for k, _ in lay], review_fractions([n for _, n in lay], design.K)))
        out[pol] = [frac[key[pol](l)] for l in labels]
    return out


def reference_coverage(labels: Sequence[str], reference: Sequence[str], design: Design,
                       benchmarks: Optional[Sequence[str]] = None,
                       draw: Optional[dict] = None,
                       E: Optional[Sequence[int]] = None) -> dict:
    """Coverage of Unsure and missing reference labels [labels: statuses only].

    These tasks are unknown: they are never counted as correct verdicts. Counts are
    given overall and by review group: invalid first verdict (I), disagreement (G1),
    U2, U1 and stable agreeing (S); and by benchmark when benchmarks are given. When E
    is given, each group also has its reference-contradicted first verdicts, and an
    invalid first verdict has none. For each policy, the expected number of such tasks it
    selects for review is sum_i pi_P(i) 1(status_i = s) over the group, with pi_P the
    exact review probability (selection_probabilities); when a seeded draw is given,
    the draw's own counts are added.
    """
    labels = list(labels)
    reference = list(reference)
    groups = (("all", STRATA),) + REVIEW_GROUPS

    def tally(idx):
        st = Counter(reference[i] for i in idx)
        return {"tasks": len(idx), "usable": st[USABLE], "unsure": st[UNSURE], "missing": st[MISSING]}

    members = {name: [i for i, l in enumerate(labels) if l in strata] for name, strata in groups}
    table = {name: tally(idx) for name, idx in members.items()}
    if E is not None:
        for name, idx in members.items():
            table[name]["contradicted"] = sum(E[i] for i in idx if labels[i] != "I")
    probs = selection_probabilities(design, labels)
    selected = {}
    for pol in ("RAND", "DIS", "AUG"):
        pr = probs[pol]
        selected[pol] = {name: {"tasks": sum((pr[i] for i in idx), Fraction(0)),
                                UNSURE: sum((pr[i] for i in idx if reference[i] == UNSURE), Fraction(0)),
                                MISSING: sum((pr[i] for i in idx if reference[i] == MISSING), Fraction(0))}
                         for name, idx in members.items()}
    out = {"groups": table, "selected_expected": selected, "selected_draw": None, "by_benchmark": None}
    if draw is not None:
        out["selected_draw"] = {}
        for pol in ("RAND", "DIS", "AUG"):
            chosen = {int(i) for i in draw["reviewed"][pol]}
            out["selected_draw"][pol] = {
                name: {UNSURE: sum(1 for i in idx if i in chosen and reference[i] == UNSURE),
                       MISSING: sum(1 for i in idx if i in chosen and reference[i] == MISSING)}
                for name, idx in members.items()}
    if benchmarks is not None:
        out["by_benchmark"] = {b: tally([i for i in range(len(labels)) if benchmarks[i] == b])
                               for b in sorted(set(benchmarks))}
    return out


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


def _primary_lines(res: dict) -> List[str]:
    """Report lines of the primary result. Existence and size are printed separately."""
    lf, cap = res["label_free"], res["captures"]
    reg = lf["regime"]
    out = []
    out.append(f"Population inference on the panel, statistical analysis plan {POPULATION_PLAN_VERSION}")
    out.append("")
    out.append("Label-free design (committed before the join)")
    out.append(f"  N = {lf['N']}, invalid A1 n_inv = {lf['n_inv']}, valid N_V = {lf['N_V']}")
    out.append(f"  disagreeing n_D1 = {lf['n_D1']}, agreeing n_D0 = {lf['n_D0']}: "
               f"unstable n1 = {lf['n1']} (U2 = {lf['n_U2']}, U1 = {lf['n_U1']}), stable n0 = {lf['n0']}")
    s_src = "held at its value in the per-100-panel-task result" if lf.get("s_held") else f"s_nom = {lf['s_nom']}"
    out.append(f"  slice s = {lf['s']} ({s_src}, cap binds: "
               f"{'yes' if lf['cap_binds'] else 'no'}); review count K = {lf['K']} "
               f"({100 * lf['K'] / lf['N']:.1f}% of N), identical for all three policies")
    if lf.get("N_per100", lf["N"]) != lf["N"]:
        out.append(f"  per-100 denominator held at the N of the per-100-panel-task result, {lf['N_per100']}")
    if reg == "COVERAGE":
        out.append(f"  regime: {reg}; L = {_f(lf['L'])}")
        if lf["delta_m"] is not None:
            out.append(f"  threshold m = {lf['m']} per 100 -> delta_m = {_f(lf['delta_m'])}; "
                       f"reachable on this panel: {'yes' if lf['reachable'] else 'no'}")
    else:
        # L and delta_m are coverage-regime quantities.
        out.append(f"  regime: {reg}")
    cov = res.get("coverage")
    if res.get("reference_supplied") and cov is not None:
        g = cov["groups"]["all"]
        out.append(f"  per-100-panel-task result (N = {lf['N']}); "
                   f"{g['unsure'] + g['missing']} tasks without a usable reference label "
                   f"(Unsure {g['unsure']}, missing {g['missing']}) count as not reference-contradicted")
    else:
        out.append("  " + REFERENCE_NOT_SUPPLIED_TEXT)
    out.append("")
    out.append("Expected captures (exact, tie-averaged)")
    out.append("  Rand is Random, DIS is disagreement-first and AUG is instability-first.")
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
        return out
    if reg == "DEGENERATE-CAP":
        out.append("Existence question: not run. The cap binds (s = n_D0): both policies "
                   "review all agreeing items, so Delta = 0 identically.")
        out.append("Size question: not applicable.")
        return out
    if reg == "DEGENERATE-NOSLICE":
        out.append("Existence question: not run. s = 0: neither policy reviews any agreeing "
                   "item, so Delta = 0 identically.")
        out.append("Size question: not applicable.")
        return out
    ex = res["existence"]
    out.append("EXISTENCE QUESTION: is there any positive gain?")
    if reg == "COVERAGE":
        out.append("  H0: p1 = p0, equivalently E is exchangeable between U and S given "
                   "the margins; H1: p1 > p0.")
        out.append(f"  Statistic: m1, the number of reference-contradicted first verdicts in U. "
                   f"p = P(M1 >= m1 | n1, n0, m_D0), one-sided Fisher exact test, which equals "
                   f"the exact within-D permutation p-value of Delta.")
        out.append(f"  m1 = {res['m1']} of n1 = {lf['n1']}; m0 = {res['m0']} of n0 = {lf['n0']}; "
                   f"m_D0 = {res['m1'] + res['m0']}")
        yes, no = "evidence at one-sided 0.025 that p1 > p0", "no evidence at one-sided 0.025 that p1 > p0"
    else:
        w = res["weights"]
        out.append("  H0: given E, the instability labels R are exchangeable within D = 0, so "
                   "p_U2 = p_U1 = p_S; H1: reference-contradicted verdicts concentrate where instability-first "
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
        if not ex.get("fragility_defined", True):
            out.append("  Fragility: not defined; no set of usable reference-label flips among "
                       "agreeing tasks reverses this decision.")
        else:
            out.append(f"  Fragility: {fr if fr is not None else 'more than ' + str(FRAGILITY_MAX)} "
                       f"usable reference-label flip{'s' if fr != 1 else ''} among agreeing tasks "
                       f"would reverse this decision.")
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
        grid_extra = "0 and δ_m" if lf["reachable"] else "0"
        out.append("  Interval: two one-sided 97.5% bounds (Chan-Zhang inversion), grid k/100 plus "
                   f"{grid_extra}, outward-rounded: both ends are rejected grid points, so the "
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
                    "of reference-contradicted agreeing verdicts; the unconditional interval takes the worst case "
                    f"over the unknown stable rate p0, and {why}. Existence is decided by Fisher's p "
                    "alone; the interval only sizes the gain.")
            else:
                out.append(
                    "  Reconciliation: the existence test does not reject, yet the interval for G "
                    "excludes 0 on the left. The two use different tests on purpose; existence is "
                    "decided by Fisher's p alone, and the interval only sizes the gain.")
        out.append("")
        out.append(COVERS_TEXT)
        out.append("")
        out.append(COVERS_NOTES_TEXT)
    else:
        b = sz["partial"]
        out.append(f"  Partial regime (n1 > s): AUG reviews only s = {lf['s']} of the n1 = {lf['n1']} "
                   f"unstable items, so the coverage identity, L and δ_m do not apply.")
        out.append(f"  Delta = {_f(cap['Delta'])}; G_partial_hat = {_f(b['G_partial_hat'])} per 100")
        out.append(f"  APPROXIMATE Jeffreys-smoothed parametric bootstrap interval for G_partial: "
                   f"[{_f(b['lower'])}, {_f(b['upper'])}] per 100 (B = {b['B']}, seed {b['seed']})")
        out.append("  What it covers, approximately: G_partial = (100/N) Σ_j w_j n_j p_j, the "
                   "expected Delta per 100 under independent binomial E in U2, U1 and S "
                   "for this design. Its coverage is not guaranteed; the synthetic checks give "
                   "informational Monte Carlo coverage.")
        out.append("  No threshold claim is made in either direction in this regime.")
        out.append("  " + PARTIAL_ASSUMPTION_TEXT)
    out.append("")
    out.append("The two questions are answered separately and are not merged into one verdict.")
    return out


def _secondary_lines(res: dict) -> List[str]:
    """Report lines of the secondary result, per 100 tasks with a usable label.

    It is the primary result rescaled, and it carries no threshold claim.
    """
    sec = res.get("secondary")
    if sec is None:
        return []
    lf = res["label_free"]
    out = [f"SECONDARY RESULT: gain per 100 tasks with a usable label "
           f"(N_Y = {sec['N_Y']} of N = {lf['N']})"]
    if sec["N_Y_per100"] != sec["N_Y"]:
        out.append(f"  per-100 denominator held at its value on the full panel, N_Y = {sec['N_Y_per100']}")
    out.append("  The label-free design, the regime and the selections are those of the full panel.")
    out.append(f"  usable labels among agreeing tasks: unstable n1Y = {sec['n1_Y']} of {lf['n1']}, "
               f"stable n0Y = {sec['n0_Y']} of {lf['n0']}")
    if sec["regime"] not in ("COVERAGE", "PARTIAL"):
        out.append(f"  Not run: Delta = 0 identically in regime {sec['regime']}.")
        return out
    if sec["not_estimable"] is not None:
        out.append(f"  Not estimable: {sec['not_estimable']}. Nothing is imputed.")
        return out
    pl = sec["plan_literal"]
    ex = pl["existence"]
    out.append("")
    out.append(f"  100 Delta / N_Y, the realized gain of the per-100-panel-task result rescaled by "
               f"N/N_Y = {_f(pl['rescale'])}")
    out.append(f"  Existence: same test as the per-100-panel-task result, p = {ex['p']:.6g}; "
               f"{'REJECT H0' if ex['reject'] else 'do not reject H0'}")
    if sec["regime"] == "COVERAGE":
        out.append(f"  100 Delta / N_Y = {_f(pl['G_hat'])} per 100 tasks with a usable label; the "
                   f"per-100-panel-task interval for G rescaled, (100 L / N_Y)(δ_L, δ_U): "
                   f"{_g_interval_text(pl['G_L'], pl['G_U'], pl['delta_L'], pl['delta_U'])}")
    else:
        b = pl["partial"]
        out.append(f"  100 Delta / N_Y = {_f(pl['G_hat'])} per 100 tasks with a usable label; the "
                   f"per-100-panel-task APPROXIMATE bootstrap interval rescaled: "
                   f"[{_f(b['lower'])}, {_f(b['upper'])}] (B = {b['B']}, seed {b['seed']})")
    out.append("")
    out.append(f"  The threshold of {lf['m']} per 100 refers to the per-100-panel-task result only: no threshold "
               f"claim is made for the secondary result.")
    return out


def _coverage_lines(res: dict) -> List[str]:
    """Report lines for the coverage of Unsure and missing reference labels."""
    cov = res.get("coverage")
    if cov is None:
        return []
    out = ["REFERENCE LABEL COVERAGE: Unsure and missing labels",
           "  These tasks are unknown, not correct verdicts. The result per 100 panel tasks counts "
           "them as not reference-contradicted; the secondary result is per 100 "
           "tasks with a usable label."]
    has_contradicted = "contradicted" in cov["groups"]["all"]
    out.append("  group | tasks | usable | Unsure | missing" + (" | reference-contradicted" if has_contradicted else ""))
    for name, g in cov["groups"].items():
        out.append(f"  {name} | {g['tasks']} | {g['usable']} | {g['unsure']} | {g['missing']}"
                   + (f" | {g['contradicted']}" if has_contradicted else ""))
    names = list(cov["groups"])
    out.append("  Selected for review, expected number (exact, tie-averaged), Unsure / missing, "
               "by group: " + ", ".join(names))
    for pol in ("RAND", "DIS", "AUG"):
        cells = [f"{_f(cov['selected_expected'][pol][n][UNSURE], 2)} / "
                 f"{_f(cov['selected_expected'][pol][n][MISSING], 2)}" for n in names]
        out.append(f"  {POLICY_NAMES[pol]}: " + "; ".join(cells))
    if cov["selected_draw"] is not None:
        out.append("  One operator's draw, Unsure / missing selected, by the same groups:")
        for pol in ("RAND", "DIS", "AUG"):
            cells = [f"{cov['selected_draw'][pol][n][UNSURE]} / {cov['selected_draw'][pol][n][MISSING]}"
                     for n in names]
            out.append(f"  {POLICY_NAMES[pol]}: " + "; ".join(cells))
    if cov["by_benchmark"] is None:
        out.append("  By benchmark: not supplied")
    else:
        out.append("  benchmark | tasks | usable | Unsure | missing")
        for b, g in cov["by_benchmark"].items():
            out.append(f"  {b} | {g['tasks']} | {g['usable']} | {g['unsure']} | {g['missing']}")
    return out


def _report_lines(res: dict) -> List[str]:
    """The population-inference results of one analysis, without the opening line."""
    out = _primary_lines(res)
    for extra in (_secondary_lines(res), _coverage_lines(res)):
        if extra:
            out += [""] + extra
    return out


def report(res: dict) -> str:
    """Human-readable report of one population-inference analysis.

    It opens with the line saying it is outside the primary result, then gives the
    plan's primary result, the secondary result and the label coverage. Existence and
    size are printed separately and never merged into one verdict.
    """
    return "\n".join([POPULATION_PATH_TEXT, ""] + _report_lines(res))


def print_report(res: dict) -> str:
    """Print the report (existence and size separately) and return its text."""
    txt = report(res)
    print(txt)
    return txt


SENSITIVITY_VARIANTS = [
    ("sensitivity: B1 INVALID treated as agreement (D = 0) instead of D = 1", {"b1_invalid": "agree"}),
    ("sensitivity: R_unobserved treated as R = 1", {"r_unobserved": "one"}),
    ("sensitivity: every trajectory with any INVALID slot excluded", {"exclude_any_invalid": True}),
    ("sensitivity: one INVALID repeat and no observed flip treated as R = 1, and R_unobserved treated as R = 1",
     {"r_unobserved": "one_any"}),
]


def analyze_with_sensitivities(slots: Sequence[Mapping[str, str]], E: Sequence[int],
                               path: str = PATH_DESCRIPTIVE, **kw) -> List[Tuple[str, dict]]:
    """Population inference plus its labeled sensitivity analyses for invalid responses.

    Runs only with path="population_inference", as analyze() does.

    Every sensitivity analysis holds s at its primary value (capped at
    the new n_D0) and the per-100 denominator N at the primary N, so K changes only
    through reclassified items and the analysis measures the invalid rule, not a
    budget change. The secondary per-100 denominator N_Y is held the same way.
    reference and benchmarks, when given, follow every analysis; the coverage table
    by benchmark is built for the primary analysis only.
    """
    _require_population_path(path, "analyze_with_sensitivities")
    kw = dict(kw)
    kw.pop("s_override", None)
    kw.pop("N_ref", None)
    kw.pop("N_Y_ref", None)
    benchmarks = kw.pop("benchmarks", None)
    lf0 = derive_label_free(slots)
    prim = analyze(lf0["stratum"], E, benchmarks=benchmarks, path=path, **kw)
    s0, N0 = prim["label_free"]["s"], prim["label_free"]["N"]
    NY0 = prim["secondary"]["N_Y"] if prim["secondary"] is not None else None
    out = [("primary", prim)]
    skw = {k: v for k, v in kw.items() if k not in ("seed",)}
    for name, opts in SENSITIVITY_VARIANTS:
        lf = derive_label_free(slots, **opts)
        out.append((name, analyze(lf["stratum"], E, s_override=s0, N_ref=N0, N_Y_ref=NY0, path=path, **skw)))
    return out


def _repeat_flags(lf: Mapping[str, np.ndarray]) -> dict:
    """Label-free counts of agreeing trajectories with an unobserved repeat."""
    g0 = np.isin(lf["stratum"], ["U2", "U1", "S"])
    return {"both_repeats_invalid": int(((lf["R_unobserved"] == 1) & g0).sum()),
            "one_repeat_invalid_no_flip": int(((lf["R_one_missing"] == 1) & (lf["R"] == 0) & g0).sum()),
            "one_repeat_invalid_flip": int(((lf["R_one_missing"] == 1) & (lf["R"] >= 1) & g0).sum())}


def analyze_panel(slots: Sequence[Mapping[str, str]], E: Optional[Sequence[int]] = None, *,
                  path: str = PATH_DESCRIPTIVE, **kw) -> dict:
    """One panel's analysis, on the path asked for.

    path="descriptive", the default, is the primary result: it passes E and the
    keywords on to describe_panel(). path="population_inference" is outside the
    primary result and separately scoped: it needs E and takes the keywords of
    _population_panel(). A keyword of one path is refused on the other, so nothing
    that was asked for is silently ignored. panel_report() prints either result.
    """
    if path == PATH_DESCRIPTIVE:
        return describe_panel(slots, E, **kw)
    if path == PATH_POPULATION:
        if E is None:
            raise ValueError("the population-inference path needs E")
        return _population_panel(slots, E, **kw)
    raise ValueError(f"path must be {PATH_DESCRIPTIVE!r} or {PATH_POPULATION!r}, not {path!r}")


def _population_panel(slots: Sequence[Mapping[str, str]], E: Sequence[int], *,
                      ids: Optional[Sequence[str]] = None,
                      benchmarks: Optional[Sequence[str]] = None,
                      templates: Optional[Sequence[str]] = None,
                      provenance: Optional[dict] = None,
                      seed: Optional[int] = None, m: float = MATERIAL_M,
                      method: Optional[str] = None, boot_B: int = 10000, boot_seed: int = 0,
                      reference: Optional[Sequence[str]] = None,
                      template_panel: Optional[Sequence[str]] = None) -> dict:
    """Everything the population-inference report needs: the plan's primary and
    secondary results, label coverage, labeled sensitivity analyses, invalid counts,
    repeat-observation counts, template sensitivity and A1 provenance.

    The panel is the default one: one trajectory per task. reference gives each
    trajectory's reference status (join_reference). templates gives each panel
    task's template id, for the multiplicities and the one-agreeing-item check.
    template_panel lists the trajectory ids of the one-task-per-template panel,
    chosen label-free before evaluation (select_panel with unit='template'); the
    full analysis is then repeated on it as a labeled sensitivity analysis, with
    its own N, s and K, since it is a different panel.
    """
    sub = None
    if template_panel is not None:
        if ids is None:
            raise ValueError("template_panel needs the trajectory ids")
        pos = {t: i for i, t in enumerate(ids)}
        if any(t not in pos for t in template_panel) or len(set(template_panel)) != len(template_panel):
            raise ValueError("template_panel must list distinct trajectory ids of this panel")
        sub = sorted(pos[t] for t in template_panel)
    if templates is not None and len(templates) != len(slots):
        raise ValueError("templates must give one template id per trajectory")
    kw = dict(m=m, method=method, boot_B=boot_B, boot_seed=boot_seed)
    kw_population = dict(kw, path=PATH_POPULATION)
    sens = analyze_with_sensitivities(slots, E, seed=seed, ids=ids, reference=reference,
                                      benchmarks=benchmarks, **kw_population)
    lf = derive_label_free(slots)
    out = {"path": PATH_POPULATION, "primary": sens[0][1], "sensitivities": sens[1:],
           "invalid_counts": invalid_counts(slots, benchmarks),
           "repeat_flags": _repeat_flags(lf), "provenance": provenance, "templates": None,
           "template_multiplicities": None, "template_panel": None,
           "n_trajectories": len(slots)}
    if templates is not None:
        out["template_multiplicities"] = template_multiplicities(templates)
        out["templates"] = template_sensitivity(list(lf["stratum"]), E, templates, ids,
                                                seed=0 if seed is None else seed, method=method,
                                                path=PATH_POPULATION)
    if sub is not None:
        ref_t = None
        if reference is not None:
            ref = _reference_list(reference, len(slots))
            ref_t = [ref[i] for i in sub]
        lf_t = derive_label_free([slots[i] for i in sub])
        res_t = analyze(lf_t["stratum"], [E[i] for i in sub], reference=ref_t,
                        ids=[ids[i] for i in sub], **kw_population)
        out["template_panel"] = {"n": len(sub), "result": res_t}
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
    sec = r.get("secondary")
    if sec is not None and sec["regime"] in ("COVERAGE", "PARTIAL"):
        head = f"    secondary (per 100 tasks with a usable label, N_Y = {sec['N_Y']}): "
        if sec["not_estimable"] is not None:
            lines.append(head + f"not estimable, {sec['not_estimable']}")
        else:
            pl = sec["plan_literal"]
            txt = head + f"100 Delta / N_Y = {_f(pl['G_hat'])}"
            if "G_L" in pl:
                txt += (f", interval for G rescaled "
                        f"{_g_interval_text(pl['G_L'], pl['G_U'], pl['delta_L'], pl['delta_U'])}")
            else:
                txt += f", approximate interval [{_f(pl['partial']['lower'])}, {_f(pl['partial']['upper'])}]"
            lines.append(txt)
    return lines


def _template_count_line(mult: Mapping[str, int]) -> str:
    """The report line on how the panel's tasks fall in templates."""
    return (f"  Panel tasks: {mult['n_tasks']} in {mult['n_templates']} templates; largest template "
            f"holds {mult['max_multiplicity']}; {mult['tasks_sharing']} tasks share a template.")


def _provenance_lines(prov: Optional[dict]) -> List[str]:
    """Report lines for the A1 rule and the attempt log's counts."""
    out = ["A1 provenance (plan rule)"]
    if prov is not None:
        a1 = prov["by_slot"]["A1"]
        cats = ", ".join(f"{k.split(':', 1)[1]} {v}" for k, v in sorted(a1.items()) if k.startswith("invalid:"))
        out.append(f"  A1 is the first complete response of slot (i, A, 1) that passes the per-response checks, under config hash "
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
        out.append("  Not supplied: resolved slots were passed without the attempt log. The plan's rule "
                   "is that A1 is the first complete response of slot (i, A, 1) that passes the per-response checks, and is never replaced.")
    return out


def _repeat_lines(rf: Mapping[str, int]) -> List[str]:
    """Report lines for the agreeing trajectories with an unobserved repeat."""
    return ["Repeat observation (label-free)",
            f"  agreeing items with both repeats INVALID (R set to 0): {rf['both_repeats_invalid']}",
            f"  agreeing items with one repeat INVALID and no observed flip (R = 0 from one repeat): "
            f"{rf['one_repeat_invalid_no_flip']}",
            f"  agreeing items with one repeat INVALID and an observed flip: {rf['one_repeat_invalid_flip']}"]


def _invalid_lines(inv: Mapping[Tuple[str, str, str], int]) -> List[str]:
    """Report lines for the invalid counts by judge slot, category and benchmark."""
    out = ["Invalid counts by judge slot, category and benchmark"]
    if not inv:
        out.append("  none")
    else:
        out.append("  slot | category | benchmark | count")
        for (slot, cat, bench), c in sorted(inv.items()):
            out.append(f"  {slot} | {cat} | {bench} | {c}")
    return out


def panel_report(panel: dict, prior_exposure: Optional[str] = None) -> str:
    """The report of analyze_panel(), on its path.

    The descriptive path needs prior_exposure and refuses to run without it
    (describe_report). The population-inference report opens with the line saying it
    is outside the primary result, and prints the statement when one is given. It
    covers provenance, the plan's primary analysis, sensitivities, invalid counts and
    templates.
    """
    if panel["path"] == PATH_DESCRIPTIVE:
        return describe_report(panel, prior_exposure)
    prim = panel["primary"]
    lf = prim["label_free"]
    out = [POPULATION_PATH_TEXT, ""]
    if prior_exposure is not None:
        out += [f"Prior exposure statement: {prior_exposure}", ""]
    out.extend(_provenance_lines(panel.get("provenance")))
    out.append("")
    out.extend(_report_lines(prim))
    out.append("")
    out.extend(_repeat_lines(panel["repeat_flags"]))
    out.append("")
    if prim.get("secondary") is not None:
        dens = f"denominators N = {lf['N']} and N_Y = {prim['secondary']['N_Y']}"
    else:
        dens = f"denominator N = {lf['N']}"
    out.append(f"Sensitivity analyses (labeled; s = {lf['s']} and the per-100 {dens} "
               f"are held at their values on the full panel, s capped at n_D0)")
    for name, r in panel["sensitivities"]:
        out.extend(_brief(name, r))
    out.append("")
    out.extend(_invalid_lines(panel["invalid_counts"]))
    tm = panel.get("templates")
    out.append("")
    out.append("Template sensitivity (labeled)")
    out.append(f"  The default panel keeps one trajectory per task (N = {lf['N']}), so tasks that share "
               "a template stay in it and can be correlated; the one-task-per-template panel is the "
               "labeled sensitivity analysis for that dependence. The test and interval of the "
               "per-100-panel-task result assume independent tasks; the inference that handles the "
               "retained dependence, and its synthetic validation, are still to come and are "
               "required before the freeze.")
    mult = panel.get("template_multiplicities")
    if mult is not None:
        out.append(_template_count_line(mult))
    tp = panel.get("template_panel")
    if tp is None:
        out.append("  One-task-per-template panel: not supplied with the panel manifest.")
    else:
        out.append("  One-task-per-template panel (chosen label-free before evaluation; its own N, s and K):")
        out.extend("  " + x for x in _brief("one task per template", tp["result"]))
    if tm is None:
        out.append("  One agreeing item per template: not run, no template ids were supplied with the "
                   "panel manifest.")
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


# ---------------------------------------------------------------------------
# Descriptive path: the primary result
# ---------------------------------------------------------------------------
NOT_DEFINED_TEXT = "not defined"
POLICY_ORDER = ("AUG", "DIS", "RAND")
DEGENERATE_REGIMES = ("DEGENERATE-NONE", "DEGENERATE-CAP", "DEGENERATE-NOSLICE")
REGIME_NOTES = {
    "DEGENERATE-NONE": ("no instability among agreeing verdicts, so instability-first and "
                        "disagreement-first review capture the same expected number"),
    "DEGENERATE-CAP": "the slice covers every agreeing task (s = n_D0), so both policies review all of them",
    "DEGENERATE-NOSLICE": "the slice is empty (s = 0), so neither policy reviews an agreeing task",
    "COVERAGE": "the slice reaches every unstable agreeing task",
    "PARTIAL": "the slice reaches only part of the unstable agreeing tasks",
}
# The sentences of the result, plan section 8, word for word.
RESULT_DEGENERATE_TEXT = {
    "DEGENERATE-NONE": "No agreeing verdict was unstable on this panel, so instability added no review candidates.",
    "DEGENERATE-CAP": "The budget reached every agreeing verdict, so the two policies reviewed the same tasks.",
    "DEGENERATE-NOSLICE": "The budget left no added review slots, so the two policies reviewed the same tasks.",
}
RESULT_NOT_MORE_TEXT = ("On this panel instability-first did not select more reference-contradicted first verdicts "
                        "than disagreement-first, in expectation over each policy's random choices.")
RESULT_MORE_TEXT = ("On this panel instability-first selected {g} more reference-contradicted first verdicts per "
                    "100 panel tasks, in expectation over each policy's random choices, {side} the practical "
                    "reference of 2 per 100.")
RESULT_ONE_TASK_TEXT = "One unstable task changes this number by {x} per 100."
RESULT_CLOSING_TEXT = ("This describes these tasks and these realized judge calls and carries no population "
                       "confidence or significance claim.")
# How to read the counts and the gain, plan section 8, word for word.
READING_U2_TEXT = "A contradicted U2 verdict has both repeats matching the released reference."
READING_U1_TEXT = "A contradicted U1 verdict whose other repeat is valid has a majority that stays with A1."
READING_GAIN_TEXT = "Instability and the reference-contradicted outcome both use A1, so part of any gain is built in."
READING_INDEPENDENT_TEXT = ("With independent calls and a judge that is right more often than wrong on a task, "
                            "contradicted verdicts are relatively more common in U2 than in U1.")
READING_FROZEN_TEXT = "This plan freezes that direction and no numeric benchmark."
# The limitations stated with the result, plan sections 7 and 8. Prior exposure is the statement at the top.
LIMITATIONS_TEXT = ("Limitations. One dataset and two judges. Released labels are a reference, not ground truth. "
                    "Runs are kept as recorded: the way a run ends could influence judge accuracy, and we have "
                    "not established that it does. Tasks that share a template and WorkArena families stay in "
                    "the panel.")
UNSTABLE_GROUPS = ("U2", "U1")
# The review cases of plan section 11. The cells in their fixed order, each with the plan's
# definition, number of cases and what the cell shows; a cell's position names its generator.
CASE_CELL_ROWS = (
    ("A", "Unstable agreeing, E = 1", "2, one U2 and one U1 when both exist",
     "Reached by instability, and by disagreement-first only by chance"),
    ("B", "Unstable agreeing, usable label, E = 0", "1",
     "Flagged by instability, the reference agrees with A1"),
    ("C", "Disagreement with valid B1, E = 1", "1", "Caught by both main policies"),
    ("D", "Stable agreeing, E = 1", "2", "Mostly missed by both main policies"),
    ("E", "Stable agreeing, usable label, E = 0", "1",
     "Control matched to the first D case on benchmark and A1 verdict, falling back to A1 verdict alone"),
    ("F", "Unsure or missing label", "1",
     "The unknown tasks. If none exist, a second control matched to the second D case"),
)
CASE_CELLS = tuple(row[0] for row in CASE_CELL_ROWS)
CASE_SPAWN_KEY = 6                     # the stream of the cases, apart from the other streams
CASE_MATCH_BOTH = "benchmark and A1 verdict"
CASE_MATCH_VERDICT = "A1 verdict alone"
CASE_MATCH_NO_D = "no D case, so the first of the cell's order"
# What the report prints with its cases. The plan's text is word for word.
CASE_INTRO_TEXT = ("A few cases, chosen by this recorded rule, explain what the policies caught and missed. "
                   "They are a selected qualitative sample by one reviewer, not an estimate of how common "
                   "anything is.")
CASE_SELECTION_TEXT = ("Selection. At most 8 cases, on valid A1 only, chosen after the sealed join by a script "
                       "with a seed in the freeze list. Seed: {seed}.")
CASE_WITHIN_CELL_TEXT = ("Within a cell, tasks are sorted by normalized task ID and permuted by a generator "
                         "derived from the frozen seed and the cell. A short cell takes all it has. Nothing is "
                         "backfilled except by the F rule. A recognized case is never replaced.")
CASE_GENERATOR_TEXT = ("Generator of a cell: SeedSequence({seed}, spawn_key=(6, cell index)), with cells A to F "
                       "at index 0 to 5. A cell A with only U2 or only U1 takes up to 2 from that group. With no "
                       "D case, cell E takes the first of its order. The control in cell F is distinct from the "
                       "first control, and there is none without a second D case. A D case with no control of "
                       "the same A1 verdict gets no control.")
CASE_NOT_REACHED_TEXT = " (not at this budget: instability-first has no chance of reviewing this task)"
CASE_NOT_SUPPLIED = "not supplied"
CASE_NO_IDS_TEXT = ("Task IDs were not supplied, so the tasks are sorted by their stored position here. The real "
                    "run must supply them.")
CASE_READING_TEXT = ("E is 1 when the released reference contradicts the first verdict. Each probability "
                     "is that policy's exact chance of reviewing the task, over its own random choices. "
                     "A contradicted task is caught with that probability and missed with the rest.")


def _rate(num: int, den: int) -> dict:
    """num/den with its denominator; the value is None, not defined and never zero, when den = 0."""
    return {"value": Fraction(num, den) if den > 0 else None, "numerator": num, "denominator": den}


def _per_100(x: Fraction, den: int) -> Optional[Fraction]:
    """100 x / den exactly, or None (not defined) when den = 0."""
    return Fraction(100) * x / den if den > 0 else None


def _judge_metrics(pairs: Sequence[Tuple[str, str]], n_usable: int) -> dict:
    """One judge's validity metrics on one set of tasks, with failure as the positive class.

    pairs holds (slot outcome, reference verdict) for every task of the set, an invalid
    output included, and n_usable is N_Y, the number of panel tasks with a usable
    reference. The confusion matrix counts valid verdicts, and invalid outputs are
    counted apart by reference class. Failure recall and success recall keep every
    reference of their class in the denominator, so an invalid output counts as not
    correct and a judge cannot gain by refusing hard cases; balanced accuracy is their
    mean. Failure precision and the contradiction rate are over valid verdicts, valid
    coverage is the valid verdicts over N_Y and prevalence the failure references over
    the tasks of the set. A metric with an empty denominator is not defined (value
    None); balanced accuracy needs both a failure and a success reference.
    """
    c = Counter((canonical_outcome(o), r) for o, r in pairs if is_valid(o))
    tp, fp = c[("failure", "failure")], c[("failure", "success")]
    fn, tn = c[("success", "failure")], c[("success", "success")]
    invalid = Counter(r for o, r in pairs if not is_valid(o))
    inv_f, inv_s = invalid["failure"], invalid["success"]
    n_valid = tp + fp + fn + tn
    failure_refs, success_refs = tp + fn + inv_f, tn + fp + inv_s
    recall, success_recall = _rate(tp, failure_refs), _rate(tn, success_refs)
    balanced = None
    if recall["value"] is not None and success_recall["value"] is not None:
        balanced = (recall["value"] + success_recall["value"]) / 2
    return {"confusion": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
            "invalid": {"failure_references": inv_f, "success_references": inv_s},
            "balanced_accuracy": {"value": balanced, "failure_references": failure_refs,
                                  "success_references": success_refs},
            "failure_recall": recall,
            "success_recall": success_recall,
            "failure_precision": _rate(tp, tp + fp),
            "contradiction": dict(_rate(fp + fn, n_valid), invalid_outputs=inv_f + inv_s),
            "valid_coverage": _rate(n_valid, n_usable),
            "prevalence": _rate(failure_refs, len(pairs))}


def validity_metrics(slots: Sequence[Mapping[str, str]], Y: Sequence[Optional[str]]) -> dict:
    """Full A1 and B1 validity metrics against the usable released labels [labels].

    Failure is the positive class. Only tasks whose released primary label is usable
    count, so an Unsure or missing label is in neither the numerator nor any
    denominator. Each judge has one row on all_usable, every task with a usable
    reference (N_Y), where an invalid output stays in the recall denominators, and one
    on shared, the tasks where A1 and B1 are both verdicts; n_excluded_from_shared is
    the number of usable tasks left out of it. See _judge_metrics for each measure. The
    contradiction rate of a row is over that judge's valid verdicts in the row's set.
    """
    usable = []
    for sl, y in zip(slots, Y):
        status, verdict = reference_status(y)
        if status == USABLE:
            usable.append((sl, verdict))
    outcomes = lambda rows, k: [(sl[k], v) for sl, v in rows]
    shared = [(sl, v) for sl, v in usable if is_valid(sl["A1"]) and is_valid(sl["B1"])]
    n = len(usable)
    return {"n_usable": n, "n_shared": len(shared), "n_excluded_from_shared": n - len(shared),
            "all_usable": {k: _judge_metrics(outcomes(usable, k), n) for k in ("A1", "B1")},
            "shared": {k: _judge_metrics(outcomes(shared, k), n) for k in ("A1", "B1")}}


def select_cases(slots: Sequence[Mapping[str, str]], E: Sequence[int], reference: Sequence[str], *,
                 benchmarks: Optional[Sequence[str]] = None, ids: Optional[Sequence[str]] = None,
                 seed: int = 0) -> List[dict]:
    """The review cases of plan section 11, chosen after the join by the recorded rule [labels].

    On valid A1 only, at most 8 cases, in the cell order A to F:
      A  U2 or U1 with E = 1: 2 cases, one U2 and one U1 when both exist; if only one
         of them has any, up to 2 from it.
      B  U2 or U1, usable reference, E = 0: 1 case.
      C  G1 whose B1 is a valid verdict, E = 1: 1 case.
      D  S with E = 1: 2 cases.
      E  S, usable reference, E = 0: 1 control matched to the first D case on benchmark
         and A1 verdict, falling back to A1 verdict alone; with no D case, the first of
         the cell's order.
      F  Reference Unsure or missing: 1 case. If F is empty, a second control from cell
         E, matched to the second D case by the same rule and distinct from the first
         control; with no second D case, none.
    Within a cell, tasks are sorted by ids (normalized task IDs; without them, by stored
    position, which the real run must not use), permuted with a generator from
    SeedSequence(seed, spawn_key=(6, cell index)) and the first cases are taken. A short
    cell takes all it has, and nothing is backfilled except by the F rule.

    Without benchmarks the controls match on the A1 verdict alone. Implementation choice:
    the plan names no further fallback, so a D case with no control of the same A1
    verdict gets no control.

    Returns one dict per case, in cell order: 'cell', 'index' (position in the panel)
    and 'matched_on' (how a control was matched, None for any other case).
    """
    slots, E, reference = list(slots), list(E), list(reference)
    n = len(slots)
    if len(E) != n or len(reference) != n:
        raise ValueError("E and reference must give one entry per trajectory")
    if benchmarks is not None and len(benchmarks) != n:
        raise ValueError("benchmarks must give one entry per trajectory")
    if ids is None:
        keys = list(range(n))
    else:
        keys = list(ids)
        if len(keys) != n or len(set(keys)) != len(keys):
            raise ValueError("ids must be unique and one per trajectory")
    labels = list(derive_label_free(slots)["stratum"])
    a1 = [canonical_outcome(sl["A1"]) for sl in slots]
    members = {"A": [i for i in range(n) if labels[i] in UNSTABLE_GROUPS and E[i] == 1],
               "B": [i for i in range(n) if labels[i] in UNSTABLE_GROUPS and reference[i] == USABLE and E[i] == 0],
               "C": [i for i in range(n) if labels[i] == "G1" and is_valid(slots[i]["B1"]) and E[i] == 1],
               "D": [i for i in range(n) if labels[i] == "S" and E[i] == 1],
               "E": [i for i in range(n) if labels[i] == "S" and reference[i] == USABLE and E[i] == 0],
               "F": [i for i in range(n) if labels[i] != "I" and reference[i] in (UNSURE, MISSING)]}
    order = {}
    for index, cell in enumerate(CASE_CELLS):
        rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(CASE_SPAWN_KEY, index)))
        order[cell] = [int(i) for i in rng.permutation(_sorted_by_key(members[cell], keys))]

    def control(d_case: int, used: Sequence[int]) -> Tuple[Optional[int], str]:
        """The first task of cell E's order, not yet used, that matches this D case."""
        pool = [i for i in order["E"] if i not in used and a1[i] == a1[d_case]]
        if benchmarks is not None:
            for i in pool:
                if benchmarks[i] == benchmarks[d_case]:
                    return i, CASE_MATCH_BOTH
        return (pool[0] if pool else None), CASE_MATCH_VERDICT

    first_u2 = [i for i in order["A"] if labels[i] == "U2"][:1]
    first_u1 = [i for i in order["A"] if labels[i] == "U1"][:1]
    cell_a = first_u2 + first_u1 if first_u2 and first_u1 else order["A"][:2]
    d_cases = order["D"][:2]
    chosen = [{"cell": "A", "index": i, "matched_on": None} for i in cell_a]
    chosen += [{"cell": "B", "index": i, "matched_on": None} for i in order["B"][:1]]
    chosen += [{"cell": "C", "index": i, "matched_on": None} for i in order["C"][:1]]
    chosen += [{"cell": "D", "index": i, "matched_on": None} for i in d_cases]
    if d_cases:
        first_control, how = control(d_cases[0], [])
    else:
        first_control, how = (order["E"] + [None])[0], CASE_MATCH_NO_D
    if first_control is not None:
        chosen.append({"cell": "E", "index": first_control, "matched_on": how})
    if order["F"]:
        chosen.append({"cell": "F", "index": order["F"][0], "matched_on": None})
    elif len(d_cases) == 2:
        second_control, how = control(d_cases[1], [first_control])
        if second_control is not None:
            chosen.append({"cell": "F", "index": second_control, "matched_on": how})
    return chosen


def _one_line(x) -> str:
    """A free-text value on one report line; a blank or absent one is 'none'."""
    if x is None or (isinstance(x, float) and math.isnan(x)) or not str(x).strip():
        return "none"
    return " ".join(str(x).split())


def _released_label_text(Y: Optional[Sequence[Optional[str]]], i: int) -> str:
    """Task i's released label as the case view prints it [labels]."""
    if Y is None:
        return CASE_NOT_SUPPLIED
    return MISSING if reference_status(Y[i])[0] == MISSING else str(Y[i])


def _case_value(values: Optional[Sequence[Optional[str]]], i: int) -> str:
    """Task i's entry of an optional input of the case view."""
    return CASE_NOT_SUPPLIED if values is None else _one_line(values[i])


def _case_rationales(rationales: Optional[Sequence[Mapping[str, Optional[str]]]], i: int) -> Optional[Dict[str, str]]:
    """Task i's rationales by slot, or None when none were given."""
    return None if rationales is None else {k: _one_line(rationales[i].get(k)) for k in SLOT_NAMES.values()}


def _case_shows(cell: str, matched_on: Optional[str], instability_probability: Fraction) -> str:
    """What a case's cell shows, in the plan's words, with how a control was matched.

    The plan's words for cell A say the task is reached by instability. In PARTIAL a
    U1 task can have no chance of review under instability-first, and the row says so.
    """
    shows = {row[0]: row[3] for row in CASE_CELL_ROWS}[cell]
    if cell == "A" and instability_probability == 0:
        shows += CASE_NOT_REACHED_TEXT
    if cell == "F":
        unknown, second_control = shows.split(". If none exist, ")
        shows = unknown if matched_on is None else second_control
    return shows if matched_on is None else f"{shows} (matched here: {matched_on})"


def inference_costs(costs: Mapping[str, object], gain: Fraction) -> dict:
    """Actual inference costs in USD, taken from the spending ledger.

    costs has 'total' and 'by_slot', the latter with one amount for each of A1, A2,
    A3 and B1, as decimal strings the way the ledger writes them or as numbers. The
    total can exceed the sum over the slots, for example through a reconciliation
    adjustment, and the rest is reported as not attributed to a slot; it can never be
    smaller. gain is the expected verdicts instability-first review captures beyond
    disagreement-first review. The extra cost is that of the repeats A2 and A3, and the
    extra cost per extra expected verdict is not defined (None) unless the gain is positive.
    """
    def amount(name: str, x) -> Fraction:
        try:
            v = Fraction(str(x))
        except (ValueError, ZeroDivisionError):
            raise ValueError(f"{name} is not an amount: {x!r}") from None
        if v < 0:
            raise ValueError(f"{name} is negative: {x!r}")
        return v
    if not {"total", "by_slot"} <= set(costs):
        raise ValueError("costs must give 'total' and 'by_slot'")
    names = tuple(SLOT_NAMES.values())
    if set(costs["by_slot"]) != set(names):
        raise ValueError(f"by_slot must give exactly the slots {names}")
    total = amount("total", costs["total"])
    by_slot = {k: amount(k, costs["by_slot"][k]) for k in names}
    attributed = sum(by_slot.values(), Fraction(0))
    if attributed > total:
        raise ValueError("the slot costs add up to more than the total")
    extra = by_slot["A2"] + by_slot["A3"]
    return {"total": total, "by_slot": by_slot, "not_attributed": total - attributed,
            "extra_repeats": extra, "gain": gain,
            "extra_per_extra_verdict": extra / gain if gain > 0 else None}


def invalid_outcome_variants(slots: Sequence[Mapping[str, str]], E: Sequence[int], s: int, N: int) -> List[dict]:
    """The difference in expected capture under each invalid-outcome rule of plan section 9 [labels].

    Each rule of SENSITIVITY_VARIANTS reclassifies the trajectories with
    derive_label_free. s is held at its primary value, capped at the variant's n_D0,
    and N is held as the per-100 denominator. Each variant gives only the difference
    between instability-first and disagreement-first, in verdicts and per 100 panel
    tasks. The report prints the difference in verdicts only, as plan section 9 says.
    """
    out = []
    for name, opts in SENSITIVITY_VARIANTS:
        labels = list(derive_label_free(slots, **opts)["stratum"])
        design = Design.from_labels(labels, s_override=s)
        counts = {j: sum(e for l, e in zip(labels, E) if l == j) for j in ("G1", "U2", "U1", "S")}
        difference = expected_captures(design, counts)["Delta"]
        out.append({"name": name.split(": ", 1)[1], "difference": difference, "per_100_panel": _per_100(difference, N)})
    return out


def lock_quantities(design: Design) -> dict:
    """What the gain on this panel can do, without reading a label.

    These three quantities are committed at the prediction lock, with the rest of the
    label-free design.

    max_gain_per_100 is the largest gain per 100 panel tasks the panel can reach: the
    sum over U2, U1 and S of max(0, w_j) n_j, times 100/N, exact, where w_j is the share
    of group j that instability-first reviews minus s/n_D0. The gain is the sum of
    w_j m_j, so it is largest when every task of a group with a positive weight is
    reference-contradicted and no other agreeing task is. reference_reachable says
    whether that largest gain reaches the practical reference of 2 per 100. x_per_100 is
    the change in the gain per 100 from one unstable task: 100/N times the largest weight
    among unstable groups with at least one task, which in COVERAGE is (n_D0 - s)/n_D0.
    In the degenerate regimes the gain is 0 whatever the labels are, so all of these
    are 0 and the reference is not reachable.
    """
    if design.regime in DEGENERATE_REGIMES:
        return {"max_gain_per_100": Fraction(0), "reference_reachable": False, "x_per_100": Fraction(0)}
    weight = delta_weights(design)
    size = {"U2": design.n_U2, "U1": design.n_U1, "S": design.n_S}
    per_100 = Fraction(100, design.N_per100)
    largest = per_100 * sum((max(Fraction(0), weight[j]) * size[j] for j in weight), Fraction(0))
    return {"max_gain_per_100": largest, "reference_reachable": largest >= MATERIAL_M,
            "x_per_100": per_100 * max(weight[j] for j in UNSTABLE_GROUPS if size[j] > 0)}


def _gain_text(g: Fraction) -> str:
    """A positive gain to 4 decimals, or with more digits when 4 would contradict the comparison.

    A gain just under the practical reference must not read as 2.0000 beside the words
    below the practical reference, nor a very small gain as 0.0000 more verdicts. Exact.
    """
    text, digits = _f(g), 4
    while Fraction(text) <= 0 or g < MATERIAL_M <= Fraction(text):
        digits += 1
        scaled = round(g * 10 ** digits)
        text = f"{scaled // 10 ** digits}.{scaled % 10 ** digits:0{digits}d}"
    return text


def result_sentences(regime: str, g_hat: Optional[Fraction], x: Fraction) -> List[str]:
    """The sentences of plan section 8 for one result, word for word.

    g_hat is the gain per 100 panel tasks and x the change in it from one unstable
    task (lock_quantities). Both are printed as decimals, the gain with as many digits
    as keep it from reading as the reference or as zero. A degenerate regime has its
    own sentence; otherwise the sentence follows g_hat, compared exactly with 0 and with
    the practical reference of 2 per 100, and is followed by the sentence on x. Every
    case ends with the closing sentence.
    """
    if regime in RESULT_DEGENERATE_TEXT:
        return [RESULT_DEGENERATE_TEXT[regime], RESULT_CLOSING_TEXT]
    if g_hat <= 0:
        first = RESULT_NOT_MORE_TEXT
    else:
        first = RESULT_MORE_TEXT.format(g=_gain_text(g_hat), side="at or above" if g_hat >= MATERIAL_M else "below")
    return [first, RESULT_ONE_TASK_TEXT.format(x=_f(x)), RESULT_CLOSING_TEXT]


def describe_panel(slots: Sequence[Mapping[str, str]], E: Optional[Sequence[int]] = None, *,
                   Y: Optional[Sequence[Optional[str]]] = None,
                   reference: Optional[Sequence[str]] = None,
                   benchmarks: Optional[Sequence[str]] = None,
                   costs: Optional[Mapping[str, object]] = None,
                   provenance: Optional[dict] = None,
                   ids: Optional[Sequence[str]] = None,
                   templates: Optional[Sequence[str]] = None,
                   rationales: Optional[Sequence[Mapping[str, Optional[str]]]] = None,
                   second_annotator: Optional[Sequence[Optional[str]]] = None,
                   case_seed: int) -> dict:
    """The primary result: a descriptive audit of the frozen panel [labels].

    Give the released primary labels Y, or E with its reference statuses as
    join_reference returns them. Y is needed for the validity metrics, since E alone
    cannot give B1's reference where A1 is invalid; with E they are not assessed.
    Without statuses every label is treated as usable, and the secondary result and
    the label coverage are not assessed, as in analyze(). costs comes from the
    spending ledger (inference_costs) and provenance from resolve_panel. ids names
    the tasks, and it, the benchmarks and case_seed fix the cases (select_cases).
    case_seed is the seed in the freeze list. It has no default, so a forgotten seed
    stops the analysis instead of choosing the cases with seed 0. templates gives each
    panel task's template id, label-free, and the report gives N with its template
    multiplicities (template_multiplicities). rationales gives, for each trajectory,
    the judge rationale of each slot, by slot name, and second_annotator its second
    annotator's label, if any. Each case shows them with its released label, and an
    input that was not given is printed as not supplied.

    The conclusions apply to these tasks and these realized judge calls. Every number
    is computed exactly in the regime at hand, degenerate ones included: the expected
    capture of each policy over its own random choices at the same K, the share of M
    each captures, the gain of instability-first review over disagreement-first and
    over random review, in verdicts and per 100 panel tasks, and, as a check,
    disagreement-first minus random on the same two scales. The secondary result
    is the same gain per 100 tasks with a usable label. A rate with an empty
    denominator is None, not defined and never zero. lock holds the label-free
    quantities committed at the prediction lock (lock_quantities), and result_sentences
    the plan's sentences for this result (result_sentences). invalid_outcome_variants
    holds the difference in expected capture under each rule of plan section 9. The
    observed gain over
    disagreement-first review is set against the practical reference of 2 per 100
    panel tasks only as observed at or above it, or below it, and only in COVERAGE and
    PARTIAL; a degenerate regime makes no comparison. contradiction gives how
    often each judge contradicts the references on its own valid tasks with a usable
    reference, with the number of its invalid outputs among usable-reference tasks;
    B1's is None without Y. review_probability holds each task's exact
    probability of review under each policy, and cases the tasks of the recorded rule.
    """
    if (E is None) == (Y is None):
        raise ValueError("give either E or the released labels Y, and not both")
    if Y is not None and reference is not None:
        raise ValueError("the released labels Y already give the reference statuses")
    if isinstance(case_seed, bool) or not isinstance(case_seed, numbers.Integral):
        raise ValueError("case_seed must be the integer seed in the freeze list")
    slots = list(slots)
    n = len(slots)
    if benchmarks is not None and len(benchmarks) != n:
        raise ValueError("benchmarks must give one entry per trajectory")
    for name, given in (("templates", templates), ("rationales", rationales), ("second_annotator", second_annotator)):
        if given is not None and len(given) != n:
            raise ValueError(f"{name} must give one entry per trajectory")
    lf = derive_label_free(slots)
    labels = list(lf["stratum"])
    if Y is not None:
        joined = join_reference(slots, Y)
        E, ref = joined["E"], joined["reference"]
    else:
        E = list(E)
        if len(E) != n or any(e not in (0, 1) for e in E):
            raise ValueError("E must give 0 or 1 for each trajectory")
        E = [int(e) for e in E]
        ref = _reference_list(reference, n)
        _require_no_contradiction_without_reference(labels, E, ref)
    design = Design.from_labels(labels)
    lock = lock_quantities(design)
    counts = {j: sum(e for l, e in zip(labels, E) if l == j) for j in ("G1", "U2", "U1", "S")}
    cap = expected_captures(design, counts)
    M = sum(counts.values())
    reviews = {pol: expected_reviewed(design, pol) for pol in POLICY_ORDER}
    over_dis, over_rand = cap["AUG"] - cap["DIS"], cap["AUG"] - cap["RAND"]
    dis_over_rand = cap["DIS"] - cap["RAND"]
    observed = _per_100(over_dis, design.N)
    if design.regime in DEGENERATE_REGIMES:             # plan section 7: printed in COVERAGE and in PARTIAL
        status = None
    elif observed >= MATERIAL_M:
        status = "observed at or above the reference"
    else:
        status = "observed below the reference"
    inv = invalid_counts(slots, benchmarks)
    by_slot_category: Counter = Counter()
    for (slot, cat, _), c in inv.items():
        by_slot_category[(slot, cat)] += c
    statuses_given = Y is not None or reference is not None
    cov = reference_coverage(labels, ref, design, benchmarks=benchmarks, E=E) if statuses_given else None
    N_Y = cov["groups"]["all"]["usable"] if statuses_given else None
    validity = validity_metrics(slots, Y) if Y is not None else None
    n_a1_usable = sum(1 for l, r in zip(labels, ref) if l != "I" and r == USABLE)
    n_a1_invalid = sum(1 for l, r in zip(labels, ref) if l == "I" and r == USABLE)
    probs = selection_probabilities(design, labels)
    cases = [{"cell": c["cell"], "task": c["index"] if ids is None else ids[c["index"]],
              "group": labels[c["index"]],
              "slots": {k: canonical_outcome(slots[c["index"]][k]) for k in SLOT_NAMES.values()},
              "E": E[c["index"]], "reference": ref[c["index"]] if statuses_given else None,
              "released_label": _released_label_text(Y, c["index"]),
              "second_annotator": _case_value(second_annotator, c["index"]),
              "rationales": _case_rationales(rationales, c["index"]),
              "review_probability": {pol: probs[pol][c["index"]] for pol in POLICY_ORDER},
              "shows": _case_shows(c["cell"], c["matched_on"], probs["AUG"][c["index"]]), "matched_on": c["matched_on"]}
             for c in select_cases(slots, E, ref, benchmarks=benchmarks, ids=ids, seed=case_seed)]
    return {"path": PATH_DESCRIPTIVE, "provenance": provenance, "N": design.N, "n_inv": design.n_inv,
            "N_V": design.N_V, "n_D1": design.n_D1, "n_D0": design.n_D0, "n_U2": design.n_U2,
            "n_U1": design.n_U1, "n_S": design.n_S, "s": design.s, "K": design.K, "regime": design.regime,
            "template_multiplicities": template_multiplicities(templates) if templates is not None else None,
            "lock": lock, "result_sentences": result_sentences(design.regime, observed, lock["x_per_100"]),
            "K_by_policy": reviews, "K_over_N": {pol: reviews[pol] / design.N if design.N > 0 else None for pol in POLICY_ORDER},
            "K_equal": all(v == design.K for v in reviews.values()),
            "R_unobserved": int((lf["R_unobserved"] == 1).sum()), "repeat_flags": _repeat_flags(lf),
            "invalid_by_slot": {k: sum(c for (slot, _), c in by_slot_category.items() if slot == k)
                                for k in SLOT_NAMES.values()},
            "invalid_by_slot_category": dict(by_slot_category),
            "invalid_counts": inv if benchmarks is not None else None,
            "reference_supplied": statuses_given, "N_Y": N_Y, "coverage": cov,
            "M": M, "m1": counts["U2"] + counts["U1"], "m0": counts["S"], "m_counts": counts,
            "captures": {pol: cap[pol] for pol in POLICY_ORDER},
            "share_of_M": {pol: cap[pol] / M if M > 0 else None for pol in POLICY_ORDER},
            "gain": {"over_disagreement": {"verdicts": over_dis, "per_100_panel": observed},
                     "over_random": {"verdicts": over_rand, "per_100_panel": _per_100(over_rand, design.N)},
                     "disagreement_over_random": {"verdicts": dis_over_rand,
                                                  "per_100_panel": _per_100(dis_over_rand, design.N)}},
            "secondary": None if N_Y is None else {"over_disagreement": _per_100(over_dis, N_Y)},
            "invalid_outcome_variants": invalid_outcome_variants(slots, E, design.s, design.N),
            "practical_reference": {"per_100": MATERIAL_M, "status": status},
            "contradiction": {"A1": dict(_rate(M, n_a1_usable), invalid_outputs=n_a1_invalid),
                              "B1": validity["all_usable"]["B1"]["contradiction"] if validity is not None else None},
            "review_probability": {pol: probs[pol] for pol in POLICY_ORDER},
            "cases": {"seed": case_seed, "rule": _case_rule_lines(case_seed, ids is not None),
                      "sort_key": "ids" if ids is not None else "stored order", "tasks": cases},
            "validity": validity,
            "costs": inference_costs(costs, over_dis) if costs is not None else None}


def _case_rule_lines(seed: int, ids_given: bool) -> List[str]:
    """The recorded rule of the cases, in the plan's words, with the seed."""
    lines = [CASE_INTRO_TEXT, CASE_SELECTION_TEXT.format(seed=seed), "Cell | Definition | Cases | What it shows"]
    lines += [" | ".join(row) for row in CASE_CELL_ROWS]
    lines += [CASE_WITHIN_CELL_TEXT, CASE_GENERATOR_TEXT.format(seed=seed)]
    return lines + ([] if ids_given else [CASE_NO_IDS_TEXT])


def _template_lines(mult: Optional[Mapping[str, object]]) -> List[str]:
    """Report lines giving N with its template multiplicities."""
    if mult is None:
        return ["  Template multiplicities: not supplied with the panel manifest."]
    sizes = ", ".join(f"{k} task{'' if k == 1 else 's'}: {v}" for k, v in mult["histogram"].items())
    return [_template_count_line(mult), f"  Templates by number of tasks: {sizes}"]


def _exact_text(x: Optional[Fraction]) -> str:
    """An exact fraction with its decimal, or 'not defined'."""
    return NOT_DEFINED_TEXT if x is None else f"{x} ({_f(x)})"


def _reference_gain_text(x: Optional[Fraction]) -> str:
    """The gain set against the practical reference, its decimal agreeing with the comparison."""
    return _exact_text(x) if x is None or x <= 0 else f"{x} ({_gain_text(x)})"


def _rate_text(r: Mapping[str, object], what: str) -> str:
    return f"{_exact_text(r['value'])}, {r['numerator']} of {r['denominator']} {what}"


def _metric_lines(name: str, mt: dict, n_set: int) -> List[str]:
    c, ba, inv = mt["confusion"], mt["balanced_accuracy"], mt["invalid"]
    return [f"  {name} (n = {n_set})",
            f"    confusion matrix of valid verdicts: TP = {c['TP']}, FP = {c['FP']}, FN = {c['FN']}, TN = {c['TN']}",
            f"    invalid outputs: {inv['failure_references']} on failure references, "
            f"{inv['success_references']} on success references",
            f"    balanced accuracy: {_exact_text(ba['value'])}, denominators "
            f"{ba['failure_references']} failure references and {ba['success_references']} success references, "
            f"an invalid output counted as not correct",
            f"    failure recall: {_rate_text(mt['failure_recall'], 'failure references, invalid outputs included')}",
            f"    success recall: {_rate_text(mt['success_recall'], 'success references, invalid outputs included')}",
            f"    failure precision: {_rate_text(mt['failure_precision'], 'valid failure verdicts')}",
            f"    contradiction rate: {_rate_text(mt['contradiction'], 'valid verdicts in this set')}; "
            f"invalid outputs in this set: {mt['contradiction']['invalid_outputs']}",
            f"    valid coverage: {_rate_text(mt['valid_coverage'], 'tasks with a usable reference')}",
            f"    prevalence of failure references: {_rate_text(mt['prevalence'], 'tasks in this set')}"]


def _validity_lines(v: Optional[dict]) -> List[str]:
    """Report lines of the A1 and B1 validity metrics."""
    out = ["A1 AND B1 VALIDITY against usable references, with failure as the positive class"]
    if v is None:
        out.append("  Not assessed: the released labels were not supplied.")
        return out
    out.append(f"  Tasks with a usable reference: N_Y = {v['n_usable']}; shared subset, where A1 and B1 are "
               f"both valid: {v['n_shared']}; usable tasks excluded from the shared subset: "
               f"{v['n_excluded_from_shared']}. An Unsure or missing reference counts in neither.")
    for k in ("A1", "B1"):
        out += _metric_lines(f"{k} on all tasks with a usable reference", v["all_usable"][k], v["n_usable"])
    for k in ("A1", "B1"):
        out += _metric_lines(f"{k} on the shared subset", v["shared"][k], v["n_shared"])
    return out


def _contradiction_lines(c: Mapping[str, Optional[dict]], statuses_given: bool) -> List[str]:
    """Report lines of how often each judge contradicts the released references."""
    where = "a usable reference" if statuses_given else "a reference, every label treated as usable"
    tasks = "tasks with a usable reference" if statuses_given else "tasks with a reference"
    out = ["JUDGE CONTRADICTION: how often each judge contradicts the released references, on its own valid tasks",
           f"  A1: {_rate_text(c['A1'], f'tasks with a valid A1 and {where}')}; "
           f"invalid outputs among {tasks}: {c['A1']['invalid_outputs']}"]
    if c["B1"] is None:
        out.append("  B1: not assessed: the released labels were not supplied, and E alone cannot give "
                   "B1's reference where A1 is invalid.")
    else:
        out.append(f"  B1: {_rate_text(c['B1'], f'tasks with a valid B1 and {where}')}; "
                   f"invalid outputs among {tasks}: {c['B1']['invalid_outputs']}")
    return out


def _variant_lines(d: Mapping[str, object]) -> List[str]:
    """Report lines of the invalid-outcome variants."""
    n_y = "not assessed" if d["N_Y"] is None else d["N_Y"]
    out = ["INVALID-OUTCOME VARIANTS: instability-first minus disagreement-first expected capture",
           f"  s is held at {d['s']}, capped at each variant's n_D0. N = {d['N']} and N_Y = {n_y} stay the per-100 "
           f"denominators, and each line is the difference in expected capture only."]
    for v in d["invalid_outcome_variants"]:
        out.append(f"  {v['name']}: {_exact_text(v['difference'])} verdicts")
    return out


def _case_lines(c: dict) -> List[str]:
    """Report lines of the cases chosen by the recorded rule."""
    out = ["CASES: what each policy caught and missed"]
    out += ["  " + line for line in c["rule"]]
    out.append("  " + CASE_READING_TEXT)
    if not c["tasks"]:
        out.append("  None: no task falls in a cell.")
        return out
    name = "task" if c["sort_key"] == "ids" else "position from 0"
    out.append(f"  cell | {name} | group | A1, A2, A3, B1 | reference | released label | second annotator | E | "
               + " | ".join(POLICY_NAMES[p] for p in POLICY_ORDER) + " | what the cell shows")
    for t in c["tasks"]:
        outcomes = ", ".join(t["slots"][k] for k in SLOT_NAMES.values())
        reference = "not assessed" if t["reference"] is None else t["reference"]
        probs = " | ".join(_exact_text(t["review_probability"][p]) for p in POLICY_ORDER)
        out.append(f"  {t['cell']} | {t['task']} | {t['group']} | {outcomes} | {reference} | {t['released_label']} | "
                   f"{t['second_annotator']} | {t['E']} | {probs} | {t['shows']}")
        rationale_text = (CASE_NOT_SUPPLIED if t["rationales"] is None
                          else "; ".join(f"{k}: {v}" for k, v in t["rationales"].items()))
        out.append(f"    rationales: {rationale_text}")
    return out


def _cost_lines(c: Optional[dict]) -> List[str]:
    """Report lines of the actual inference costs."""
    out = ["ACTUAL INFERENCE COSTS in USD, from the spending ledger"]
    if c is None:
        out.append("  Not supplied.")
        return out
    out.append(f"  total: {_f(c['total'])}")
    out.append("  by slot: " + ", ".join(f"{k} {_f(v)}" for k, v in c["by_slot"].items()))
    out.append(f"  not attributed to a slot: {_f(c['not_attributed'])}")
    out.append(f"  extra cost of A2 and A3: {_f(c['extra_repeats'])}")
    if c["extra_per_extra_verdict"] is None:
        out.append(f"  extra cost per extra expected verdict captured: {NOT_DEFINED_TEXT}, since the gain over "
                   f"disagreement-first review is not positive ({_exact_text(c['gain'])} verdicts)")
    else:
        out.append(f"  extra cost per extra expected verdict captured: {_f(c['extra_per_extra_verdict'])}, "
                   f"the extra cost over the gain of {_exact_text(c['gain'])} verdicts")
    return out


def describe_report(desc: dict, prior_exposure: Optional[str] = None) -> str:
    """The primary report of one panel, from describe_panel().

    prior_exposure is the statement of what was exposed before the prediction lock.
    The report refuses to run without one. It prints the fixed scope sentence and the
    panel wording, and carries no inferential quantity of any kind.
    """
    if not isinstance(prior_exposure, str) or not prior_exposure.strip():
        raise ValueError("the descriptive report refuses to run without a prior exposure statement")
    d = desc
    equal = "equal across the three policies" if d["K_equal"] else "NOT equal across the three policies"
    out = [f"Primary result: descriptive audit of the frozen panel, analysis plan version {PLAN_VERSION}",
           SCOPE_TEXT,
           f"Panel: {PANEL_WORDING}.",
           f"Prior exposure statement: {prior_exposure}",
           "",
           "RESULT"]
    out += ["  " + sentence for sentence in d["result_sentences"]]
    out += ["", LIMITATIONS_TEXT, ""]
    out += _provenance_lines(d["provenance"])
    out += ["", "Label-free design",
            f"  N = {d['N']}, invalid A1 n_inv = {d['n_inv']}, valid N_V = {d['N_V']}",
            *_template_lines(d["template_multiplicities"]),
            f"  disagreeing n_D1 = {d['n_D1']}, agreeing n_D0 = {d['n_D0']}",
            f"  agreeing groups: U2 = {d['n_U2']}, U1 = {d['n_U1']}, S = {d['n_S']}",
            f"  valid first verdicts with both repeats invalid, disagreeing tasks included "
            f"(R_unobserved): {d['R_unobserved']}",
            f"  slice s = {d['s']}; review count K = {d['K']}, "
            f"K/N = {_exact_text(Fraction(d['K'], d['N']) if d['N'] else None)}, {equal} ("
            + ", ".join(f"{POLICY_NAMES[p]} {d['K_by_policy'][p]}" for p in POLICY_ORDER) + ")",
            f"  regime: {d['regime']}; {REGIME_NOTES[d['regime']]}",
            f"  largest gain per 100 panel tasks the panel can reach: {_exact_text(d['lock']['max_gain_per_100'])}",
            f"  2 per 100 reachable on this panel: {'yes' if d['lock']['reference_reachable'] else 'no'}",
            f"  change in the gain per 100 panel tasks from one unstable task, x: {_exact_text(d['lock']['x_per_100'])}",
            ""]
    out += _repeat_lines(d["repeat_flags"])
    out += ["", "INVALID OUTPUTS by judge slot and category", "  slot | invalid | categories"]
    for k, total in d["invalid_by_slot"].items():
        cats = ", ".join(f"{cat} {c}" for (slot, cat), c in sorted(d["invalid_by_slot_category"].items())
                         if slot == k)
        out.append(f"  {k} | {total} | {cats if cats else 'none'}")
    if d["invalid_counts"] is not None:
        out += [""] + _invalid_lines(d["invalid_counts"])
    out.append("")
    if d["reference_supplied"]:
        out += _coverage_lines(d)
    else:
        out += ["REFERENCE LABEL COVERAGE: Unsure and missing labels", "  " + REFERENCE_NOT_SUPPLIED_TEXT]
    mc = d["m_counts"]
    out += [""] + _contradiction_lines(d["contradiction"], d["reference_supplied"])
    out += ["", "REFERENCE-CONTRADICTED FIRST VERDICTS",
            f"  M = {d['M']}: disagreement m_G1 = {mc['G1']}; unstable agreeing m1 = {d['m1']} "
            f"(m_U2 = {mc['U2']}, m_U1 = {mc['U1']}); stable agreeing m0 = {d['m0']}",
            "  " + READING_U2_TEXT,
            "  " + READING_U1_TEXT,
            "  " + READING_INDEPENDENT_TEXT,
            "  " + READING_FROZEN_TEXT,
            "", "EXPECTED CAPTURE at the same review count, exact over each policy's random choices",
            "  policy | expected capture | K | K/N | share of M"]
    for p in POLICY_ORDER:
        out.append(f"  {POLICY_NAMES[p]} | {_exact_text(d['captures'][p])} | {d['K_by_policy'][p]} | "
                   f"{_exact_text(d['K_over_N'][p])} | {_exact_text(d['share_of_M'][p])}")
    g = d["gain"]
    out += ["", "GAIN OF INSTABILITY-FIRST REVIEW, in expected reference-contradicted first verdicts"]
    for key, name in (("over_disagreement", "disagreement-first"), ("over_random", "random")):
        out.append(f"  over {name}: {_exact_text(g[key]['verdicts'])} verdicts; "
                   f"{_exact_text(g[key]['per_100_panel'])} per 100 panel tasks (N = {d['N']})")
    check = g["disagreement_over_random"]
    out.append(f"  check, disagreement-first minus random: {_exact_text(check['verdicts'])} verdicts; "
               f"{_exact_text(check['per_100_panel'])} per 100 panel tasks (N = {d['N']})")
    out.append("  " + READING_GAIN_TEXT)
    out.append("")
    if d["secondary"] is None:
        out += ["SECONDARY RESULT: the same gain per 100 tasks with a usable label",
                "  Not assessed: the reference statuses were not supplied."]
    else:
        out.append(f"SECONDARY RESULT: the same gain per 100 tasks with a usable label "
                   f"(N_Y = {d['N_Y']} of N = {d['N']})")
        out.append(f"  over disagreement-first: {_exact_text(d['secondary']['over_disagreement'])} per 100")
    pr = d["practical_reference"]
    out += ["", f"PRACTICAL REFERENCE: {pr['per_100']} per 100 panel tasks, descriptive only",
            f"  gain over disagreement-first per 100 panel tasks: "
            f"{_reference_gain_text(g['over_disagreement']['per_100_panel'])}; "
            f"{pr['status'] if pr['status'] is not None else 'no comparison is defined'}.",
            ""]
    out += _variant_lines(d) + [""]
    out += _case_lines(d["cases"])
    out.append("")
    out += _validity_lines(d["validity"])
    out.append("")
    out += _cost_lines(d["costs"])
    return "\n".join(out)
