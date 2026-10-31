"""Spending controls: prices, reservations, the cap, a write-ahead ledger and reconciliation.

**Prices.** One versioned table, ``PRICES``, in dollars per million tokens, from the setup
note and the provider facts it relies on: ``claude-sonnet-5`` $2 input, $2.50 cache write
(five minutes), $0.20 cache read and $10 output; ``gpt-6-sol`` and its first fallback
``gpt-6.1-sol`` $2 input and $10 output, with no caching. A model outside the table
cannot be reserved. Every amount is an exact decimal.

**Price modifiers.** ``MODIFIERS`` holds the documented rates a response can show beyond
the table: for A, ``inference_geo`` "us" at 1.1x every token and one-hour cache writes at
$4 per million; for B, cached input ($0.20, or $0.10 on ``gpt-6.1-sol``), cache writes at
the 12-hour rate (2.5x input, the higher of the two documented write rates), input above
272,000 tokens at 2x input and cache and 1.5x output, and the flex, priority and
ultrafast tiers. A complete response that shows a modifier settles at the larger of its
reservation and its cost under the modifier. One that shows a modifier with no
documented price (an A tier other than standard, an A region other than global or us, a
B tier not listed, or a value that is not reported) holds its whole reservation. A B
call whose input count is above 272,000 tokens is refused before it is reserved.

**Reservation.** Before each call, r = (I x 1.02 + 256) x p_in x 1.25 + max_output x p_out.
I is the provider's own pre-send token count; the stage runner refuses paid mode until a
provider token-count client exists. In fake mode only, where no provider exists, I may be
``ESTIMATE_FACTOR`` (2) times the request layer's estimate. The 1.25 prices every input
token as a cache write, so a repeat that misses the cache stays inside its reservation.
max_output is the request's own output cap, 32,000 tokens. The ledger header keeps the
SHA-256 of the price table, the modifiers and these reservation constants, and a ledger
written under other values is refused.

**Admission.** A call is admitted only if settled spend plus open and held reservations
plus r stays at or below the budget's cap: $95 for the pilot stages together, $285 for
the evaluation. Otherwise a refusal is written and ``SpendHalt`` is raised before
anything is sent. In the evaluation a call is also refused (``reconcile_due``) while the
oldest settled or held call that no passing reconciliation follows is more than 24 hours
old.

**Ledger.** One append-only JSONL file per budget; each line is flushed and fsynced
before the next step, so a reservation is on disk before its request leaves. Its header
names the budget, the cap, the pricing hash, the mode and a random ledger id. One process
holds the file (an exclusive lock) and at most one reservation is open at a time, so
calls go out one at a time. ``open_budget_ledger`` keeps each budget's ledger at one fixed
place shared by every run, with a register of every ledger created there, so a moved or
deleted ledger is never silently replaced by a fresh cap: a second ledger for a budget
needs a new-budget note, which its header keeps. A reservation closes from its attempt
record in the attempt log, which keeps the raw response:

- a complete response settles at the cost of the usage it reports (input, cache write,
  cache read and output tokens), or as the price modifiers above say;
- a non-2xx error response settles at zero: Anthropic does not bill failed requests;
  OpenAI's behaviour is unverified and covered by reconciliation;
- no response, a stream that did not complete, an Anthropic stream whose
  ``message_delta`` did not report final usage, or usage not reported in full holds the
  whole reservation, which then counts as spent for good;
- no attempt record at all also holds it.

After a restart, an open reservation counts as spent until ``recover`` matches it to its
attempt record in the same way. A reservation names its run by the run's own id, and its
attempt log relative to that run's folder.

**Reconciliation.** ``read_usage_export`` reads a usage export in the runner's own format
(``EXPORT_FIELDS``, as CSV or JSON), not a provider's format. Every row states the
export's period and its scope (the Anthropic workspace or the OpenAI project), the same
for all of one provider's rows. The export is cumulative from the ledger's creation: a
period that starts after the ledger's ``created_at`` or ends before the provider's last
settled or held call is refused. ``Ledger.reconcile`` prices it with the same table and
compares each provider's total with the ledger's spend on that provider. Usage above the
ledger by more than 1% fails, and a failed reconciliation refuses every later reservation
until ``record_correction`` appends a note. The note first appends, for each provider, a
settled adjustment of the export's excess over the ledger in that failed reconciliation,
so committed spend never stays below what the provider reported.
"""

import csv
import fcntl
import hashlib
import io
import json
import os
import secrets
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from . import dispatch as D
from .storage import AttemptLog
from .streams import anthropic_summary, openai_summary

PRICE_TABLE_VERSION = "prices-2026-10-28-r2"
PRICES = {
    "claude-sonnet-5": {"provider": "anthropic", "input": "2", "cache_write": "2.50", "cache_read": "0.20",
                        "output": "10"},
    "gpt-6-sol": {"provider": "openai", "input": "2", "cache_write": None, "cache_read": None, "output": "10"},
    "gpt-6.1-sol": {"provider": "openai", "input": "2", "cache_write": None, "cache_read": None, "output": "10"},
}
MODIFIERS = {
    "anthropic": {
        "inference_geo": {"global": "1", "us": "1.1"},
        "service_tier": {"standard": "1"},
        "cache_write_1h_per_million": {"claude-sonnet-5": "4"},
    },
    "openai": {
        "service_tier": {"default": "1", "flex": "0.5", "priority": "2", "ultrafast": "6"},
        "cache_read_per_million": {"gpt-6-sol": "0.20", "gpt-6.1-sol": "0.10"},
        "cache_write_per_million": {"gpt-6-sol": "5", "gpt-6.1-sol": "5"},
        "long_context": {"above_input_tokens": 272000, "input_and_cache": "2", "output": "1.5"},
    },
}
TOKEN_KINDS = ("input", "cache_write", "cache_read", "output")
MILLION = Decimal(1000000)

INPUT_MARGIN = Decimal("1.02")
INPUT_PAD_TOKENS = 256
WRITE_FACTOR = Decimal("1.25")
ESTIMATE_FACTOR = 2
CAPS = {"pilot": Decimal("95"), "evaluation": Decimal("285")}
RECONCILE_TOLERANCE = Decimal("0.01")
DAILY_RECONCILE_S = 24 * 3600
B_INPUT_LIMIT = MODIFIERS["openai"]["long_context"]["above_input_tokens"]
RESERVATION_RULE = "r = (I x 1.02 + 256) x p_in x 1.25 + max_output x p_out"
OUTPUT_CAP_RULE = "max_output is the request body's max_tokens (A) or max_output_tokens (B)"

SETTLE, HOLD = "settle", "hold"
COVERAGE_FIELDS = ("period_start", "period_end", "scope")
EXPORT_FIELDS = ("provider", "model", "input_tokens", "cache_write_tokens", "cache_read_tokens",
                 "output_tokens", "cost_usd") + COVERAGE_FIELDS
REGISTER_NAME = "budgets.jsonl"


class LedgerError(RuntimeError):
    """The ledger file is unusable, held by another run, or inconsistent."""


class ConcurrentCall(RuntimeError):
    """A second call was started while one was in flight."""


class SpendHalt(RuntimeError):
    """A call was refused before sending: the cap, a reconciliation, or another reservation check."""

    def __init__(self, kind: str, message: str, entry: dict):
        super().__init__(message)
        self.kind, self.entry = kind, entry


def usd(amount: Decimal) -> str:
    return format(amount, "f")


def _now(clock) -> str:
    return datetime.fromtimestamp(clock(), timezone.utc).isoformat(timespec="milliseconds")


def timestamp(text) -> float:
    """Seconds since the epoch for an ISO 8601 time that names its time zone."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{text!r} is not an ISO 8601 time")
    t = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    if t.tzinfo is None:
        raise ValueError(f"{text!r} names no time zone")
    return t.timestamp()


def pricing_record() -> dict:
    """What fixes every amount the ledger records: prices, modifiers and the reservation constants."""
    return {"version": PRICE_TABLE_VERSION, "prices_per_million_tokens": PRICES, "modifiers": MODIFIERS,
            "reservation": {"rule": RESERVATION_RULE, "input_margin": usd(INPUT_MARGIN),
                            "input_pad_tokens": INPUT_PAD_TOKENS, "write_factor": usd(WRITE_FACTOR),
                            "estimate_factor": ESTIMATE_FACTOR, "output_cap_rule": OUTPUT_CAP_RULE,
                            "b_input_limit_tokens": B_INPUT_LIMIT}}


def pricing_sha256() -> str:
    return hashlib.sha256(json.dumps(pricing_record(), sort_keys=True).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- prices and reservations

def price(model: str) -> dict:
    """Dollars per token for ``model``: {kind: Decimal or None}."""
    if model not in PRICES:
        raise ValueError(f"{model!r} is not in price table {PRICE_TABLE_VERSION}; it cannot be reserved")
    row = PRICES[model]
    return {k: (Decimal(row[k]) / MILLION if row[k] is not None else None) for k in TOKEN_KINDS}


def call_cost(model: str, tokens: dict) -> Decimal:
    """The cost of disjoint token counts at the table's prices; a kind with no price (B's
    cache) is priced as input."""
    p = price(model)
    return sum((Decimal(tokens[k]) * (p[k] if p[k] is not None else p["input"]) for k in TOKEN_KINDS),
               Decimal(0))


def reservation(model: str, input_tokens: int, max_output_tokens: int) -> Decimal:
    """r = (I x 1.02 + 256) x p_in x 1.25 + max_output x p_out."""
    if not (isinstance(input_tokens, int) and input_tokens >= 0 and isinstance(max_output_tokens, int)
            and max_output_tokens > 0):
        raise ValueError("a reservation needs a token count and an output cap")
    p = price(model)
    return ((Decimal(input_tokens) * INPUT_MARGIN + INPUT_PAD_TOKENS) * p["input"] * WRITE_FACTOR
            + Decimal(max_output_tokens) * p["output"])


def estimated_input_tokens(request_estimate: int) -> int:
    """The reservation's input count in fake mode when no count is supplied."""
    return ESTIMATE_FACTOR * request_estimate


def max_output_tokens(judge: str, body: dict) -> int:
    """The output cap a request body sets."""
    return body["max_tokens"] if judge == "A" else body["max_output_tokens"]


# ---------------------------------------------------------------- what one attempt cost

def _int(v):
    return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else None


def priced_usage(provider: str, model: str, summary: dict):
    """(tokens, cost, modifiers, undocumented) for a complete response, or None when its
    usage is not reported in full. ``tokens`` are disjoint counts; ``cost`` applies every
    documented modifier the response shows, each listed in ``modifiers``; a modifier with
    no documented price is listed in ``undocumented``."""
    u, p, m = summary.get("usage") or {}, price(model), MODIFIERS[provider]
    mods, unknown = [], []
    if provider == "anthropic":
        tokens = {"input": _int(u.get("input_tokens")), "cache_write": _int(u.get("cache_write_tokens")),
                  "cache_read": _int(u.get("cache_read_tokens")), "output": _int(u.get("output_tokens"))}
        one_hour = _int(u.get("cache_write_1h_tokens"))
        if None in tokens.values() or (tokens["cache_write"] and one_hour is None):
            return None                       # the five-minute and one-hour split is not reported
        one_hour = one_hour or 0
        if one_hour > tokens["cache_write"]:
            return None
        cost = call_cost(model, dict(tokens, cache_write=tokens["cache_write"] - one_hour))
        if one_hour:
            rate = m["cache_write_1h_per_million"].get(model)
            if rate is None:
                unknown.append("one-hour cache writes")
            else:
                cost += Decimal(one_hour) * Decimal(rate) / MILLION
                mods.append(f"{one_hour} one-hour cache write tokens at ${rate} per million")
        for field in ("service_tier", "inference_geo"):
            value = u.get(field)
            factor = m[field].get(value) if isinstance(value, str) else None
            if factor is None:
                unknown.append(f"{field} {value!r}")
            elif Decimal(factor) != 1:
                cost *= Decimal(factor)
                mods.append(f"{field} {value} at {factor}x every token")
        return tokens, cost, mods, unknown
    inp, out = _int(u.get("input_tokens")), _int(u.get("output_tokens"))
    read, write = _int(u.get("cache_read_tokens")), _int(u.get("cache_write_tokens"))
    if None in (inp, out, read, write) or read + write > inp:
        return None
    tokens = {"input": inp - read - write, "cache_write": write, "cache_read": read, "output": out}
    cost = call_cost(model, dict(tokens, cache_write=0, cache_read=0))
    if read or write:
        r_read, r_write = m["cache_read_per_million"].get(model), m["cache_write_per_million"].get(model)
        if r_read is None or r_write is None:
            unknown.append("cache tokens")
        else:
            cost += (Decimal(read) * Decimal(r_read) + Decimal(write) * Decimal(r_write)) / MILLION
            mods.append(f"{read} cached and {write} cache-write input tokens at ${r_read} and ${r_write} "
                        "per million")
    lc = m["long_context"]
    if inp > lc["above_input_tokens"]:
        out_cost = Decimal(out) * p["output"]
        cost = (cost - out_cost) * Decimal(lc["input_and_cache"]) + out_cost * Decimal(lc["output"])
        mods.append(f"{inp} input tokens, above {lc['above_input_tokens']}: input and cache at "
                    f"{lc['input_and_cache']}x, output at {lc['output']}x")
    tier = summary.get("service_tier")
    factor = m["service_tier"].get(tier) if isinstance(tier, str) else None
    if factor is None:
        unknown.append(f"service_tier {tier!r}")
    elif Decimal(factor) != 1:
        cost *= Decimal(factor)
        mods.append(f"service_tier {tier} at {factor}x every token")
    return tokens, cost, mods, unknown


def charge(provider: str, model: str, status, raw: bytes, reserved=Decimal(0)):
    """(SETTLE, cost, tokens, detail) or (HOLD, None, None, reason) for one finished attempt
    whose reservation was ``reserved``."""
    if status is None:
        return HOLD, None, None, "no response arrived; the provider may still bill the request"
    if not 200 <= status < 300:
        return SETTLE, Decimal(0), dict.fromkeys(TOKEN_KINDS, 0), (
            f"HTTP {status} error response, settled at zero: Anthropic does not bill failed requests; "
            "OpenAI's behaviour is unverified and covered by reconciliation")
    s = anthropic_summary(raw) if provider == "anthropic" else openai_summary(raw)
    if not s["complete"]:
        return HOLD, None, None, "the stream did not complete, so its usage is unknown"
    if provider == "anthropic" and not s["usage_final"]:
        return HOLD, None, None, "the stream's message_delta did not report final usage"
    priced = priced_usage(provider, model, s)
    if priced is None:
        return HOLD, None, None, "the response did not report its usage in full"
    tokens, cost, mods, unknown = priced
    if unknown:
        return HOLD, None, None, "the response shows a price modifier with no documented price: " + "; ".join(unknown)
    if mods:
        return SETTLE, max(Decimal(reserved), cost), tokens, (
            "the larger of the reservation and the response usage under documented modifiers: " + "; ".join(mods))
    return SETTLE, cost, tokens, "response usage"


def attempt_log_root(entry: dict, base=None, run_roots=None) -> Path:
    """Where a reservation's attempt log is: ``log_root`` under the run's folder, found by
    the run's id in ``run_roots``, or else by ``run_root`` relative to ``base``, the ledger's
    folder. An absolute ``log_root`` is used as it is."""
    log_root = Path(entry["log_root"])
    if log_root.is_absolute():
        return log_root
    run_root = (run_roots or {}).get(entry.get("run"))
    if run_root is None:
        run_root = Path(entry.get("run_root", "."))
        if base is not None and not run_root.is_absolute():
            run_root = Path(base) / run_root
    return Path(run_root) / log_root


def recorded_attempt(entry: dict, base=None, run_roots=None):
    """(attempt record, raw bytes) for a reservation, or None if no attempt was recorded."""
    log = AttemptLog(attempt_log_root(entry, base, run_roots))
    records = log.attempts(entry["trace_key"], entry["config_sha256"], entry["judge"], entry["repeat"])
    if len(records) < entry["attempt"]:
        return None
    record = records[entry["attempt"] - 1]
    folder = log.slot_dir(entry["trace_key"], entry["config_sha256"], entry["judge"], entry["repeat"])
    raw = (folder / f"{entry['attempt']}.raw").read_bytes()
    if record["request_sha256"] != entry["request_sha256"] or hashlib.sha256(raw).hexdigest() != record["raw_sha256"]:
        raise LedgerError(f"reservation {entry['id']} does not match its attempt record")
    return record, raw


def next_attempt(log, trace_key: str, config_sha256: str, judge: str, slots) -> tuple:
    """(slot, attempt index) of the dispatcher's next send for ``judge``: its first slot,
    in the stage's order, that has not ended."""
    for j, repeat in slots:
        if j != judge:
            continue
        records = log.attempts(trace_key, config_sha256, j, repeat)
        if not D._slot_done(records):
            return (j, repeat), len(records) + 1
    raise LedgerError(f"no open {judge} slot for {trace_key}")


# ---------------------------------------------------------------- ledger

class Ledger:
    """The write-ahead ledger for one budget (see the module docstring)."""

    def __init__(self, path, budget: str, clock=time.time, mode=None, new_budget_note=None, replaces=()):
        if budget not in CAPS:
            raise ValueError(f"unknown budget {budget!r}; use one of {sorted(CAPS)}")
        self.path, self.budget, self.cap, self.clock = Path(path), budget, CAPS[budget], clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        created = not self.path.exists()
        self._f = open(self.path, "a+b")
        try:
            fcntl.flock(self._f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._f.close()
            raise LedgerError(f"{self.path} is held by another run; one run at a time") from None
        self.entries, self.reservations, self.adjustments, self._blocking = [], {}, [], None
        try:
            self._f.seek(0)
            data = self._f.read()
            if data and not data.endswith(b"\n"):
                raise LedgerError(f"{self.path} ends in a partial line; inspect it before any further call")
            for line in data.decode("utf-8").splitlines():
                self._apply(json.loads(line))
            if not self.entries:
                self._append({"type": "ledger", "budget": budget, "cap_usd": usd(self.cap),
                              "ledger_id": secrets.token_hex(8), "mode": mode, "price_table": PRICE_TABLE_VERSION,
                              "pricing_sha256": pricing_sha256(), "pricing": pricing_record(),
                              "new_budget_note": new_budget_note, "replaces": list(replaces),
                              "created_at": _now(clock)})
                if created:
                    fd = os.open(self.path.parent, os.O_RDONLY)
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)
            head = self.entries[0]
            if (head.get("type"), head.get("budget"), head.get("cap_usd"), head.get("pricing_sha256")) != (
                    "ledger", budget, usd(self.cap), pricing_sha256()):
                raise LedgerError(f"{self.path} is not a {budget} ledger under cap ${usd(self.cap)}, price table "
                                  f"{PRICE_TABLE_VERSION} and these reservation constants")
            if mode is not None and head.get("mode") != mode:
                raise LedgerError(f"{self.path} is a {head.get('mode')} ledger, not a {mode} one")
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def header(self) -> dict:
        return self.entries[0]

    @property
    def ledger_id(self) -> str:
        return self.header.get("ledger_id")

    @property
    def created_at(self) -> str:
        return self.header["created_at"]

    @property
    def mode(self):
        return self.header.get("mode")

    # -------------------------------------------------------------- entries

    def _apply(self, e: dict) -> None:
        if e.get("seq") != len(self.entries):
            raise LedgerError(f"{self.path}: entry {len(self.entries)} has sequence {e.get('seq')}")
        t = e.get("type")
        if (t == "ledger") != (not self.entries):
            raise LedgerError(f"{self.path}: the header must be the first entry and only there")
        if t == "reserve":
            if e["id"] in self.reservations or self.open_reservations():
                raise LedgerError(f"{self.path}: reservation {e['id']} is a repeat or a second open one")
            self.reservations[e["id"]] = dict(e, state="open", cost=None)
        elif t in ("settle", "hold"):
            r = self.reservations.get(e["id"])
            if r is None or r["state"] != "open":
                raise LedgerError(f"{self.path}: {t} for {e['id']}, which is not open")
            r.update(state="settled" if t == "settle" else "held",
                     cost=Decimal(e["cost_usd"]) if t == "settle" else None, closed_at=e.get("at"))
        elif t == "reconcile":
            if not e["ok"]:
                self._blocking = e                # a later passing one does not clear it; a note does
        elif t in ("adjust", "correction"):
            if self._blocking is None or e["reconcile_seq"] != self._blocking["seq"]:
                raise LedgerError(f"{self.path}: an {t} entry must answer the failed reconciliation")
            if t == "adjust":
                self.adjustments.append(e)
            else:
                self._blocking = None
        elif t not in ("ledger", "refuse"):
            raise LedgerError(f"{self.path}: unknown entry type {t!r}")
        self.entries.append(e)

    def _append(self, entry: dict) -> dict:
        entry = dict(entry, seq=len(self.entries))
        line = (json.dumps(entry, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        self._f.write(line)
        self._f.flush()
        os.fsync(self._f.fileno())
        self._apply(entry)
        return entry

    # -------------------------------------------------------------- totals

    def open_reservations(self) -> list:
        return [r for r in self.reservations.values() if r["state"] == "open"]

    @staticmethod
    def spent(r: dict) -> Decimal:
        """What one reservation counts as: its cost once settled, its full amount otherwise."""
        return r["cost"] if r["state"] == "settled" else Decimal(r["amount_usd"])

    def adjusted(self, provider=None) -> Decimal:
        return sum((Decimal(e["cost_usd"]) for e in self.adjustments if provider in (None, e["provider"])),
                   Decimal(0))

    def totals(self, rows=None) -> dict:
        """Settled, held, open and committed spend; for the whole budget, settled includes
        the adjustments recorded after failed reconciliations."""
        selected = rows is not None
        rows = list(self.reservations.values()) if rows is None else rows
        out = {s: sum((self.spent(r) for r in rows if r["state"] == s), Decimal(0))
               for s in ("settled", "held", "open")}
        if not selected:
            out["settled"] += self.adjusted()
        out["committed"] = out["settled"] + out["held"] + out["open"]
        return out

    # -------------------------------------------------------------- calls

    def refuse(self, fields: dict, amount: Decimal, kind: str, message: str, **extra):
        """Write a refusal and raise ``SpendHalt``; nothing is sent."""
        entry = dict(fields, amount_usd=usd(amount), committed_before_usd=usd(self.totals()["committed"]),
                     cap_usd=usd(self.cap), at=_now(self.clock))
        refused = self._append(dict(entry, type="refuse", reason=kind, **extra))
        raise SpendHalt(kind, message, refused)

    def reconcile_due(self):
        """The oldest settled or held call that no passing reconciliation follows, when it is
        more than a day old; otherwise None."""
        last_ok = max((e["seq"] for e in self.entries if e["type"] == "reconcile" and e["ok"]), default=-1)
        for e in self.entries:
            if (e["type"] == "reserve" and e["seq"] > last_ok
                    and self.reservations[e["id"]]["state"] in ("settled", "held")):
                return e if self.clock() - timestamp(e["at"]) > DAILY_RECONCILE_S else None
        return None

    def reserve(self, fields: dict, amount: Decimal) -> str:
        """Admit and record one reservation before its call, or refuse it (``SpendHalt``)."""
        if self.open_reservations():
            raise ConcurrentCall("a reservation is still open; calls go out one at a time")
        if self._blocking is not None:
            self.refuse(fields, amount, "reconciliation", "a failed reconciliation has no correction note yet",
                        reconcile_seq=self._blocking["seq"])
        due = self.reconcile_due() if self.budget == "evaluation" else None
        if due is not None:
            self.refuse(fields, amount, "reconcile_due",
                        f"the call reserved at {due['at']} is more than a day old and no passing reconciliation "
                        "follows it; reconcile the evaluation ledger, record a note, and resume",
                        oldest_unreconciled=due["id"])
        if fields.get("provider") == "openai" and (fields.get("input_tokens") or 0) > B_INPUT_LIMIT:
            self.refuse(fields, amount, "long_context",
                        f"the B request's input count {fields['input_tokens']} is above {B_INPUT_LIMIT} tokens, "
                        "which the provider prices at a long-context rate")
        t = self.totals()
        if not t["committed"] + amount <= self.cap:
            self.refuse(fields, amount, "spend_cap", f"the reservation of ${usd(amount)} on ${usd(t['committed'])} "
                        f"already committed would pass the {self.budget} cap of ${usd(self.cap)}")
        rid = f"r{len(self.reservations) + 1:06d}"
        self._append(dict(fields, amount_usd=usd(amount), committed_before_usd=usd(t["committed"]),
                          cap_usd=usd(self.cap), at=_now(self.clock), type="reserve", id=rid))
        return rid

    def close_reservation(self, rid: str, found) -> dict:
        """Settle or hold an open reservation from its attempt record (or its absence)."""
        r = self.reservations[rid]
        if found is None:
            return self._append({"type": "hold", "id": rid, "reason": "no attempt was recorded for it",
                                 "at": _now(self.clock)})
        record, raw = found
        kind, cost, tokens, detail = charge(r["provider"], r["model"], record["http_status"], raw,
                                            reserved=Decimal(r["amount_usd"]))
        if kind == HOLD:
            return self._append({"type": "hold", "id": rid, "reason": detail, "at": _now(self.clock)})
        return self._append({"type": "settle", "id": rid, "cost_usd": usd(cost), "tokens": tokens,
                             "basis": detail, "price_table": PRICE_TABLE_VERSION,
                             "over_reservation": cost > Decimal(r["amount_usd"]), "at": _now(self.clock)})

    def recover(self, find=None, run_roots=None) -> list:
        """Close every reservation left open by an earlier process (see the module docstring)."""
        find = find or (lambda entry: recorded_attempt(entry, self.path.parent, run_roots))
        closed = []
        for r in self.open_reservations():
            found = find(r)
            if found is None:
                closed.append(self._append({"type": "hold", "id": r["id"], "at": _now(self.clock),
                                            "reason": "open at restart with no recorded attempt"}))
            else:
                closed.append(self.close_reservation(r["id"], found))
        return closed

    # -------------------------------------------------------------- reconciliation

    def last_closed_at(self, provider: str):
        """When the provider's last reservation was settled or held, or None."""
        times = [r["closed_at"] for r in self.reservations.values()
                 if r["provider"] == provider and r["state"] in ("settled", "held") and r.get("closed_at")]
        return max(times, key=timestamp) if times else None

    def reconcile(self, rows: list, export_sha256: str) -> dict:
        """Compare a usage export with the ledger; a failure blocks later calls. An export
        that does not cover the ledger's whole life is refused (``ValueError``)."""
        if self.open_reservations():
            raise LedgerError("recover the open reservation before reconciling")
        coverage = export_coverage(rows)
        for p, c in sorted(coverage.items()):
            if timestamp(c["period_start"]) > timestamp(self.created_at):
                raise ValueError(f"the {p} export starts at {c['period_start']}, after the ledger was created at "
                                 f"{self.created_at}; an export is cumulative from the ledger's creation")
            last = self.last_closed_at(p)
            if last is not None and timestamp(c["period_end"]) < timestamp(last):
                raise ValueError(f"the {p} export ends at {c['period_end']}, before the ledger's last {p} call "
                                 f"was settled or held at {last}")
        by_provider = {}
        for row in rows:
            by_provider.setdefault(row["provider"], Decimal(0))
            by_provider[row["provider"]] += export_usd(row)
        ledger_by = {}
        for r in self.reservations.values():
            ledger_by[r["provider"]] = ledger_by.get(r["provider"], Decimal(0)) + self.spent(r)
        for e in self.adjustments:
            ledger_by[e["provider"]] = ledger_by.get(e["provider"], Decimal(0)) + Decimal(e["cost_usd"])
        missing = sorted(p for p, v in ledger_by.items() if v > 0 and p not in by_provider)
        if missing:
            raise ValueError(f"the export has no rows for {missing}; export both providers' usage")
        providers, ok = {}, True
        for p in sorted(set(by_provider) | set(ledger_by)):
            exported, ledgered = by_provider.get(p, Decimal(0)), ledger_by.get(p, Decimal(0))
            within = exported <= ledgered * (1 + RECONCILE_TOLERANCE)
            ok &= within
            providers[p] = {"export_usd": usd(exported), "ledger_usd": usd(ledgered), "within": within,
                            "excess_share": (f"{(exported - ledgered) / ledgered:.6f}" if ledgered else None)}
        return self._append({"type": "reconcile", "ok": ok, "export_sha256": export_sha256,
                             "export_rows": len(rows), "coverage": coverage, "tolerance": usd(RECONCILE_TOLERANCE),
                             "providers": providers, "at": _now(self.clock)})

    def record_correction(self, note: str) -> dict:
        """Answer the failed reconciliation: first a settled adjustment per provider of the
        export's excess over the ledger in it, then the note, which clears the block."""
        if self._blocking is None:
            raise ValueError("no failed reconciliation needs a correction")
        if not (isinstance(note, str) and note.strip()):
            raise ValueError("a correction note is required")
        seq = self._blocking["seq"]
        for p, v in sorted(self._blocking["providers"].items()):
            excess = Decimal(v["export_usd"]) - Decimal(v["ledger_usd"])
            done = sum((Decimal(e["cost_usd"]) for e in self.adjustments
                        if e["reconcile_seq"] == seq and e["provider"] == p), Decimal(0))
            if excess > done:
                self._append({"type": "adjust", "provider": p, "cost_usd": usd(excess - done), "reconcile_seq": seq,
                              "basis": "the provider's usage above the ledger in the failed reconciliation",
                              "at": _now(self.clock)})
        return self._append({"type": "correction", "reconcile_seq": seq, "note": note, "at": _now(self.clock)})

    def blocking_reconciliation(self):
        return self._blocking

    def reconciled_after_last_call(self, run=None, stage=None) -> bool:
        """Whether a passing reconciliation follows the last reservation, of one run's stage
        when they are given, and no failed one is waiting for a note."""
        last_call = max((e["seq"] for e in self.entries if e["type"] == "reserve"
                         and (run is None or e["run"] == run) and (stage is None or e["stage"] == stage)),
                        default=None)
        last_ok = max((e["seq"] for e in self.entries if e["type"] == "reconcile" and e["ok"]), default=None)
        return self._blocking is None and (last_call is None or (last_ok is not None and last_ok > last_call))

    def unreconciled_calls(self) -> list:
        """Reservations made after the last passing reconciliation."""
        last_ok = max((e["seq"] for e in self.entries if e["type"] == "reconcile" and e["ok"]), default=-1)
        return [e for e in self.entries if e["type"] == "reserve" and e["seq"] > last_ok]

    # -------------------------------------------------------------- reports

    def summary(self, run=None, stage=None) -> dict:
        rows = [r for r in self.reservations.values()
                if (run is None or r["run"] == run) and (stage is None or r["stage"] == stage)]
        t, ta = self.totals(rows), self.totals()
        refusals = [e for e in self.entries if e["type"] == "refuse"
                    and (run is None or e.get("run") == run) and (stage is None or e.get("stage") == stage)]
        ids = {r["id"] for r in rows}
        over = [e for e in self.entries if e["type"] == "settle" and e["over_reservation"] and e["id"] in ids]
        amounts = [Decimal(r["amount_usd"]) for r in rows]
        return {"budget": self.budget, "cap_usd": usd(self.cap),
                "ledger": {"file": self.path.name, "id": self.ledger_id, "last_seq": len(self.entries) - 1},
                "price_table": PRICE_TABLE_VERSION, "budget_committed_usd": usd(ta["committed"]),
                "budget_settled_usd": usd(ta["settled"]), "budget_held_usd": usd(ta["held"]),
                "budget_open_usd": usd(ta["open"]), "budget_adjusted_usd": usd(self.adjusted()),
                "headroom_usd": usd(self.cap - ta["committed"]),
                "selected_spent_usd": usd(t["committed"]), "selected_settled_usd": usd(t["settled"]),
                "selected_held_usd": usd(t["held"]),
                "reservations": len(rows), "settled": sum(r["state"] == "settled" for r in rows),
                "held": sum(r["state"] == "held" for r in rows), "open": sum(r["state"] == "open" for r in rows),
                "reserved_usd": usd(sum(amounts, Decimal(0))),
                "largest_reservation_usd": usd(max(amounts)) if amounts else None,
                "refusals": [{"reason": e["reason"], "trace_key": e.get("trace_key"), "slot": e.get("slot"),
                              "amount_usd": e["amount_usd"], "committed_before_usd": e["committed_before_usd"]}
                             for e in refusals],
                "over_reservation": [{"id": e["id"], "cost_usd": e["cost_usd"],
                                      "amount_usd": self.reservations[e["id"]]["amount_usd"]} for e in over]}


def ledger_name(budget: str) -> str:
    return f"ledger-{budget}.jsonl"


def _note_given(note) -> bool:
    return isinstance(note, str) and bool(note.strip())


def _register(folder: Path, led: Ledger) -> None:
    line = json.dumps({"budget": led.budget, "ledger_id": led.ledger_id, "file": led.path.name, "mode": led.mode,
                       "created_at": led.created_at, "new_budget_note": led.header.get("new_budget_note"),
                       "replaces": led.header.get("replaces", [])}, sort_keys=True) + "\n"
    with open(folder / REGISTER_NAME, "ab") as f:
        f.write(line.encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())


def open_budget_ledger(folder, budget: str, mode: str, *, create=False, new_budget_note=None,
                       clock=time.time) -> Ledger:
    """The budget's one ledger, ``folder``/ledger-<budget>.jsonl, shared by every run.

    ``budgets.jsonl`` in ``folder`` lists every ledger created there. A ledger is created
    only with ``create``. Once a ledger for the budget has been created, another one (after
    the first was moved aside or deleted) needs ``new_budget_note``, which the new ledger's
    header keeps; otherwise a fresh cap would quietly replace the spend already committed.
    """
    folder = Path(folder)
    path = folder / ledger_name(budget)
    register = folder / REGISTER_NAME
    entries = ([json.loads(line) for line in register.read_text(encoding="utf-8").splitlines() if line.strip()]
               if register.is_file() else [])
    earlier = [e for e in entries if e.get("budget") == budget]
    if path.exists():
        if _note_given(new_budget_note):
            raise LedgerError(f"{path} exists; a new-budget note starts a new {budget} ledger only after this one "
                              "is moved aside")
        led = Ledger(path, budget, clock, mode=mode)
        if led.ledger_id not in {e.get("ledger_id") for e in earlier}:
            if earlier and not _note_given(led.header.get("new_budget_note")):
                led.close()
                raise LedgerError(f"{path} is not the {budget} ledger registered in {register}")
            _register(folder, led)            # created, then stopped before its registration
        return led
    if not create:
        raise LedgerError(f"no {budget} ledger at {path}")
    others = sorted(p.name for p in folder.glob(f"ledger-{budget}*")) if folder.is_dir() else []
    if (earlier or others) and not _note_given(new_budget_note):
        before = ", ".join([f"{e['ledger_id']} created {e['created_at']}" for e in earlier] + others)
        raise LedgerError(f"a {budget} ledger was created here before ({before}) and is not at {path}; restore it, "
                          f"or give a new-budget note to start a new {budget} budget")
    folder.mkdir(parents=True, exist_ok=True)
    led = Ledger(path, budget, clock, mode=mode, new_budget_note=new_budget_note if _note_given(new_budget_note)
                 else None, replaces=[e["ledger_id"] for e in earlier])
    _register(folder, led)
    return led


# ---------------------------------------------------------------- metered transport

class MeteredTransport:
    """A transport wrapper: each send reserves on the ledger first, one call at a time.

    ``begin`` names the trajectory and the input count each judge's request is priced at.
    Before a send, the previous reservation is closed from the attempt record the
    dispatcher wrote for it; ``finish`` closes the last one after the trajectory. A slot
    folder holding a raw response without its attempt record (a stop between the two
    writes) is refused before anything is reserved or sent.
    """

    def __init__(self, inner, ledger: Ledger, run: str, stage: str, log_root, run_root=None):
        self.inner, self.ledger, self.run, self.stage = inner, ledger, run, stage
        led_dir = ledger.path.parent.resolve()
        if run_root is None:
            self.run_root, self.run_root_rel = None, "."
            self.log_root = os.path.relpath(Path(log_root).resolve(), led_dir)
        else:
            self.run_root = Path(run_root).resolve()
            self.run_root_rel = os.path.relpath(self.run_root, led_dir)
            self.log_root = os.path.relpath(Path(log_root).resolve(), self.run_root)
        self._busy = threading.Lock()
        self._open = None
        self.ctx = None

    def begin(self, trace_key: str, config_sha256: str, log, slots, inputs: dict) -> None:
        self.finish()
        self.ctx = {"trace_key": trace_key, "config_sha256": config_sha256, "log": log,
                    "slots": tuple(slots), "inputs": inputs}

    def _close_open(self) -> None:
        if self._open is not None:
            rid, self._open = self._open, None
            roots = {self.run: self.run_root} if self.run_root is not None else None
            self.ledger.close_reservation(rid, recorded_attempt(self.ledger.reservations[rid],
                                                                self.ledger.path.parent, roots))

    def finish(self) -> None:
        if not self._busy.acquire(blocking=False):
            raise ConcurrentCall("a call is in flight")
        try:
            self._close_open()
        finally:
            self._busy.release()

    def send(self, request):
        if not self._busy.acquire(blocking=False):
            raise ConcurrentCall("a call is already in flight; calls go out one at a time")
        try:
            self._close_open()
            c = self.ctx
            basis = c["inputs"][request.judge]
            if request.sha256 != basis["request_sha256"]:
                raise ValueError(f"the {request.judge} request is not the one this trajectory was priced for")
            (judge, repeat), attempt = next_attempt(c["log"], c["trace_key"], c["config_sha256"],
                                                    request.judge, c["slots"])
            amount = reservation(request.model, basis["input_tokens"], basis["max_output_tokens"])
            fields = {"run": self.run, "stage": self.stage, "trace_key": c["trace_key"],
                      "config_sha256": c["config_sha256"], "judge": judge, "repeat": repeat,
                      "slot": f"{judge}{repeat}", "attempt": attempt, "provider": request.provider,
                      "model": request.model, "request_sha256": request.sha256,
                      "input_tokens": basis["input_tokens"], "input_source": basis["input_source"],
                      "max_output_tokens": basis["max_output_tokens"], "run_root": self.run_root_rel,
                      "log_root": self.log_root}
            folder = c["log"].slot_dir(c["trace_key"], c["config_sha256"], judge, repeat)
            orphans = (sorted(p.name for p in folder.glob("*.raw") if not p.with_suffix(".json").exists())
                       if folder.is_dir() else [])
            if orphans:
                self.ledger.refuse(fields, amount, "orphan_attempt",
                                   f"{folder} holds {', '.join(orphans)} without its attempt record; inspect it, "
                                   "nothing sent")
            self._open = self.ledger.reserve(fields, amount)
            return self.inner.send(request)
        finally:
            self._busy.release()


# ---------------------------------------------------------------- usage export

def _coverage_value(path, i, field, value):
    if not (isinstance(value, str) and value.strip()):
        raise ValueError(f"{path}: row {i}: {field} must be stated")
    if field != "scope":
        timestamp(value)
    return value.strip()


def export_coverage(rows: list) -> dict:
    """{provider: {"period_start", "period_end", "scope"}} from export rows; every row of one
    provider must state the same period and scope (``ValueError`` otherwise)."""
    coverage = {}
    for i, row in enumerate(rows):
        c = {k: _coverage_value("export", i, k, row.get(k)) for k in COVERAGE_FIELDS}
        if timestamp(c["period_start"]) > timestamp(c["period_end"]):
            raise ValueError(f"export row {i}: the period ends before it starts")
        if coverage.setdefault(row["provider"], c) != c:
            raise ValueError(f"export row {i}: every {row['provider']} row must state the same period and scope")
    return coverage


def read_usage_export(path):
    """(rows, SHA-256) from a usage export in the runner's own format: a CSV with the header
    ``EXPORT_FIELDS``, or JSON, either a list of objects with those keys or {"rows": [...]}.

    Token columns are disjoint counts: ``input_tokens`` excludes cache writes and reads.
    ``cost_usd`` may be empty; when given, the larger of it and the priced tokens counts.
    ``period_start`` and ``period_end`` (ISO 8601 with a time zone) and ``scope`` (the
    Anthropic workspace or OpenAI project) state what the export covers.
    """
    path = Path(path)
    data = path.read_bytes()
    if path.suffix == ".csv":
        reader = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""))
        if tuple(reader.fieldnames or ()) != EXPORT_FIELDS:
            raise ValueError(f"{path}: the header must be {','.join(EXPORT_FIELDS)}")
        raw_rows = list(reader)
    elif path.suffix == ".json":
        obj = json.loads(data)
        raw_rows = obj.get("rows") if isinstance(obj, dict) else obj
        if not isinstance(raw_rows, list):
            raise ValueError(f"{path}: expected a list of rows")
    else:
        raise ValueError(f"{path}: a usage export is a .csv or .json file")
    rows = []
    for i, row in enumerate(raw_rows):
        if not isinstance(row, dict) or set(row) != set(EXPORT_FIELDS):
            raise ValueError(f"{path}: row {i} must have exactly the fields {EXPORT_FIELDS}")
        model = row["model"]
        if model not in PRICES or PRICES[model]["provider"] != row["provider"]:
            raise ValueError(f"{path}: row {i}: {row['provider']}/{model} is not in the price table")
        tokens = {}
        for kind in TOKEN_KINDS:
            v = row[f"{kind}_tokens"]
            v = int(v) if isinstance(v, str) and v.strip().isdigit() else v
            if not (isinstance(v, int) and not isinstance(v, bool) and v >= 0):
                raise ValueError(f"{path}: row {i}: {kind}_tokens must be a whole number")
            tokens[kind] = v
        cost = row["cost_usd"]
        cost = None if cost in (None, "") else Decimal(str(cost))
        if cost is not None and not (cost.is_finite() and cost >= 0):
            raise ValueError(f"{path}: row {i}: cost_usd must be a non-negative amount")
        rows.append(dict({"provider": row["provider"], "model": model, "tokens": tokens, "cost_usd": cost},
                         **{k: _coverage_value(path, i, k, row[k]) for k in COVERAGE_FIELDS}))
    export_coverage(rows)
    return rows, hashlib.sha256(data).hexdigest()


def export_usd(row: dict) -> Decimal:
    priced = call_cost(row["model"], row["tokens"])
    return priced if row["cost_usd"] is None else max(priced, row["cost_usd"])


def write_usage_export(path, rows) -> None:
    """Write rows of {"provider", "model", "tokens", "cost_usd", "period_start",
    "period_end", "scope"} as an export CSV, replacing the file in one step."""
    path = Path(path)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(EXPORT_FIELDS)
    for row in rows:
        w.writerow([row["provider"], row["model"]] + [row["tokens"][k] for k in TOKEN_KINDS]
                   + ["" if row.get("cost_usd") is None else usd(row["cost_usd"])]
                   + [row[k] for k in COVERAGE_FIELDS])
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(buf.getvalue(), encoding="utf-8")
    os.replace(tmp, path)
