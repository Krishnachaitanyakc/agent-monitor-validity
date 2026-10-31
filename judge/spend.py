"""Spending controls: prices, reservations, the cap, a write-ahead ledger and reconciliation.

**Prices.** One versioned table, ``PRICES``, in dollars per million tokens, from the setup
note's section 4.1: ``claude-sonnet-5`` $2 input, $2.50 cache write (five minutes), $0.20
cache read and $10 output; ``gpt-6-sol`` $2 input and $10 output, with no caching, so any
cached token it reports is priced as input. A model outside the table cannot be
reserved. Every amount is an exact decimal.

**Reservation.** Before each call, r = (I x 1.02 + 256) x p_in x 1.25 + max_output x p_out.
I is the provider's pre-send token count when one is supplied, and otherwise
``ESTIMATE_FACTOR`` (2) times the request layer's estimate, whose text rate and B image
count are unmeasured. The 1.25 prices every input token as a cache write, so a repeat that
misses the cache stays inside its reservation. max_output is the request's own output
cap, 32,000 tokens.

**Admission.** A call is admitted only if settled spend plus open and held reservations
plus r stays at or below the budget's cap: $95 for the pilot stages together, $285 for
the evaluation. Otherwise a refusal is written and ``SpendHalt`` is raised before
anything is sent.

**Ledger.** One append-only JSONL file per budget; each line is flushed and fsynced
before the next step, so a reservation is on disk before its request leaves. One process
holds the file (an exclusive lock) and at most one reservation is open at a time, so
calls go out one at a time. A reservation closes from its attempt record in the attempt
log, which keeps the raw response:

- a complete response settles at the cost of the usage it reports (input, cache write,
  cache read and output tokens);
- a non-2xx error response settles at zero, since providers do not bill errors;
- no response, a stream that did not complete, or usage not reported in full holds the
  whole reservation, which then counts as spent for good;
- no attempt record at all also holds it.

After a restart, an open reservation counts as spent until ``recover`` matches it to its
attempt record in the same way.

**Reconciliation.** ``read_usage_export`` reads a provider usage export (``EXPORT_FIELDS``,
as CSV or JSON). ``Ledger.reconcile`` prices it with the same table and compares each
provider's total with the ledger's spend on that provider. Usage above the ledger by
more than 1% fails, and a failed reconciliation refuses every later reservation until
``record_correction`` appends a note.
"""

import csv
import fcntl
import hashlib
import io
import json
import os
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from . import dispatch as D
from .storage import AttemptLog
from .streams import anthropic_summary, openai_summary

PRICE_TABLE_VERSION = "prices-2026-10-28"
PRICES = {
    "claude-sonnet-5": {"provider": "anthropic", "input": "2", "cache_write": "2.50", "cache_read": "0.20",
                        "output": "10"},
    "gpt-6-sol": {"provider": "openai", "input": "2", "cache_write": None, "cache_read": None, "output": "10"},
}
PRICE_TABLE_SHA256 = hashlib.sha256(json.dumps({"version": PRICE_TABLE_VERSION, "prices": PRICES},
                                               sort_keys=True).encode("utf-8")).hexdigest()
TOKEN_KINDS = ("input", "cache_write", "cache_read", "output")
MILLION = Decimal(1000000)

INPUT_MARGIN = Decimal("1.02")
INPUT_PAD_TOKENS = 256
WRITE_FACTOR = Decimal("1.25")
ESTIMATE_FACTOR = 2
CAPS = {"pilot": Decimal("95"), "evaluation": Decimal("285")}
RECONCILE_TOLERANCE = Decimal("0.01")
RESERVATION_RULE = "r = (I x 1.02 + 256) x p_in x 1.25 + max_output x p_out"

SETTLE, HOLD = "settle", "hold"
EXPORT_FIELDS = ("provider", "model", "input_tokens", "cache_write_tokens", "cache_read_tokens",
                 "output_tokens", "cost_usd")


class LedgerError(RuntimeError):
    """The ledger file is unusable, held by another run, or inconsistent."""


class ConcurrentCall(RuntimeError):
    """A second call was started while one was in flight."""


class SpendHalt(RuntimeError):
    """A call was refused before sending: the cap, or a failed reconciliation."""

    def __init__(self, kind: str, message: str, entry: dict):
        super().__init__(message)
        self.kind, self.entry = kind, entry


def usd(amount: Decimal) -> str:
    return format(amount, "f")


def _now(clock) -> str:
    return datetime.fromtimestamp(clock(), timezone.utc).isoformat(timespec="milliseconds")


# ---------------------------------------------------------------- prices and reservations

def price(model: str) -> dict:
    """Dollars per token for ``model``: {kind: Decimal or None}."""
    if model not in PRICES:
        raise ValueError(f"{model!r} is not in price table {PRICE_TABLE_VERSION}; it cannot be reserved")
    row = PRICES[model]
    return {k: (Decimal(row[k]) / MILLION if row[k] is not None else None) for k in TOKEN_KINDS}


def call_cost(model: str, tokens: dict) -> Decimal:
    """The cost of one call's usage; a kind with no price (B's cache) is priced as input."""
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
    """The reservation's input count when no provider count is supplied."""
    return ESTIMATE_FACTOR * request_estimate


def max_output_tokens(judge: str, body: dict) -> int:
    """The output cap a request body sets."""
    return body["max_tokens"] if judge == "A" else body["max_output_tokens"]


# ---------------------------------------------------------------- what one attempt cost

def _int(v):
    return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else None


def billable_tokens(provider: str, usage: dict):
    """{kind: tokens} from a response's normalized usage, or None when it is incomplete."""
    if provider == "anthropic":
        tokens = {"input": _int(usage.get("input_tokens")), "cache_write": _int(usage.get("cache_write_tokens")),
                  "cache_read": _int(usage.get("cache_read_tokens")), "output": _int(usage.get("output_tokens"))}
        one_hour = usage.get("cache_write_1h_tokens")
        if None in tokens.values() or one_hour or (tokens["cache_write"] and one_hour is None):
            return None                       # one-hour writes have no price in the table
        return tokens
    inp, out = _int(usage.get("input_tokens")), _int(usage.get("output_tokens"))
    if inp is None or out is None:
        return None
    return {"input": inp, "cache_write": 0, "cache_read": 0, "output": out}   # input includes any cached


def charge(provider: str, model: str, status, raw: bytes):
    """(SETTLE, cost, tokens, detail) or (HOLD, None, None, reason) for one finished attempt."""
    if status is None:
        return HOLD, None, None, "no response arrived; the provider may still bill the request"
    if not 200 <= status < 300:
        return SETTLE, Decimal(0), dict.fromkeys(TOKEN_KINDS, 0), f"HTTP {status} error response, not billed"
    s = anthropic_summary(raw) if provider == "anthropic" else openai_summary(raw)
    if not s["complete"]:
        return HOLD, None, None, "the stream did not complete, so its usage is unknown"
    tokens = billable_tokens(provider, s["usage"])
    if tokens is None:
        return HOLD, None, None, "the response did not report its usage in full"
    return SETTLE, call_cost(model, tokens), tokens, "response usage"


def recorded_attempt(entry: dict):
    """(attempt record, raw bytes) for a reservation, or None if no attempt was recorded."""
    log = AttemptLog(entry["log_root"])
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

    def __init__(self, path, budget: str, clock=time.time):
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
        self.entries, self.reservations, self._blocking = [], {}, None
        try:
            self._f.seek(0)
            data = self._f.read()
            if data and not data.endswith(b"\n"):
                raise LedgerError(f"{self.path} ends in a partial line; inspect it before any further call")
            for line in data.decode("utf-8").splitlines():
                self._apply(json.loads(line))
            if not self.entries:
                self._append({"type": "ledger", "budget": budget, "cap_usd": usd(self.cap),
                              "price_table": PRICE_TABLE_VERSION, "price_table_sha256": PRICE_TABLE_SHA256,
                              "prices_per_million_tokens": PRICES, "reservation_rule": RESERVATION_RULE,
                              "estimate_factor": ESTIMATE_FACTOR, "created_at": _now(clock)})
                if created:
                    fd = os.open(self.path.parent, os.O_RDONLY)
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)
            head = self.entries[0]
            if (head.get("type"), head.get("budget"), head.get("cap_usd"), head.get("price_table_sha256")) != (
                    "ledger", budget, usd(self.cap), PRICE_TABLE_SHA256):
                raise LedgerError(f"{self.path} is not a {budget} ledger under cap ${usd(self.cap)} and "
                                  f"price table {PRICE_TABLE_VERSION}")
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
                     cost=Decimal(e["cost_usd"]) if t == "settle" else None)
        elif t == "reconcile":
            if not e["ok"]:
                self._blocking = e                # a later passing one does not clear it; a note does
        elif t == "correction":
            if self._blocking is None or e["reconcile_seq"] != self._blocking["seq"]:
                raise LedgerError(f"{self.path}: a correction must answer the failed reconciliation")
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

    def totals(self, rows=None) -> dict:
        rows = list(self.reservations.values()) if rows is None else rows
        out = {s: sum((self.spent(r) for r in rows if r["state"] == s), Decimal(0))
               for s in ("settled", "held", "open")}
        out["committed"] = out["settled"] + out["held"] + out["open"]
        return out

    # -------------------------------------------------------------- calls

    def reserve(self, fields: dict, amount: Decimal) -> str:
        """Admit and record one reservation before its call, or refuse it (``SpendHalt``)."""
        if self.open_reservations():
            raise ConcurrentCall("a reservation is still open; calls go out one at a time")
        t = self.totals()
        entry = dict(fields, amount_usd=usd(amount), committed_before_usd=usd(t["committed"]),
                     cap_usd=usd(self.cap), at=_now(self.clock))
        if self._blocking is not None:
            refused = self._append(dict(entry, type="refuse", reason="reconciliation",
                                        reconcile_seq=self._blocking["seq"]))
            raise SpendHalt("reconciliation", "a failed reconciliation has no correction note yet", refused)
        if not t["committed"] + amount <= self.cap:
            refused = self._append(dict(entry, type="refuse", reason="spend_cap"))
            raise SpendHalt("spend_cap", f"the reservation of ${usd(amount)} on ${usd(t['committed'])} already "
                            f"committed would pass the {self.budget} cap of ${usd(self.cap)}", refused)
        rid = f"r{len(self.reservations) + 1:06d}"
        self._append(dict(entry, type="reserve", id=rid))
        return rid

    def close_reservation(self, rid: str, found) -> dict:
        """Settle or hold an open reservation from its attempt record (or its absence)."""
        r = self.reservations[rid]
        if found is None:
            return self._append({"type": "hold", "id": rid, "reason": "no attempt was recorded for it",
                                 "at": _now(self.clock)})
        record, raw = found
        kind, cost, tokens, detail = charge(r["provider"], r["model"], record["http_status"], raw)
        if kind == HOLD:
            return self._append({"type": "hold", "id": rid, "reason": detail, "at": _now(self.clock)})
        return self._append({"type": "settle", "id": rid, "cost_usd": usd(cost), "tokens": tokens,
                             "basis": detail, "price_table": PRICE_TABLE_VERSION,
                             "over_reservation": cost > Decimal(r["amount_usd"]), "at": _now(self.clock)})

    def recover(self, find=recorded_attempt) -> list:
        """Close every reservation left open by an earlier process (see the module docstring)."""
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

    def reconcile(self, rows: list, export_sha256: str) -> dict:
        """Compare a provider usage export with the ledger; a failure blocks later calls."""
        if self.open_reservations():
            raise LedgerError("recover the open reservation before reconciling")
        by_provider = {}
        for row in rows:
            by_provider.setdefault(row["provider"], Decimal(0))
            by_provider[row["provider"]] += export_usd(row)
        ledger_by = {}
        for r in self.reservations.values():
            ledger_by[r["provider"]] = ledger_by.get(r["provider"], Decimal(0)) + self.spent(r)
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
                             "export_rows": len(rows), "tolerance": usd(RECONCILE_TOLERANCE),
                             "providers": providers, "at": _now(self.clock)})

    def record_correction(self, note: str) -> dict:
        if self._blocking is None:
            raise ValueError("no failed reconciliation needs a correction")
        if not (isinstance(note, str) and note.strip()):
            raise ValueError("a correction note is required")
        return self._append({"type": "correction", "reconcile_seq": self._blocking["seq"], "note": note,
                             "at": _now(self.clock)})

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
        return {"budget": self.budget, "cap_usd": usd(self.cap), "ledger": str(self.path),
                "price_table": PRICE_TABLE_VERSION, "budget_committed_usd": usd(ta["committed"]),
                "budget_settled_usd": usd(ta["settled"]), "budget_held_usd": usd(ta["held"]),
                "budget_open_usd": usd(ta["open"]), "headroom_usd": usd(self.cap - ta["committed"]),
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


# ---------------------------------------------------------------- metered transport

class MeteredTransport:
    """A transport wrapper: each send reserves on the ledger first, one call at a time.

    ``begin`` names the trajectory and the input count each judge's request is priced at.
    Before a send, the previous reservation is closed from the attempt record the
    dispatcher wrote for it; ``finish`` closes the last one after the trajectory.
    """

    def __init__(self, inner, ledger: Ledger, run: str, stage: str, log_root):
        self.inner, self.ledger, self.run, self.stage, self.log_root = inner, ledger, run, stage, str(log_root)
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
            self.ledger.close_reservation(rid, recorded_attempt(self.ledger.reservations[rid]))

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
                      "max_output_tokens": basis["max_output_tokens"], "log_root": self.log_root}
            self._open = self.ledger.reserve(fields, amount)
            return self.inner.send(request)
        finally:
            self._busy.release()


# ---------------------------------------------------------------- provider usage export

def read_usage_export(path):
    """(rows, SHA-256) from a usage export: a CSV with the header ``EXPORT_FIELDS``, or JSON,
    either a list of objects with those keys or {"rows": [...]}.

    Token columns are disjoint counts: ``input_tokens`` excludes cache writes and reads.
    ``cost_usd`` may be empty; when given, the larger of it and the priced tokens counts.
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
        rows.append({"provider": row["provider"], "model": model, "tokens": tokens, "cost_usd": cost})
    return rows, hashlib.sha256(data).hexdigest()


def export_usd(row: dict) -> Decimal:
    priced = call_cost(row["model"], row["tokens"])
    return priced if row["cost_usd"] is None else max(priced, row["cost_usd"])


def write_usage_export(path, rows) -> None:
    """Write rows of {"provider", "model", "tokens", "cost_usd"} as an export CSV."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(EXPORT_FIELDS)
        for row in rows:
            w.writerow([row["provider"], row["model"]] + [row["tokens"][k] for k in TOKEN_KINDS]
                       + ["" if row.get("cost_usd") is None else usd(row["cost_usd"])])
