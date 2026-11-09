#!/usr/bin/env python3
"""One runner for every stage: P1, P2 and P3 of the pilot, then the evaluation.

    export SPAR_FENCE_CONFIG=/path/to/local/fence_config.json
    python3 scripts/run_stage.py run --stage P1 --run-root ~/spar169-arb-data/out/pilot-1 \\
        --data-root ~/spar169-arb-data --packs ~/spar169-arb-data/out/packs-dev5 \\
        --provenance ~/spar169-arb-data/out/dev5/provenance.jsonl
    python3 scripts/run_stage.py reconcile --run-root ~/spar169-arb-data/out/pilot-1 \\
        --data-root ~/spar169-arb-data --budget pilot --export ~/spar169-arb-data/out/pilot-1/usage_export.csv
    python3 scripts/run_stage.py correct --run-root ... --data-root ... --stage P1 --halt halt-0001 --note "..."
    python3 scripts/run_stage.py correct --run-root ... --data-root ... --budget pilot --note "..."

Stages. Each refuses to start without the previous stage's complete report, and a stage
after P1 also needs a passing reconciliation of the previous stage's ledger after its
last call:
  P1    five development cases, A1, A2, A3 then B1 each
  P2    one prompt pass on 20 development traces, A1 and B1, uncached
  P3    a frozen development pass on 20 tasks with the two planned repeats
  EVAL  the frozen panel; needs --panel-manifest, its recorded --panel-sha256 and --freeze-tag,
        in the run root whose P3 the panel was frozen after, and that P3 must have sent a call
P1 to P3 install the dev_only guard and share the pilot budget ($95); EVAL installs
pre_join and has its own budget ($285). The guard is installed before anything is read.
P3 and EVAL refuse a rubric whose status is not "frozen". --allocation names the
development allocation from scripts/draw_allocation.py, which paid mode needs at every
stage, with --allocation-sha256, its SHA-256 as recorded in the plan or the approval note,
and which fake mode takes when given: each trace's normalized task ID must be in the
stage's list (P1 in p1, P2 in p2, P3 in p3), P2 takes a task of p1 only when the rubric
differs from the one P1 recorded, no panel task may be one of the allocation's
development tasks, drawn or not, and stage.json records the allocation's SHA-256. EVAL
in paid mode loads only records of the test split. EVAL refuses a configuration hash other than
the one P3 ran under, unless --config-change-note names a non-empty file declaring the
change, which stage.json keeps. --b-model picks judge B's model, gpt-6-sol or its first
fallback; it is part of the configuration hash, so a stage resumes only under the model
it started with. In the evaluation, a trajectory does not start, with a reconcile_due
halt, once the oldest call that no passing reconciliation follows is more than 23 hours
old (a single call is refused at 24 hours), and EVAL's report is complete only after a
passing reconciliation that follows its last call: reconcile, then run the same command
again to write it. A pilot stage records an ineligible case and goes on. Eligibility is
settled before the panel is frozen, so in EVAL an ineligible task halts the stage, is not
counted as ended and does not change N, and a note does not drop it.

**Transport.** With none of --paid, --confirm-paid-calls and --approval-note, only the
fake transport runs: every socket connection is refused, no API key is read, and
responses carry scripted usage (--scripted-usage central, high or extreme). Paid mode
needs all three together, --paid, --confirm-paid-calls (passed to the HTTPS transport as
``confirm_paid_calls``) and --approval-note naming an existing, non-empty file, plus
--workspace and --openai-project, the study's Anthropic workspace and OpenAI project.
Every OpenAI request carries the project in its OpenAI-Project header, added at send time
so the request bytes and the configuration hash do not depend on it, and a response that
reports another project halts the stage. A partial set of paid flags, or paid mode
without both, is refused before any transport exists. Every reservation in paid mode
needs the provider's own pre-send count (``dispatch.HttpsTokenCounter``), made under the
same credential and scope as the send: a count that fails, is malformed or is missing
writes a token_count halt before anything of the trajectory is sent, and no estimate
stands in for it. Every call reserves on the budget's ledger first (``judge.spend``), and
the existing dispatcher sends A1, A2 and A3 serially, then B1. A call that settles above
its reservation halts the stage once its trajectory finishes; the cap still counts actual
settled spend.

**Ledgers**, one per budget and mode, at one fixed place under --data-root, shared by
every run root, so committed spend adds up across runs:
  <data-root>/out/ledgers/<mode>/ledger-pilot.jsonl, ledger-evaluation.jsonl
  <data-root>/out/ledgers/budgets.jsonl             every ledger created, in every mode
  <data-root>/out/ledgers/marks-<budget>.jsonl      the ledger's mark after every entry
  <data-root>/out/ledgers/fake/fake_provider_export-<budget>-<ledger id>.csv
                                                    the fake provider's usage, per ledger
Every command refuses a --run-root that is not a folder directly under <data-root>/out
(other than ledgers/ in any letter case), in fake and paid mode alike, so every start
reads every run root's marks. A ledger shorter than, or differing from, a mark that the
marks file or a run root recorded is refused. Once the register or a mark names a ledger
for a budget, a second one (after it was moved or deleted, even with the whole ledgers/
folder) is created only with --new-budget-note, which the new ledger's header keeps.
These checks hold within one data root; a copy of the whole data root carries copies of
its ledgers.

**Outputs**, under --run-root, directly under <data-root>/out:
  run.json                          the mode, workspace, OpenAI project and the run's id,
                                    fixed at the first stage
  ledger_marks-<budget>.jsonl       the ledger's mark after each trajectory and command
  identity/                         the first model string per judge and every provider
                                    response id, shared by all stages of the run
  <stage>/stage.json                the stage's inputs; a resumed stage must match them
  <stage>/attempt_log/              the write-once attempt log, halts and corrections
  <stage>/progress.json             the gates so far, rewritten after each trajectory
  <stage>/report.json               written once, when every case has ended
A raw response left without its attempt record by a stop gets that record rebuilt at the
next start: a complete stream is recorded as it was, so a slot it ended is never sent
again; an error event that opens a stream, or a provider error body, halts, ends the
slot as FILTERED or is retried, as its error shows, and a B error body that would
otherwise be a transport failure halts unless it is a known transient error, since
OpenAI marks some halting errors by the status alone; a cut or unrecognized response is
a transport failure whose slot is retried under the usual retry rule. A response that
raises while it is read is recorded with a classification_error halt; if it may be
complete, its slot has ended and it is never sent again.
The report prints the gates for review and never acts on them to select a judge. A halt
stops the stage; rerun the same command after recording a correction note.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import secrets
import subprocess
import sys
import time
from collections import Counter, namedtuple
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import draw_allocation as DA  # noqa: E402
from datapath import fence, fetch  # noqa: E402
from datapath.ids import normalize_task_id, trace_key as key_of_trace  # noqa: E402
from judge import dispatch as D  # noqa: E402
from judge import pack as P  # noqa: E402
from judge import requests as R  # noqa: E402
from judge import spend as S  # noqa: E402
from judge.outcomes import CLASSIFICATION_ERROR, HALT, OUTCOMES, VALID, Exchange  # noqa: E402
from judge.storage import AttemptLog, WriteOnceError, write_once  # noqa: E402
from judge.streams import anthropic_events, openai_events, sse  # noqa: E402

Stage = namedtuple("Stage", "name about budget profile previous slots caching planned")
STAGES = {
    "P1": Stage("P1", "five development cases, A1, A2, A3 then B1 each", "pilot", "dev_only", None,
                R.SLOTS, True, 5),
    "P2": Stage("P2", "one prompt pass on 20 development traces, A1 and B1, uncached", "pilot", "dev_only",
                "P1", D.PROMPT_SLOTS, False, 20),
    "P3": Stage("P3", "a frozen development pass on 20 tasks with the two planned repeats", "pilot",
                "dev_only", "P2", R.SLOTS, True, 20),
    "EVAL": Stage("EVAL", "the frozen panel", "evaluation", "pre_join", "P3", R.SLOTS, True, None),
}
FROZEN_RUBRIC_STAGES = ("P3", "EVAL")
ALLOCATION_LIST = {"P1": "p1", "P2": "p2", "P3": "p3"}
ONE_TRACE_PER_TASK = ("P3", "EVAL")
TASK_ID_FORM = re.compile(r"[a-z0-9._-]+")
USAGE_PROFILES = {"central": (60000, 3000), "high": (60000, 12000), "extreme": (100000, 20000)}
FAKE_WORKSPACE = "wrkspc_fake"
FAKE_PROJECT = "proj_fake"
SCOPE_FLAGS = {"anthropic": "--workspace", "openai": "--openai-project"}
NO_COUNTER = ("paid mode needs the provider's own pre-send token count for every reservation, and no provider "
              "token-count client was given")
GATES_NOTE = ("Reported for review only. None of these gates selects or changes a judge, and a refusal "
              "is a recorded outcome, never a reason for a fallback.")
RESUME_RULES = {
    "spend_cap": "nothing was sent; a note does not raise the cap, and the gate is applied again",
    "reconciliation": "nothing was sent; answer the failed reconciliation (correct --budget), then note this halt",
    "reconcile_due": "nothing was sent; reconcile the evaluation ledger with both providers' usage, note this "
                     "halt, and run the stage again",
    "long_context": "nothing of the trajectory was sent; the B request is above the long-context threshold, and "
                    "the gate is applied again on resume",
    "orphan_attempt": "nothing was sent; leave the raw response where it is, since the next start rebuilds its "
                      "attempt record from it, then note this halt and resume",
    "over_reservation": "check the token count behind the reservation before resuming",
    "token_count": "nothing of the trajectory was sent and nothing was reserved; find why the provider gave no "
                   "count, note this halt, and run the stage again, which asks for the count again",
    "ineligible_frozen_task": "nothing of the trajectory was sent; eligibility was settled before the panel was "
                              "frozen, so a note does not drop the task or change N, and the gate is applied again "
                              "on resume",
}
FAKE_TRANSPORT = "fake transport with scripted provider-shaped streams"

Item = namedtuple("Item", "trace_key pack pack_sha256 entries private task_id", defaults=(None,))

_network_refused = False


class StageRefused(SystemExit):
    """A stage may not start or resume; nothing was sent."""


def _now(clock=time.time) -> str:
    return datetime.fromtimestamp(clock(), timezone.utc).isoformat(timespec="milliseconds")


def _share(n, d):
    return round(n / d, 4) if d else None


def _outside_repo(path: Path, flag: str) -> Path:
    path = path.expanduser().resolve()
    if path == REPO or REPO in path.parents:
        raise SystemExit(f"{flag} must be outside the repository: {path}")
    return path


def _forbid_network() -> None:
    global _network_refused

    def hook(event, args):
        if event in ("socket.connect", "socket.getaddrinfo", "urllib.Request"):
            raise PermissionError(f"fake mode makes no network calls: {event}")
    sys.addaudithook(hook)
    _network_refused = True


def fake_transport_line() -> str:
    """How fake mode's transport is described; it says every socket is refused only once
    the audit hook is installed in this process."""
    return FAKE_TRANSPORT + ("; every socket refused" if _network_refused else "; no socket guard installed")


def _git(repo, *args):
    """git with only PATH passed on, so no credential in the environment reaches it."""
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")})


def code_version(repo=REPO) -> dict:
    try:
        head, status = _git(repo, "rev-parse", "HEAD"), _git(repo, "status", "--porcelain")
    except OSError:
        return {"commit": "unknown", "uncommitted_changes": None}
    if head.returncode != 0:
        return {"commit": "unknown", "uncommitted_changes": None}
    return {"commit": head.stdout.strip(), "uncommitted_changes": bool(status.stdout.strip())}


def _jsonl(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _record_bytes(record: dict) -> bytes:
    return (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")


def ledger_root(data_root) -> Path:
    """The folder of every ledger, its register and the marks files, shared by every run root."""
    return Path(data_root) / "out" / "ledgers"


def ledger_dir(data_root, mode: str) -> Path:
    """The one fixed folder of a mode's ledgers."""
    return ledger_root(data_root) / mode


def check_run_root(data_root: Path, run_root: Path) -> Path:
    """The run root, which must be a folder directly under <data-root>/out other than the
    ledgers' folder, so that every start reads the marks of every run root. The ledgers'
    folder is matched without regard to letter case and by file identity, since the file
    system may ignore case."""
    out_dir, ledgers = (Path(data_root) / "out").resolve(), ledger_root(data_root)
    if (run_root.parent != out_dir or run_root.name.casefold() == ledgers.name.casefold()
            or (run_root.exists() and ledgers.exists() and run_root.samefile(ledgers))):
        raise StageRefused(f"--run-root must be a folder directly under {out_dir}, other than "
                           f"{ledgers.name}/, so that every start reads its ledger marks: {run_root}")
    return run_root


def ledger_path(folder, budget: str) -> Path:
    return Path(folder) / S.ledger_name(budget)


def export_path(folder, budget: str, ledger_id: str) -> Path:
    """The fake provider's usage export for one ledger: a new ledger starts a new export."""
    return Path(folder) / f"fake_provider_export-{budget}-{ledger_id}.csv"


# ---------------------------------------------------------------- inputs

def task_of_trace(key: str, prov: dict):
    """The record's normalized task ID, once it is tied to the trace it describes: the
    key is the hash of the record's trace ID, the trace ID ends in the raw task ID, and
    the normalized ID is the raw one's. None for a record with no normalized ID, which an
    allocation refuses."""
    normalized = prov.get("task_id_normalized")
    if normalized is None:
        return None
    task, trace = prov.get("task_id"), prov.get("trace_id")
    if (not isinstance(trace, str) or key_of_trace(trace) != key or trace.rsplit("/", 1)[-1] != task
            or normalize_task_id(task) != normalized):
        raise StageRefused(f"{key}: the provenance record's task ID is not tied to its trace")
    return normalized


def load_items(packs, provenance, split="dev") -> list:
    """Items from scripts/build_packs.py folders, each with its provenance file (same order).

    ``split`` is the split every record must come from, or None (EVAL in fake mode, where
    the panel manifest names the tasks and a rehearsal may use other records).
    """
    if len(packs) != len(provenance):
        raise StageRefused("give one --provenance file for each --packs folder, in the same order")
    items = []
    for folder, prov_path in zip(packs, provenance):
        folder = Path(folder).expanduser().resolve()
        provs = {p["trace_key"]: p for p in _jsonl(Path(prov_path).expanduser().resolve())}
        manifest = {}
        for e in _jsonl(folder / "image_manifest.jsonl"):
            manifest.setdefault(e["trace_key"], []).append(e)
        for line in _jsonl(folder / "packs.jsonl"):
            key, pack = line["trace_key"], line["pack"]
            if P.sha256_text(P.serialize(pack)) != line["pack_sha256"]:
                raise ValueError(f"{key}: the pack does not match its hash")
            prov = provs.get(key)
            if prov is None or (split is not None and prov.get("split") != split):
                raise StageRefused(f"{key}: no {split or 'matching'}-split provenance record")
            entries = manifest.get(key, [])
            if any(e["pack_sha256"] != line["pack_sha256"] for e in entries):
                raise ValueError(f"{key}: the image manifest belongs to another pack")
            private = sorted(set(P.private_strings(prov)) | set(R.manifest_strings(entries))
                             | {key, line["pack_sha256"]})
            items.append(Item(key, pack, line["pack_sha256"], entries, private, task_of_trace(key, prov)))
    keys = [i.trace_key for i in items]
    if len(set(keys)) != len(keys):
        raise ValueError("a trace key appears twice across the --packs folders")
    return items


# ---------------------------------------------------------------- transports

def _note_ok(path) -> bool:
    if not path:
        return False
    p = Path(path).expanduser()
    return p.is_file() and p.stat().st_size > 0 and bool(p.read_text(encoding="utf-8").strip())


def paid_conditions(paid, confirm_paid_calls, approval_note) -> list:
    """The paid-mode conditions that do not hold; empty when all three do."""
    missing = []
    if paid is not True:
        missing.append("--paid")
    if confirm_paid_calls is not True:
        missing.append("--confirm-paid-calls")
    if not _note_ok(approval_note):
        missing.append("--approval-note naming an existing, non-empty file")
    return missing


def select_transport(paid, confirm_paid_calls, approval_note, workspace=None, openai_project=None):
    """("fake", None) when no paid condition is given; ("paid", HttpsTransport) when all
    three hold and the study workspace and OpenAI project are named; any other combination
    is refused before a transport exists."""
    if not (paid or confirm_paid_calls or approval_note):
        return "fake", None
    missing = paid_conditions(paid, confirm_paid_calls, approval_note)
    if missing:
        raise StageRefused("paid mode needs --paid, --confirm-paid-calls and --approval-note together; "
                           "missing: " + "; ".join(missing))
    if not workspace:
        raise StageRefused("paid mode also needs --workspace, the study's Anthropic workspace ID")
    if not openai_project:
        raise StageRefused("paid mode also needs --openai-project, the study's OpenAI project ID")
    return "paid", D.HttpsTransport(confirm_paid_calls=confirm_paid_calls, openai_project=openai_project)


class ScriptedProvider(D.FakeTransport):
    """The fake transport for fake mode: one provider-shaped response per request.

    Every call reports the profile's input and output tokens. An A request with cache
    control writes the cache on its first send and reads it afterwards. ``events`` maps
    (trace key, slot, attempt) to "timeout", "http_400", "refusal", "output_cap" or
    "cache_miss". Each response the fake provider bills is kept in ``billed``;
    ``write_export`` adds them to the usage export at ``export_path``, which states one
    period from ``period_start`` (the ledger's creation) to the time it is written, by
    ``clock``. A provider that was sent requests but billed none gets a zero-token row.
    ``count`` stands in for the provider's pre-send token count.
    """

    def __init__(self, profile="central", workspace=FAKE_WORKSPACE, events=None, export_path=None,
                 project=FAKE_PROJECT, clock=time.time):
        super().__init__()
        self.profile = profile
        self.input_tokens, self.output_tokens = USAGE_PROFILES[profile]
        self.workspace, self.project, self.events, self.export = workspace, project, dict(events or {}), export_path
        self.cached, self.billed, self.ctx, self.period_start, self._exported = set(), [], None, None, 0
        self.clock = clock

    def count(self, request) -> int:
        return self.input_tokens

    def begin(self, stage: Stage, trace_key, config_sha256, log) -> None:
        self.ctx = (stage, trace_key, config_sha256, log)

    def send(self, request):
        stage, trace_key, config, log = self.ctx
        (judge, repeat), attempt = S.next_attempt(log, trace_key, config, request.judge, stage.slots)
        slot = f"{judge}{repeat}"
        kind = self.events.get((trace_key, slot, attempt), "valid")
        self.add(self._exchange(request, stage, f"{stage.name}_{trace_key}_{slot}_{attempt}", kind))
        return super().send(request)

    def export_rows(self, period_start, period_end=None, rows=None) -> list:
        """Export rows for ``rows`` (default: everything billed), each stating the period
        and the provider's scope."""
        if not period_start:
            raise ValueError("an export period starts at the ledger's creation; give period_start")
        end = period_end or _now(self.clock)
        scope = {"anthropic": self.workspace, "openai": self.project}
        rows = list(self.billed if rows is None else rows)
        for request in self.sent:
            if request.provider not in {r["provider"] for r in rows}:
                rows.append({"provider": request.provider, "model": request.model,
                             "tokens": dict.fromkeys(S.TOKEN_KINDS, 0), "cost_usd": None})
        return [dict(r, period_start=period_start, period_end=end, scope=scope[r["provider"]]) for r in rows]

    def write_export(self, period_end=None) -> None:
        """Add the calls billed since the last write to the usage export and restate its period."""
        if self.export is None:
            return
        path = Path(self.export)
        earlier = S.read_usage_export(path)[0] if path.exists() else []
        rows = earlier + self.billed[self._exported:]
        S.write_usage_export(path, self.export_rows(self.period_start, period_end, rows))
        self._exported = len(self.billed)

    @staticmethod
    def _answer(ref):
        verdict = "success" if int(hashlib.sha256(ref.encode()).hexdigest(), 16) % 2 else "failure"
        return json.dumps({"verdict": verdict, "p_failure": 0.3, "evidence_sufficient": "yes",
                           "evidence_step_ids": [0], "rationale": "A scripted answer from the fake provider."})

    def _bill(self, request, tokens) -> None:
        self.billed.append({"provider": request.provider, "model": request.model, "tokens": tokens, "cost_usd": None})

    def _exchange(self, request, stage: Stage, ref: str, kind: str) -> Exchange:
        if kind == "timeout":
            return Exchange(None, {}, b"", "TimeoutError: timed out (scripted)", None)
        if kind == "http_400":
            err = ({"type": "error", "error": {"type": "invalid_request_error", "message": "scripted bad request"}}
                   if request.provider == "anthropic" else
                   {"error": {"message": "scripted bad request", "type": "invalid_request_error",
                              "param": None, "code": None}})
            return Exchange(400, {}, json.dumps(err).encode(), None, None)
        inp = self.input_tokens
        out = R.OUTPUT_CAP if kind == "output_cap" else self.output_tokens
        if request.provider == "anthropic":
            write = read = 0
            if stage.caching:
                if kind == "cache_miss" or request.sha256 not in self.cached:
                    write = inp
                else:
                    read = inp
                self.cached.add(request.sha256)
            uncached = 0 if stage.caching else inp
            usage = {"input_tokens": uncached, "cache_creation_input_tokens": write, "cache_read_input_tokens": read,
                     "cache_creation": {"ephemeral_5m_input_tokens": write, "ephemeral_1h_input_tokens": 0}}
            stop = {"refusal": "refusal", "output_cap": "max_tokens"}.get(kind, "end_turn")
            text = self._answer(ref) if stop == "end_turn" else ("" if stop == "refusal" else None)
            events = anthropic_events(text, stop_reason=stop, model=request.model, msg_id=f"msg_{ref}", usage=usage)
            for _, obj in events:
                if obj.get("type") == "message_delta":
                    obj["usage"] = {"output_tokens": out}
            self._bill(request, {"input": uncached, "cache_write": write, "cache_read": read, "output": out})
            return Exchange(200, {"anthropic-workspace-id": self.workspace, "request-id": f"req_{ref}"},
                            sse(events), None, None)
        usage = {"input_tokens": inp, "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                 "output_tokens": out, "output_tokens_details": {"reasoning_tokens": max(out - 200, 0)},
                 "total_tokens": inp + out}
        if kind == "refusal":
            events = openai_events(None, refusal="I can't help with that.", model=request.model,
                                   resp_id=f"resp_{ref}", usage=usage)
        elif kind == "output_cap":
            events = openai_events(None, status="incomplete", incomplete_reason="max_output_tokens",
                                   model=request.model, resp_id=f"resp_{ref}", usage=usage)
        else:
            events = openai_events(self._answer(ref), model=request.model, resp_id=f"resp_{ref}", usage=usage)
        self._bill(request, {"input": inp, "cache_write": 0, "cache_read": 0, "output": out})
        return Exchange(200, {"x-request-id": f"req_{ref}", D.OPENAI_PROJECT_HEADER: self.project}, sse(events),
                        None, None)


# ---------------------------------------------------------------- the run's records

class StageLog(AttemptLog):
    """A stage's attempt log that keeps the first model string per judge and requested
    model, and every provider response id, once under the run root, so a silent model
    change or a repeated response id is caught across stages."""

    def __init__(self, root, run_root, stage: str, models: dict):
        super().__init__(root)
        self.shared, self.stage, self.models = Path(run_root) / "identity", stage, dict(models)

    def _model_path(self, judge) -> Path:
        return self.shared / "first_model" / f"{judge}-{self.models[judge]}.json"

    def first_model(self, config_sha256, judge):
        path = self._model_path(judge)
        return json.loads(path.read_text(encoding="utf-8"))["model"] if path.exists() else None

    def set_first_model(self, config_sha256, judge, model, attempt_ref) -> None:
        write_once(self._model_path(judge), _record_bytes({
            "judge": judge, "model_requested": self.models[judge], "model": model,
            "first_attempt": f"{self.stage}/{attempt_ref}", "config_sha256": config_sha256}))

    def _response_id_path(self, provider: str, response_id: str) -> Path:
        digest = hashlib.sha256(f"{provider}\n{response_id}".encode("utf-8")).hexdigest()
        return self.shared / "response_ids" / f"{digest}.json"

    def claim_response_id(self, provider: str, response_id: str, attempt_ref: str) -> bool:
        try:
            write_once(self._response_id_path(provider, response_id), _record_bytes(
                {"provider": provider, "id": response_id, "attempt": f"{self.stage}/{attempt_ref}"}))
        except WriteOnceError:
            return False
        return True

    def owns_response_id(self, provider: str, response_id: str, attempt_ref: str) -> bool:
        path = self._response_id_path(provider, response_id)
        return (path.is_file()
                and json.loads(path.read_text(encoding="utf-8")).get("attempt") == f"{self.stage}/{attempt_ref}")


def check_ledger_scopes(ledger, scopes: dict) -> None:
    """Refuse a run whose scope for a provider differs from one that the budget's ledger
    already names: every run root shares the ledger, and one export covers one scope, so a
    second scope would leave the ledger unreconcilable."""
    for provider, named in sorted(ledger.scopes().items()):
        other = sorted(named - {scopes.get(provider)}) if scopes.get(provider) else []
        if other:
            raise StageRefused(f"the {ledger.budget} ledger's {provider} calls ran under {other}, not "
                               f"{scopes[provider]!r}; one ledger is reconciled under one scope, so fix "
                               f"{SCOPE_FLAGS[provider]}")


def run_record(run_root: Path, mode: str, scopes: dict, ledger) -> dict:
    """run.json: the mode, the study workspace and OpenAI project, fixed at the first
    stage, and the run's own id, by which the ledger names it wherever the folder is.
    Scopes that differ from the ledger's are refused before run.json is written."""
    path = run_root / "run.json"
    wanted = {"mode": mode, "workspace": scopes["anthropic"], "openai_project": scopes["openai"]}
    record = None
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        given = {k: record.get(k) for k in wanted}
        if given != wanted:
            raise StageRefused(f"{path} records {given}; every stage of a run uses the same ones, not {wanted}")
    check_ledger_scopes(ledger, scopes)
    if record is None:
        record = dict(wanted, run_id="run-" + secrets.token_hex(6))
        write_once(path, _record_bytes(record))
    return record


def run_mode(run_root: Path) -> dict:
    path = run_root / "run.json"
    if not path.is_file():
        raise StageRefused(f"no run in {run_root}: {path} is missing")
    return json.loads(path.read_text(encoding="utf-8"))


def marks_path(run_root: Path, budget: str) -> Path:
    return Path(run_root) / f"ledger_marks-{budget}.jsonl"


def record_mark(run_root: Path, ledger, mode: str, clock=time.time) -> None:
    """Add the ledger's mark to the run root's marks file for its budget, under the run's mode."""
    line = json.dumps(dict(ledger.mark(), mode=mode, at=_now(clock)), sort_keys=True) + "\n"
    with open(marks_path(run_root, ledger.budget), "ab") as f:
        f.write(line.encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())


def run_root_marks(run_root: Path, budget: str, mode: str) -> list:
    """(where, mark) for every mark of the budget's ledger in ``mode`` that a run root
    recorded: its marks file and the progress files of the budget's stages."""
    run_root = Path(run_root)
    path = marks_path(run_root, budget)
    out = [(str(path), m) for m in (_jsonl(path) if path.is_file() else []) if m.get("mode") == mode]
    for name in (n for n, st in STAGES.items() if st.budget == budget):
        progress = run_root / name / "progress.json"
        if progress.is_file():
            report = json.loads(progress.read_text(encoding="utf-8"))
            led = report.get("ledger") or {}
            if report.get("mode") == mode and led.get("id"):
                out.append((str(progress), {"ledger_id": led["id"], "seq": led.get("last_seq"),
                                            "sha256": led.get("sha256")}))
    return out


def peer_marks(data_root: Path, run_root: Path, budget: str, mode: str) -> list:
    """The marks of ``run_root`` and of every run root directly under <data-root>/out."""
    out_dir = Path(data_root) / "out"
    roots = {Path(run_root)}
    if out_dir.is_dir():
        roots |= {c for c in out_dir.iterdir() if c.is_dir() and (c / "run.json").is_file()}
    return [m for root in sorted(roots) for m in run_root_marks(root, budget, mode)]


# ---------------------------------------------------------------- stage checks

def check_previous(spec: Stage, run_root: Path, ledger, mode: str, run: dict):
    """The previous stage's complete report, in the same mode, with its ledger reconciled."""
    if spec.previous is None:
        return None
    prev = STAGES[spec.previous]
    path = run_root / prev.name / "report.json"
    if not path.is_file():
        raise StageRefused(f"{spec.name} needs {prev.name}'s report at {path}; run {prev.name} first")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("stage") != prev.name or report.get("complete") is not True:
        raise StageRefused(f"{path} is not a complete {prev.name} report")
    if report.get("mode") != mode:
        raise StageRefused(f"{prev.name} ran in {report.get('mode')} mode; {spec.name} would run in {mode} mode")
    if prev.budget == ledger.budget:
        reconciled = ledger.reconciled_after_last_call(run["run_id"], prev.name)
    else:
        prev_path = ledger_path(ledger.path.parent, prev.budget)
        if not prev_path.is_file():
            raise StageRefused(f"no {prev.budget} ledger at {prev_path}")
        recorded = json.loads((run_root / prev.name / "stage.json").read_text(encoding="utf-8"))["ledger"]
        with S.Ledger(prev_path, prev.budget, mode=ledger.mode) as prev_ledger:
            if prev_ledger.ledger_id != recorded.get("id"):
                raise StageRefused(f"{prev.name} ran on {prev.budget} ledger {recorded.get('id')}, not on "
                                   f"{prev_ledger.ledger_id}")
            reconciled = prev_ledger.reconciled_after_last_call(run["run_id"], prev.name)
            if not any(e["type"] == "reserve" and e["run"] == run["run_id"] and e["stage"] == prev.name
                       for e in prev_ledger.entries):
                raise StageRefused(f"{prev.name} of this run sent no call, so no configuration ran that "
                                   f"{spec.name} could follow")
    if not reconciled:
        raise StageRefused(f"reconcile the {prev.budget} ledger with the providers' usage after {prev.name} "
                           f"before starting {spec.name}")
    return str(path)


def check_ledger_against_logs(spec: Stage, run_root: Path, ledger, run_id: str, mode: str) -> None:
    """Every stage of this budget in the run root ran on this ledger, the ledger is not
    shorter than, and does not differ from, any mark the run root recorded, and every
    attempt its stages logged has its reservation in it. Otherwise the ledger was rolled
    back or replaced."""
    reserved = {(e.get("run"), e.get("stage"), e.get("trace_key"), e.get("judge"), e.get("repeat"),
                 e.get("attempt"), e.get("request_sha256")) for e in ledger.entries if e["type"] == "reserve"}
    for where, mark in run_root_marks(run_root, spec.budget, mode):
        if mark["ledger_id"] == ledger.ledger_id:
            try:
                ledger.check_mark(mark, where)
            except S.LedgerError as e:
                raise StageRefused(str(e)) from None
        elif mark["ledger_id"] not in ledger.header.get("replaces", []):
            raise StageRefused(f"{where} recorded {spec.budget} ledger {mark['ledger_id']}; this one is "
                               f"{ledger.ledger_id}, so nothing is sent")
    for name in (n for n, s in STAGES.items() if s.budget == spec.budget):
        folder = run_root / name
        if (folder / "stage.json").is_file():
            used = json.loads((folder / "stage.json").read_text(encoding="utf-8")).get("ledger") or {}
            if used.get("id") != ledger.ledger_id:
                raise StageRefused(f"{name} ran on {spec.budget} ledger {used.get('id')}; this one is "
                                   f"{ledger.ledger_id}, so nothing is sent")
        for path in sorted((folder / "attempt_log" / "attempts").glob("*/*/*/*.json")):
            rec = json.loads(path.read_text(encoding="utf-8"))
            key = (run_id, name, rec.get("trace_key"), rec.get("judge"), rec.get("repeat"), rec.get("attempt"),
                   rec.get("request_sha256"))
            if key not in reserved:
                raise StageRefused(f"{path} has no reservation in the {spec.budget} ledger: it was rolled back "
                                   "or replaced, so nothing is sent")


def rebuild_orphans(log, ledger, items, run_id: str, stage: str, config_sha256: str, slots) -> list:
    """Attempt records rebuilt from raw responses whose records were never written (a stop
    before the record), classified as ``dispatch.recover_attempt`` describes: a
    complete stream is recorded as it was and a slot it ended is never sent again, an
    error event that opens a stream or an error body halts, ends the slot as FILTERED or
    is retried, and a cut or unrecognized response is a transport failure retried under
    the usual rule. Each must have its reservation in the ledger, which then closes from
    the rebuilt record. A stop during a send closes that attempt's reservation with no
    record and the next start reserves the attempt again, so the last reservation is the
    one the raw response answers; the ledger keeps only one open, so the others are
    closed."""
    rebuilt = []
    for item in items:
        for j, r in slots:
            n = len(log.attempts(item.trace_key, config_sha256, j, r)) + 1
            raw = log.slot_dir(item.trace_key, config_sha256, j, r) / f"{n}.raw"
            if not raw.exists():
                continue
            match = [e for e in ledger.entries if e["type"] == "reserve" and e.get("run") == run_id
                     and e.get("stage") == stage and (e.get("trace_key"), e.get("config_sha256"), e.get("judge"),
                                                      e.get("repeat"), e.get("attempt"))
                     == (item.trace_key, config_sha256, j, r, n)]
            if not match:
                raise StageRefused(f"{raw} has no attempt record and no reservation in the {ledger.budget} "
                                   "ledger, so nothing is sent")
            record = D.recover_attempt(log, match[-1])
            rebuilt.append({"trace_key": item.trace_key, "slot": f"{j}{r}", "attempt": n,
                            "outcome": record["outcome"], "http_status": record["http_status"]})
    return rebuilt


def record_over_reservation_halts(log, ledger, run_id: str, stage: str, config_sha256: str, clock) -> list:
    """Halt records for this run's stage's calls that settled above their reservation with
    no halt record naming them (a stop before it was written); each is then unresolved
    until a correction note."""
    named = {rid for h in log.halts() if h.get("halt_kind") == "over_reservation"
             for rid in (h.get("reservations") or [])}
    by_case = {}
    for e in ledger.entries:
        if e["type"] == "settle" and e.get("over_reservation") and e["id"] not in named:
            r = ledger.reservations[e["id"]]
            if r.get("run") == run_id and r.get("stage") == stage:
                by_case.setdefault(r.get("trace_key"), []).append(e["id"])
    return [log.write_halt({
        "trace_key": key, "config_sha256": config_sha256, "slot": None, "attempt": None,
        "halt_kind": "over_reservation", "reservations": ids, "failed_checks": [], "error": None, "http_status": None,
        "detail": f"{len(ids)} call(s) settled above their reservation with no halt record; recorded at stage start",
        "at": _now(clock), "recorded_at_start": True, "resume_rule": RESUME_RULES["over_reservation"]})
        for key, ids in by_case.items()]


def record_unrecorded_halts(log, items, config_sha256: str, slots, clock) -> list:
    """Halt records for slots whose last attempt stopped the run, as a HALT or a
    classification failure, with no halt record (a stop before it was written); each is
    then unresolved until a correction note."""
    recorded = {(h.get("trace_key"), h.get("slot"), h.get("attempt")) for h in log.halts()}
    written = []
    for item in items:
        for j, r in slots:
            recs = log.attempts(item.trace_key, config_sha256, j, r)
            stops = recs and (recs[-1]["outcome"] == HALT or recs[-1].get("halt_kind") == CLASSIFICATION_ERROR)
            if stops and (item.trace_key, f"{j}{r}", recs[-1]["attempt"]) not in recorded:
                last = recs[-1]
                written.append(log.write_halt({
                    "trace_key": item.trace_key, "config_sha256": config_sha256, "slot": f"{j}{r}",
                    "attempt": last["attempt"], "halt_kind": last.get("halt_kind"),
                    "failed_checks": last.get("failed_checks") or [], "error": last.get("error"),
                    "http_status": last.get("http_status"),
                    "detail": "the slot's last attempt stopped the run with no halt record; recorded at stage start",
                    "at": _now(clock), "recorded_at_start": True, "resume_rule": D.resume_rule(last["outcome"])}))
    return written


def freeze_commit(repo, tag: str) -> str:
    """The commit of the freeze tag, which must be HEAD with no uncommitted changes."""
    tagged = _git(repo, "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}^{{commit}}")
    if tagged.returncode != 0 or not tagged.stdout.strip():
        raise StageRefused(f"no freeze tag {tag!r} in {repo}")
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    if head != tagged.stdout.strip():
        raise StageRefused(f"the code is at {head[:12]}, not at the freeze tag {tag!r}")
    status = _git(repo, "status", "--porcelain")
    if status.returncode != 0 or status.stdout.strip():
        raise StageRefused("the repository has uncommitted changes; EVAL runs only the frozen code")
    return head


def verify_panel(panel, config_sha256: str, items: list, repo=REPO, allocation=None) -> dict:
    """EVAL's frozen panel manifest, checked against its recorded hash and freeze tag.

    The manifest is JSON: {"kind": "panel-manifest", "frozen": true, "freeze_tag",
    "config_sha256", "allocation_sha256", "p3_run_id", "n", "tasks": [{"trace_key",
    "pack_sha256"}, ...]}. The packs given must be its tasks, in its order, the code must
    be at the freeze tag, and ``allocation``, when one is in use, must be the one the
    manifest names. ``p3_run_id`` is the run_id in the run.json of the run root whose P3
    the panel was frozen after, which ``check_panel_run`` compares with this run.
    """
    if not panel or not all(panel.get(k) for k in ("manifest", "sha256", "freeze_tag")):
        raise StageRefused("EVAL needs --panel-manifest, its recorded --panel-sha256 and --freeze-tag")
    path = Path(panel["manifest"]).expanduser()
    if not path.is_file():
        raise StageRefused(f"no panel manifest at {path}")
    data = path.read_bytes()
    if R.sha256_bytes(data) != panel["sha256"]:
        raise StageRefused("the panel manifest does not match its recorded SHA-256")
    m = json.loads(data)
    if not isinstance(m, dict) or m.get("kind") != "panel-manifest" or m.get("frozen") is not True:
        raise StageRefused("the manifest is not a frozen panel manifest")
    if m.get("freeze_tag") != panel["freeze_tag"]:
        raise StageRefused(f"the manifest names freeze tag {m.get('freeze_tag')!r}, not {panel['freeze_tag']!r}")
    if m.get("config_sha256") != config_sha256:
        raise StageRefused("the manifest's config hash is not this runner's configuration")
    if allocation is not None and m.get("allocation_sha256") != allocation["sha256"]:
        raise StageRefused(f"the manifest was frozen under allocation {m.get('allocation_sha256')!r}, not "
                           f"{allocation['sha256']}")
    tasks = m.get("tasks")
    if not isinstance(tasks, list) or not tasks or m.get("n") != len(tasks):
        raise StageRefused("the manifest's task list and N do not agree")
    if [(t.get("trace_key"), t.get("pack_sha256")) for t in tasks] != [(i.trace_key, i.pack_sha256) for i in items]:
        raise StageRefused("the packs given are not the panel's tasks in the manifest's order")
    return {"manifest": str(path.resolve()), "sha256": panel["sha256"], "freeze_tag": panel["freeze_tag"],
            "freeze_commit": freeze_commit(repo, panel["freeze_tag"]), "n": len(tasks),
            "p3_run_id": m.get("p3_run_id")}


def check_panel_run(panel_record: dict, run: dict) -> None:
    """EVAL runs in the run root whose P3 the panel was frozen after. A fresh run root gets
    a new random run ID, so its own short P3 cannot stand in for that one."""
    if panel_record["p3_run_id"] != run["run_id"]:
        raise StageRefused(f"the panel was frozen after the P3 of run {panel_record['p3_run_id']!r}; this run root "
                           f"is run {run['run_id']!r}, so its P3 is not the one the panel followed")


def load_allocation(path) -> dict:
    """The allocation file from scripts/draw_allocation.py, once it passes that script's checks."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise StageRefused(f"no allocation file at {path}")
    try:
        return DA.verify_allocation(json.loads(path.read_bytes()))
    except ValueError as e:                     # a refusal, or a file that is not JSON
        raise StageRefused(f"the allocation is refused: {e}") from None


def _is_task_id(task) -> bool:
    """Whether ``task`` is a non-empty task ID in the split table's form: lower-case, with
    no space, and normalized."""
    return isinstance(task, str) and TASK_ID_FORM.fullmatch(task) is not None and normalize_task_id(task) == task


def check_allocation(stage: str, items: list, allocation: dict) -> None:
    """Refuse a trace whose task is outside its stage's list, and any panel task among
    the allocation's development tasks, drawn or not. A trace maps to its task through its
    provenance record's ``task_id_normalized``, the ID the split table uses, so any other
    form of ID is refused here, and P3 and EVAL take one trace per task."""
    unmapped = [i.trace_key for i in items if not _is_task_id(i.task_id)]
    if unmapped:
        raise StageRefused(f"{unmapped[0]} has no normalized task ID in its provenance, so the allocation cannot "
                           "be applied")
    tasks = {i.task_id for i in items}
    if stage in ONE_TRACE_PER_TASK and len(tasks) != len(items):
        raise StageRefused(f"{stage} runs one trace per task; {len(items)} traces of {len(tasks)} tasks were given")
    if stage == "EVAL":
        used = sorted(tasks & set(allocation[DA.DEVELOPMENT]))
        if used:
            raise StageRefused(f"EVAL's panel holds tasks of the development allocation: {used}")
        return
    outside = sorted(tasks - set(allocation[ALLOCATION_LIST[stage]]))
    if outside:
        raise StageRefused(f"{stage} runs only tasks in the allocation's {ALLOCATION_LIST[stage]}; these are not: "
                           f"{outside}")


def check_prompt_pass(items: list, run_root: Path, allocation: dict, rubric_sha256: str) -> None:
    """P2 is P1's five plus the 15, and P1's five go out again only after a rubric change,
    which the rubric hash in P1's stage.json shows. With no hash there, nothing shows one."""
    again = sorted({i.task_id for i in items} & set(allocation["p1"]))
    if not again:
        return
    recorded = json.loads((run_root / "P1" / "stage.json").read_text(encoding="utf-8")).get("rubric_sha256")
    if recorded in (None, rubric_sha256):
        raise StageRefused(f"P2 sends P1's tasks again only after a rubric change, and this is the rubric P1 ran "
                           f"under: {again}")


def check_recorded_allocation(allocation: dict, recorded, mode: str) -> None:
    """``recorded`` is --allocation-sha256, the hash in the plan or the approval note. Paid
    mode needs it, and paid mode also needs the file to be the study's draw, since the
    file's own hash shows only that it was not edited after it was written."""
    if mode == "paid" and not recorded:
        raise StageRefused("paid mode needs --allocation-sha256, the allocation's SHA-256 as recorded in the plan "
                           "or the approval note")
    if recorded and allocation["sha256"] != recorded:
        raise StageRefused(f"the allocation's SHA-256 is {allocation['sha256']}, not the recorded {recorded}")
    if mode == "paid":
        try:
            DA.verify_study_draw(allocation)
        except DA.DrawRefused as e:
            raise StageRefused(f"the allocation is refused: {e}") from None


def check_same_allocation(stage: str, run_root: Path, mode: str, allocation: dict) -> None:
    """Every stage that recorded an allocation in this mode used this one, in this run
    root or any run root beside it, so a fresh run root cannot use another. A resumed
    stage's own record is checked with its other inputs."""
    for root in sorted(c for c in run_root.parent.iterdir() if c.is_dir() and (c / "run.json").is_file()):
        for name in STAGES:
            path = root / name / "stage.json"
            if (root, name) == (run_root, stage) or not path.is_file():
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("mode") == mode and record.get("allocation") not in (None, allocation["sha256"]):
                raise StageRefused(f"{root.name}/{name} ran under allocation {record['allocation']}, not "
                                   f"{allocation['sha256']}; every stage of a run uses the same one")


def check_eval_config(run_root: Path, config_sha256: str, note) -> dict:
    """EVAL runs under the configuration P3 ran under, or declares the difference in a
    note. Returns the note's record for stage.json, or None when there is no difference."""
    p3 = json.loads((run_root / "P3" / "stage.json").read_text(encoding="utf-8")).get("config_sha256")
    if p3 == config_sha256:
        if note:
            raise StageRefused("--config-change-note declares a change, but EVAL's configuration hash is the one P3 "
                               "ran under")
        return None
    if not _note_ok(note):
        raise StageRefused(f"EVAL's configuration hash {config_sha256} is not the one P3 ran under, {p3}; "
                           "--config-change-note, a non-empty file, must declare the change")
    path = Path(note).expanduser()
    return {"sha256": fetch.sha256_file(path), "text": path.read_text(encoding="utf-8"), "p3_config_sha256": p3}


def _fixed_record(path: Path, record: dict, what: str) -> None:
    """Write ``record`` once; later starts must present the same one."""
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != record:
            raise StageRefused(f"{path} records different {what}; a resumed stage must use the same ones")
        return
    write_once(path, _record_bytes(record))


def _slots_ended(log, key, config_sha256, slots) -> bool:
    return all(D._slot_done(log.attempts(key, config_sha256, j, r)) for j, r in slots)


# ---------------------------------------------------------------- run

def run_stage(stage: str, run_root, items: list, *, mode: str, transport, ledger, read_images, workspace,
              openai_project=None, counter=None, panel=None, repo=REPO, sleep=None, clock=time.time, settings=None,
              b_model=R.B_MODEL, rubric_check=True, allocation=None, allocation_sha256=None,
              config_change_note=None) -> dict:
    """Run one stage, or resume it, and return its report (see the module docstring).

    ``workspace`` and ``openai_project`` are the study's scopes; fake mode's project
    defaults to the fake provider's.
    ``counter`` gives each request's input count before it is reserved; paid mode needs an
    ``HttpsTokenCounter`` on its transport and workspace. ``rubric_check=False`` waives the
    frozen-rubric rule of P3 and EVAL, which only the fake-mode stage dry run does, and the
    report says so. A call that settles above its reservation halts the stage once its
    trajectory finishes; the cap still counts actual settled spend. ``allocation`` is the
    path of the development allocation, which paid mode needs, with ``allocation_sha256``,
    its recorded hash; ``config_change_note`` is the path of a note declaring that EVAL's
    configuration differs from P3's.
    """
    spec = STAGES[stage]
    run_root = Path(run_root)
    if ledger.budget != spec.budget:
        raise ValueError(f"{stage} spends from the {spec.budget} budget, not {ledger.budget}")
    if mode not in ("fake", "paid"):
        raise ValueError(f"mode is fake or paid, not {mode!r}")
    if mode == "fake" and not isinstance(transport, D.FakeTransport):
        raise ValueError("fake mode runs only a fake transport")
    if mode == "paid" and not isinstance(transport, D.HttpsTransport):
        raise ValueError("paid mode runs only the HTTPS transport")
    if ledger.mode != mode:
        raise ValueError(f"{ledger.path} is a {ledger.mode} ledger; a {mode} stage spends only from a {mode} one")
    if not rubric_check and not isinstance(transport, D.FakeTransport):
        raise ValueError("the frozen-rubric rule is waived only with the fake transport")
    if mode == "paid" and counter is None:
        raise StageRefused(NO_COUNTER)
    if mode == "paid" and not (workspace and openai_project):
        raise StageRefused("paid mode needs the study's workspace and OpenAI project")
    if mode == "paid" and transport.openai_project != openai_project:
        raise ValueError("the HTTPS transport is bound to another OpenAI project than this stage's")
    if mode == "paid" and not isinstance(counter, D.HttpsTokenCounter):
        raise ValueError("paid mode counts tokens only with the providers' own endpoints")
    if mode == "paid" and (counter.transport is not transport or counter.workspace != workspace):
        raise ValueError("the token counter is bound to another transport or workspace than this stage's")
    if mode == "fake" and isinstance(counter, D.HttpsTokenCounter):
        raise ValueError("fake mode counts no tokens with a provider")
    scopes = {"anthropic": workspace, "openai": openai_project or (FAKE_PROJECT if mode == "fake" else None)}
    config_hash = R.config_sha256(b_model=b_model, caching=spec.caching)
    rubric = R.config_record(b_model=b_model, caching=spec.caching)["rubric"]
    rubric_status = rubric["status"]
    if rubric_check and spec.name in FROZEN_RUBRIC_STAGES and rubric_status != "frozen":
        raise StageRefused(f"{stage} needs a frozen rubric; the rubric's status is {rubric_status!r}")
    allocation_record = load_allocation(allocation) if allocation else None
    if mode == "paid" and allocation_record is None:
        raise StageRefused("paid mode needs --allocation, the development allocation from "
                           "scripts/draw_allocation.py")
    if allocation_record is not None:
        check_recorded_allocation(allocation_record, allocation_sha256, mode)
        check_allocation(stage, items, allocation_record)
    elif allocation_sha256:
        raise StageRefused("--allocation-sha256 names the hash of --allocation, and no --allocation was given")
    run_root.mkdir(parents=True, exist_ok=True)
    panel_record = None
    if spec.name == "EVAL":
        panel_record = verify_panel(panel, config_hash, items, repo, allocation_record)
    run = run_record(run_root, mode, scopes, ledger)
    previous = check_previous(spec, run_root, ledger, mode, run)
    if panel_record is not None:
        check_panel_run(panel_record, run)
    if allocation_record is not None:
        check_same_allocation(stage, run_root, mode, allocation_record)
    if allocation_record is not None and spec.name == "P2":
        check_prompt_pass(items, run_root, allocation_record, rubric["sha256"])
    change_note = check_eval_config(run_root, config_hash, config_change_note) if spec.name == "EVAL" else None
    planned = panel_record["n"] if panel_record else spec.planned
    if not items or len(items) > planned:
        raise StageRefused(f"{stage} runs 1 to {planned} cases; {len(items)} were given")
    stage_dir = run_root / stage
    if (stage_dir / "report.json").exists():
        raise StageRefused(f"{stage} is already complete: {stage_dir / 'report.json'}")
    check_ledger_against_logs(spec, run_root, ledger, run["run_id"], mode)
    stage_dir.mkdir(parents=True, exist_ok=True)
    _fixed_record(stage_dir / "stage.json", {
        "stage": stage, "mode": mode, "config_sha256": config_hash, "rubric_sha256": rubric["sha256"],
        "b_model": b_model, "workspace": workspace,
        "openai_project": scopes["openai"],
        "ledger": {"budget": ledger.budget, "file": ledger.path.name, "id": ledger.ledger_id},
        "scripted_usage": getattr(transport, "profile", None), "panel": panel_record and panel_record["sha256"],
        "allocation": allocation_record and allocation_record["sha256"], "config_change_note": change_note,
        "cases": [[i.trace_key, i.pack_sha256] for i in items]}, "inputs")
    log_root = (stage_dir / "attempt_log").resolve()
    log = StageLog(log_root, run_root, stage, {"A": R.A_MODEL, "B": b_model})
    rebuilt = rebuild_orphans(log, ledger, items, run["run_id"], stage, config_hash, spec.slots)
    recovered = ledger.recover(run_roots={run["run_id"]: run_root})
    record_over_reservation_halts(log, ledger, run["run_id"], stage, config_hash, clock)
    if spec.budget == "evaluation" and ledger.reconcile_due(S.TRAJECTORY_MARGIN_S) is not None:
        raise StageRefused("the evaluation ledger has calls more than 23 hours old since its last reconciliation; "
                           "reconcile before resuming")
    record_unrecorded_halts(log, items, config_hash, spec.slots, clock)
    pending = log.unresolved_halts()
    if pending:
        raise StageRefused(f"{stage} has halts without a correction note: "
                           + ", ".join(h["halt_id"] for h in pending))
    settings = dict(settings or {}, workspace=workspace, openai_project=scopes["openai"], b_model=b_model,
                    rubric_status=rubric_status,
                    rubric_check=("applied" if rubric_check else "waived in a fake-mode dry run")
                    if spec.name in FROZEN_RUBRIC_STAGES else "not required")
    metered = S.MeteredTransport(transport, ledger, run["run_id"], stage, log_root, run_root=run_root, scopes=scopes)
    sleep = sleep or (lambda s: None)
    rows, stopped = [], None
    report_args = (spec, run, items, log, ledger, config_hash, mode, b_model, rebuilt)
    for item in items:
        if _slots_ended(log, item.trace_key, config_hash, spec.slots):
            rows.append({"trace_key": item.trace_key, "eligible": True})
            continue
        prep = R.prepare(item.pack, read_images(item.entries), b_model=b_model, caching=spec.caching)
        if not prep["eligible"]:
            rows.append({"trace_key": item.trace_key, "eligible": False, "reasons": prep["reasons"]})
            if spec.name == "EVAL":                 # eligibility was settled before the panel was frozen
                halt_id = log.write_halt({
                    "trace_key": item.trace_key, "config_sha256": config_hash, "slot": None, "attempt": None,
                    "halt_kind": "ineligible_frozen_task", "failed_checks": [], "error": None, "http_status": None,
                    "detail": "a task of the frozen panel is ineligible: " + "; ".join(prep["reasons"]),
                    "at": _now(clock), "resume_rule": RESUME_RULES["ineligible_frozen_task"]})
                stopped = {"halt_id": halt_id, "kind": "ineligible_frozen_task", "trace_key": item.trace_key,
                           "slot": None}
                break
            continue
        reqs, inputs = prep["requests"], {}
        for j in "AB":
            body = json.loads(reqs[j].body)
            problems = R.check_body(j, body, caching=spec.caching)
            leak = R.check_request(j, reqs[j].body, item.private)
            if problems or not leak["ok"]:
                raise SystemExit(f"{item.trace_key}: the {j} request fails its checks, so nothing is sent: "
                                 f"{problems or leak}")
            estimate = R.token_estimate(j, reqs[j], prep["image_sizes"])["input_tokens"]
            try:
                count = counter.count(reqs[j]) if counter is not None else None
                if count is None and mode != "fake":
                    raise D.TokenCountError(f"the {reqs[j].provider} token counter gave no count")
            except D.TokenCountError as e:      # nothing was sent or reserved, and no estimate stands in
                halt_id = log.write_halt({
                    "trace_key": item.trace_key, "config_sha256": config_hash, "slot": None, "attempt": None,
                    "halt_kind": "token_count", "failed_checks": [], "error": e.error, "http_status": e.status,
                    "detail": str(e), "at": _now(clock), "resume_rule": RESUME_RULES["token_count"]})
                stopped = {"halt_id": halt_id, "kind": "token_count", "trace_key": item.trace_key, "slot": None}
                break
            inputs[j] = {"request_sha256": reqs[j].sha256, "request_estimate": estimate,
                         "input_tokens": count if count is not None else S.estimated_input_tokens(estimate),
                         "input_source": "provider_count" if count is not None else "estimate",
                         "max_output_tokens": S.max_output_tokens(j, body)}
            del body
        if len(inputs) < 2:                     # a count failed and its halt is written
            rows.append({"trace_key": item.trace_key, "eligible": True})
            break
        rows.append({"trace_key": item.trace_key, "eligible": True, "fallback_applied": prep["fallback"]["applied"],
                     "inputs": {j: {k: v for k, v in inputs[j].items() if k != "request_sha256"} for j in "AB"}})
        del prep
        seq = len(ledger.entries)
        result = None
        try:
            metered.begin(item.trace_key, config_hash, log, spec.slots, inputs, reqs)
            if hasattr(transport, "begin"):
                transport.begin(spec, item.trace_key, config_hash, log)
            result = D.dispatch_trajectory(item.trace_key, reqs, config_hash, metered, log, workspace,
                                           sleep=sleep, clock=clock, slots=spec.slots, project=scopes["openai"])
        except S.SpendHalt as e:
            halt_id = log.write_halt({
                "trace_key": item.trace_key, "config_sha256": config_hash, "slot": e.entry.get("slot"),
                "attempt": e.entry.get("attempt"), "halt_kind": e.kind, "detail": str(e), "failed_checks": [],
                "error": None, "http_status": None, "at": _now(clock), "ledger_seq": e.entry["seq"],
                "resume_rule": RESUME_RULES.get(e.kind, "nothing was sent")})
            stopped = {"halt_id": halt_id, "kind": e.kind, "trace_key": item.trace_key, "slot": e.entry.get("slot")}
        finally:
            metered.finish()
            del reqs
        if result is not None and result["halted"] and stopped is None:
            halt = next(h for h in log.halts() if h["halt_id"] == result["halt_id"])
            stopped = {"halt_id": halt["halt_id"], "kind": halt["halt_kind"], "trace_key": item.trace_key,
                       "slot": halt["slot"]}
        over = [e for e in ledger.entries[seq:] if e["type"] == "settle" and e["over_reservation"]]
        if over:                                # recorded even when another halt ended the trajectory
            halt_id = log.write_halt({
                "trace_key": item.trace_key, "config_sha256": config_hash, "slot": None, "attempt": None,
                "halt_kind": "over_reservation", "reservations": [e["id"] for e in over], "failed_checks": [],
                "error": None, "http_status": None,
                "detail": f"{len(over)} call(s) settled above their reservation; the stage halts once the "
                          "trajectory finishes, and the cap counts the actual settled spend", "at": _now(clock),
                "resume_rule": RESUME_RULES["over_reservation"]})
            if stopped is None:
                stopped = {"halt_id": halt_id, "kind": "over_reservation", "trace_key": item.trace_key, "slot": None}
        report = stage_report(*report_args, rows, stopped, recovered, previous, panel_record, planned, settings)
        _write_progress(stage_dir, report)
        record_mark(run_root, ledger, mode, clock)
        if stopped is not None:
            break
    report = stage_report(*report_args, rows, stopped, recovered, previous, panel_record, planned, settings)
    _write_progress(stage_dir, report)
    record_mark(run_root, ledger, mode, clock)
    if report["complete"]:
        write_once(stage_dir / "report.json", _record_bytes(report))
    return report


def _write_progress(stage_dir: Path, report: dict) -> None:
    tmp = stage_dir / ".progress.json.tmp"
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, stage_dir / "progress.json")


# ---------------------------------------------------------------- report

def stage_report(spec, run, items, log, ledger, config_hash, mode, b_model, rebuilt, rows, stopped, recovered,
                 previous, panel_record, planned, settings) -> dict:
    names = [f"{j}{r}" for j, r in spec.slots]
    finals, records, b_records, ended = {s: [] for s in names}, [], [], 0
    ineligible = [r["trace_key"] for r in rows if not r["eligible"]]
    for item in items:
        done = True
        for j, r in spec.slots:
            recs = log.attempts(item.trace_key, config_hash, j, r)
            records += recs
            if j == "B":
                b_records += recs
            if D._slot_done(recs):
                finals[f"{j}{r}"].append(recs[-1])
            else:
                done = False
        ended += done or (spec.name != "EVAL" and item.trace_key in ineligible)

    invalid = {}
    for s, fs in finals.items():
        c = Counter(r["outcome"] for r in fs)
        bad = len(fs) - c[VALID]
        invalid[s] = {"ended": len(fs), "invalid": bad, "invalid_share": _share(bad, len(fs)),
                      "by_outcome": {o: {"n": c[o], "share": _share(c[o], len(fs))}
                                     for o in OUTCOMES if o not in (VALID, "HALT")}}
    if spec.caching:
        cache = {}
        for s in ("A2", "A3"):
            fs = [r for r in finals.get(s, []) if r.get("response_id")]
            u = [((r["usage"].get("cache_read_tokens") or 0), (r["usage"].get("cache_write_tokens") or 0)) for r in fs]
            hit = sum(rd > 0 and wr == 0 for rd, wr in u)
            partial = sum(rd > 0 and wr > 0 for rd, wr in u)
            cache[s] = {"responses": len(fs), "hit": hit, "partial": partial, "miss": len(fs) - hit - partial,
                        "hit_share": _share(hit, len(fs)),
                        "within_cache_lifetime_share": _share(sum(r["within_cache_lifetime"] is True for r in fs),
                                                              len(fs))}
    else:
        cache = {"not_applicable": "this stage sends no cache control"}

    spend = ledger.summary(run=run["run_id"], stage=spec.name)
    spent = Decimal(spend["selected_spent_usd"])
    per_case = spent / ended if ended else None

    def projected(cases):
        return None if per_case is None else S.usd((spent + per_case * (cases - ended)).quantize(Decimal("0.01")))

    reconciled = ledger.reconciled_after_last_call(run["run_id"], spec.name)
    spend.update(
        stage_spent_usd=spend.pop("selected_spent_usd"), stage_settled_usd=spend.pop("selected_settled_usd"),
        stage_held_usd=spend.pop("selected_held_usd"), projected_stage_usd=projected(len(items)),
        projected_at_planned_count_usd=projected(planned), reconciled_after_last_call=reconciled)
    capped = [{"trace_key": r["trace_key"], "slot": r["slot"], "attempt": r["attempt"],
               "output_tokens": r["usage"].get("output_tokens"), "stop": r["stop"], "detail": r["detail"]}
              for r in records
              if (r["usage"].get("output_tokens") or 0) >= R.OUTPUT_CAP
              or r["stop"] in ("max_tokens", "model_context_window_exceeded")
              or "max_output_tokens" in (r["detail"] or "")]
    halts = []
    for h in log.halts():
        note = log.correction(h["halt_id"])
        halts.append({"halt_id": h["halt_id"], "trace_key": h.get("trace_key"), "slot": h.get("slot"),
                      "kind": h.get("halt_kind"), "detail": h.get("detail"),
                      "correction": note["note"] if note else None})
    capability = dict(D.capability_check(b_records, b_model), acted_on=False,
                      note="reported only; the runner never changes a judge")
    finished = stopped is None and ended == len(items) and not log.unresolved_halts()
    needs_reconciliation = spec.budget == "evaluation" and not reconciled
    complete = finished and not needs_reconciliation
    return {
        "stage": spec.name, "about": spec.about, "mode": mode, "complete": complete, "stopped": stopped,
        "awaiting": ("a passing reconciliation after the stage's last call; reconcile, then run the same "
                     "command again" if finished and needs_reconciliation else None),
        "run_id": run["run_id"],
        "ledger": spend["ledger"],
        "settings": dict(settings or {}, config_sha256=config_hash, caching=spec.caching, slots=names,
                         budget=spec.budget, fence_profile=spec.profile),
        "previous_report": previous, "panel": panel_record,
        "cases": {"planned": planned, "given": len(items), "ended": ended, "ineligible": ineligible},
        "recovered_at_start": [{"id": e["id"], "type": e["type"]} for e in recovered],
        "attempts_rebuilt_at_start": rebuilt,
        "gates": {"note": GATES_NOTE, "invalid_share": invalid, "cache_hit_share": cache, "spend": spend,
                  "output_cap_reached": capped, "halts": halts, "capability_check_b": capability},
        "trajectories": rows,
    }


def _cents(amount) -> str:
    return "none" if amount is None else S.usd(Decimal(amount).quantize(Decimal("0.01")))


def print_report(report: dict) -> None:
    g = report["gates"]
    sp = g["spend"]
    print(f"{report['stage']}  {report['mode']}  cases {report['cases']['ended']}/{report['cases']['given']} "
          f"(planned {report['cases']['planned']})  config {report['settings']['config_sha256'][:12]}  "
          f"{'complete' if report['complete'] else 'NOT COMPLETE'}")
    if report.get("awaiting"):
        print(f"  awaiting        {report['awaiting']}")
    print("  invalid share   " + "  ".join(
        f"{s} {v['invalid']}/{v['ended']}" + ("" if not v["invalid"] else " (" + ", ".join(
            f"{o} {x['n']}" for o, x in v["by_outcome"].items() if x["n"]) + ")")
        for s, v in g["invalid_share"].items()))
    cache = g["cache_hit_share"]
    print("  cache hits      " + ("  ".join(f"{s} {v['hit']}/{v['responses']}" for s, v in cache.items())
                                  if "not_applicable" not in cache else "not applicable (uncached)"))
    print(f"  spend           stage ${_cents(sp['stage_spent_usd'])}  projected ${_cents(sp['projected_stage_usd'])}  "
          f"{sp['budget']} committed ${_cents(sp['budget_committed_usd'])} of ${_cents(sp['cap_usd'])} cap  "
          f"reservations {sp['reservations']} (held {sp['held']})  refusals {len(sp['refusals'])}")
    print("  output cap      " + (", ".join(f"{c['trace_key']} {c['slot']}#{c['attempt']}"
                                            for c in g["output_cap_reached"]) or "none"))
    print("  halts           " + ("; ".join(f"{h['halt_id']} {h['kind']} {h['slot']}: "
                                            f"{h['correction'] or 'no correction note yet'}" for h in g["halts"])
                                  or "none"))
    cap = g["capability_check_b"]
    print("  capability B    " + ", ".join(f"{k} {v}" for k, v in cap["checks"].items())
          + f"; fallback indicated {cap['fallback_indicated']}, not acted on")


# ---------------------------------------------------------------- commands

def _guard(profile: str, args) -> tuple:
    data_root = _outside_repo(Path(args.data_root), "--data-root")
    run_root = check_run_root(data_root, _outside_repo(Path(args.run_root), "--run-root"))
    fence.install(profile, data_root, fence.load_config(required=True))
    return data_root, run_root


def _budget_profile(budget: str) -> str:
    return "pre_join" if budget == "evaluation" else "dev_only"


def cmd_run(args) -> int:
    spec = STAGES[args.stage]
    data_root, run_root = _guard(spec.profile, args)
    for flag, value in (("--approval-note", args.approval_note), ("--panel-manifest", args.panel_manifest),
                        ("--allocation", args.allocation), ("--config-change-note", args.config_change_note)):
        if value:
            _outside_repo(Path(value), flag)
    if spec.name != "EVAL" and (args.panel_manifest or args.panel_sha256 or args.freeze_tag):
        raise StageRefused("only EVAL takes a panel manifest")
    if spec.name != "EVAL" and args.config_change_note:
        raise StageRefused("only EVAL takes --config-change-note")
    if args.scripted_usage and (args.paid or args.confirm_paid_calls or args.approval_note):
        raise StageRefused("--scripted-usage is for fake mode only")
    mode, paid = select_transport(args.paid, args.confirm_paid_calls, args.approval_note, args.workspace,
                                  args.openai_project)
    if mode == "fake":
        _forbid_network()
        transport = ScriptedProvider(args.scripted_usage or "central")
        counter, workspace, project = transport, FAKE_WORKSPACE, FAKE_PROJECT
        sleep = None
    else:
        transport, workspace, project, sleep = paid, args.workspace, args.openai_project, time.sleep
        counter = D.HttpsTokenCounter(transport, workspace, sleep=sleep)
    items = load_items(args.packs, args.provenance,
                       split="dev" if spec.budget == "pilot" else "test" if mode == "paid" else None)
    panel = ({"manifest": args.panel_manifest, "sha256": args.panel_sha256, "freeze_tag": args.freeze_tag}
             if spec.name == "EVAL" else None)
    config_env = os.environ.get(fence.CONFIG_ENV)
    note = Path(args.approval_note).expanduser() if mode == "paid" else None
    settings = {"python": platform.python_version(), "code": code_version(),
                "fence": {"profiles": fence.installed_profiles(),
                          "config_sha256": fetch.sha256_file(Path(config_env).expanduser()) if config_env else None},
                "transport": ("HttpsTransport, confirmed paid calls" if mode == "paid" else fake_transport_line()),
                "scripted_usage": (args.scripted_usage or "central") if mode == "fake" else None,
                "approval_note": {"path": str(note), "sha256": fetch.sha256_file(note)} if note else None,
                "inputs": {"packs": [str(Path(p).expanduser().resolve()) for p in args.packs],
                           "provenance": [str(Path(p).expanduser().resolve()) for p in args.provenance]}}
    with S.open_budget_ledger(ledger_root(data_root), spec.budget, mode, create=True,
                              new_budget_note=args.new_budget_note,
                              marks=peer_marks(data_root, run_root, spec.budget, mode)) as ledger:
        if mode == "fake":
            transport.period_start = ledger.created_at
            transport.export = export_path(ledger.path.parent, spec.budget, ledger.ledger_id)
        try:
            report = run_stage(spec.name, run_root, items, mode=mode, transport=transport, ledger=ledger,
                               read_images=lambda entries: R.read_images(entries, data_root), workspace=workspace,
                               openai_project=project, counter=counter, panel=panel, sleep=sleep, settings=settings,
                               b_model=args.b_model, allocation=args.allocation,
                               allocation_sha256=args.allocation_sha256,
                               config_change_note=args.config_change_note)
        finally:
            if mode == "fake":
                transport.write_export()
    print_report(report)
    return 0 if report["complete"] else 2


def cmd_reconcile(args) -> int:
    data_root, run_root = _guard(_budget_profile(args.budget), args)
    export = _outside_repo(Path(args.export), "--export")
    run = run_mode(run_root)
    try:
        rows, sha = S.read_usage_export(export)
    except ValueError as e:
        raise StageRefused(f"the export is refused: {e}") from None
    with S.open_budget_ledger(ledger_root(data_root), args.budget, run["mode"],
                              marks=peer_marks(data_root, run_root, args.budget, run["mode"])) as ledger:
        ledger.recover(run_roots={run["run_id"]: run_root})
        try:
            entry = ledger.reconcile(rows, sha, scopes={"anthropic": run.get("workspace"),
                                                        "openai": run.get("openai_project")})
        except ValueError as e:
            raise StageRefused(f"the export is refused: {e}") from None
        finally:
            record_mark(run_root, ledger, run["mode"])
    for p, v in entry["providers"].items():
        c = entry["coverage"].get(p) or {}
        print(f"{p}: export ${v['export_usd']}, ledger ${v['ledger_usd']}, excess share {v['excess_share']}; "
              f"{c.get('scope')} from {c.get('period_start')} to {c.get('period_end')}")
    print("RECONCILED" if entry["ok"] else "USAGE ABOVE THE LEDGER BY MORE THAN 1%: no further call is admitted "
          "until a correction note is recorded")
    return 0 if entry["ok"] else 3


def cmd_correct(args) -> int:
    if bool(args.stage) == bool(args.budget):
        raise StageRefused("give --stage with --halt for a halt, or --budget for a failed reconciliation")
    profile = STAGES[args.stage].profile if args.stage else _budget_profile(args.budget)
    data_root, run_root = _guard(profile, args)
    if args.stage:
        spec = STAGES[args.stage]
        stage_record = json.loads((run_root / spec.name / "stage.json").read_text(encoding="utf-8"))
        AttemptLog(run_root / spec.name / "attempt_log").record_correction(
            args.halt, args.note, stage_record["config_sha256"], at=_now())
    else:
        run = run_mode(run_root)
        with S.open_budget_ledger(ledger_root(data_root), args.budget, run["mode"],
                                  marks=peer_marks(data_root, run_root, args.budget, run["mode"])) as ledger:
            entry = ledger.record_correction(args.note)
            adjusted = [e for e in ledger.adjustments if e["reconcile_seq"] == entry["reconcile_seq"]]
            record_mark(run_root, ledger, run["mode"])
        for e in adjusted:
            print(f"{e['provider']}: ${e['cost_usd']} settled as an adjustment to the provider's reported usage")
    print("correction recorded")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--run-root", required=True, help="the run's folder, directly under <data-root>/out")
    common.add_argument("--data-root", required=True, help="local data directory the guard covers; the ledgers "
                        "live under its out/ledgers/")
    run = sub.add_parser("run", parents=[common], help="run or resume one stage")
    run.add_argument("--stage", required=True, choices=sorted(STAGES))
    run.add_argument("--packs", action="append", required=True, help="a scripts/build_packs.py output folder")
    run.add_argument("--provenance", action="append", required=True, help="its provenance.jsonl, one per --packs")
    run.add_argument("--scripted-usage", choices=sorted(USAGE_PROFILES), help="fake mode: tokens per call")
    run.add_argument("--b-model", default=R.B_MODEL, choices=(R.B_MODEL,) + R.B_FALLBACK_MODELS,
                     help="judge B's model; the fallback is a configuration change")
    run.add_argument("--paid", action="store_true", help="paid mode, with the two conditions below")
    run.add_argument("--confirm-paid-calls", action="store_true", help="passed to the HTTPS transport")
    run.add_argument("--approval-note", help="the file written after the mentor's written go-ahead")
    run.add_argument("--workspace", help="paid mode: the study's Anthropic workspace ID; ANTHROPIC_API_KEY must "
                     "be scoped to it")
    run.add_argument("--openai-project", help="paid mode: the study's OpenAI project ID, sent on every OpenAI request")
    run.add_argument("--new-budget-note", help="why a second ledger for this budget is started, after the first "
                     "was moved aside")
    run.add_argument("--panel-manifest", help="EVAL: the frozen panel manifest")
    run.add_argument("--panel-sha256", help="EVAL: the manifest's recorded SHA-256")
    run.add_argument("--freeze-tag", help="EVAL: the git tag of the frozen code")
    run.add_argument("--allocation", help="the development allocation from scripts/draw_allocation.py; paid mode "
                     "needs it at every stage")
    run.add_argument("--allocation-sha256", help="the allocation's SHA-256 as recorded in the plan or the approval "
                     "note; paid mode needs it")
    run.add_argument("--config-change-note", help="EVAL: a file declaring why its configuration differs from P3's")
    rec = sub.add_parser("reconcile", parents=[common], help="compare a usage export with a budget's ledger")
    rec.add_argument("--budget", required=True, choices=sorted(S.CAPS))
    rec.add_argument("--export", required=True, help="the usage export, .csv or .json, outside the repository")
    cor = sub.add_parser("correct", parents=[common], help="record a correction note")
    cor.add_argument("--stage", choices=sorted(STAGES))
    cor.add_argument("--halt", help="the halt id, with --stage")
    cor.add_argument("--budget", choices=sorted(S.CAPS), help="answer the budget's failed reconciliation")
    cor.add_argument("--note", required=True)
    args = parser.parse_args(argv)
    try:
        return {"run": cmd_run, "reconcile": cmd_reconcile, "correct": cmd_correct}[args.command](args)
    except S.LedgerError as e:
        raise StageRefused(f"ledger refused: {e}") from None


if __name__ == "__main__":
    sys.exit(main())
