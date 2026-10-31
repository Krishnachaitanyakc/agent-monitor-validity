#!/usr/bin/env python3
"""One runner for every stage: P1, P2 and P3 of the pilot, then the evaluation.

    export SPAR_FENCE_CONFIG=/path/to/local/fence_config.json
    python3 scripts/run_stage.py run --stage P1 --run-root ~/spar169-arb-data/out/pilot-1 \\
        --data-root ~/spar169-arb-data --packs ~/spar169-arb-data/out/packs-dev5 \\
        --provenance ~/spar169-arb-data/out/dev5/provenance.jsonl
    python3 scripts/run_stage.py reconcile --run-root ... --data-root ... --budget pilot --export usage.csv
    python3 scripts/run_stage.py correct --run-root ... --data-root ... --stage P1 --halt halt-0001 --note "..."
    python3 scripts/run_stage.py correct --run-root ... --data-root ... --budget pilot --note "..."

Stages. Each refuses to start without the previous stage's complete report, and a stage
after P1 also needs a passing reconciliation of the previous stage's ledger after its
last call:
  P1    five development cases, A1, A2, A3 then B1 each
  P2    one prompt pass on 20 development traces, A1 and B1, uncached
  P3    a frozen development pass on 20 tasks with the two planned repeats
  EVAL  the frozen panel; needs --panel-manifest, its recorded --panel-sha256 and --freeze-tag
P1 to P3 install the dev_only guard and share the pilot budget ($95); EVAL installs
pre_join and has its own budget ($285). The guard is installed before anything is read.

**Transport.** Paid mode needs --paid, --confirm-paid-calls (passed to the HTTPS
transport as ``confirm_paid_calls``) and --approval-note naming an existing, non-empty
file, together, plus --workspace. Without all three, only the fake transport runs: every
socket connection is refused, no API key is read, and responses carry scripted usage
(--scripted-usage central, high or extreme). Every call reserves on the budget's ledger
first (``judge.spend``), and the existing dispatcher sends A1, A2 and A3 serially, then B1.

**Outputs**, under --run-root (outside the repository):
  run.json                          the mode, fixed at the first stage
  ledger-pilot.jsonl, ledger-evaluation.jsonl     the write-ahead ledgers
  fake_provider_export-<budget>.csv the fake provider's usage, in fake mode
  <stage>/stage.json                the stage's inputs; a resumed stage must match them
  <stage>/attempt_log/              the write-once attempt log, halts and corrections
  <stage>/progress.json             the gates so far, rewritten after each trajectory
  <stage>/report.json               written once, when every case has ended
The report prints the gates for review and never acts on them to select a judge. A halt
stops the stage; rerun the same command after recording a correction note.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter, namedtuple
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from datapath import fence, fetch  # noqa: E402
from judge import dispatch as D  # noqa: E402
from judge import pack as P  # noqa: E402
from judge import requests as R  # noqa: E402
from judge import spend as S  # noqa: E402
from judge.outcomes import OUTCOMES, VALID, Exchange  # noqa: E402
from judge.storage import AttemptLog, write_once  # noqa: E402
from judge.streams import anthropic_events, openai_events, sse  # noqa: E402

Stage = namedtuple("Stage", "name about budget profile previous slots caching planned")
STAGES = {
    "P1": Stage("P1", "five development cases, A1, A2, A3 then B1 each", "pilot", "dev_only", None,
                R.SLOTS, True, 5),
    "P2": Stage("P2", "one prompt pass on 20 development traces, A1 and B1, uncached", "pilot", "dev_only",
                "P1", (("A", 1), ("B", 1)), False, 20),
    "P3": Stage("P3", "a frozen development pass on 20 tasks with the two planned repeats", "pilot",
                "dev_only", "P2", R.SLOTS, True, 20),
    "EVAL": Stage("EVAL", "the frozen panel", "evaluation", "pre_join", "P3", R.SLOTS, True, None),
}
USAGE_PROFILES = {"central": (60000, 3000), "high": (60000, 12000), "extreme": (100000, 20000)}
FAKE_WORKSPACE = "wrkspc_fake"
DAILY_RECONCILE_S = 24 * 3600
GATES_NOTE = ("Reported for review only. None of these gates selects or changes a judge, and a refusal "
              "is a recorded outcome, never a reason for a fallback.")

Item = namedtuple("Item", "trace_key pack pack_sha256 entries private")


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
    def hook(event, args):
        if event in ("socket.connect", "socket.getaddrinfo", "urllib.Request"):
            raise PermissionError(f"fake mode makes no network calls: {event}")
    sys.addaudithook(hook)


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


def ledger_path(run_root, budget: str) -> Path:
    return Path(run_root) / f"ledger-{budget}.jsonl"


def export_path(run_root, budget: str) -> Path:
    return Path(run_root) / f"fake_provider_export-{budget}.csv"


# ---------------------------------------------------------------- inputs

def load_items(packs, provenance, split="dev") -> list:
    """Items from scripts/build_packs.py folders, each with its provenance file (same order).

    ``split`` is the split every record must come from, or None (EVAL, where the panel
    manifest names the tasks).
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
            items.append(Item(key, pack, line["pack_sha256"], entries, private))
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


def select_transport(paid, confirm_paid_calls, approval_note, workspace=None):
    """("fake", None) when no paid condition is given; ("paid", HttpsTransport) when all
    three hold and a workspace is named; any other combination is refused before a
    transport exists."""
    if not (paid or confirm_paid_calls or approval_note):
        return "fake", None
    missing = paid_conditions(paid, confirm_paid_calls, approval_note)
    if missing:
        raise StageRefused("paid mode needs --paid, --confirm-paid-calls and --approval-note together; "
                           "missing: " + "; ".join(missing))
    if not workspace:
        raise StageRefused("paid mode also needs --workspace, the study's Anthropic workspace ID")
    return "paid", D.HttpsTransport(confirm_paid_calls=confirm_paid_calls)


class ScriptedProvider(D.FakeTransport):
    """The fake transport for fake mode: one provider-shaped response per request.

    Every call reports the profile's input and output tokens. An A request with cache
    control writes the cache on its first send and reads it afterwards. ``events`` maps
    (trace key, slot, attempt) to "timeout", "http_400", "refusal", "output_cap" or
    "cache_miss". Each response the fake provider bills is kept in ``billed`` and, with
    ``export_path``, appended to that usage export. ``count`` stands in for the
    provider's pre-send token count.
    """

    def __init__(self, profile="central", workspace=FAKE_WORKSPACE, events=None, export_path=None):
        super().__init__()
        self.profile = profile
        self.input_tokens, self.output_tokens = USAGE_PROFILES[profile]
        self.workspace, self.events, self.export = workspace, dict(events or {}), export_path
        self.cached, self.billed, self.ctx = set(), [], None

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

    @staticmethod
    def _answer(ref):
        verdict = "success" if int(hashlib.sha256(ref.encode()).hexdigest(), 16) % 2 else "failure"
        return json.dumps({"verdict": verdict, "p_failure": 0.3, "evidence_sufficient": "yes",
                           "evidence_step_ids": [0], "rationale": "A scripted answer from the fake provider."})

    def _bill(self, request, tokens) -> None:
        row = {"provider": request.provider, "model": request.model, "tokens": tokens, "cost_usd": None}
        self.billed.append(row)
        if self.export is not None:
            new = not Path(self.export).exists()
            with open(self.export, "a", encoding="utf-8", newline="") as f:
                w = csv.writer(f, lineterminator="\n")
                if new:
                    w.writerow(S.EXPORT_FIELDS)
                w.writerow([row["provider"], row["model"]] + [tokens[k] for k in S.TOKEN_KINDS] + [""])

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
            events = openai_events(None, refusal="I can't help with that.", resp_id=f"resp_{ref}", usage=usage)
        elif kind == "output_cap":
            events = openai_events(None, status="incomplete", incomplete_reason="max_output_tokens",
                                   resp_id=f"resp_{ref}", usage=usage)
        else:
            events = openai_events(self._answer(ref), resp_id=f"resp_{ref}", usage=usage)
        self._bill(request, {"input": inp, "cache_write": 0, "cache_read": 0, "output": out})
        return Exchange(200, {"x-request-id": f"req_{ref}"}, sse(events), None, None)


# ---------------------------------------------------------------- stage checks

def check_previous(spec: Stage, run_root: Path, ledger, mode: str):
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
        reconciled = ledger.reconciled_after_last_call(str(run_root), prev.name)
    elif not ledger_path(run_root, prev.budget).is_file():
        raise StageRefused(f"no {prev.budget} ledger in {run_root}")
    else:
        with S.Ledger(ledger_path(run_root, prev.budget), prev.budget) as prev_ledger:
            reconciled = prev_ledger.reconciled_after_last_call(str(run_root), prev.name)
    if not reconciled:
        raise StageRefused(f"reconcile the {prev.budget} ledger with the providers' usage after {prev.name} "
                           f"before starting {spec.name}")
    return str(path)


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


def verify_panel(panel, config_sha256: str, items: list, repo=REPO) -> dict:
    """EVAL's frozen panel manifest, checked against its recorded hash and freeze tag.

    The manifest is JSON: {"kind": "panel-manifest", "frozen": true, "freeze_tag",
    "config_sha256", "n", "tasks": [{"trace_key", "pack_sha256"}, ...]}. The packs given
    must be its tasks, in its order, and the code must be at the freeze tag.
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
    tasks = m.get("tasks")
    if not isinstance(tasks, list) or not tasks or m.get("n") != len(tasks):
        raise StageRefused("the manifest's task list and N do not agree")
    if [(t.get("trace_key"), t.get("pack_sha256")) for t in tasks] != [(i.trace_key, i.pack_sha256) for i in items]:
        raise StageRefused("the packs given are not the panel's tasks in the manifest's order")
    return {"manifest": str(path.resolve()), "sha256": panel["sha256"], "freeze_tag": panel["freeze_tag"],
            "freeze_commit": freeze_commit(repo, panel["freeze_tag"]), "n": len(tasks)}


def _fixed_record(path: Path, record: dict, what: str) -> None:
    """Write ``record`` once; later starts must present the same one."""
    data = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != record:
            raise StageRefused(f"{path} records different {what}; a resumed stage must use the same ones")
        return
    write_once(path, data)


def _slots_ended(log, key, config_sha256, slots) -> bool:
    return all(D._slot_done(log.attempts(key, config_sha256, j, r)) for j, r in slots)


# ---------------------------------------------------------------- run

def run_stage(stage: str, run_root, items: list, *, mode: str, transport, ledger, read_images, workspace,
              counter=None, panel=None, repo=REPO, sleep=None, clock=time.time, settings=None) -> dict:
    """Run one stage, or resume it, and return its report (see the module docstring)."""
    spec = STAGES[stage]
    run_root = Path(run_root)
    if ledger.budget != spec.budget:
        raise ValueError(f"{stage} spends from the {spec.budget} budget, not {ledger.budget}")
    run_root.mkdir(parents=True, exist_ok=True)
    config_hash = R.config_sha256(caching=spec.caching)
    panel_record = None
    if spec.name == "EVAL":
        panel_record = verify_panel(panel, config_hash, items, repo)
    _fixed_record(run_root / "run.json", {"mode": mode}, "a mode")
    previous = check_previous(spec, run_root, ledger, mode)
    planned = panel_record["n"] if panel_record else spec.planned
    if not items or len(items) > planned:
        raise StageRefused(f"{stage} runs 1 to {planned} cases; {len(items)} were given")
    stage_dir = run_root / stage
    if (stage_dir / "report.json").exists():
        raise StageRefused(f"{stage} is already complete: {stage_dir / 'report.json'}")
    stage_dir.mkdir(parents=True, exist_ok=True)
    _fixed_record(stage_dir / "stage.json", {
        "stage": stage, "mode": mode, "config_sha256": config_hash, "ledger": str(ledger.path),
        "scripted_usage": getattr(transport, "profile", None), "panel": panel_record and panel_record["sha256"],
        "cases": [[i.trace_key, i.pack_sha256] for i in items]}, "inputs")
    recovered = ledger.recover()
    if spec.budget == "evaluation":
        waiting = ledger.unreconciled_calls()
        if waiting and clock() - datetime.fromisoformat(waiting[0]["at"]).timestamp() > DAILY_RECONCILE_S:
            raise StageRefused("the evaluation ledger has calls older than a day since its last reconciliation; "
                               "reconcile before resuming")
    log_root = (stage_dir / "attempt_log").resolve()
    log = AttemptLog(log_root)
    pending = log.unresolved_halts()
    if pending:
        raise StageRefused(f"{stage} has halts without a correction note: "
                           + ", ".join(h["halt_id"] for h in pending))
    metered = S.MeteredTransport(transport, ledger, str(run_root), stage, log_root)
    sleep = sleep or (lambda s: None)
    rows, stopped = [], None
    for item in items:
        if _slots_ended(log, item.trace_key, config_hash, spec.slots):
            rows.append({"trace_key": item.trace_key, "eligible": True})
            continue
        prep = R.prepare(item.pack, read_images(item.entries), caching=spec.caching)
        if not prep["eligible"]:
            rows.append({"trace_key": item.trace_key, "eligible": False, "reasons": prep["reasons"]})
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
            count = counter.count(reqs[j]) if counter is not None else None
            inputs[j] = {"request_sha256": reqs[j].sha256, "request_estimate": estimate,
                         "input_tokens": count if count is not None else S.estimated_input_tokens(estimate),
                         "input_source": "provider_count" if count is not None else "estimate",
                         "max_output_tokens": S.max_output_tokens(j, body)}
            del body
        rows.append({"trace_key": item.trace_key, "eligible": True, "fallback_applied": prep["fallback"]["applied"],
                     "inputs": {j: {k: v for k, v in inputs[j].items() if k != "request_sha256"} for j in "AB"}})
        del prep
        metered.begin(item.trace_key, config_hash, log, spec.slots, inputs)
        if hasattr(transport, "begin"):
            transport.begin(spec, item.trace_key, config_hash, log)
        seq = len(ledger.entries)
        result = None
        try:
            result = D.dispatch_trajectory(item.trace_key, reqs, config_hash, metered, log, workspace,
                                           sleep=sleep, clock=clock, slots=spec.slots)
        except S.SpendHalt as e:
            halt_id = log.write_halt({
                "trace_key": item.trace_key, "config_sha256": config_hash, "slot": e.entry.get("slot"),
                "attempt": e.entry.get("attempt"), "halt_kind": e.kind, "detail": str(e), "failed_checks": [],
                "error": None, "http_status": None, "at": _now(clock), "ledger_seq": e.entry["seq"],
                "resume_rule": "nothing was sent; a note does not raise the cap, and the gate is applied again"})
            stopped = {"halt_id": halt_id, "kind": e.kind, "trace_key": item.trace_key, "slot": e.entry.get("slot")}
        finally:
            metered.finish()
            del reqs
        over = [e for e in ledger.entries[seq:] if e["type"] == "settle" and e["over_reservation"]]
        if over and stopped is None:
            halt_id = log.write_halt({
                "trace_key": item.trace_key, "config_sha256": config_hash, "slot": None, "attempt": None,
                "halt_kind": "over_reservation", "failed_checks": [], "error": None, "http_status": None,
                "detail": f"{len(over)} call(s) settled above their reservation", "at": _now(clock),
                "resume_rule": "check the token count behind the reservation before resuming"})
            stopped = {"halt_id": halt_id, "kind": "over_reservation", "trace_key": item.trace_key, "slot": None}
        if result is not None and result["halted"] and stopped is None:
            halt = next(h for h in log.halts() if h["halt_id"] == result["halt_id"])
            stopped = {"halt_id": halt["halt_id"], "kind": halt["halt_kind"], "trace_key": item.trace_key,
                       "slot": halt["slot"]}
        report = stage_report(spec, run_root, items, log, ledger, config_hash, mode, rows, stopped, recovered,
                              previous, panel_record, planned, settings)
        _write_progress(stage_dir, report)
        if stopped is not None:
            break
    report = stage_report(spec, run_root, items, log, ledger, config_hash, mode, rows, stopped, recovered,
                          previous, panel_record, planned, settings)
    _write_progress(stage_dir, report)
    if report["complete"]:
        write_once(stage_dir / "report.json", (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return report


def _write_progress(stage_dir: Path, report: dict) -> None:
    tmp = stage_dir / ".progress.json.tmp"
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, stage_dir / "progress.json")


# ---------------------------------------------------------------- report

def stage_report(spec, run_root, items, log, ledger, config_hash, mode, rows, stopped, recovered, previous,
                 panel_record, planned, settings) -> dict:
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
        ended += done or item.trace_key in ineligible

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

    spend = ledger.summary(run=str(run_root), stage=spec.name)
    spent = Decimal(spend["selected_spent_usd"])
    per_case = spent / ended if ended else None

    def projected(cases):
        return None if per_case is None else S.usd((spent + per_case * (cases - ended)).quantize(Decimal("0.01")))

    spend.update(
        stage_spent_usd=spend.pop("selected_spent_usd"), stage_settled_usd=spend.pop("selected_settled_usd"),
        stage_held_usd=spend.pop("selected_held_usd"), projected_stage_usd=projected(len(items)),
        projected_at_planned_count_usd=projected(planned))
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
    capability = dict(D.capability_check(b_records, R.B_MODEL), acted_on=False,
                      note="reported only; the runner never changes a judge")
    complete = stopped is None and ended == len(items) and not log.unresolved_halts()
    return {
        "stage": spec.name, "about": spec.about, "mode": mode, "complete": complete, "stopped": stopped,
        "settings": dict(settings or {}, config_sha256=config_hash, caching=spec.caching, slots=names,
                         budget=spec.budget, fence_profile=spec.profile),
        "previous_report": previous, "panel": panel_record,
        "cases": {"planned": planned, "given": len(items), "ended": ended, "ineligible": ineligible},
        "recovered_at_start": [{"id": e["id"], "type": e["type"]} for e in recovered],
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
    run_root = _outside_repo(Path(args.run_root), "--run-root")
    fence.install(profile, data_root, fence.load_config(required=True))
    return data_root, run_root


def cmd_run(args) -> int:
    spec = STAGES[args.stage]
    data_root, run_root = _guard(spec.profile, args)
    if spec.name != "EVAL" and (args.panel_manifest or args.panel_sha256 or args.freeze_tag):
        raise StageRefused("only EVAL takes a panel manifest")
    if args.scripted_usage and (args.paid or args.confirm_paid_calls or args.approval_note):
        raise StageRefused("--scripted-usage is for fake mode only")
    mode, paid = select_transport(args.paid, args.confirm_paid_calls, args.approval_note, args.workspace)
    if mode == "fake":
        _forbid_network()
        transport = ScriptedProvider(args.scripted_usage or "central",
                                     export_path=export_path(run_root, spec.budget))
        counter, workspace = transport, FAKE_WORKSPACE
        sleep = None
    else:
        transport, counter, workspace, sleep = paid, None, args.workspace, time.sleep
    items = load_items(args.packs, args.provenance, split="dev" if spec.budget == "pilot" else None)
    panel = ({"manifest": args.panel_manifest, "sha256": args.panel_sha256, "freeze_tag": args.freeze_tag}
             if spec.name == "EVAL" else None)
    config_env = os.environ.get(fence.CONFIG_ENV)
    note = Path(args.approval_note).expanduser() if mode == "paid" else None
    settings = {"python": platform.python_version(), "code": code_version(),
                "fence": {"profiles": fence.installed_profiles(),
                          "config_sha256": fetch.sha256_file(Path(config_env).expanduser()) if config_env else None},
                "transport": ("HttpsTransport, confirmed paid calls" if mode == "paid" else
                              "fake transport with scripted provider-shaped streams; every socket refused"),
                "scripted_usage": (args.scripted_usage or "central") if mode == "fake" else None,
                "approval_note": {"path": str(note), "sha256": fetch.sha256_file(note)} if note else None,
                "inputs": {"packs": [str(Path(p).expanduser().resolve()) for p in args.packs],
                           "provenance": [str(Path(p).expanduser().resolve()) for p in args.provenance]}}
    with S.Ledger(ledger_path(run_root, spec.budget), spec.budget) as ledger:
        report = run_stage(spec.name, run_root, items, mode=mode, transport=transport, ledger=ledger,
                           read_images=lambda entries: R.read_images(entries, data_root), workspace=workspace,
                           counter=counter, panel=panel, sleep=sleep, settings=settings)
    print_report(report)
    return 0 if report["complete"] else 2


def cmd_reconcile(args) -> int:
    _, run_root = _guard("pre_join" if args.budget == "evaluation" else "dev_only", args)
    rows, sha = S.read_usage_export(Path(args.export).expanduser())
    with S.Ledger(ledger_path(run_root, args.budget), args.budget) as ledger:
        ledger.recover()
        entry = ledger.reconcile(rows, sha)
    for p, v in entry["providers"].items():
        print(f"{p}: export ${v['export_usd']}, ledger ${v['ledger_usd']}, excess share {v['excess_share']}")
    print("RECONCILED" if entry["ok"] else "USAGE ABOVE THE LEDGER BY MORE THAN 1%: no further call is admitted "
          "until a correction note is recorded")
    return 0 if entry["ok"] else 3


def cmd_correct(args) -> int:
    if bool(args.stage) == bool(args.budget):
        raise StageRefused("give --stage with --halt for a halt, or --budget for a failed reconciliation")
    profile = (STAGES[args.stage].profile if args.stage else
               "pre_join" if args.budget == "evaluation" else "dev_only")
    _, run_root = _guard(profile, args)
    if args.stage:
        spec = STAGES[args.stage]
        AttemptLog(run_root / spec.name / "attempt_log").record_correction(
            args.halt, args.note, R.config_sha256(caching=spec.caching), at=_now())
    else:
        with S.Ledger(ledger_path(run_root, args.budget), args.budget) as ledger:
            ledger.record_correction(args.note)
    print("correction recorded")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--run-root", required=True, help="the run's folder, outside the repository")
    common.add_argument("--data-root", required=True, help="local data directory the guard covers")
    run = sub.add_parser("run", parents=[common], help="run or resume one stage")
    run.add_argument("--stage", required=True, choices=sorted(STAGES))
    run.add_argument("--packs", action="append", required=True, help="a scripts/build_packs.py output folder")
    run.add_argument("--provenance", action="append", required=True, help="its provenance.jsonl, one per --packs")
    run.add_argument("--scripted-usage", choices=sorted(USAGE_PROFILES), help="fake mode: tokens per call")
    run.add_argument("--paid", action="store_true", help="paid mode, with the two conditions below")
    run.add_argument("--confirm-paid-calls", action="store_true", help="passed to the HTTPS transport")
    run.add_argument("--approval-note", help="the file written after the mentor's written go-ahead")
    run.add_argument("--workspace", help="paid mode: the study's Anthropic workspace ID")
    run.add_argument("--panel-manifest", help="EVAL: the frozen panel manifest")
    run.add_argument("--panel-sha256", help="EVAL: the manifest's recorded SHA-256")
    run.add_argument("--freeze-tag", help="EVAL: the git tag of the frozen code")
    rec = sub.add_parser("reconcile", parents=[common], help="compare a provider usage export with a ledger")
    rec.add_argument("--budget", required=True, choices=sorted(S.CAPS))
    rec.add_argument("--export", required=True, help="the usage export, .csv or .json")
    cor = sub.add_parser("correct", parents=[common], help="record a correction note")
    cor.add_argument("--stage", choices=sorted(STAGES))
    cor.add_argument("--halt", help="the halt id, with --stage")
    cor.add_argument("--budget", choices=sorted(S.CAPS), help="answer the budget's failed reconciliation")
    cor.add_argument("--note", required=True)
    args = parser.parse_args(argv)
    return {"run": cmd_run, "reconcile": cmd_reconcile, "correct": cmd_correct}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
