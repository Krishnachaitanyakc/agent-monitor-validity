#!/usr/bin/env python3
"""Development dry run of the request layer: real requests, scripted responses, no network.

    export SPAR_FENCE_CONFIG=/path/to/local/fence_config.json
    python3 scripts/dry_run.py --data-root ~/spar169-arb-data \\
        --packs ~/spar169-arb-data/out/packs-dev5 --provenance ~/spar169-arb-data/out/dev5/provenance.jsonl \\
        --example-trace-key t48af8c403a820068 --out ~/spar169-arb-data/out/dry-run-1

The development-only guard is installed over --data-root before anything is read, and a
second audit hook refuses every socket connection and URL request, so the run cannot reach
any provider. Each --packs folder is the output of scripts/build_packs.py, and its
--provenance file (same order) supplies the identifiers the leak check looks for.

For every development pack the run reads its screenshots through the private image
manifest, checks each file's hash, builds the A and B requests twice and compares the
bytes, applies the fit check and the shared fallback, checks both bodies against the
request-field table and for leaks, and then dispatches A1, A2, A3 and B1 through
FakeTransport. The fake's scripted responses cycle through 13 scenarios that together
cover every outcome class, the retry path and the halt and resume path, so a run over 13
or more packs exercises all of them; a scripted halt gets a correction note that says so.

Outputs go to a staging folder next to --out and are published in one rename, only if
every check passes; --out must not exist yet. They are never committed:
  report.json             settings, config hash, per-pack request hashes, sizes, image
                          counts, token estimates, fit, checks, scenario and outcome counts
  request_hashes.jsonl    one line per request; byte-identical across runs on the same inputs
  example_request_A.json  the full A request for --example-trace-key, with each image's
  example_request_B.json  base64 replaced by its length and SHA-256
  attempt_log/            the write-once attempt log of the scripted dispatch
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from datapath import fence, fetch  # noqa: E402
from judge import pack as P  # noqa: E402
from judge import requests as R  # noqa: E402
from judge import dispatch as D  # noqa: E402
from judge.outcomes import OUTCOMES, TERMINAL, Exchange  # noqa: E402
from judge.storage import AttemptLog  # noqa: E402
from judge.streams import anthropic_events, openai_events, sse  # noqa: E402

WORKSPACE = "wrkspc_dryrun"


def _outside_repo(path: Path, flag: str) -> Path:
    path = path.expanduser().resolve()
    if path == REPO or REPO in path.parents:
        raise SystemExit(f"{flag} must be outside the repository: {path}")
    return path


def _forbid_network() -> None:
    def hook(event, args):
        if event in ("socket.connect", "socket.getaddrinfo", "urllib.Request"):
            raise PermissionError(f"the dry run makes no network calls: {event}")
    sys.addaudithook(hook)


def _code_version() -> dict:
    try:
        commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], check=True,
                                capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], check=True,
                                    capture_output=True, text=True).stdout.strip())
        return {"commit": commit, "uncommitted_changes": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "uncommitted_changes": None}


def _jsonl(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------- scripted responses

class Script:
    """Provider-shaped synthetic exchanges, each with a unique response id."""

    def __init__(self):
        self.n = 0

    def _id(self, prefix):
        self.n += 1
        return f"{prefix}_dryrun_{self.n:05d}"

    @staticmethod
    def answer(verdict="success", p_failure=0.2, sufficient="yes", ids=(0,), rationale=None):
        return json.dumps({"verdict": verdict, "p_failure": p_failure, "evidence_sufficient": sufficient,
                           "evidence_step_ids": list(ids),
                           "rationale": rationale or "A scripted answer for the dry run."})

    def a(self, text=None, stop="end_turn", cache="read", model=R.A_MODEL, verdict="success"):
        text = self.answer(verdict) if text is None and stop == "end_turn" else text
        usage = {"input_tokens": 40, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
        if cache == "write":
            usage.update(cache_creation_input_tokens=50000,
                         cache_creation={"ephemeral_5m_input_tokens": 50000, "ephemeral_1h_input_tokens": 0})
        elif cache == "read":
            usage.update(cache_read_input_tokens=50000)
        events = anthropic_events(text, stop_reason=stop, model=model, msg_id=self._id("msg"), usage=usage)
        return Exchange(200, {"anthropic-workspace-id": WORKSPACE, "request-id": self._id("req")},
                        sse(events), None, None)

    def a_cut(self):
        raw = sse(anthropic_events(self.answer(), msg_id=self._id("msg")))
        return Exchange(200, {"anthropic-workspace-id": WORKSPACE}, raw[:len(raw) // 2],
                        "IncompleteRead: connection closed", None)

    def a_stream_error(self):
        events = anthropic_events(None, msg_id=self._id("msg"))[:2]
        events.append(("error", {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}))
        return Exchange(200, {"anthropic-workspace-id": WORKSPACE}, sse(events), None, None)

    def b(self, text=None, status="completed", reason=None, refusal=None, error=None, verdict="success"):
        if text is None and status == "completed" and refusal is None:
            text = self.answer(verdict)
        events = openai_events(text, status=status, resp_id=self._id("resp"), incomplete_reason=reason,
                               refusal=refusal, error=error)
        return Exchange(200, {"x-request-id": self._id("req")}, sse(events), None, None)

    @staticmethod
    def a_error(status, etype, message, code=None):
        err = {"type": etype, "message": message}
        if code:
            err["details"] = {"error_code": code}
        return Exchange(status, {"request-id": "req_dryrun_error"},
                        json.dumps({"type": "error", "error": err}).encode(), None, None)

    @staticmethod
    def b_error(status, code, etype, message):
        return Exchange(status, {"x-request-id": "req_dryrun_error"}, json.dumps(
            {"error": {"message": message, "type": etype, "param": None, "code": code}}).encode(), None, None)

    @staticmethod
    def no_response():
        return Exchange(None, {}, b"", "TimeoutError: timed out", None)


def scenario(i: int, s: Script):
    """(name, exchanges in send order) for the i-th pack; covers every outcome class."""
    malformed = '{"verdict": "success", "p_failure": 0.1}'
    abstain = Script.answer(verdict="unsure")
    void_p = Script.answer(verdict="failure", p_failure=1.7)
    table = [
        ("all valid", [s.a(cache="write"), s.a(), s.a(verdict="failure"), s.b()]),
        ("A1 timeout then valid; A3 malformed; B1 refusal",
         [s.no_response(), s.a(cache="write"), s.a(), s.a(text=malformed), s.b(refusal="I can't help with that.")]),
        ("A1 refusal; B1 filtered",
         [s.a(text="", stop="refusal", cache="write"), s.a(), s.a(), s.b(status="incomplete", reason="content_filter")]),
        ("A1 truncated; A2 abstain; A3 p_failure voided; B1 truncated",
         [s.a(text=None, stop="max_tokens", cache="write"), s.a(text=abstain), s.a(text=void_p),
          s.b(status="incomplete", reason="max_output_tokens")]),
        ("A1 four timeouts, invalid; A2 cache miss",
         [s.no_response() for _ in range(4)] + [s.a(cache="write"), s.a(), s.b()]),
        ("A2 HTTP 400 halts, then resumes",
         [s.a(cache="write"), Script.a_error(400, "invalid_request_error", "messages.0.content: bad block"),
          s.a(), s.a(), s.b()]),
        ("B1 credit balance exhausted halts, then resumes",
         [s.a(cache="write"), s.a(), s.a(),
          Script.b_error(429, "credit_balance_exhausted", "insufficient_quota", "No credits remain."), s.b()]),
        ("A1 HTTP 413 halts, then resumes",
         [Script.a_error(413, "request_too_large", "Request exceeds the maximum size"),
          s.a(cache="write"), s.a(), s.a(), s.b()]),
        ("cut stream, 529 and 503 retried",
         [s.a_cut(), s.a(cache="write"), Script.a_error(529, "overloaded_error", "Overloaded"), s.a(), s.a(),
          Script.b_error(503, "server_is_overloaded", "service_unavailable_error", "Overloaded"), s.b()]),
        ("A3 model string differs, halts, then resumes",
         [s.a(cache="write"), s.a(), s.a(model="claude-sonnet-5-changed"), s.a(), s.b()]),
        ("A1 spend cap halts, then resumes; B1 policy failure filtered",
         [Script.a_error(429, "rate_limit_error", "You have reached your API usage limits: monthly cap.",
                         code="enforced_spend_limit_reached"),
          s.a(cache="write"), s.a(), s.a(), s.b(status="failed", error={"code": "invalid_prompt", "message": "flagged"})]),
        ("B1 malformed", [s.a(cache="write"), s.a(verdict="failure"), s.a(), s.b(text='{"verdict": "success"')]),
        ("A1 stream error retried; B1 abstain",
         [s.a_stream_error(), s.a(cache="write"), s.a(), s.a(), s.b(text=abstain)]),
    ]
    return table[i % len(table)]


# ---------------------------------------------------------------- run

def run(args) -> int:
    data_root = _outside_repo(Path(args.data_root), "--data-root")
    out = _outside_repo(Path(args.out), "--out")
    if out.exists():
        raise SystemExit(f"--out {out} already exists; outputs are never overwritten, so choose a new folder")
    if len(args.packs) != len(args.provenance):
        raise SystemExit("give one --provenance file for each --packs folder, in the same order")
    fence_config = fence.load_config(required=True)
    fence.install("dev_only", data_root, fence_config)
    _forbid_network()
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.staging-", dir=out.parent))
    try:
        ok = _run_into(args, data_root, staging)
    except BaseException:
        print(f"run failed; nothing was published to {out}; partial files are in {staging}", file=sys.stderr)
        raise
    if not ok:
        print(f"CHECKS FAILED; nothing was published to {out}; the report is {staging / 'report.json'}")
        return 1
    staging.rename(out)
    print(f"ALL CHECKS PASS; outputs published to {out}")
    return 0


def _load(args):
    """[(trace key, pack, manifest entries, provenance)] over every --packs folder, in order."""
    items = []
    for folder, prov_path in zip(args.packs, args.provenance):
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
            if prov is None or prov.get("split") != "dev":
                raise SystemExit(f"{key}: no development-split provenance record; this run reads "
                                 "development packs only")
            entries = manifest.get(key, [])
            if any(e["pack_sha256"] != line["pack_sha256"] for e in entries):
                raise ValueError(f"{key}: the image manifest belongs to another pack")
            items.append((key, pack, line["pack_sha256"], entries, prov))
    keys = [k for k, *_ in items]
    if len(set(keys)) != len(keys):
        raise ValueError("a trace key appears twice across the --packs folders")
    return items


def _run_into(args, data_root: Path, out: Path) -> bool:
    items = _load(args)
    config = R.config_record()
    config_hash = R.config_sha256()
    log = AttemptLog(out / "attempt_log")
    script = Script()
    hashes = open(out / "request_hashes.jsonl", "w", encoding="utf-8", newline="\n")
    summary, all_ok, totals, slot_totals, b_records = [], True, Counter(), Counter(), []
    cache = Counter()
    example = None
    for i, (key, pack, pack_hash, entries, prov) in enumerate(items):
        images = R.read_images(entries, data_root)
        prep = R.prepare(pack, images)
        again = R.prepare(json.loads(P.serialize(pack)), dict(images))
        deterministic = prep["eligible"] == again["eligible"] and (
            not prep["eligible"] or all(prep["requests"][j].body == again["requests"][j].body for j in "AB"))
        private = sorted(set(P.private_strings(prov)) | set(R.manifest_strings(entries)) | {key, pack_hash})
        row = {"trace_key": key, "pack_sha256": pack_hash, "steps": len(pack["steps"]),
               "images": len(entries), "native_max_side": max((max(s) for s in prep["native_sizes"]), default=0),
               "sent_max_side": max((max(s) for s in prep["image_sizes"]), default=0),
               "fit": {"eligible": prep["eligible"], "reasons": prep["reasons"],
                       "fallback_applied": prep["fallback"]["applied"],
                       "fallback_triggered_by": prep["fallback"]["triggered_by"]},
               "deterministic": deterministic}
        if not prep["eligible"]:
            row["ok"] = deterministic
            all_ok &= deterministic
            summary.append(row)
            continue
        reqs = prep["requests"]
        row["requests"], checks_ok = {}, deterministic
        for j in "AB":
            body = json.loads(reqs[j].body)
            leak = R.check_request(j, reqs[j].body, private)
            table = R.check_body(j, body)
            texts = [v for kind, v in R.neutral_content(j, body) if kind == "text"]
            rebuilt = P.serialize(R.rebuild_pack(texts)) == P.serialize(pack)
            row["requests"][j] = {"sha256": reqs[j].sha256, "body_bytes": len(reqs[j].body),
                                  "token_estimate": R.token_estimate(j, reqs[j], prep["image_sizes"]),
                                  "leak_check": leak, "table_problems": table, "rebuilds_pack": rebuilt}
            checks_ok &= leak["ok"] and not table and rebuilt
            hashes.write(P.serialize({"trace_key": key, "judge": j, "request_sha256": reqs[j].sha256,
                                      "body_bytes": len(reqs[j].body)}) + "\n")
        a_parts, b_parts = (R.neutral_content(j, json.loads(reqs[j].body)) for j in "AB")
        row["same_evidence_both_judges"] = a_parts == b_parts
        checks_ok &= a_parts == b_parts
        if key == args.example_trace_key:
            example = (key, reqs)

        # Scripted dispatch through the fake transport, resuming after each scripted halt.
        name, exchanges = scenario(i, script)
        transport = D.FakeTransport(exchanges)
        halts = []
        while True:
            result = D.dispatch_trajectory(key, reqs, config_hash, transport, log, WORKSPACE,
                                           sleep=lambda s: None)
            if not result["halted"]:
                break
            halt = next(h for h in log.halts() if h["halt_id"] == result["halt_id"])
            halts.append({"halt_id": halt["halt_id"], "slot": halt["slot"], "kind": halt["halt_kind"]})
            log.record_correction(halt["halt_id"], f"Dry run: the {halt['halt_kind']} halt on {halt['slot']} "
                                  "was scripted; nothing was changed, and the run resumed under the same "
                                  "configuration.", config_hash, at="dry run")
        sent_a = {r.sha256 for r in transport.sent if r.judge == "A"}
        sent_b = {r.sha256 for r in transport.sent if r.judge == "B"}
        identical = sent_a == {reqs["A"].sha256} and sent_b == {reqs["B"].sha256}
        checks_ok &= identical and not transport.script
        counts = Counter()
        for judge, repeat in R.SLOTS:
            records = log.attempts(key, config_hash, judge, repeat)
            counts.update(r["outcome"] for r in records)
            if judge == "B":
                b_records += records
            last = records[-1]
            if judge == "A" and last["outcome"] in TERMINAL:
                u = last["usage"]
                cache[f"A{repeat}:" + ("read" if u.get("cache_read_tokens") else
                                       "write" if u.get("cache_write_tokens") else "none")] += 1
        totals.update(counts)
        slot_totals.update(f"{slot}:{res['outcome']}" for slot, res in result["slots"].items())
        row.update(scenario=name, slots=result["slots"], halts=halts, outcome_counts=dict(sorted(counts.items())),
                   sent_bytes_identical_per_judge=identical)
        row["ok"] = checks_ok
        all_ok &= checks_ok
        summary.append(row)
    hashes.close()

    # Example requests for the mentor, with image data replaced by length and SHA-256.
    examples = {}
    if args.example_trace_key:
        if example is None:
            raise SystemExit(f"--example-trace-key {args.example_trace_key} is not among the eligible packs")
        key, reqs = example
        for j in "AB":
            body = json.loads(reqs[j].body)
            red = R.redacted(j, body)
            b64 = [v for kind, v in R.neutral_content(j, body) if kind == "image"]
            restored = R.restore(j, red, b64) == reqs[j].body
            all_ok &= restored
            doc = {"note": "The exact request body, with each image's base64 data replaced by its length "
                           "and the SHA-256 of the base64 text. Keys appear in the canonical order of the "
                           "sent bytes. The credential header is added at send time and is not shown.",
                   "judge": j, "endpoint": reqs[j].url, "headers": dict(reqs[j].headers),
                   "request_sha256": reqs[j].sha256, "body_bytes": len(reqs[j].body),
                   "config_sha256": config_hash, "images_replaced": len(b64),
                   "placeholders_restore_the_exact_bytes": restored, "body": red}
            path = out / f"example_request_{j}.json"
            path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            examples[j] = {"file": path.name, "request_sha256": reqs[j].sha256, "restores": restored}

    capability = D.capability_check(b_records, R.B_MODEL)
    covered = {o: totals[o] for o in OUTCOMES}
    report = {
        "ok": all_ok,
        "settings": {"python": platform.python_version(), "code": _code_version(),
                     "fence": {"profiles": fence.installed_profiles(),
                               "config_sha256": (fetch.sha256_file(Path(os.environ[fence.CONFIG_ENV]).expanduser())
                                                 if os.environ.get(fence.CONFIG_ENV) else None)},
                     "network": "every socket connection and URL request refused by an audit hook",
                     "transport": "FakeTransport with scripted provider-shaped streams",
                     "inputs": {"packs": [str(Path(p).expanduser().resolve()) for p in args.packs],
                                "provenance": [str(Path(p).expanduser().resolve()) for p in args.provenance]}},
        "config_sha256": config_hash,
        "config": config,
        "packs": len(summary),
        "eligible": sum(r["fit"]["eligible"] for r in summary),
        "fallback_applied": sum(r["fit"]["fallback_applied"] for r in summary),
        "deterministic": all(r["deterministic"] for r in summary),
        "outcome_counts_all_attempts": covered,
        "outcome_classes_not_exercised": [o for o in OUTCOMES if not covered[o]],
        "slot_results": dict(sorted(slot_totals.items())),
        "cache_use_on_final_a_attempts": {"note": "from the scripted usage; real counts come from the "
                                                  "providers' responses", **dict(sorted(cache.items()))},
        "halts": len(log.halts()), "corrections": sum(log.correction(h["halt_id"]) is not None for h in log.halts()),
        "capability_check_b": capability,
        "examples": examples,
        "summary": summary,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for r in summary:
        req = r.get("requests") or {}
        print(f"{r['trace_key']}  images={r['images']:<3} max_side={r['sent_max_side']:<5} "
              + "  ".join(f"{j}={req[j]['sha256'][:12]} {req[j]['body_bytes']:>9}B ~{req[j]['token_estimate']['input_tokens']}tok"
                          for j in req)
              + f"  {'ok' if r['ok'] else 'FAIL'}")
    print(f"packs: {len(summary)}; config {config_hash[:12]}; outcomes "
          + " ".join(f"{o}={covered[o]}" for o in OUTCOMES) + f"; halts {report['halts']}")
    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, help="local data directory the guard covers")
    parser.add_argument("--packs", action="append", required=True,
                        help="a scripts/build_packs.py output folder; repeat for several")
    parser.add_argument("--provenance", action="append", required=True,
                        help="the provenance.jsonl those packs were built from; one per --packs")
    parser.add_argument("--example-trace-key", default=None,
                        help="trace key of the pack whose two requests are written as examples")
    parser.add_argument("--out", required=True, help="new output directory outside the repository")
    return run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
