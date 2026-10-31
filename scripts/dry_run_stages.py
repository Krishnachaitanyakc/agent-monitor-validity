#!/usr/bin/env python3
"""Dry run of the stage runner: P1, P2 and P3 through the fake transport, with the ledger.

    export SPAR_FENCE_CONFIG=/path/to/local/fence_config.json
    python3 scripts/dry_run_stages.py --data-root ~/spar169-arb-data \\
        --packs ~/spar169-arb-data/out/packs-dev5 --provenance ~/spar169-arb-data/out/dev5/provenance.jsonl \\
        --packs ~/spar169-arb-data/out/packs-long14 --provenance ~/spar169-arb-data/out/long14/provenance.jsonl \\
        --out ~/spar169-arb-data/out/dry-run-stages-1

The development-only guard is installed over --data-root before anything is read, and
every socket connection is refused. P1 runs the packs of the first --packs folder (the
five development cases); P2 and P3 run every pack given, in order. Each call reports
scripted usage, and the reservations use the scripted count as the provider's count:

  central/    60,000 input and 3,000 output tokens per call
  extreme/    100,000 input and 20,000 output tokens per call
  cap-check/  the extreme pilot, all calls valid, repeated on one pilot ledger in new run
              folders until the $95 cap halts it

Each scenario keeps its pilot ledger in its own ledgers/ folder, apart from its run
folders, as the runner keeps one ledger per budget for every run. The rubric is still a
draft, so P3 runs here as a fake-mode rehearsal with the runner's frozen-rubric rule
waived; the summary says so, and the runner itself never waives it.

In central and extreme the same scripted events run, keyed by each stage's case order:
P1 case 3's A2 misses the cache; P2 case 2's A1 times out once, case 5's B1 refuses and
case 8's A1 gets an HTTP 400, which halts the stage until a correction note is recorded;
P3 case 4's A3 and case 7's B1 reach the output cap. After each stage the fake provider's
usage export, stating its period from the ledger's creation and each provider's scope, is
reconciled with the ledger, and the run checks that the ledger's settled spend equals
that export exactly.

Outputs go to a staging folder next to --out and are published in one rename, only if
every check passes; --out must not exist yet. They are never committed:
  summary.json   per scenario and stage: cases, spend, projection, the gates, the
                 reconciliation, the reservations, and where the cap check halted
  <scenario>/    ledgers/ (the pilot ledger and the fake usage export) and the run
                 folders: attempt logs and stage reports
"""

import argparse
import json
import os
import platform
import sys
import tempfile
from decimal import Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from datapath import fence, fetch  # noqa: E402
from judge import requests as R  # noqa: E402
from judge import spend as S  # noqa: E402
from judge.storage import AttemptLog  # noqa: E402
import run_stage as RS  # noqa: E402

PILOT = ("P1", "P2", "P3")
EVENTS = {"P1": [(2, "A2", 1, "cache_miss")],
          "P2": [(1, "A1", 1, "timeout"), (4, "B1", 1, "refusal"), (7, "A1", 1, "http_400")],
          "P3": [(3, "A3", 1, "output_cap"), (6, "B1", 1, "output_cap")]}
MAX_PILOTS = 50


def stage_items(stage, first, everything):
    return first if stage == "P1" else everything


def events_for(stage, items):
    return {(items[i].trace_key, slot, attempt): kind for i, slot, attempt, kind in EVENTS[stage] if i < len(items)}


def run_pilot(root, ledger, profile, first, everything, read_images, export, with_events, settings):
    """P1, P2 and P3 on one ledger, reconciling after each; stops at a halt that is not scripted."""
    stages, ok = {}, True
    for stage in PILOT:
        items = stage_items(stage, first, everything)
        provider = RS.ScriptedProvider(profile, events=events_for(stage, items) if with_events else None,
                                       export_path=export)
        provider.period_start = ledger.created_at
        corrections = []
        while True:
            try:
                report = RS.run_stage(stage, root, items, mode="fake", transport=provider, ledger=ledger,
                                      read_images=read_images, workspace=RS.FAKE_WORKSPACE, counter=provider,
                                      settings=settings, rubric_check=False)
            finally:
                provider.write_export()
            stop = report["stopped"]
            if stop is None or stop["kind"] != "request_error" or not with_events:
                break
            note = (f"Dry run: the HTTP 400 on {stop['slot']} was scripted; nothing was changed, and the "
                    "stage resumed under the same configuration.")
            AttemptLog(Path(root) / stage / "attempt_log").record_correction(
                stop["halt_id"], note, report["settings"]["config_sha256"], at="dry run")
            corrections.append(stop["halt_id"])
        entry, exported, settled = None, None, None
        if report["complete"]:
            rows, sha = S.read_usage_export(export)
            entry = ledger.reconcile(rows, sha)
            exported = sum((S.export_usd(r) for r in rows), Decimal(0))
            settled = ledger.totals()["settled"]
            ok &= entry["ok"] and exported == settled
        stages[stage] = {"report": report, "reconciliation": entry, "corrections": corrections,
                         "export_equals_settled": entry is not None and exported == settled}
        if not report["complete"]:
            break
    return stages, ok


def _brief(report):
    g = report["gates"]
    sp = g["spend"]
    return {"complete": report["complete"], "stopped": report["stopped"], "cases": report["cases"],
            "config_sha256": report["settings"]["config_sha256"],
            "invalid_share": {s: {"invalid": v["invalid"], "ended": v["ended"],
                                  "by_outcome": {o: x["n"] for o, x in v["by_outcome"].items() if x["n"]}}
                              for s, v in g["invalid_share"].items()},
            "cache_hit_share": g["cache_hit_share"],
            "spend": {k: sp[k] for k in ("stage_spent_usd", "stage_settled_usd", "stage_held_usd",
                                         "projected_stage_usd", "projected_at_planned_count_usd",
                                         "budget_committed_usd", "headroom_usd", "cap_usd", "reservations",
                                         "held", "reserved_usd", "largest_reservation_usd", "refusals",
                                         "over_reservation")},
            "output_cap_reached": g["output_cap_reached"], "halts": g["halts"],
            "capability_check_b": {k: g["capability_check_b"][k] for k in ("checks", "fallback_indicated",
                                                                          "acted_on")}}


def _cents(s):
    return f"{Decimal(s).quantize(Decimal('0.01'))}"


def run(args) -> int:
    data_root = RS._outside_repo(Path(args.data_root), "--data-root")
    out = RS._outside_repo(Path(args.out), "--out")
    if out.exists():
        raise SystemExit(f"--out {out} already exists; outputs are never overwritten, so choose a new folder")
    fence.install("dev_only", data_root, fence.load_config(required=True))
    RS._forbid_network()
    first = RS.load_items(args.packs[:1], args.provenance[:1])
    everything = RS.load_items(args.packs, args.provenance)
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.staging-", dir=out.parent))
    try:
        ok, summary = _run_into(staging, data_root, first, everything, args)
    except BaseException:
        print(f"run failed; nothing was published to {out}; partial files are in {staging}", file=sys.stderr)
        raise
    (staging / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not ok:
        print(f"CHECKS FAILED; nothing was published to {out}; the summary is {staging / 'summary.json'}")
        return 1
    staging.rename(out)
    print(f"ALL CHECKS PASS; outputs published to {out}")
    return 0


def _run_into(staging, data_root, first, everything, args):
    read_images = lambda entries: R.read_images(entries, data_root)  # noqa: E731
    settings = {"python": platform.python_version(), "code": RS.code_version(),
                "fence": {"profiles": fence.installed_profiles(),
                          "config_sha256": fetch.sha256_file(Path(os.environ[fence.CONFIG_ENV]).expanduser())},
                "transport": RS.fake_transport_line()}
    summary = {"settings": settings, "price_table": S.PRICE_TABLE_VERSION, "pricing_sha256": S.pricing_sha256(),
               "reservation_rule": S.RESERVATION_RULE,
               "rubric": {"status": R.RUBRIC_STATUS, "frozen_rubric_rule": "waived for this fake-mode rehearsal"},
               "events": {k: [list(e) for e in v] for k, v in EVENTS.items()},
               "packs": {"P1": len(first), "P2": len(everything), "P3": len(everything)}, "scenarios": {}}
    all_ok = True

    for profile in ("central", "extreme"):
        root, ledgers = staging / profile / "run", staging / profile / "ledgers"
        with S.Ledger(RS.ledger_path(ledgers, "pilot"), "pilot", mode="fake") as ledger:
            stages, ok = run_pilot(root, ledger, profile, first, everything, read_images,
                                   RS.export_path(ledgers, "pilot"), True, settings)
            totals = ledger.totals()
        complete = all(s["report"]["complete"] for s in stages.values()) and len(stages) == 3
        all_ok &= ok and complete and totals["committed"] <= S.CAPS["pilot"]
        summary["scenarios"][profile] = {
            "tokens_per_call": RS.USAGE_PROFILES[profile], "complete": complete, "checks_ok": ok,
            "ledger": str(RS.ledger_path(ledgers, "pilot").relative_to(staging)),
            "totals_usd": {k: S.usd(v) for k, v in totals.items()},
            "stages": {s: dict(_brief(v["report"]), reconciliation=v["reconciliation"],
                               corrections=v["corrections"], export_equals_settled=v["export_equals_settled"])
                       for s, v in stages.items()}}

    # The cap check: the extreme pilot, repeated on one ledger until the cap halts it.
    cap_root = staging / "cap-check"
    cap_ledgers = cap_root / "ledgers"
    halt, pilots = None, []
    with S.Ledger(RS.ledger_path(cap_ledgers, "pilot"), "pilot", mode="fake") as ledger:
        for k in range(1, MAX_PILOTS + 1):
            root = cap_root / f"pilot-{k:02d}"
            stages, ok = run_pilot(root, ledger, "extreme", first, everything, read_images,
                                   RS.export_path(cap_ledgers, "pilot"), False, settings)
            all_ok &= ok
            pilots.append({s: _brief(v["report"])["spend"]["stage_spent_usd"] for s, v in stages.items()})
            stopped = [(s, v["report"]) for s, v in stages.items() if not v["report"]["complete"]]
            if stopped:
                stage, report = stopped[0]
                refusal = report["gates"]["spend"]["refusals"][-1] if report["gates"]["spend"]["refusals"] else None
                halt = {"pilot": k, "stage": stage, "stopped": report["stopped"], "refused_reservation": refusal,
                        "committed_at_halt_usd": S.usd(ledger.totals()["committed"])}
                break
        totals = ledger.totals()
        sent = sum(1 for e in ledger.entries if e["type"] == "reserve")
    cap_ok = (halt is not None and halt["stopped"]["kind"] == "spend_cap" and totals["committed"] <= S.CAPS["pilot"]
              and halt["refused_reservation"] is not None
              and Decimal(halt["refused_reservation"]["committed_before_usd"])
              + Decimal(halt["refused_reservation"]["amount_usd"]) > S.CAPS["pilot"])
    all_ok &= cap_ok
    summary["scenarios"]["cap-check"] = {
        "tokens_per_call": RS.USAGE_PROFILES["extreme"], "checks_ok": cap_ok,
        "ledger": str(RS.ledger_path(cap_ledgers, "pilot").relative_to(staging)),
        "pilots": pilots, "halt": halt, "calls_reserved": sent, "totals_usd": {k: S.usd(v) for k, v in totals.items()}}
    summary["ok"] = all_ok

    for profile in ("central", "extreme"):
        sc = summary["scenarios"][profile]
        print(f"{profile}: {sc['tokens_per_call'][0]:,} input and {sc['tokens_per_call'][1]:,} output tokens per call")
        for stage, b in sc["stages"].items():
            sp = b["spend"]
            inv = "  ".join(f"{s} {v['invalid']}/{v['ended']}" for s, v in b["invalid_share"].items())
            hits = b["cache_hit_share"]
            hit = ("  ".join(f"{s} {v['hit']}/{v['responses']}" for s, v in hits.items())
                   if "not_applicable" not in hits else "uncached")
            print(f"  {stage} cases {b['cases']['ended']}/{b['cases']['given']}  "
                  f"stage ${_cents(sp['stage_spent_usd'])} "
                  f"(held ${_cents(sp['stage_held_usd'])})  pilot ${_cents(sp['budget_committed_usd'])} of "
                  f"${_cents(sp['cap_usd'])}  invalid {inv}  cache {hit}  output cap {len(b['output_cap_reached'])}  "
                  f"halts {len(b['halts'])}  reconciled {b['reconciliation']['ok'] if b['reconciliation'] else None}")
        print(f"  total ${_cents(sc['totals_usd']['committed'])} committed, ${_cents(sc['totals_usd']['settled'])} "
              f"settled, ${_cents(sc['totals_usd']['held'])} held")
    cc = summary["scenarios"]["cap-check"]
    if cc["halt"]:
        h = cc["halt"]
        r = h["refused_reservation"]
        print(f"cap-check: halted in pilot {h['pilot']} {h['stage']} at {h['stopped']['slot']} of "
              f"{h['stopped']['trace_key']}: ${_cents(r['committed_before_usd'])} committed, reservation "
              f"${Decimal(r['amount_usd']).normalize()} refused against the ${_cents(S.usd(S.CAPS['pilot']))} cap; "
              f"{cc['calls_reserved']} calls reserved")
    print(f"transport: {settings['transport']}; P3 rubric rule waived (rubric {R.RUBRIC_STATUS})")
    return all_ok, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, help="local data directory the guard covers")
    parser.add_argument("--packs", action="append", required=True,
                        help="a scripts/build_packs.py output folder; the first one is P1's")
    parser.add_argument("--provenance", action="append", required=True,
                        help="the provenance.jsonl those packs were built from; one per --packs")
    parser.add_argument("--out", required=True, help="new output directory outside the repository")
    return run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
