#!/usr/bin/env python3
"""Deliberate breaks that the test suite must catch.

Each mutation copies the code to a temporary folder, makes one exact edit that
reintroduces a leak or a weak check, runs the unit tests there, and expects them to
fail. Run from the repository root:

    python3 tests/mutations.py

Exit code 0 means every mutation was caught.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

MUTATIONS = [
    ("read a test row's label field before the split check", "datapath/labels.py",
     '            if splits.split(row["task_id"]) not in allowed:\n',
     '            _ = row["trajectory_success"]\n            if splits.split(row["task_id"]) not in allowed:\n'),
    ("read every field of a test row before the split check", "datapath/labels.py",
     '            if splits.split(row["task_id"]) not in allowed:\n',
     '            _ = list(row.values())\n            if splits.split(row["task_id"]) not in allowed:\n'),
    ("copy the agent's prompts into the monitor view", "datapath/normalize.py",
     '"elements": step.get("extra_element_properties") or {},',
     '"elements": step.get("extra_element_properties") or {},\n            "prompt": step.get("chat_messages"),'),
    ("remove the schema allowlist", "datapath/leakage.py",
     "    l0 = _unexpected_keys(rec)", "    l0 = []"),
    ("skip the hostname rewrite", "datapath/normalize.py",
     'v, n = HOST_COPY.subn(r"\\1\\2", v)', "n = 0"),
    ("put screenshot hashes back in the monitor view", "datapath/normalize.py",
     'shot = {"width": info["width"],', 'shot = {"sha256": sha256_file(f), "width": info["width"],'),
    ("check LFS downloads by size only", "datapath/fetch.py",
     'size == lfs["size"] and sha == lfs["oid"]', 'size == lfs["size"]'),
    ("check Git blob downloads by size only", "datapath/fetch.py",
     'size == entry["size"] and git_blob_sha1(path) == entry["oid"]', 'size == entry["size"]'),
    ("re-enable evaluation label unlocking", "datapath/labels.py",
     'if tuple(allowed_splits) != ("dev",) or lock_file is not None:', "if False:"),
    ("publish into an existing output folder", "scripts/arb_datapath.py",
     "    if out.exists():", "    if False:"),
    ("write outputs straight into --out without staging", "scripts/arb_datapath.py",
     'staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.staging-", dir=out.parent))', "staging = out"),
    ("fetch screenshots before checking whose they are", "scripts/arb_datapath.py",
     "        check_screenshot_paths(source, task_id)     # before any screenshot is fetched or opened",
     "        pass"),
    ("accept nested values in text fields", "datapath/normalize.py",
     "if isinstance(v, (dict, list, tuple)):", "if False:"),
    ("accept non-finite numbers in element properties", "datapath/leakage.py",
     "return isinstance(v, float) and math.isfinite(v)", "return isinstance(v, float)"),
    ("check only the first screenshot's trajectory and step", "datapath/normalize.py",
     "    for step in steps:\n", "    for step in steps[:1]:\n"),
    ("accept any element property value", "datapath/leakage.py",
     'bad += [f"/events[{i}]/elements/{bid}/{k} (shape)" for k in element_shape_errors(props)]',
     "bad += []"),
    ("check hostnames without URL-unquoting", "datapath/leakage.py",
     "for t in (text, unquoted)", "for t in (text,)"),
    ("remove the label guard from the data-path command", "scripts/arb_datapath.py",
     '    fence.install("label_free", data_root, fence_config)', "    pass"),
    ("remove the development-only guard from the data-path command", "scripts/arb_datapath.py",
     '        fence.install("dev_only", data_root, fence.load_config(required=True))', "        pass"),
    ("let any caller open the label file", "datapath/fence.py",
     'if kind in ("label", "label_part") and opener_admitted(kind):', 'if kind in ("label", "label_part"):'),
    ("trust the caller's split table in load_labels", "datapath/fence.py",
     "return claimed_dev <= canonical_dev()", "return True"),
    ("recognize the label file by path only", "datapath/fence.py",
     "if ident is not None and ident in {_identity(p) for p in label_paths}:", "if False:"),
    ("compare guarded paths case-sensitively", "datapath/fence.py",
     'return unicodedata.normalize("NFC", p).casefold()', "return p"),
    ("ignore dataset URL requests", "datapath/fence.py",
     "kind = url_refusal(args[0])", "kind = None"),
    ("run without the local fence config", "datapath/fence.py",
     "        if required:\n", "        if False:\n"),
    ("look up acted elements in the next step's tree", "judge/pack.py",
     'describe_action(action, obs.get("axtree"), event.get("elements"))',
     'describe_action(action, (events[event["step"] + 1:] or [event])[0]["observation"]["axtree"], '
     'event.get("elements"))'),
    ("look up acted elements in the next step's element properties", "judge/pack.py",
     'describe_action(action, obs.get("axtree"), event.get("elements"))',
     'describe_action(action, obs.get("axtree"), (events[event["step"] + 1:] or [event])[0]["elements"])'),
    ("report a nonvisible target as absent", "judge/acted.py",
     "        status = NONVISIBLE", "        status = ABSENT"),
    ("report a target listed at another visibility as nonvisible", "judge/acted.py",
     "        status = NOT_IN_TREE", "        status = NONVISIBLE"),
    ("accept calls to unknown functions", "judge/acted.py",
     "        if name not in ELEMENT_PARAMS and name not in NO_TARGET_ACTIONS:", "        if False:"),
    ("resolve an ambiguous ID to its first tree line", "judge/acted.py",
     "    if len(lines) == 1:", "    if lines:"),
    ("drop the canonical key sort from the pack bytes", "judge/pack.py",
     "json.dumps(obj, sort_keys=True,", "json.dumps(obj, sort_keys=False,"),
    ("put screenshot paths and hashes into the pack", "judge/pack.py",
     "    pack = build_pack(view)", '    pack = dict(build_pack(view), images=provenance["screenshot_files"])'),
    ("cut the final tree without a marker", "judge/pack.py",
     "text[:FINAL_TREE_HEAD] + marker + text[-FINAL_TREE_TAIL:]",
     "text[:FINAL_TREE_HEAD] + text[-FINAL_TREE_TAIL:]"),
    ("remove the guard install from the pack command", "scripts/build_packs.py",
     '    fence.install("dev_only", data_root, fence_config)', "    pass"),
    ("pack evaluation-split records", "scripts/build_packs.py",
     '        if prov.get("split") != "dev":', "        if False:"),
    ("publish packs into an existing output folder", "scripts/build_packs.py",
     "    if out.exists():", "    if False:"),
    ("write packs straight into --out without staging", "scripts/build_packs.py",
     'staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.staging-", dir=out.parent))', "staging = out"),
    ("send a repeat whose body differs from A1", "judge/dispatch.py",
     '        slot = f"{judge}{repeat}"\n        request = requests[judge]\n',
     '        slot = f"{judge}{repeat}"\n        request = requests[judge] if repeat == 1 else '
     'requests[judge]._replace(body=requests[judge].body + b" ")\n'),
    ("skip the repeats when A1 is invalid", "judge/dispatch.py",
     '        slot = f"{judge}{repeat}"\n        request = requests[judge]\n',
     '        slot = f"{judge}{repeat}"\n        if repeat > 1 and not results.get("A1", {}).get("valid", True):\n'
     '            continue\n        request = requests[judge]\n'),
    ("count an HTTP 400 or 413 as an invalid verdict instead of a halt", "judge/outcomes.py",
     'if status in (400, 413) or etype in ("invalid_request_error", "request_too_large"):\n'
     '        return HALT, "request_error", label',
     'if status in (400, 413) or etype in ("invalid_request_error", "request_too_large"):\n'
     '        return MALFORMED, None, label'),
    ("treat a spend-cap 429 as a rate limit", "judge/outcomes.py",
     'A_SPEND_CODES = frozenset({"enforced_spend_limit_reached"})', "A_SPEND_CODES = frozenset()"),
    ("treat a cut stream as a complete response", "judge/streams.py",
     's["complete"] = started and stopped and s["stop_reason"] is not None and s["error"] is None',
     's["complete"] = started and s["error"] is None'),
    ("retry outcomes other than transport failures", "judge/dispatch.py",
     '            if oc["retryable"]:', "            if outcome != VALID:"),
    ("allow a fourth retry", "judge/requests.py", "MAX_RETRIES = 3", "MAX_RETRIES = 4"),
    ("resume after a halt without a correction note", "judge/dispatch.py",
     "    pending = log.unresolved_halts()", "    pending = []"),
    ("accept a changed model string", "judge/dispatch.py",
     '    if first_model is not None and summary.get("model") != first_model:', "    if False:"),
    ("allow an attempt record to be overwritten", "judge/storage.py",
     "        os.link(tmp, path)", "        os.replace(tmp, path)"),
    ("send a sampling parameter to judge A", "judge/requests.py",
     '    "max_tokens": OUTPUT_CAP,\n', '    "max_tokens": OUTPUT_CAP,\n    "temperature": 0,\n'),
    ("drop cache control from judge A", "judge/requests.py",
     '    "cache_control": {"type": "ephemeral"},        # omitted in prompt development\n', ""),
    ("enable prompt caching on judge B", "judge/requests.py",
     '"prompt_cache_options": {"mode": "explicit"},', '"prompt_cache_options": {"mode": "implicit"},'),
    ("put each screenshot's hash into judge B's image blocks", "judge/requests.py",
     '            out.append({"type": "input_image", "detail": "original",',
     '            out.append({"type": "input_image", "detail": "original", '
     '"file_id": hashlib.sha256(images[value]).hexdigest(),'),
    ("skip the private-string search in the request leak check", "judge/requests.py",
     "    found = [s for s in private_strings if s and s in text]", "    found = []"),
    ("drop the canonical key sort from request bytes", "judge/requests.py",
     "        items = list(obj.items()) if keep_order else sorted(obj.items())", "        items = list(obj.items())"),
    ("drop judge A's size limit for requests with more than 20 images", "judge/requests.py",
     'side = lim["many_images_max_side"] if len(sizes) > lim["many_images_above"] else lim["max_side"]',
     'side = lim["max_side"]'),
    ("compare verdicts case-sensitively", "judge/schema.py",
     '    verdict = obj["verdict"].casefold()', '    verdict = obj["verdict"]'),
    ("let an out-of-range p_failure void the verdict", "judge/schema.py",
     '        voided["p_failure"] = "not a finite number in [0, 1]"', "        return out"),
    ("remove the guard install from the dry run", "scripts/dry_run.py",
     '    fence.install("dev_only", data_root, fence_config)', "    pass"),
]


def main() -> int:
    missed = 0
    for name, rel, old, new in MUTATIONS:
        with tempfile.TemporaryDirectory() as tmp:
            for part in ("datapath", "judge", "scripts", "tests"):
                shutil.copytree(REPO / part, Path(tmp) / part,
                                ignore=shutil.ignore_patterns("__pycache__"))
            target = Path(tmp) / rel
            text = target.read_text(encoding="utf-8")
            if text.count(old) != 1:
                print(f"SETUP ERROR  {name}: expected exactly one match in {rel}")
                missed += 1
                continue
            target.write_text(text.replace(old, new), encoding="utf-8")
            result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"],
                                    cwd=tmp, capture_output=True, text=True)
            caught = result.returncode != 0
            missed += not caught
            print(f"{'caught' if caught else 'MISSED'}  {name}")
    print(f"{len(MUTATIONS) - missed} of {len(MUTATIONS)} mutations caught")
    return 0 if missed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
