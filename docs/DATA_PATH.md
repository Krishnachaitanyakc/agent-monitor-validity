# AgentRewardBench data path

One command turns pinned AgentRewardBench (ARB) files into three JSONL files: what a detector may see, where each record came from, and the released labels, kept apart. Standard library only; tested on Python 3.11 and 3.13, with byte-identical outputs.

## Run it

From a clone of the repository, in the repository root. The command installs a file-open guard (`datapath.fence`) before it reads anything; set `SPAR_FENCE_CONFIG` to a local JSON file that names any further folders and files to guard (keys: `label_files`, `fenced_dirs`, `fenced_files`, `archive_dirs`, `transcript_dirs`, `test_output_dirs`, `eval_output_dirs`, `join_dirs`). No paths are kept in the repository.

```bash
export SPAR_FENCE_CONFIG=/path/to/local/fence_config.json
python3 scripts/arb_datapath.py run --data-root ~/spar169-arb-data --out ~/spar169-arb-data/out/dev5-$(date +%Y%m%d-%H%M%S) \
  --trajectory webarena/GenericAgent-gpt-4o-2024-11-20/webarena.370 \
  --trajectory webarena/GenericAgent-anthropic_claude-3.7-sonnet/webarena.427 \
  --trajectory webarena/GenericAgent-gpt-4o-2024-11-20/webarena.344 \
  --trajectory visualwebarena/GenericAgent-gpt-4o-2024-11-20/visualwebarena.118 \
  --trajectory workarena/GenericAgent-anthropic_claude-3.7-sonnet/workarena.servicenow.navigate-and-order-apple-mac-book-pro15-l2
```

The first run downloads about 29 MB, 17 MB of data and 12 MB of file listings, and takes about 35 seconds on a home connection. It prints one line per trajectory and ends with `ALL CHECKS PASS`; the exit code is 0 only then. `--out` must be a new folder: the run writes everything to a staging folder beside it and publishes it to `--out` in one rename only after every check passes, so a failed or repeated run never overwrites or truncates an earlier output set. On Oct 8, 2026 these five gave the hashes below; on Oct 15 the same five trajectories, run with the current code into a new `--out`, gave the same hashes. `report.json` also records the code commit the run used.

| Output | SHA-256 |
|---|---|
| `monitor_view.jsonl` | `862590c6104a4933639dcef12aea34619a3408a1411be563d5a802c1f74bf0fc` |
| `provenance.jsonl` | `adc72c4d6586d1b49118a5bf5f346598799df2cfb2fc5b72a940a8e02052b8b7` |
| `labels/labels_dev.jsonl` | `0c731c8a45798ec6c03e3e368b712cc29b822a2a718044730aea277575d78ea3` |

To check them, read `outputs_sha256` in `report.json`, or run `shasum -a 256` on macOS or `sha256sum` on Linux over the three files. To check that a rerun is identical, run again with a different `--out` and compare the files with `cmp`. A rerun re-checks the files on disk instead of downloading them; `report.json` shows `files_downloaded: 0` and lists the three listings under `listings_read_from_cache`. The cached listings are the Hugging Face API's output and may change format over time without changing the pinned files; on Oct 8 their SHA-256 values began `30df6579` for `data`, `abf2dd2b` for `cleaned` and `1f228a7a` for `screenshots`, and `report.json` records them in full.

If a python.org install on macOS fails with `CERTIFICATE_VERIFY_FAILED`, run its `Install Certificates.command` once.

> **Never open `<data-root>/raw/data/annotations.csv` directly.** It is ARB's full label file and holds the test-split labels, which must stay unseen until the predictions are locked. The split guard lives in the loader, not in the file: read labels only through `datapath.labels.load_labels`.

Tests use a synthetic fixture and need no network. The second command applies seventeen deliberate breaks, one at a time, to a temporary copy and checks that the tests catch each:

```bash
python3 -m unittest discover -s tests
python3 tests/mutations.py
```

`--data-root` and `--out` must be outside the repository; the command refuses otherwise. As a project policy, nothing derived from ARB's trajectories or screenshots is committed or re-hosted. ARB's own Terms of Use, which derivatives must carry, are reproduced in [ARB_TERMS_OF_USE.md](ARB_TERMS_OF_USE.md).

## What it does

1. **Pinned listings.** The file listings of `data/`, `cleaned/` and `screenshots/` at revision `b6d17e646009d6cb63d5dd7be78807b680693f61` are fetched once and cached under `<data-root>/listing/<revision>/`.
2. **Hash-checked downloads.** Every file resolves against that revision and lands under `<data-root>/raw/` at its dataset path. It is accepted only if it matches the listing: the SHA-256 for files stored in Git LFS, which are all 27,299 screenshots and 747 trajectories, and the Git blob SHA-1 for the rest, which are the `data/` files and the other 555 trajectories, including all five above. A download goes to a `.part` file and is moved into place only after it passes. A file on disk that fails is downloaded again once; if the new copy also fails, the run stops.
3. **Normalize,** one trajectory at a time in trace-key order, so memory holds one trajectory. Fifteen runs, fourteen of them at the 31-step maximum, peaked at 0.46 GB.
4. **Labels, development split only.** The label store skips every row whose task is not in the development split before it accesses any label field. Evaluation labels cannot be read in this version: `load_labels` refuses every split other than development, with or without a lock file, before it opens the file. The runner will restore that path only after it validates complete prediction records bound to the frozen panel and run manifest.
5. **Checks.** Validation, leakage and determinism, all written to `report.json`.

## The file-open guard

`datapath.fence.install(profile, data_root, config)` adds a Python audit hook that raises `PermissionError` when the process opens a guarded path. It sees only file opens made inside a Python process that installs it; it does not see other processes, native code that opens files without Python, or directory listings.

| Profile | Guards |
|---|---|
| `label_free` | the label file, any `annotations.csv.part` and any `judgments` folder under the data root, and the config's fenced folders and files, archive and transcript folders |
| `dev_only` | the above, plus trajectories and screenshots of tasks outside the development split (by task ID from `data/splits.csv`) and named test outputs |
| `pre_join` | the above, plus named evaluation output and join folders |

Only three readers may open the label file: `fetch.fetch_file` (download and hash check, and the only code that may touch `annotations.csv.part`), `labels.load_labels` for the development split with no lock file, and `labels.annotation_structure`.

Callers: `scripts/arb_datapath.py run` installs `label_free` before reading anything and `dev_only` once the split file is verified. `--owner-label-free` keeps only `label_free`, for the data owner's own steps on evaluation data. Later scripts install `dev_only` (the evidence packer and the pilot runner) or `pre_join` (the panel draw, the evaluation runner, the ending-category, heuristic, readings and rerun scripts, the lock and the review tooling) when run on development or synthetic data, and `label_free` with the owner flag on evaluation data. `report.json` records the installed profiles and the config file's SHA-256.

## The three records

**Monitor view** (`monitor_view.jsonl`, schema `0.2-arb`): what a detector may see.

- `trace_key`: a plain hash of the trace ID. The readable ID names the agent, so it stays in provenance. The key is opaque to a reader but not secret: hashing the public trace IDs inverts it.
- `task.goal`, plus any image URLs found in the goal. These benchmarks state a task's constraints inside the goal, which is kept verbatim apart from the hostname rewrite below.
- `events`, in step order. Each holds the agent's `action` and `reasoning`, `url`, `open_pages_urls`, `focused_element`, `last_action_error`, an `observation` with the screenshot's dimensions and PNG status, the full and pruned accessibility trees, and the element properties. The last event is the final observation and normally has no action. Every screenshot path must name this trajectory's benchmark, agent, task and step number, with no `.` or `..` segment; any other path stops the run before any screenshot is fetched and before anything is published. Text fields must be text: a nested mapping or list in the goal, an action, the reasoning, a URL, the focused element, the last action error or a tree stops the run, and `open_pages_urls` must be a list. Element properties must have their supported shapes: `bbox` null or exactly four finite numbers, `visibility` a finite number, `clickable` and `set_of_marks` booleans. Non-finite numbers are never written: the serializer refuses them.
- ARB hosts each WebArena and VisualWebArena site twice, and the agents split across the two copies: in the 11 development runs checked, GPT-4o and Claude used `-xl-1` and Qwen and Llama used `-xl-2`. The copy suffix is rewritten to the site name in every text field, for example `wa-forum-xl-2.mcgill-nlp.org` becomes `wa-forum.mcgill-nlp.org`. Each rewrite is counted in provenance, and the source file's hash still pins the original bytes. Screenshot pixels are not altered, so the suffix can stay visible where a page prints a URL in the image, as in one development run.

Screenshots are held by reference: the view records each one's dimensions and PNG status, and the evidence packer, in a later PR, attaches each hash-checked PNG by trace key and step, so judges see every screenshot. The monitor view is the fenced pool every detector draws from; the protocol's judge view is a narrower, deterministic selection from it, made and tested by the packer.

**Provenance** (`provenance.jsonl`): the readable `trace_id`, benchmark, agent, raw and normalized task IDs, split, `source_url`, the source file's revision, bytes and SHA-256, each screenshot's path, SHA-256 and bytes, event and action counts, the number of hostname rewrites, how the run ended (development runs only, below), and which source fields were dropped and why. Screenshot hashes and byte counts are file bookkeeping, not evidence, so they stay here.

**Labels** (`labels/labels_dev.jsonl`, in its own folder so a loader that reads every top-level JSONL file cannot pick it up): the four released label fields for development-split records, the primary annotator, the number of annotation rows, and any second annotator's labels under `secondary_labels`. The first row in file order is primary, as in ARB's own scorer; secondary rows are kept for reporting and never used as the reference. `success_binary` keeps ARB's polarity, Successful as 1; the metrics code treats failure as the positive class.

Two codebook provenance fields are handled differently: `collection_date` is replaced by the pinned revision, which makes outputs reproducible, and `length_tokens` waits for the pilot's provider token counts, since it depends on the judge's tokenizer.

## What never reaches the monitor view

| Source field | Why it is dropped |
|---|---|
| `summary_info` | holds `cum_reward`, the benchmark checker's verdict |
| `agent`, `model`, `model_args`, `experiment`, `flags` | identify the agent or its configuration |
| `valid`, `seed`, `package_version` | run bookkeeping |
| step `chat_messages` | the agent's own prompts, which repeat evidence the view already holds |
| step `stats` | token and cost telemetry |
| step `axtree_obj`, `bounding_boxes` | duplicates of the kept tree text and element properties |

Only fields named in the schema are copied, so a new top-level or step field from ARB is dropped and reported as unrecognized. Element properties are copied whole, so a new property name fails the allowlist check instead.

## Checks in `report.json`

| Check | Fails when |
|---|---|
| V1 required | trace key, schema version, goal or events missing |
| V2 event order | steps are not 0, 1, 2 and so on |
| V3 actions | an event before the last has no action; an action on the last event is reported, not failed |
| V4 screenshots | a referenced screenshot is missing or is not a valid PNG: signature, chunk CRCs, a 13-byte IHDR first, positive dimensions, IDAT that decompresses to the expected size, IEND last with nothing after it |
| V5 missingness | never; reports empty-field counts |
| V6 labels | a primary or secondary label value is outside ARB's vocabulary, or a development record has no label |
| L0 allowlist | any key at any level is not in the schema, or an element property has an unsupported shape |
| L1 forbidden keys | a fenced or label key appears at any depth |
| L2 forbidden text | a fenced field name or the label file's name appears anywhere |
| L3 identifiers | the agent name, model name, trace ID, raw or normalized task ID, or the `task_<n>/` folder of a goal image appears anywhere, or a hosted-site copy suffix appears in any case, also after URL-unquoting |
| L4 import | the normalizer imports the label store |
| L5 package | the pip package that bundles ARB's labels is importable in the interpreter running the check; the judge runner must run the same check in its own environment |
| determinism | normalizing a trajectory twice gives different bytes |

L3 fails on image-goal VisualWebArena tasks, whose goal names the task folder of the input image. The agreed scope excludes those tasks.

The 44 unit tests plant canary strings in the reward field, the agent's prompts and two test-split label rows, one with a `.resized` task ID, and check that none reaches any output file. A spy on the CSV rows checks that `load_labels` accesses no field of a test-split row except its task ID; `annotation_structure` reads every row's keys and annotator name for its label-free counts, never a label field. `tests/mutations.py` holds the seventeen deliberate breaks, each an exact one-line edit: reading one or all fields of a test row before the split check, copying the agent's prompts into the view, removing the allowlist, skipping the hostname rewrite, putting screenshot hashes back in the view, checking LFS or Git blob downloads by size only, re-enabling evaluation label unlocking, publishing into an existing output folder, writing outputs without staging, fetching screenshots before checking whose they are, accepting nested values in text fields, accepting non-finite numbers, checking only the first screenshot, accepting any element property value, and checking hostnames without URL-unquoting.

For test runs, the console and the report summary show only whether the checks passed, not lengths or endings.

## What the monitor view hides, and what it cannot

The monitor view keeps identifiers out of the text a judge reads: no agent or model name, no task ID, no checker reward, no hostname copy suffix. It does not hide a run from a person who looks it up. The key, the goal text or the run's length can be matched against the public dataset; for example, in 86 of the 196 development runs, the number of screenshots alone tells which agent ran that task. Keeping reviewers blind during the error analysis is therefore a procedure, backed by the label store's split guard, not a property of the file.

Cues a judge can see in the evidence itself:

1. **How a run ended.** A run either ends with an agent message, reaches the 30-action cap, or ends early without a message. In BrowserGym 0.13.3, an early end in WebArena or VisualWebArena means the checker scored the run above zero, or, at reward 0, that a tab left the benchmark's sites. In WorkArena 0.4.1 L2, it means every subtask validated, or, at reward 0, that a subtask stopped the episode. Harness errors can also end a run. WorkArena L1 and AssistantBench were not checked. On the 20 development runs checked so far, all three early endings had a checker reward of 1. Provenance records `ending` as `agent_message`, `step_cap` or `environment` for development runs only, and leaves it empty for test runs, since it mostly stands in for the checker's verdict. That restriction is procedural: a test run's events still let anyone work out how it ended. Runs are kept as recorded whatever their ending. The ending categories used in the study, including an unknown category, will be defined on development data from label-free trace evidence and frozen before evaluation; they are applied without evaluation labels and reported descriptively after the predictions are locked. Trajectories are never selected on how well their endings predict the labels.
2. **Claude runs carry no free-text reasoning.** In all six Claude 3.7 Sonnet runs checked, `reasoning` only repeats the action inside `<action>` tags. The other three agents write free text. The field therefore hints at the agent and gives judges less evidence on Claude runs.

## Not done yet

- The evidence packer and the pack checks that build each judge's input from the monitor view.
- Evaluation label unlocking, which the runner restores after validating complete prediction records, valid verdicts and model roles, and their binding to the frozen panel and run manifest.
- Unsure or missing primary labels are handled at the label join (proposed): excluded from the error-based quantities and reported by count. No test label field is read before the join.
- The README, charter and codebook still describe the earlier three-mentee design. A separate docs PR reconciles them before the main run, as the brief asks, and retires the multi-source `scripts/collect_traces.py` stub.

- Template IDs are not in ARB's metadata. The task-to-template mapping, joined label-free from the original WebArena and VisualWebArena configs pinned to a commit, and the counts behind 243 eligible tasks and 209 templates go to review with the setup materials. The working rule is one trajectory per eligible task.
- VisualWebArena goal images are referenced by URL on a McGill host that returned HTTP 530 on Oct 8. The same relative path exists in the VisualWebArena GitHub repository. The agreed scope excludes image-goal tasks, so this matters only if that rule changes.
