# Development allocation record

The study's real allocation, how to reproduce it, and which of its tasks were used before the draw. [RUNNER.md](RUNNER.md) describes the rules the runner applies to it.

## The draw

The draw ran at commit `3af9a4b` with seed 169. It reads two label-free counts files from the project's `counts/` folder.

| Input | SHA-256 |
|---|---|
| `task_cluster_mapping.json` | `7d527819e334d69bbf92a7925a354c010c3373b18d996546f383bf72e2821faa` |
| `eligible_minus_exposed.json` | `640b73697a24afb9e0c0b92b4e77325b5df083a00d0d75f0a9f5bf0dea7514f0` |

Run at that commit, these commands give the same bytes.

```bash
python3 scripts/build_dev_table.py table --counts COUNTS_DIR --out dev_table.json
python3 scripts/draw_allocation.py --table dev_table.json --seed 169 --out allocation.json --p1 \
  visualwebarena.118 webarena.344 webarena.370 webarena.427 \
  workarena.servicenow.navigate-and-order-apple-mac-book-pro15-l2
python3 scripts/build_dev_table.py exclusion --counts COUNTS_DIR --allocation allocation.json --out template_exclusion.json
```

| Output | SHA-256 of the file |
|---|---|
| `dev_table.json`, 42 tasks | `f15d4445f5b5d6dc843009c85e6804037d9f86efbbb531fbd778bdc939fac421` |
| `allocation.json` | `5c97a223d049a298a1842599517130458e8753e8f6f6497a32c9fcd1fc9a17a8` |
| `template_exclusion.json` | `cdaf6a5144cadd30f495f1e8dd02eb273d99cb7649fcfb4b0ce29612d5270a95` |

The value for `--allocation-sha256` is the `sha256` field inside `allocation.json`, `4f654915f6e44bf445c02feec42f28b46fc32a390a88addb2e7762556bf08e0a`. It covers the file's content apart from that field.

The prompt tasks use three templates that hold panel tasks. Those templates hold three panel tasks, which takes N from 226 to 223 before attrition.

## Known use before the draw

`counts/scripts/build_mapping.py` lists the tasks with development traces on disk before the draw. Its comment says 20 traces, and the list names 16 tasks. This record is built from the repository's documents and the project's notes. The data root and past session transcripts were not opened for it. The Nov 5 plan does not count earlier pack inspection as tuning.

"Read" below means that a person or a session read the run's content. Scripts have also read these files: the Oct 9 audit's count of run endings and rewards and the screenshot check cover 20 development runs, and the element counts and the request dry run cover 19.

### Frozen-pass tasks (P3)

| Task | List | Read | Where |
|---|---|---|---|
| `visualwebarena.419` | p3 | Yes. Its goal and steps 6 and 9 to 13 are written out in the walkthrough. | [PACK_WALKTHROUGH.md](PACK_WALKTHROUGH.md), Example 2 |
| `webarena.430` | p3 | No record of it being read. | No document quotes or describes this run. |
| `workarena.servicenow.dashboard-retrieve-incident-and-max-request-galaxy-note20-l2` | p3 | No record of it being read. | No document quotes or describes this run. |
| `workarena.servicenow.filter-three-items-uniform-expenses-and-find-total-return-large-l2` | p3 | No record of it being read. | No document quotes or describes this run. |

### Other tasks with traces on disk

| Task | List | Read | Where |
|---|---|---|---|
| `webarena.344` | p1 | Yes. | [DATASET_NOTE_ARB.md](DATASET_NOTE_ARB.md), table of examples |
| `webarena.370` | p1 | Yes. | [DATASET_NOTE_ARB.md](DATASET_NOTE_ARB.md), table of examples |
| `webarena.427` | p1 | Yes. | [DATASET_NOTE_ARB.md](DATASET_NOTE_ARB.md), table of examples |
| `visualwebarena.118` | p1 | Yes. | [DATASET_NOTE_ARB.md](DATASET_NOTE_ARB.md), table of examples |
| `workarena.servicenow.navigate-and-order-apple-mac-book-pro15-l2` | p1 | Yes. | [DATASET_NOTE_ARB.md](DATASET_NOTE_ARB.md), table of examples |
| `visualwebarena.133` | p2_new | No record of it being read. | No document quotes or describes this run. |
| `webarena.578` | p2_new | No record of it being read. | No document quotes or describes this run. |
| `webarena.583` | p2_new | No record of it being read. | No document quotes or describes this run. |
| `workarena.servicenow.infeasible-navigate-and-create-hardware-asset-with-reason-l2` | p2_new | No record of it being read. | No document quotes or describes this run. |
| `workarena.servicenow.three-changes-fix-wide-priority-uniform-risk-change-request-scheduling-l2` | p2_new | No record of it being read. | No document quotes or describes this run. |
| `webarena.380` | undrawn | Yes. Step 16 is written out in the walkthrough, and it is the example request. | [PACK_WALKTHROUGH.md](PACK_WALKTHROUGH.md), Example 1; [REQUESTS.md](REQUESTS.md) |
| `visualwebarena.64` | not in the table, an image-goal task | No record of it being read. | No document quotes or describes this run. |

### Packs

The Oct 30 dry run built 19 packs from those traces, so packs of P1, P2 and P3 tasks already exist. By that list, 10 of the 15 new prompt tasks and 16 of the 20 frozen-pass tasks have no trace on disk and so no pack.

### Still open

- Which of the 20 traces is outside the 19 packs, and so which of the four frozen-pass tasks has a pack. The pack folder's provenance would settle it, and it was not opened for this record.
- Whether anyone read the runs marked "No record of it being read" during a session. Only the owner can say, since session transcripts are fenced.
