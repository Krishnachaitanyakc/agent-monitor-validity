# Dataset note: AgentRewardBench

This note covers the brief's week-1 inspection items for AgentRewardBench (ARB). Facts marked "checked" come from files downloaded at the pinned revision; the rest come from the paper, the ARB code, or the Hugging Face file listing. How to reproduce the outputs is in [DATA_PATH.md](DATA_PATH.md).

## 1. Revisions and access

| Item | Value |
|---|---|
| Paper | arXiv 2504.08942 v2, COLM 2025 |
| Code | github.com/McGill-NLP/agent-reward-bench, commit `05899fcfe52c925978944a23920373b7a9c63740`, still HEAD on Oct 8 |
| Data | huggingface.co/datasets/McGill-NLP/agent-reward-bench, revision `b6d17e646009d6cb63d5dd7be78807b680693f61` |
| Download | `scripts/arb_datapath.py run`: pinned URLs, every file checked against the listing's hash before use, SHA-256 recorded. It fetches only the metadata, the label file, and the named trajectories with their screenshots. It never fetches `judgments/` |
| Agent harness recorded in the trajectories | browsergym 0.13.3, browsergym-workarena 0.4.1, agentlab 0.3.0 |
| Our environment | Python 3.13.13, standard library only; branch `datapath-arb-v1` from commit `8a75de2` |

## 2. Label fields

- **Released fields.** `trajectory_success`, `trajectory_side_effect`, `trajectory_optimality` and `trajectory_looping`, plus `annotator_name`, `benchmark`, `task_id`, `model_name` and `exp_name`.
- **Primary outcome.** `trajectory_success`, answering "Was the sequence of actions successful in achieving the goal?" Allowed answers: Successful, Unsuccessful, Unsure. Unsure cases were reviewed by other annotators, and annotators were told to resolve ambiguous instructions toward "the most lenient option".
- **Rows, checked without reading labels.** 1,408 rows for 1,302 trajectories. 106 trajectories have a second row: 100 GPT-4o WebArena, 3 Qwen WebArena, 3 Llama WorkArena. No trajectory has more than two rows, and no annotator appears twice on one trajectory. The file has 8 distinct annotator names, which are anonymized; the paper reports six experts.
- **Primary row.** The first row per benchmark, agent and task in file order, which is the rule ARB's scorer uses. Second rows are kept in the label store for reporting, never as the reference.
- **Join, checked.** The 1,302 annotation keys match the 1,302 trajectory files one to one, with none left over on either side.
- **Unsure or missing primary labels.** Handled at the label join (proposed): excluded from the error-based quantities and reported by count. No test-split label field is read before the join, so the test-split count is unknown until then.
- **What the labels are not.** Mostly one annotation per trajectory. The only reported agreement is 89.3% raw, on GPT-4o WebArena runs, with no denominator. Annotators could consult the live environment, which judges cannot. We treat the labels as an expert reference, not as verified correctness.

## 3. Trajectory contents and modalities

- **Per step.** URL, open pages, action, agent reasoning, focused element, last action error, a screenshot, the full accessibility tree, the pruned visible-only tree, and element properties. The last step is the final observation and has no action.
- **Screenshots.** 27,299 PNGs. Checked: all 464 screenshots in 20 development runs are valid PNGs at 1280×720, including the Claude and Qwen VisualWebArena runs whose task IDs say "resized". Content below the fold is not captured.
- **Length.** Median 31 screenshots and mean 21 per trajectory. 702 of 1,302, or 53.9%, reach the cap of 30 actions plus the final observation. Counted from the listing, all splits, label-free.
- **How a run ended is visible.** Checked in the BrowserGym 0.13.3 code for WebArena and VisualWebArena and the WorkArena 0.4.1 code for L2 tasks: a run that stops before the step cap without a final agent message was ended by the environment: either because the checker gave it a score above zero, or, at reward 0, because a tab left the benchmark's sites, a WorkArena subtask stopped the episode, or the harness failed. How often the reward-0 exits occur is unknown. In all three such runs among the 20 development runs checked, the checker reward was 1. The cue comes from the benchmark harness, not from our data path, and any judge that sees the whole trajectory can see it. Provenance records it as `ending` for development runs only; it is never computed for test runs before the label join, and never used to select.
- **Claude runs carry no free-text reasoning.** Checked: in all six Claude 3.7 Sonnet runs, the reasoning field only repeats the action. The other three agents write free text. Judges get less evidence on Claude runs, and the field hints at which agent ran.
- **Hostnames hinted at the agent; removed from text fields.** ARB hosts each site twice. In the 11 development WebArena and VisualWebArena runs checked, GPT-4o and Claude used the `-xl-1` copy and Qwen and Llama the `-xl-2` copy. The data path rewrites the copy suffix to the site name in every text field. Screenshot pixels are not altered, so the suffix can stay visible where a page prints a URL in the image, as in one development run.
- **What the monitor view does not hide.** It keeps identifiers out of the text a judge reads. A person can still identify a run by matching the goal, the key or the run's length against the public dataset; in 86 of the 196 development runs, the screenshot count alone tells which agent ran that task. Keeping reviewers blind during the error analysis is a procedure, not a property of the file.
- **Known gaps.**
  - VisualWebArena goal images are referenced by URL on a McGill host that returned HTTP 530 on Oct 8. The same file exists in the VisualWebArena GitHub repository, commit `89f5af29305c3d1e9f97ce4421462060a70c9a03`, at the matching path. The agreed scope excludes image-goal tasks: the images are not in ARB's release, the hosted copy is down, and the GitHub copy cannot be shown to match what the agents and annotators saw. The claim is narrowed to text-goal tasks, and the excluded count will be reported.
  - AssistantBench success depends on a gold answer and on live web pages the record does not capture.
- **Fenced out of every judge input.** `summary_info`, which holds the checker's `cum_reward`; `agent`, `model`, `model_args`, `experiment` and `flags`; the agent's own prompts in `chat_messages`; all of `judgments/`; and `annotations.csv`. The pip package bundles the labels, so it is never installed in the judge runtime; a check fails if it is importable where the data path runs, and the judge runner will run the same check.

## 4. Task IDs and repeated runs

- **Split by task.** 351 tasks: 51 development and 300 test, holding 196 and 1,106 trajectories. Each agent ran each task once, so a task has up to four trajectories.

  | Benchmark | Dev tasks | Test tasks | Dev trajectories | Test trajectories |
  |---|---|---|---|---|
  | WebArena | 22 | 78 | 88 | 310 |
  | VisualWebArena | 8 | 92 | 24 | 276 |
  | WorkArena L1 and L2 | 15 | 103 | 60 | 412 |
  | AssistantBench | 6 | 27 | 24 | 108 |

- **ID normalization.** `.resized` and `.improved` are stripped before the split lookup. Checked: no normalized task appears in both splits.
- **Template IDs are not in ARB.** The metadata files carry task names, sites and evaluator types but no template IDs. The join with the original task configs (WebArena at commit `dce04686a56253aefba7b18a4fa0937cf1dc987b`, VisualWebArena at the commit above) has been computed label-free, reading only task IDs, template IDs, sites and whether a goal has an input image. The task-to-template mapping, the development overlap and the counts go to review with the setup materials.
- **Eligible tasks.** Excluding AssistantBench and image-goal VisualWebArena leaves 243 test tasks: 78 WebArena, 62 text-goal VisualWebArena and 103 WorkArena. The working rule is one trajectory per eligible task. Grouping WebArena and VisualWebArena tasks by template gives 209 units, since WorkArena tasks carry no template id. Individually exposed tasks, with all their trajectories, will be removed from the count.
- **One trajectory per task (proposed rule).** A seeded permutation of the task's runs; take the first run with complete evidence that fits the judge view, with runs skipped for incompleteness or length reported.

## 5. Terms

- **Dataset.** No declared license. The Hugging Face card's custom Terms of Use, reproduced in [ARB_TERMS_OF_USE.md](ARB_TERMS_OF_USE.md), require derivatives to carry those terms, and third-party site terms also apply.
- **Code.** No license.
- **Release plan.** IDs, hashes, our code, our prompts, our judge outputs and aggregate results, with the terms attached. No re-hosted trajectories or screenshots, as a project policy; ARB's terms themselves ask that derivatives carry its terms. The repository ignores `data/` and `out/`, and the command refuses data or output folders inside the repository.

## 6. Five development examples

All five are development-split runs; reading their labels is within the protocol and is recorded in the exposure log. The labels and short quotes below come from AgentRewardBench and fall under its Terms of Use on the dataset card at huggingface.co/datasets/McGill-NLP/agent-reward-bench. "Checker" is ARB's rule-based `cum_reward`, read here for development runs only. Where a second annotator exists, both labels are shown, primary first.

| Run | Human label | Ending and checker | Evidence for judging completion | What gives the answer away |
|---|---|---|---|---|
| webarena.370, GPT-4o | Successful; second annotator Unsuccessful | ended early, 1 | Goal: pull up the description page of Carnegie Mellon University on the map. The agent searched and stopped on the results list, which shows three CMU entries; it never opened one. Judging needs the final page and a reading of "description page" | `cum_reward`; the early stop without a message |
| webarena.427, Claude 3.7 Sonnet | Unsuccessful | ended early, 1 | Goal: find the page of the university with the most Turing Award winners. The agent searched "MIT" and stopped on results led by Massachusetts Institute of Technology. Judging needs outside knowledge and the final page. No free-text reasoning | the same two cues, which here contradict the human label |
| webarena.344, GPT-4o | Unsure; second annotator Successful | agent message, 0 | Goal: how many reviews the shop has received. The agent opened the reviews-by-customer report, saw one customer with one review, and answered "1". It never opened the full review list, so the total is not settled from what it saw | `cum_reward` |
| visualwebarena.118, GPT-4o | Unsuccessful | agent message, 0 | Goal: the time on a phone screen in the third row. The agent opened a listing and said the time "is not directly readable". The message shows the failure; judging a stated time would need the screenshot, since the trees carry no image content | `cum_reward` |
| WorkArena order 5 MacBook Pro 15, Claude 3.7 Sonnet | Successful | ended early, 1 | Four clicks from the All menu through Service Catalog and Hardware to the MacBook Pro 15 item, quantity set to 5, a click on Order Now, and a final page reading "Thank you, your request has been submitted" with a request number | `cum_reward`; the early stop |

Rows 1 and 2 have the same shape, a search that stops on a results page, and the same checker score of 1. The primary labels are opposite, and on 370 the two annotators themselves disagree. So the human reference carries a judgment the checker does not, and on runs like these that judgment is contested. It also shows the ending cue can mislead a judge that leans on it.
