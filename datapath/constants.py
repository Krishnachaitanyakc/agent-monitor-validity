"""Pinned source, fenced fields and label vocabulary for the ARB data path."""

DATASET = "McGill-NLP/agent-reward-bench"
# Pinned dataset revision (Hugging Face commit). Every download resolves against it.
REVISION = "b6d17e646009d6cb63d5dd7be78807b680693f61"
RESOLVE_URL = "https://huggingface.co/datasets/{dataset}/resolve/{rev}/{path}"
TREE_URL = "https://huggingface.co/api/datasets/{dataset}/tree/{rev}/{path}?recursive=true"

# Small metadata files under data/ (labels excluded; see LABEL_FILE).
META_FILES = (
    "data/splits.csv",
    "data/complete_task_ids.csv",
    "data/webarena.csv",
    "data/visualwebarena.csv",
    "data/workarena.csv",
    "data/assistantbench.csv",
    "data/webarena.task_ids.json",
    "data/visualwebarena.task_ids.json",
    "data/workarena_l2.task_ids.json",
)
LABEL_FILE = "data/annotations.csv"

# Top-level fields of a cleaned ARB trajectory that never enter the monitor view.
#   summary_info  holds cum_reward (the benchmark's rule-based verdict), err_msg, etc.
#   experiment, model_args, flags, agent, model identify the agent or its run config.
FENCED_TOP_LEVEL = {
    "summary_info": "rule-based reward (cum_reward) and run summary: label proxy",
    "experiment": "identifies the agent run",
    "model_args": "identifies the agent model and its decoding settings",
    "flags": "agent configuration",
    "agent": "identifies the agent",
    "model": "identifies the agent model",
    "valid": "run bookkeeping",
    "seed": "run bookkeeping",
    "package_version": "run bookkeeping",
}
# Per-step fields that never enter the monitor view.
FENCED_STEP = {
    "chat_messages": "the agent's own prompts, which repeat evidence the view already holds",
    "stats": "agent token and cost telemetry, not evidence of the outcome",
    "axtree_obj": "raw object form of the axtree text, which is kept",
    "bounding_boxes": "duplicate of extra_element_properties, which is kept as elements",
}

# Keys that must not appear at any depth of a monitor-view record.
FORBIDDEN_KEYS = frozenset(
    set(FENCED_TOP_LEVEL) | set(FENCED_STEP) | {
        "cum_reward", "cum_raw_reward", "reward", "rewards", "raw_reward",
        "trajectory_success", "trajectory_side_effect", "trajectory_optimality",
        "trajectory_looping", "annotator_name", "label", "labels", "judgment",
        "judgments", "terminated", "truncated", "err_msg", "stack_trace",
    }
)
# Substrings that must not appear anywhere in a serialized monitor-view record.
FORBIDDEN_SUBSTRINGS = (
    "cum_reward", "cum_raw_reward", "trajectory_success", "trajectory_side_effect",
    "trajectory_optimality", "trajectory_looping", "annotator_name", "summary_info",
    "judgments/", "annotations.csv",
)

# Label vocabulary of data/annotations.csv.
LABEL_COLUMNS = (
    "trajectory_success", "trajectory_side_effect",
    "trajectory_optimality", "trajectory_looping",
)
ALLOWED_LABEL_VALUES = {
    "trajectory_success": {"Successful", "Unsuccessful", "Unsure"},
    "trajectory_side_effect": {"Yes", "No", "Unsure"},
    "trajectory_looping": {"Yes", "No", "Unsure"},
    "trajectory_optimality": {
        "1. Complete Failure", "2. Suboptimal", "3. Somewhat Optimal",
        "4. Completely Optimal", "Unsure",
    },
}
SUCCESS_BINARY = {"Successful": 1, "Unsuccessful": 0, "Unsure": None}

# Python package that bundles annotations.csv; it must not be importable by the
# detector runtime.
LABEL_BUNDLING_PACKAGE = "agent_reward_bench"

# How a trajectory ended, recorded in provenance for development runs only (see
# normalize.ending_type). Every ARB trajectory has at most 31 screenshots, i.e. at most
# 30 actions plus the final observation, so 30 actions means the run reached the cap.
STEP_CAP = 30
# 'environment' means the run ended before the cap without a final agent message. In
# BrowserGym 0.13.3 (webarena and visualwebarena task.validate) that happens when the
# checker scores above zero, or at reward 0 when a tab leaves the benchmark's sites. In
# WorkArena 0.4.1 L2 tasks it happens when every subtask validates, or at reward 0 when a
# subtask stops the episode. Harness errors can also end a run. Never select or order
# trajectories on this field: it mostly stands in for the rule-based verdict.
AGENT_STOP_ACTIONS = ("send_msg_to_user", "report_infeasible")
