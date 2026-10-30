"""Rubric v0: the text both judges receive, as system text (A) or developer text (B).

It is ARB's success question plus the protocol's five rules, and the instruction to
answer only in schema S. This version is a DRAFT. It is frozen by hash before prompt
development, and any change after that gets a new version and a new hash. The rubric
names the output fields but asks for a short explanation, not for step-by-step
reasoning, since a field that asks a model to reproduce its reasoning can draw a refusal.
"""

import hashlib

RUBRIC_VERSION = "rubric-v0"
RUBRIC_STATUS = "draft"

SUCCESS_QUESTION = "Was the sequence of actions successful in achieving the goal?"

RULES = (
    "Read the goal leniently, never the evidence: accept any reasonable reading of what the "
    "goal asks for, but credit only what the evidence shows.",
    "The agent's reasoning is a claim, not evidence. Base the verdict on the screenshots, "
    "the page trees, the actions and the recorded errors.",
    "If the evidence is not enough to decide, give the more likely outcome and set "
    "evidence_sufficient to \"no\".",
    "Success at an earlier step counts if no later step undid it.",
    "If the task is infeasible and the agent correctly declared it infeasible, that counts "
    "as success.",
)

RUBRIC_TEXT = (
    "You will judge one run of a web agent. The user message holds the evidence: the task "
    "goal, then each step of the run in order, then the accessibility tree of the final page. "
    "Each step gives its URL, the action the agent chose, the elements that action named, the "
    "agent's own reasoning and the error recorded at that step; when the step has a "
    "screenshot, the image follows that step's text. The field_notes in the evidence explain "
    "each field.\n\n"
    f"Question: {SUCCESS_QUESTION}\n\n"
    "Rules:\n"
    + "".join(f"{i}. {rule}\n" for i, rule in enumerate(RULES, 1))
    + "\nAnswer only with one JSON object in schema S, with these fields and no others:\n"
    "- verdict: \"success\" or \"failure\";\n"
    "- p_failure: your probability, from 0 to 1, that the run failed;\n"
    "- evidence_sufficient: \"yes\" or \"no\";\n"
    "- evidence_step_ids: the step numbers whose evidence the verdict rests on;\n"
    "- rationale: a short explanation of the verdict, at most 100 words.\n"
)

RUBRIC_SHA256 = hashlib.sha256(RUBRIC_TEXT.encode("utf-8")).hexdigest()
