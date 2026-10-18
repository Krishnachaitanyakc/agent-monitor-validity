# Can We Trust the Failure Detectors?

A validity audit of trace-based monitoring for LLM agents.

**Status: pre-round scaffolding.** This repository hosts the scaffolding for a proposed SPAR Fall 2026 research project (mentor: Kc Balusu). Nothing here is a result. If the project is selected, the study runs September to December 2026, and this repository becomes its working home.

## The question

Agent monitoring runs on failure detectors: trace-signal heuristics, LLM judges, and hybrids. This project measures how valid they are, not merely how consistent: do they recover the judgments careful humans make about the same traces? We will audit deployed detector classes against double-annotated, reliability-quantified human ground truth over real agent traces, with a preregistered analysis and blinding, and release the labeled corpus openly. Safety-salient failure classes (unsafe tool use, guardrail bypass, specification-gaming-shaped behavior) are deliberately oversampled and reported per class.

## Design commitments (integrity by construction)

1. A candidate trace pool with per-trace provenance is collected under written inclusion rules (see `scripts/collect_traces.py`) and committed to this repository before preregistration.
2. The stratified study sample is drawn from that pool by a project member other than the mentor, by executing the preregistered selection rules. The mentor never selects traces.
3. The mentor never produces or adjudicates a label.
4. Annotators are blind to detector outputs and to injection status.
5. Judge prompts and detector configurations are frozen at preregistration (OSF), before main-set annotation begins.

These commitments exist because the mentor is the sole author of AgentTelemetry, one of the detectors under audit; the design keeps the audited party out of both labeling and test-set selection.

## Repository layout

- `CHARTER.md`: roles, cadence, gates, authorship policy, escalation rules.
- `codebook/CODEBOOK_v0.md`: draft annotation codebook. Not for annotation use until v1.
- `samples/`: two synthetic sample traces and instructions for the applicant screening exercise. Not part of the study corpus.
- `scripts/collect_traces.py`: trace-collection stub implementing the written inclusion rules.
- `datapath/`, `scripts/arb_datapath.py`, `tests/`: the AgentRewardBench data path (fetch, normalize, dev-only labels, validation, leakage checks). See `docs/DATA_PATH.md`.

## References

- Reliability and validity of LLM judges: https://arxiv.org/abs/2606.19544
- TRAIL (human-annotated agent traces): https://arxiv.org/abs/2505.08638
- MAST (multi-agent failure taxonomy): https://arxiv.org/abs/2503.13657
- Production incident catalog: https://arxiv.org/abs/2606.04056
- AgentTelemetry (AIware 2026): https://doi.org/10.1145/3805760.3814931

## License

Apache-2.0 for code and text in this repository. The labeled trace corpus, when released, will carry its own data license after a documented provenance, terms-of-service, and PII scrub (benchmark rollouts only, never real-user traces).
