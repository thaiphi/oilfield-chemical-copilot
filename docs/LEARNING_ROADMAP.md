# Learning Roadmap

**Project:** Oilfield Chemical Troubleshooting Copilot
**Purpose:** Keep the learning sequence explicit while preserving work that was implemented ahead of its lesson.

## Status Vocabulary

- **Active lesson:** the module currently being studied, explained, tested, and locked.
- **Implemented early:** working code and tests exist, but the feature is not yet being taught or formally locked in its intended module.
- **Scaffold only:** a placeholder, schema, plan, or helper exists without the complete runtime behavior and acceptance evidence.
- **Not started:** no meaningful implementation or scaffold exists yet.

## Module Map

| Module | Learning focus | Status | Existing evidence | Teaching boundary |
| --- | --- | --- | --- | --- |
| 1. Core RAG fundamentals | Inventory, parsing, chunking, embeddings, keyword/vector/hybrid retrieval, source-grounded Streamlit answers, and weak-evidence fallback. | Active lesson | Milestones 1-5 in `docs/PROJECT_STATUS.md`; local Ollama RAG and RRF tests. | Learn the retrieval-to-answer path before treating advanced controls as part of the lesson. |
| 2. Evaluation | Retrieval metrics, synthetic-answer checks, live RAG comparison, and interpretation of aggregate results. | Implemented early | `eval/`, `src/oilfield_chemical_copilot/evaluation/`, and associated tests. | Preserve the evaluators; revisit their metrics, fixtures, and limitations as the evaluation lesson. |
| 3. Safety and claim scope | Abstention policy, sealed holdout, production policy boundary, and citation-selection control. | Implemented early | `abstention_policy.py`, production-hardening materials, citation diagnostics, and service tests. | Revisit as a safety-boundary lesson; it does not establish chemistry correctness or field readiness. |
| 4. Tool calling | Explicit product-ppm water-basis calculation, validation, and scope-first routing. | Implemented early | `tools/chemical_dosage.py`, Streamlit route tests, and tool contract tests. | Revisit as constrained deterministic tool use, not agentic or field-ready chemical treatment. |
| 5. Monitoring | Aggregate response/routing outcomes and latency measurement; later persistence and dashboard design. | Implemented early | `observability/aggregate_monitoring.py` and its unit/route tests. | Revisit only after the monitoring lesson begins. The in-memory collector is not persistent telemetry. |
| 6. Orchestration | Coordinate ingestion, retrieval, generation, tools, and safeguards. | Scaffold only | `flows/kestra/ingest.yml` and the project workflow-role configuration. | No claim of a complete runtime orchestration system. |
| 7. Capstone readiness | End-to-end quality gates, deployment readiness, and documented operational limits. | Not started | Docker and deployment files are prerequisites, not readiness evidence. | Start only after the preceding modules are taught and locked. |

## Cross-Cutting Scaffolds

- Database tables for conversations, feedback, latency, retrieved chunks, and tool calls are **scaffold only**. The current runtime does not write raw-content telemetry to them.
- Grafana provisioning material is **scaffold only**. No dashboard is treated as operational monitoring evidence.
- The water-analysis helper is a starter utility. Detailed water analysis, recommendations, and field-treatment decisions are outside the active lesson.
- Sol, Luna, and Terra workflow files are project-delivery infrastructure. They are not a substitute for the course orchestration module.

## Working Rule

Implemented-early code stays in the repository with its tests. When its intended module begins, the lesson starts by reviewing the existing boundary, explaining the design, rerunning its verification, and then deciding whether to extend it. No preserved feature is deleted merely because it was implemented early.
