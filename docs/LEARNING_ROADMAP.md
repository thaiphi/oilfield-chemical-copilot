# Learning Roadmap

**Project:** Oilfield Chemical Troubleshooting Copilot
**Purpose:** Keep the learning sequence explicit while preserving work that was implemented ahead of its lesson.

## Status Vocabulary

- **Active lesson:** the module currently being studied, explained, tested, and locked.
- **Implemented early:** working code and tests exist, but the feature is not yet being taught or formally locked in its intended module.
- **Scaffold only:** a placeholder, schema, plan, or helper exists without the complete runtime behavior and acceptance evidence.
- **Not started:** no meaningful implementation or scaffold exists yet.

## Authoritative Course Module Map

This map follows the comprehensive handout at `C:\Users\Thai Phi\Documents\Codex\oilfield-chemical-copilot\docs\02_MODULE_MAPPING.md`. It supersedes the earlier local numbering, which incorrectly treated evaluation and safety work as Modules 2 and 3.

| Official module | Learning focus | Status | Existing evidence | Remaining teaching or implementation boundary |
| --- | --- | --- | --- | --- |
| 1. Agentic RAG | Keyword search, prompt construction, function calling, and an agentic routing loop. | Locked | Source-grounded RAG, an explicit dosage contract, and an opt-in one-decision Ollama planner with two controller-owned tools are public-test verified. A local Granite smoke test returned exactly one tool call. | Teaching review completed on 2026-08-14. Future changes require a new scoped plan and verification. |
| 2. Vector Search | Embeddings, semantic search, minsearch/sqlitesearch/PGVector, and metadata filtering. | Locked | Ollama embeddings, PGVector, keyword/vector/hybrid retrieval, and RRF tests. Focused public tests passed; a count-only live Granite-to-PGVector search returned one topic-filtered result. | Teaching review completed on 2026-08-15. Future changes require a new scoped plan and verification. |
| 3. Orchestration | Reliable multi-step ingestion and RAG workflows with Kestra. | Locked | A public-only Kestra flow completed inventory -> parse/chunk -> embed/load -> count validation -> aggregate dlt publication using local Granite embeddings. | Teaching review completed on 2026-08-15. Future changes require a new scoped plan and verification. |
| 4. Evaluation | Ground truth, Hit Rate, MRR, answer evaluation, and LLM-as-a-judge. | Locked | Public and sealed-local v2 evaluation completed. Local hybrid outperformed vector on Hit Rate@5 (`0.833` vs `0.500`), MRR@5 (`0.722` vs `0.500`), citations, and abstention, while failures remained. | Locked on 2026-08-15. Any RAG change needs a new approved experiment and fresh fixture. |
| 5. Monitoring | Streamlit chat, stored feedback, dashboards, Grafana, and Docker Compose. | Locked | Streamlit writes only closed hourly aggregates and aggregate feedback; Docker Compose provisions a localhost-only Grafana viewer, a read-only database role, six panels with built-in explanations, and an explicit fixed synthetic seed. The local reviewer path, dashboard review, and teaching review are complete. | Locked on 2026-08-16. Future monitoring expansion needs a separate approved scope. |
| 6. Best Practices | Hybrid search, reranking, RRF, and query rewriting. | Implemented early | Hybrid retrieval and RRF are complete. | Evaluate reranking and query rewriting only when the metrics identify a retrieval gap. |
| 7. End-to-End Project | Reproducible application, ingestion, evaluation, monitoring, interface, and deployment evidence. | Locked | A public reviewer guide, rubric evidence map, static documentation contract, live public 11-chunk Granite index, local endpoint checks, aggregate-only readiness report, and the locked Module 5 dashboard join the Streamlit app, Docker Compose, and sample corpus. | Locked on 2026-08-16. The package is reproducible locally, not a hosted or production-ready deployment. |

## Learning Lock Rule

No official module is locked merely because related code was implemented early. A module locks only after its official handout objectives, practical check, and acceptance evidence have been reviewed under this corrected mapping.

## Cross-Cutting Scaffolds

- Database tables for conversations, feedback, latency, retrieved chunks, and tool calls are **scaffold only**. The current runtime does not write raw-content telemetry to them.
- Grafana provisioning is **implemented for local synthetic review**. It is not hosted, production telemetry, alerting, or operational evidence.
- The water-analysis helper is a starter utility. Detailed water analysis, recommendations, and field-treatment decisions are outside the active lesson.
- Sol, Luna, and Terra workflow files are project-delivery infrastructure. They are not a substitute for the course orchestration module.

## Working Rule

Implemented-early code stays in the repository with its tests. When its intended module begins, the lesson starts by reviewing the existing boundary, explaining the design, rerunning its verification, and then deciding whether to extend it. No preserved feature is deleted merely because it was implemented early.
