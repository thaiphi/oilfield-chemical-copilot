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
| 4. Evaluation | Ground truth, Hit Rate, MRR, answer evaluation, and LLM-as-a-judge. | Implemented early | Retrieval, synthetic answer, live RAG, diagnosis, and policy evaluation modules. | Teach the metrics and review evaluation boundaries under the official module number. |
| 5. Monitoring | Streamlit chat, stored feedback, dashboards, Grafana, and Docker Compose. | Scaffold only | Database schema, Grafana material, feedback placeholders, and a process-local aggregate monitor exist. | Build persistent safe telemetry and the required dashboard evidence; the in-memory collector alone does not complete this module. |
| 6. Best Practices | Hybrid search, reranking, RRF, and query rewriting. | Implemented early | Hybrid retrieval and RRF are complete. | Evaluate reranking and query rewriting only when the metrics identify a retrieval gap. |
| 7. End-to-End Project | Reproducible application, ingestion, evaluation, monitoring, interface, and deployment evidence. | Scaffold only | Streamlit app, Docker Compose, sample corpus, and README exist. | Complete the remaining orchestration, monitoring, corpus, and reviewer-ready documentation evidence. |

## Learning Lock Rule

No official module is locked merely because related code was implemented early. A module locks only after its official handout objectives, practical check, and acceptance evidence have been reviewed under this corrected mapping.

## Cross-Cutting Scaffolds

- Database tables for conversations, feedback, latency, retrieved chunks, and tool calls are **scaffold only**. The current runtime does not write raw-content telemetry to them.
- Grafana provisioning material is **scaffold only**. No dashboard is treated as operational monitoring evidence.
- The water-analysis helper is a starter utility. Detailed water analysis, recommendations, and field-treatment decisions are outside the active lesson.
- Sol, Luna, and Terra workflow files are project-delivery infrastructure. They are not a substitute for the course orchestration module.

## Working Rule

Implemented-early code stays in the repository with its tests. When its intended module begins, the lesson starts by reviewing the existing boundary, explaining the design, rerunning its verification, and then deciding whether to extend it. No preserved feature is deleted merely because it was implemented early.
