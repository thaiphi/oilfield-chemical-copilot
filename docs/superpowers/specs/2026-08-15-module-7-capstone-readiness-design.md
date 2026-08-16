# Module 7 Capstone Readiness Design

## Goal

Make the Oilfield Chemical Troubleshooting Copilot reproducible and reviewable from a public GitHub commit without publishing private source material, private evaluation fixtures, credentials, or raw application telemetry.

## Reviewer Model

A reviewer clones a fixed commit, uses the committed public sample corpus, installs the documented local prerequisites, starts the local services, indexes only `data/sample`, opens Streamlit and Grafana, and follows the evidence links. The project remains local-first: Ollama runs on the reviewer's machine and Grafana is bound to localhost.

## Scope

In scope:

- A public reviewer guide with exact setup, public ingestion/indexing, app, evaluation, and monitoring steps.
- A rubric-to-evidence map for the official course criteria.
- Automated checks that the reviewer-facing docs retain the public-only and privacy boundaries.
- A live public-sample verification using the existing application boundaries.
- A sanitized readiness report and status updates.

Out of scope:

- Private corpus onboarding or claims that every private file has been handled.
- Retrieval, generation, claim-scope, tool, or evaluation behavior changes.
- Reranking and query rewriting, because the existing evidence has not identified a measured retrieval gap requiring either technique.
- Hosted deployment, alerts, external credentials, commits, or pushes.

## Evidence Model

The project already contains the required technical pieces: public ingestion through Kestra and Python, PGVector keyword/vector/hybrid retrieval, public evaluation artifacts, Streamlit, closed aggregate feedback, and a six-panel local Grafana dashboard. Module 7 makes their public reproduction path explicit and tests that the documentation never directs a reviewer toward private data.

The final report may record only command status, public aggregate counts, panel availability, test results, and residual limitations. It cannot include private paths, private corpus content, raw prompt/answer text, source excerpts, secrets, or detailed private evaluation results.

## Residual Gate

Module 5 remains active until the user reviews and locks it. Module 7 may complete its implementation and evidence package before that decision, but it cannot claim final capstone readiness or lock the end-to-end module while Module 5 remains unreviewed.
