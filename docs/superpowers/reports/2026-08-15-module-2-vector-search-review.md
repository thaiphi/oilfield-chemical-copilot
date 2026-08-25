# Module 2 Vector Search Teaching Review

**Date:** 2026-08-15  
**Status:** Locked after public verification, count-only live evidence, and practical teaching review.

## Course Objectives Mapped To The Project

- **Embeddings:** the configured Granite embedding provider converts chunks and questions into 384-dimensional vectors.
- **Semantic search:** PGVector ranks stored vectors by cosine distance to the question vector.
- **PGVector storage:** each persisted chunk includes its vector, embedding-model label, topic, and provenance metadata.
- **Metadata filtering:** a topic filter narrows eligible chunks before vector ranking.

## Verification Evidence

- Focused public embedding, vector-retriever, retrieval-pipeline, and PGVector tests: 35 passed.
- Local Granite embedding smoke check: one compatible 384-dimensional vector returned.
- Count-only local PGVector smoke check: the project database connected, the chunks schema was present, and one result was returned with the `scale` topic filter.

## Teaching Review

An embedding is useful only relative to the model that created it. The project therefore stores an `embedding_model` label and searches only chunks created by the same model. Semantic similarity helps find related material with different wording, but it does not replace citations, answer grounding, or claim-scope safety controls.

Module 2 was reviewed and locked on 2026-08-15. No application behavior changed during this teaching review.
