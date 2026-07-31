# Hybrid RRF Retrieval Design

## Goal

Improve Module 1 retrieval by combining exact keyword matches with semantic vector matches through Reciprocal Rank Fusion (RRF), then provide the fused evidence to the existing source-grounded RAG flow.

## Scope

- Run keyword and vector retrieval for each non-empty question.
- Fuse ranked results by `chunk_id` with RRF and preserve source metadata.
- Support vector-only and hybrid modes so their behavior can be compared in Streamlit.
- Show the retrieval method provenance for each cited source.
- Add focused tests for exact technical terms, semantic questions, tie handling, topic filtering, and weak-evidence behavior.

## Non-Goals

- No cross-encoder reranker, learned ranker, agentic tools, evaluation dataset, monitoring, or database migration.
- No replacement of PGVector or MinSearch.
- No raw-score normalization or weighted blending.

## RRF Algorithm

For every chunk in either ranked list:

```text
rrf_score = sum(1 / (rrf_k + rank))
```

The default `rrf_k` is `60`. Ranks start at `1`; an absent result contributes no score. A chunk found by both methods is rewarded, while a high result from either method remains eligible.

RRF scores are ranking scores, not similarity percentages. They must not use the vector-only `RAG_MIN_SCORE` threshold.

## Components And Data Flow

1. At app startup, build the existing keyword index from stored chunks and construct the current PGVector retriever.
2. In `vector` mode, retain the current vector-only pipeline and `RAG_MIN_SCORE` gate.
3. In `hybrid` mode, retrieve `HYBRID_CANDIDATE_LIMIT` results from each method, fuse them with `HYBRID_RRF_K`, then apply `HYBRID_MIN_RRF_SCORE` and the current context budget.
4. Each fused hit records `retrieval_method="hybrid"` and metadata identifying whether it was found by `keyword`, `vector`, or both.
5. The existing prompt builder, answer generation, citation validation, and path redaction remain unchanged.

```text
question -> keyword search -> ranked list --\
                                        -> RRF -> evidence -> Granite -> cited answer
question -> vector search  -> ranked list --/
```

## Configuration

```dotenv
RETRIEVAL_MODE=hybrid
HYBRID_CANDIDATE_LIMIT=10
HYBRID_RRF_K=60
HYBRID_MIN_RRF_SCORE=0.015
```

`RETRIEVAL_MODE=vector` remains available for a direct comparison. `RAG_MIN_SCORE` applies only to vector mode; hybrid mode uses `HYBRID_MIN_RRF_SCORE`.

## Errors And Safety

- Empty questions return no hits without calling either retriever.
- If keyword retrieval is empty, vector results can still be fused; likewise for vector retrieval.
- Retrieval failures remain controlled app errors without raw provider details.
- Source paths remain hidden in citations; hybrid provenance may show only retrieval method names and ranks.

## Validation

- Unit tests calculate expected RRF scores and deterministic tie ordering.
- Pipeline tests show exact chemical terms survive keyword retrieval, semantic wording survives vector retrieval, and shared results receive a higher fused rank.
- Streamlit tests cover mode selection and provenance display without source-path exposure.
- Existing vector-only tests retain their current behavior.

## Learning Outcome

This milestone demonstrates retrieval-system composition. Keyword search supplies lexical precision, vector search supplies semantic recall, and RRF combines their separate judgments without assuming their scores share a scale.
