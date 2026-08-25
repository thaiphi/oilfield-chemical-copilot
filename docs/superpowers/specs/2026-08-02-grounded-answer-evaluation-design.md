# Grounded Answer Evaluation Design

## Goal

Measure whether public-sample RAG answers are grounded, relevant, appropriately limited, and free of unsafe certainty. This is a learning baseline, not chemistry validation or operational approval.

## Design

Each public case contains a question, retrieved public evidence IDs, and an expected evidence-sufficiency outcome. Deterministic checks reject missing/invalid citations, unsupported citation IDs, and answers that fail to abstain when evidence is intentionally insufficient.

A judge returns strict JSON rubric scores from 1 to 5 for `groundedness`, `relevance`, `limitation_awareness`, and `operational_certainty`. Ollama/Granite is the default local judge; OpenAI is optional. Judge output is advisory and reported separately from deterministic failures because a judge can be wrong or biased, especially when it resembles the answer model.

## Privacy

Use public synthetic cases only. Reports contain case IDs, aggregate scores, rubric labels, provider/model-safe identifiers, and deterministic failure labels. They never contain prompt text, answer text, evidence excerpts, filenames, source paths, database URLs, or private data. Judge failures produce an explicit unavailable status, not a fabricated score.

## Teaching Limits

Citation validity proves structural grounding, not chemical correctness. A high judge score does not prove safe operational advice. Comparing judge providers and human review are later work; this milestone establishes a repeatable public baseline.
