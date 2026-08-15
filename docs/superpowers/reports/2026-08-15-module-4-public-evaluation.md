# Module 4 Public Evaluation

Status: success

The approved public evaluation pack ran against the active local vector and hybrid RAG boundaries with the claim-scope policy enabled. The committed public answer-evaluation dataset SHA-256 was `0271efed1c11af594a6816ab4478632c84a4f630e64575c54f9856089f5fa4d2`.

| Mode | Retrieval cases | Hit Rate@5 | MRR@5 | Citation pass/fail | Abstention pass/fail |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vector | 6 | 1.000 | 1.000 | 9/3 | 12/0 |
| Hybrid | 6 | 1.000 | 0.917 | 10/2 | 10/2 |

## Interpretation

Both modes retrieved expected public evidence in the first five results for every answerable case. Vector placed the first expected evidence earlier on this small pack. That retrieval result does not establish answer quality: vector still had three deterministic citation failures, while hybrid had two citation failures and two abstention failures.

The result is a transparent classroom measurement, not chemistry validation, operational advice, a private-corpus score, or proof that either retrieval mode is production-ready. No score-driven RAG change follows from this run. The sealed local handout pack remains the separate project-relevant measurement.
