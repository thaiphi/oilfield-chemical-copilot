# Module 4 Teaching Review

## What Was Measured

The public pack is a transparent classroom example. The sealed local v2 pack measured the same active vector and hybrid RAG boundaries against six reviewed handout-grounded cases and six closed-scope cases. Only its aggregate report is durable.

| Scope | Mode | Retrieval cases | Hit Rate@5 | MRR@5 | Citation pass/fail | Abstention pass/fail |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Public | Vector | 6 | 1.000 | 1.000 | 9/3 | 12/0 |
| Public | Hybrid | 6 | 1.000 | 0.917 | 10/2 | 10/2 |
| Local handout v2 | Vector | 6 | 0.500 | 0.500 | 7/5 | 8/4 |
| Local handout v2 | Hybrid | 6 | 0.833 | 0.722 | 8/4 | 10/2 |

## Core Concepts

Ground truth is the reviewed mapping from a question to the evidence that should support it. It is not an answer written by the model and it is not whichever source retrieval happened to return.

Hit Rate@5 asks whether at least one reviewed evidence chunk appeared among the first five retrieved chunks. On the six local grounded cases, vector succeeded on three and hybrid on five. The public result was perfect because the public corpus is tiny and intentionally simple; the local result is therefore the more useful retrieval signal for this project.

MRR@5 rewards placing the first expected chunk earlier. A first-place hit contributes `1`; a second-place hit contributes `1/2`; a miss contributes `0`. Hybrid's local `0.722` exceeds vector's `0.500`, so expected evidence tended to appear earlier for hybrid as well as more often.

Citation checks are structural: did the final answer cite only allowed retrieved evidence when a citation was expected? Abstention checks are behavioral: did a closed-scope question safely decline to answer? Neither proves that an answer is chemically true, safe for a named well, or a valid operating recommendation.

## Decision

Hybrid is the better-supported future retrieval-improvement candidate on this local measurement. It is not a production winner: four citation and two abstention failures remain. Do not tune production RAG from this single pack. Any retrieval or prompt change requires a separate approved experiment and a fresh sealed evaluation fixture.
