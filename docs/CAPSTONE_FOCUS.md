# Capstone Focus

## Product Thesis

The Oilfield Chemical Troubleshooting Copilot is an evidence-grounded assistant for production-chemistry investigation. Its primary value is helping an engineer understand what to investigate, which evidence supports that direction, what information is missing, and when the available material cannot support a safe conclusion.

It is not a field-treatment prescribing system and it is not a calculator product.

## Primary User Workflow

1. An engineer describes a scale, corrosion, iron-sulfide, paraffin, or produced-water concern and supplies the operating context available to them.
2. The system retrieves relevant approved source chunks.
3. The response presents a bounded troubleshooting brief:
   - evidence-grounded factors to investigate;
   - citations that the user can inspect;
   - missing inputs needed for a stronger assessment; and
   - safe next checks and stated limitations.
4. If the request requires a site-specific determination or field-ready prescription, the system abstains before retrieval or generation.

## Feature Hierarchy

### Primary Experience

- Source-grounded troubleshooting questions.
- Evidence citations and source inspection.
- Weak-evidence fallback and claim-scope abstention.
- Retrieval and answer evaluation that measure whether the system is finding and using evidence correctly.

### Supporting Experience

- Structured identification of missing operational, water-analysis, and process-context inputs.
- Simple evidence review of produced-water and deposit indicators.
- Comparison of vector and hybrid retrieval behavior during evaluation.

### Optional Utility

- The product-ppm water-basis calculator is a constrained deterministic helper. It is not a diagnosis engine, recommendation engine, or primary workflow.

## Explicit Non-Goals

- Field-ready chemical prescriptions.
- Autonomous root-cause determination for a named asset.
- Replacing complete water analysis, compatibility studies, or qualified engineering review.
- Model-selected arbitrary function calls.
- Persistent raw prompt, answer, source, tool-input, or error logging.

## Capstone Success Conditions

- A user can ask a realistic troubleshooting question and receive a useful, cited investigation brief.
- A user can trace each visible citation back to an approved source chunk without seeing an absolute local path.
- The system abstains when evidence is weak or the requested claim exceeds its supported scope.
- Evaluation separates retrieval quality from answer/citation behavior and records only safe aggregates.
- Supporting utilities remain bounded and do not overshadow the evidence-grounded troubleshooting workflow.

## Next Product Decision

Before changing the interface, define the first troubleshooting brief layout: the user context to collect, the evidence panel, the missing-inputs section, and the limitations/next-checks section. This is the highest-value capstone enhancement because it strengthens the primary workflow rather than adding another calculator.
