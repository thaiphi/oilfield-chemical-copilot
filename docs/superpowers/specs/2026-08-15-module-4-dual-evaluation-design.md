# Module 4 Dual Evaluation Design

## Goal

Build a course-aligned evaluation workflow that produces a transparent public teaching example and a meaningful local evaluation against the comprehensive handout corpus, without committing private questions, source mappings, answers, or detailed results.

## Why Two Evaluation Packs

The committed public corpus is intentionally small. Its retrieval and live-RAG fixtures are appropriate for explaining metric mechanics and verifying reproducibility, but they cannot establish quality on the local course handouts.

The handouts are source material, not an evaluation set. Meaningful retrieval evaluation requires reviewed questions with expected evidence. Meaningful answer evaluation also requires an expected citation or abstention expectation. Those labels must remain local because they reveal how the private corpus is organized and interpreted.

## Scope

### Public Teaching Pack

Reuse the committed public retrieval and live-RAG fixtures to run the real local system in `vector` and `hybrid` modes. Produce an aggregate-only teaching report showing:

- question count;
- Hit Rate@5 and MRR@5 for retrieval;
- citation pass/fail counts;
- abstention pass/fail counts;
- configuration labels and safe runtime status.

The public pack may identify its public fixture version, but it must not write runtime answers, evidence excerpts, source paths, chunk IDs, raw provider errors, or credentials into durable reports.

### Local Handout Pack

Create one controller-owned directory outside Git:

```text
.private/evaluation/module4_handouts/
  dataset/
  review/
  sealed/
  results/
```

The local dataset contains reviewed questions, expected evidence identifiers, answer expectations, and optional category labels. It stays in the primary workspace, not in worktrees, commits, reports, or agent packets.

The local runner must verify the dataset is sealed before scoring. It runs once per sealed dataset hash and writes detailed local diagnostics only under `.private`. Its durable project report contains aggregates only: case count, pass/fail totals, metric values, category totals, model labels, sealed-dataset SHA-256, and a clear limitation statement.

## Architecture

Build a small Module 4 runner around the existing evaluation primitives rather than introducing a second metric implementation:

1. Retrieval evaluation uses the existing `EvaluationCase`, ranked-hit, Hit Rate@k, and MRR primitives.
2. Live-RAG evaluation uses the existing citation and abstention checks around the actual `BasicRagService` path.
3. The new orchestration layer standardizes preflight, run provenance, aggregate-only rendering, and the public-versus-local privacy boundary.
4. The semantic-grounding holdout is not rerun or changed. It remains separate safety evidence and is explained as a formatter-boundary test rather than an answer-quality metric.

The production RAG service, prompt builder, retrievers, generator settings, and private corpus files are out of scope. Evaluation observes the system; it does not tune the system after viewing results.

## Data Flow

```text
public fixtures or sealed local handout fixture
  -> preflight and manifest validation
  -> retrieval metrics
  -> actual RAG capture
  -> citation and abstention checks
  -> local detailed diagnostics only
  -> aggregate-only durable report
```

For the handout pack, questions and evidence identifiers remain in memory during the run and are written only to the local `.private` result directory. Aggregate reports must reject unsafe keys and string values that resemble source content, paths, raw errors, prompts, answers, excerpts, or identifiers.

## Failure Policy

- Invalid, unsealed, previously scored, or out-of-bound local datasets fail before any RAG call.
- A public run rejects a database whose chunk manifest differs from the committed public sample manifest.
- A local run rejects a database whose stored embedding model does not match the configured model label.
- Missing local Ollama, PostgreSQL, or configuration prerequisites produce a sanitized unavailable status rather than a partial report.
- A run may not alter retrieval configuration, prompts, models, corpus files, or evaluation cases after it observes results. A follow-up change requires a separate approved experiment.

## Acceptance Evidence

1. Focused tests cover public/local path boundaries, sealing, one-shot protection, report safety, metric aggregation, and preflight failure before RAG construction.
2. A fresh public live run produces a reviewable aggregate report using the actual vector and hybrid pipelines.
3. A reviewed local handout dataset is sealed and scored once, with detailed artifacts remaining under `.private` and an aggregate-only durable report.
4. The Module 4 teaching review explains ground truth, Hit Rate@5, MRR@5, citation checks, abstention checks, and the limits of all observed metrics.
5. Module 4 is locked only after the public and local evidence are reviewed, tests pass, and the user approves the lock commit.

## Out Of Scope

- Changing RAG behavior to improve a score.
- Rerunning the sealed semantic-grounding V6 holdout.
- Sending private handout material to OpenAI or committing it to Git.
- Building a dashboard before the metrics and reports are trustworthy.

## Design Self-Review

- No placeholders: all required artifacts, privacy boundaries, and acceptance evidence are explicit.
- Consistency: both packs use the same existing metric primitives; only corpus and artifact visibility differ.
- Scope: evaluation orchestration and evidence are in scope; production RAG changes and dashboards are not.
- Ambiguity resolved: the local handout fixture is controller-owned in the primary workspace, sealed before use, and scored once per dataset hash.
