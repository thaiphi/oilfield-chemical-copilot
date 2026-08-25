# Module 3 Kestra Live Run

## Scope

One local public-sample execution of the five-task ingestion flow using `granite-embedding:latest`.

## Result

Overall status: `SUCCESS`.

| Task | Status |
| --- | --- |
| Inventory | SUCCESS |
| Parse and chunk | SUCCESS |
| Embed and load | SUCCESS |
| Validate counts | SUCCESS |
| Publish aggregate metrics | SUCCESS |

## Aggregate Evidence

| Field | Value |
| --- | --- |
| Source files | 10 |
| Chunks | 11 |
| Expected indexed chunks | 11 |
| Actual indexed chunks | 11 |
| Embedding model | granite-embedding:latest |
| dlt aggregate publication | SUCCESS |

## Privacy Boundary

The worker image includes the public sample corpus only. The flow passes only declared public artifacts between tasks. The dlt destination receives the fixed six-field aggregate record. This report deliberately excludes source text, source names, paths, chunk identifiers, vectors, credentials, raw provider errors, and execution identifiers.
