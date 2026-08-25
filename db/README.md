# Database Migrations

`migrations/0001_oilfield_chemical_copilot_schema.sql` creates the initial PostgreSQL + PGVector storage model.

`migrations/0002_milestone_3_retrieval.sql` upgrades chunk storage for Milestone 3 retrieval. It adds dedicated parser/source metadata columns, records the embedding model used for each row, and changes `chunks.embedding` to `vector(384)` for local `sentence-transformers/all-MiniLM-L6-v2` and deterministic test embeddings.

Apply migrations to an existing empty chunk table:

```powershell
$env:DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/oilfield_copilot"
uv run python ingestion/apply_migrations.py --database-url $env:DATABASE_URL
```

With Docker Compose:

```powershell
docker compose up -d postgres
docker compose run --rm migrate
```

The Milestone 3 migration refuses to run when `chunks` already contains rows because it changes the vector dimension from 1536 to 384. For an old local volume with disposable sample data, reset the volume and re-index from `data/processed/chunks.jsonl`. For any non-disposable database, back it up first and plan a controlled re-embedding migration instead of forcing the schema change in place.