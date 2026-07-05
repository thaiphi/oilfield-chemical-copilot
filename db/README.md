# Database Scaffold

`migrations/0001_oilfield_chemical_copilot_schema.sql` creates the initial PostgreSQL + PGVector storage model.

The migration is storage-only. It prepares tables for chunks, embeddings, conversations, feedback, latency events, retrieved chunks, and tool calls.

TODO: Confirm the embedding dimension before production use. The current scaffold uses `vector(1536)` for OpenAI `text-embedding-3-small`.
