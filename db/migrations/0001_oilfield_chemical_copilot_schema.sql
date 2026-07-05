create extension if not exists vector;
create extension if not exists pgcrypto;

create table if not exists chunks (
    id uuid primary key default gen_random_uuid(),
    chunk_id text not null unique,
    source_file text not null,
    topic text not null,
    page_sheet text not null,
    content text not null,
    embedding vector(1536),
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists chunks_source_file_idx on chunks (source_file);
create index if not exists chunks_topic_idx on chunks (topic);
create index if not exists chunks_embedding_hnsw_idx on chunks using hnsw (embedding vector_cosine_ops);

create table if not exists conversations (
    id uuid primary key default gen_random_uuid(),
    session_id text not null,
    user_message text not null,
    assistant_message text,
    model text,
    created_at timestamptz not null default now()
);

create index if not exists conversations_session_id_idx on conversations (session_id);

create table if not exists feedback (
    id uuid primary key default gen_random_uuid(),
    conversation_id uuid references conversations (id) on delete cascade,
    rating text not null,
    comment text,
    created_at timestamptz not null default now()
);

create index if not exists feedback_conversation_id_idx on feedback (conversation_id);

create table if not exists latency_events (
    id uuid primary key default gen_random_uuid(),
    conversation_id uuid references conversations (id) on delete cascade,
    event_name text not null,
    latency_ms integer not null,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create index if not exists latency_events_conversation_id_idx on latency_events (conversation_id);

create table if not exists retrieved_chunks (
    id uuid primary key default gen_random_uuid(),
    conversation_id uuid references conversations (id) on delete cascade,
    chunk_id text not null references chunks (chunk_id) on delete cascade,
    retrieval_method text not null,
    rank integer not null,
    score double precision,
    created_at timestamptz not null default now()
);

create index if not exists retrieved_chunks_conversation_id_idx on retrieved_chunks (conversation_id);
create index if not exists retrieved_chunks_chunk_id_idx on retrieved_chunks (chunk_id);

create table if not exists tool_calls (
    id uuid primary key default gen_random_uuid(),
    conversation_id uuid references conversations (id) on delete cascade,
    tool_name text not null,
    arguments jsonb not null default '{}'::jsonb,
    result jsonb,
    latency_ms integer,
    created_at timestamptz not null default now()
);

create index if not exists tool_calls_conversation_id_idx on tool_calls (conversation_id);
