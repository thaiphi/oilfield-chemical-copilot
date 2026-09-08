-- Candidate Corpus V2 only.  This directory is intentionally outside db/migrations.
create extension if not exists vector;

create table corpus_release (
    singleton boolean primary key default true check (singleton),
    release_id text not null unique,
    index_manifest_sha256 char(64) not null,
    source_register_sha256 char(64) not null,
    created_at timestamptz not null default now(),
    unique (release_id, index_manifest_sha256)
);

create function reject_corpus_release_mutation() returns trigger language plpgsql as $$
begin
    raise exception 'corpus_release is immutable';
end;
$$;

create trigger corpus_release_immutable
before update or delete on corpus_release
for each row execute function reject_corpus_release_mutation();

create table chunks (
    chunk_id text not null,
    release_id text not null,
    source_id text not null,
    source_sha256 char(64) not null,
    manifest_sha256 char(64) not null,
    page_or_sheet text not null,
    embedding_model text not null,
    embedding vector(384) not null,
    content text not null,
    created_at timestamptz not null default now(),
    primary key (release_id, chunk_id),
    unique (release_id, chunk_id),
    foreign key (release_id, manifest_sha256)
        references corpus_release (release_id, index_manifest_sha256)
);

create index v2_chunks_release_source_idx on chunks (release_id, source_id);
create index v2_chunks_embedding_hnsw_idx on chunks using hnsw (embedding vector_cosine_ops);
