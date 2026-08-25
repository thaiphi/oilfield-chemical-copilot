do $$
begin
    if exists (select 1 from chunks limit 1) then
        raise exception 'Milestone 3 migration changes chunks.embedding to vector(384); refusing to run while chunks contains rows';
    end if;
end $$;

drop index if exists chunks_embedding_hnsw_idx;

alter table chunks
    add column if not exists source_path text,
    add column if not exists parser_type text,
    add column if not exists page_or_sheet text,
    add column if not exists chunk_index integer,
    add column if not exists embedding_model text;

update chunks set
    source_path = coalesce(source_path, source_file),
    parser_type = coalesce(parser_type, 'unknown'),
    page_or_sheet = coalesce(page_or_sheet, page_sheet),
    chunk_index = coalesce(chunk_index, 0)
where false;

alter table chunks
    alter column source_path set not null,
    alter column parser_type set not null,
    alter column page_or_sheet set not null,
    alter column chunk_index set not null,
    alter column embedding type vector(384);

create index if not exists chunks_parser_type_idx on chunks (parser_type);
create index if not exists chunks_embedding_model_idx on chunks (embedding_model);
create index if not exists chunks_embedding_hnsw_idx on chunks using hnsw (embedding vector_cosine_ops);
