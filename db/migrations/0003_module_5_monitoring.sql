create table if not exists monitoring_request_hourly (
    bucket_start timestamptz not null,
    outcome text not null check (outcome in (
        'rag_answered',
        'rag_weak_evidence',
        'scope_abstained',
        'tool_calculated',
        'tool_input_invalid',
        'rag_configuration_error'
    )),
    retrieval_mode text not null check (retrieval_mode in ('vector', 'hybrid', 'not_applicable')),
    request_count bigint not null check (request_count >= 0),
    latency_count bigint not null check (latency_count >= 0),
    latency_total_ms double precision not null check (latency_total_ms >= 0),
    latency_minimum_ms double precision check (latency_minimum_ms >= 0),
    latency_maximum_ms double precision check (latency_maximum_ms >= 0),
    primary key (bucket_start, outcome, retrieval_mode),
    check (latency_count <= request_count),
    check (latency_minimum_ms is null or latency_maximum_ms is null or latency_minimum_ms <= latency_maximum_ms)
);

create index if not exists monitoring_request_hourly_bucket_start_idx
    on monitoring_request_hourly (bucket_start);

create table if not exists monitoring_feedback_hourly (
    bucket_start timestamptz not null,
    feedback_value text not null check (feedback_value in ('helpful', 'needs_work')),
    retrieval_mode text not null check (retrieval_mode in ('vector', 'hybrid', 'not_applicable')),
    feedback_count bigint not null check (feedback_count >= 0),
    primary key (bucket_start, feedback_value, retrieval_mode)
);

create index if not exists monitoring_feedback_hourly_bucket_start_idx
    on monitoring_feedback_hourly (bucket_start);
