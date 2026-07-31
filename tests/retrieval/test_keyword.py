from __future__ import annotations

from oilfield_chemical_copilot.ingest.models import ChunkMetadata, LoadedChunk
from oilfield_chemical_copilot.retrieval.keyword import KeywordSearchIndex
from oilfield_chemical_copilot.retrieval.models import RetrievalHit


def _chunk(chunk_id: str, text: str, topic: str, source_file: str) -> LoadedChunk:
    return LoadedChunk(
        text=text,
        metadata=ChunkMetadata(
            source_file=source_file,
            source_path=f"C:/sample/{source_file}",
            topic=topic,
            parser_type="text",
            page_or_sheet="document",
            chunk_index=0,
            chunk_id=chunk_id,
        ),
    )


def _retrieval_hit(chunk_id: str, text: str, topic: str) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        text=text,
        score=0.5,
        retrieval_method="vector",
        source_file=f"{chunk_id}.md",
        source_path=f"C:/stored/{chunk_id}.md",
        topic=topic,
        parser_type="stored-text",
        page_or_sheet="sheet 2",
        chunk_index=4,
        metadata={"storage": "persisted"},
    )


def test_keyword_search_ranks_relevant_oilfield_chunks() -> None:
    index = KeywordSearchIndex.from_chunks(
        [
            _chunk(
                "iron", "Iron sulfide black solids and THPS treatment", "iron_sulfide", "iron.md"
            ),
            _chunk("scale", "Scale water analysis chloride sulfate barium", "scale", "scale.md"),
            _chunk(
                "dosage", "Chemical dosage gallons per day ppm calculation", "dosage", "dosage.md"
            ),
        ]
    )

    hits = index.search("iron sulfide deposits", limit=2)

    assert hits[0].chunk_id == "iron"
    assert hits[0].retrieval_method == "keyword"
    assert hits[0].score > 0


def test_keyword_search_supports_topic_filter_and_empty_query() -> None:
    index = KeywordSearchIndex.from_chunks(
        [
            _chunk("scale", "Scale chloride sulfate", "scale", "scale.md"),
            _chunk("water", "Water analysis chloride", "water_analysis", "water.md"),
        ]
    )

    assert index.search("", limit=5) == []
    hits = index.search("chloride", limit=5, topic="water_analysis")
    assert [hit.chunk_id for hit in hits] == ["water"]


def test_keyword_index_can_be_built_from_stored_hits() -> None:
    stored_hit = _retrieval_hit("scale", "SCALE-X inhibitor compatibility", "scale")
    index = KeywordSearchIndex.from_hits(
        [stored_hit, _retrieval_hit("corrosion", "oxygen scavenger program", "corrosion")]
    )

    hits = index.search("SCALE-X", limit=5, topic="scale")

    assert [hit.chunk_id for hit in hits] == ["scale"]
    assert hits[0].retrieval_method == "keyword"
    assert hits[0].source_file == stored_hit.source_file
    assert hits[0].source_path == stored_hit.source_path
    assert hits[0].parser_type == stored_hit.parser_type
    assert hits[0].page_or_sheet == stored_hit.page_or_sheet
    assert hits[0].chunk_index == stored_hit.chunk_index
    assert hits[0].metadata == stored_hit.metadata
