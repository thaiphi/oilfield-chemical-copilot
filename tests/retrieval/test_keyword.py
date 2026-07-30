from __future__ import annotations

from oilfield_chemical_copilot.ingest.models import ChunkMetadata, LoadedChunk
from oilfield_chemical_copilot.retrieval.keyword import KeywordSearchIndex


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


def test_keyword_search_ranks_relevant_oilfield_chunks() -> None:
    index = KeywordSearchIndex.from_chunks(
        [
            _chunk("iron", "Iron sulfide black solids and THPS treatment", "iron_sulfide", "iron.md"),
            _chunk("scale", "Scale water analysis chloride sulfate barium", "scale", "scale.md"),
            _chunk("dosage", "Chemical dosage gallons per day ppm calculation", "dosage", "dosage.md"),
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
