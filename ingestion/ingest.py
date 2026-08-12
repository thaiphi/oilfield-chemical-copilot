from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from ingestion.inventory import infer_topic
except ModuleNotFoundError:  # Allows python ingestion/ingest.py script execution.
    from inventory import infer_topic
from oilfield_chemical_copilot.ingest.models import LoadedChunk
from oilfield_chemical_copilot.ingest.parsers import parse_document
from oilfield_chemical_copilot.ingest.scanner import scan_sources


def generate_chunks(
    data_dir: str | Path,
    output_dir: str | Path = "data/processed",
    *,
    max_files: int | None = None,
) -> list[LoadedChunk]:
    root = Path(data_dir).expanduser().resolve()
    if not root.exists():
        raise ValueError(f"Data directory does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"Data path is not a directory: {root}")
    if max_files is not None and max_files < 1:
        raise ValueError("--max-files must be at least 1")

    sources = scan_sources(root)
    if max_files is not None:
        sources = sources[:max_files]

    chunks: list[LoadedChunk] = []
    failures: list[dict[str, str]] = []
    empty_sources: list[str] = []
    for source in sources:
        relative = source.resolve().relative_to(root)
        topic = infer_topic(relative)
        try:
            source_chunks = parse_document(source, source_root=root, topic=topic)
        except Exception as error:
            failures.append(
                {
                    "source_file": relative.as_posix(),
                    "parser_type": source.suffix.lower().lstrip("."),
                    "error_type": type(error).__name__,
                }
            )
            continue
        if not source_chunks:
            empty_sources.append(relative.as_posix())
        chunks.extend(source_chunks)

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    _write_jsonl(destination / "chunks.jsonl", chunks)
    _write_ingestion_report(
        destination / "ingestion_report.json",
        source_count=len(sources),
        chunk_count=len(chunks),
        failures=failures,
        empty_sources=empty_sources,
    )
    return chunks


def _write_jsonl(path: Path, chunks: Sequence[LoadedChunk]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for chunk in chunks:
            output.write(json.dumps(_serialize_chunk(chunk), sort_keys=True) + "\n")


def _write_ingestion_report(
    path: Path,
    *,
    source_count: int,
    chunk_count: int,
    failures: list[dict[str, str]],
    empty_sources: list[str],
) -> None:
    report = {
        "source_files": source_count,
        "parsed_files": source_count - len(failures),
        "failed_files": len(failures),
        "empty_files": len(empty_sources),
        "chunks_written": chunk_count,
        "failures": failures,
        "empty_sources": empty_sources,
    }
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _serialize_chunk(chunk: LoadedChunk) -> dict[str, object]:
    return {
        "text": chunk.text,
        "metadata": asdict(chunk.metadata),
    }


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse public or private source files into local metadata-rich chunks."
    )
    parser.add_argument("--data-dir", required=True, help="Directory to scan recursively.")
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory for chunks.jsonl.",
    )
    parser.add_argument("--max-files", type=int, help="Stop after N supported files in sorted order.")
    return parser


def main() -> int:
    parser = _build_argument_parser()
    args = parser.parse_args()
    try:
        chunks = generate_chunks(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            max_files=args.max_files,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    output_path = Path(args.output_dir).expanduser().resolve() / "chunks.jsonl"
    print(f"Generated {len(chunks)} chunk(s).")
    print(f"Chunks: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
