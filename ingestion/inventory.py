from __future__ import annotations

import argparse
import csv
import mimetypes
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

CSV_COLUMNS = (
    "relative_path",
    "absolute_path",
    "file_name",
    "extension",
    "mime_guess",
    "size_bytes",
    "size_mb",
    "topic_folder",
    "parent_folder",
    "parser_type",
    "ingest_priority",
    "needs_ocr",
    "notes",
)

EXPLICIT_TOPIC_FOLDERS = (
    ("iron_sulfide", ("iron sulfide", "iron_sulfide")),
    ("scale", ("scale",)),
    ("water_analysis", ("water analysis", "water analyses")),
    ("corrosion", ("corrosion",)),
    ("paraffin", ("paraffin",)),
    ("asphaltene", ("asphaltene",)),
    ("useful_charts", ("useful charts", "useful_charts")),
    ("reference_papers", ("reference papers", "reference_papers")),
    ("workover", ("workover", "completion")),
    ("dosage", ("chemical dosage", "dosage")),
)

TOPIC_MATCHERS = (
    ("iron_sulfide", ("iron sulfide", "iron_sulfide", "fes", "schmoo", "thps")),
    ("scale", ("scale", "deposits", "deposit", "inorganic")),
    ("water_analysis", ("water analysis", "water analyses", "water", "chloride", "tds", "analysis", "sulfate")),
    ("corrosion", ("corrosion", "coupon", "inhibitor", "failure", "nace")),
    ("paraffin", ("paraffin", "wax")),
    ("asphaltene", ("asphaltene", "sara", "oliensis")),
    ("useful_charts", ("useful charts", "useful_charts", "conversion", "ppm")),
    ("reference_papers", ("reference papers", "reference_papers")),
    ("workover", ("workover", "completion")),
    ("dosage", ("dosage", "ppm", "gallons")),
)

PARSER_TYPES = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".xlsx": "spreadsheet",
    ".xls": "spreadsheet",
    ".csv": "spreadsheet",
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
    ".txt": "text",
    ".md": "text",
}

HIGH_PRIORITY_TOPICS = {"dosage", "water_analysis", "iron_sulfide", "scale"}
MEDIUM_PRIORITY_TOPICS = {"corrosion", "paraffin", "asphaltene"}
SYSTEM_FILE_NAMES = {"desktop.ini", "thumbs.db", ".ds_store"}

InventoryRow = dict[str, str | int | float | bool]


def infer_topic(path: str | Path) -> str:
    normalized_parts = _normalized_path_parts(path)
    directory_parts = normalized_parts[:-1]
    normalized = " / ".join(normalized_parts)
    for part in reversed(directory_parts):
        for topic, keywords in EXPLICIT_TOPIC_FOLDERS:
            if any(_matches_keyword(part, keyword) for keyword in keywords):
                return topic
    for topic, keywords in TOPIC_MATCHERS:
        if any(_matches_keyword(normalized, keyword) for keyword in keywords):
            return topic
    return "unknown"


def _normalized_path_parts(path: str | Path) -> list[str]:
    return [part.replace("-", " ").replace("_", " ").lower() for part in Path(path).parts]


def _matches_keyword(value: str, keyword: str) -> bool:
    normalized_keyword = keyword.replace("_", " ").lower()
    pattern = rf"(?<![a-z0-9]){re.escape(normalized_keyword)}(?![a-z0-9])"
    return re.search(pattern, value) is not None


def infer_parser_type(path_or_extension: str | Path) -> str:
    value = str(path_or_extension)
    extension = value.lower() if value.startswith(".") and "/" not in value else Path(value).suffix.lower()
    return PARSER_TYPES.get(extension, "unsupported")


def normalize_extensions(extensions: Iterable[str] | None) -> set[str] | None:
    if extensions is None:
        return None

    normalized = {
        extension.strip().lower()
        if extension.strip().startswith(".")
        else f".{extension.strip().lower()}"
        for extension in extensions
        if extension.strip()
    }
    return normalized or None


def infer_ingest_priority(path: Path, topic: str, parser_type: str) -> str:
    normalized_name = path.name.lower()
    normalized_stem = path.stem.lower()
    normalized_path = str(path).replace("\\", "/").lower()
    normalized_parts = [part.lower() for part in path.parts]
    is_temp_file = normalized_stem in {"temp", "tmp"} or normalized_stem.startswith(
        ("temp_", "temp-", "tmp_", "tmp-")
    ) or any(part in {"temp", "tmp"} for part in normalized_parts)
    is_system_file = (
        normalized_name in SYSTEM_FILE_NAMES
        or normalized_name.startswith("~$")
        or normalized_name.startswith(".")
    )

    if is_system_file or is_temp_file or parser_type == "unsupported":
        return "skip"
    if topic in HIGH_PRIORITY_TOPICS or "main handout" in normalized_path:
        return "high"
    if topic in MEDIUM_PRIORITY_TOPICS or "flow assurance" in normalized_path:
        return "medium"
    return "low"


def build_inventory_rows(
    data_dir: str | Path,
    *,
    include_extensions: Iterable[str] | None = None,
    max_files: int | None = None,
) -> list[InventoryRow]:
    root = Path(data_dir).expanduser().resolve()
    if not root.exists():
        raise ValueError(f"Data directory does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"Data path is not a directory: {root}")
    if max_files is not None and max_files < 1:
        raise ValueError("--max-files must be at least 1")

    extension_filter = normalize_extensions(include_extensions)
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix().lower(),
    )
    if extension_filter is not None:
        files = [path for path in files if path.suffix.lower() in extension_filter]
    if max_files is not None:
        files = files[:max_files]

    rows: list[InventoryRow] = []
    for path in files:
        absolute_path = path.resolve()
        relative_path = absolute_path.relative_to(root)
        extension = path.suffix.lower()
        parser_type = infer_parser_type(extension)
        topic = infer_topic(relative_path)
        priority = infer_ingest_priority(relative_path, topic, parser_type)
        needs_ocr = parser_type == "image"
        notes = _inventory_notes(path, parser_type, priority)
        size_bytes = path.stat().st_size

        rows.append(
            {
                "relative_path": relative_path.as_posix(),
                "absolute_path": str(absolute_path),
                "file_name": path.name,
                "extension": extension,
                "mime_guess": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                "size_bytes": size_bytes,
                "size_mb": round(size_bytes / (1024 * 1024), 6),
                "topic_folder": topic,
                "parent_folder": relative_path.parent.name or root.name,
                "parser_type": parser_type,
                "ingest_priority": priority,
                "needs_ocr": needs_ocr,
                "notes": notes,
            }
        )
    return rows


def generate_inventory(
    data_dir: str | Path,
    output_dir: str | Path = "data/processed",
    *,
    include_extensions: Iterable[str] | None = None,
    max_files: int | None = None,
    summary_only: bool = False,
) -> list[InventoryRow]:
    root = Path(data_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    rows = build_inventory_rows(
        root,
        include_extensions=include_extensions,
        max_files=max_files,
    )
    destination.mkdir(parents=True, exist_ok=True)
    _write_csv(destination / "inventory.csv", rows)
    _write_summary(
        destination / "inventory_summary.md",
        rows,
        root=root,
        summary_only=summary_only,
    )
    return rows


def _inventory_notes(path: Path, parser_type: str, priority: str) -> str:
    if priority == "skip" and parser_type == "unsupported":
        return "Unsupported file type; skip ingestion."
    if priority == "skip":
        return "Temporary or system file; skip ingestion."
    if parser_type == "image":
        return "Image metadata only; OCR required in a later milestone."
    if parser_type == "pdf":
        # TODO: Detect scanned PDFs without running heavy OCR.
        return "Scanned-PDF detection deferred to a later milestone."
    return ""


def _write_csv(path: Path, rows: Sequence[InventoryRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            serialized["needs_ocr"] = str(bool(row["needs_ocr"])).lower()
            writer.writerow(serialized)


def _write_summary(
    path: Path,
    rows: Sequence[InventoryRow],
    *,
    root: Path,
    summary_only: bool,
) -> None:
    generated_at = datetime.now(timezone.utc).isoformat()
    total_size = sum(int(row["size_bytes"]) for row in rows)
    extension_counts = Counter(str(row["extension"]) or "(no extension)" for row in rows)
    parser_counts = Counter(str(row["parser_type"]) for row in rows)
    topic_counts = Counter(str(row["topic_folder"]) for row in rows)
    unsupported_rows = [row for row in rows if row["parser_type"] == "unsupported"]
    ocr_rows = [row for row in rows if row["needs_ocr"]]
    largest_rows = sorted(rows, key=lambda row: int(row["size_bytes"]), reverse=True)[:20]

    lines = [
        "# Corpus Inventory Summary",
        "",
        f"- Scanned root path: `{root}`",
        f"- Generated timestamp (UTC): `{generated_at}`",
        f"- Scan mode: `{'summary-only' if summary_only else 'metadata-only'}`",
        f"- Total file count: **{len(rows)}**",
        f"- Total size: **{total_size:,} bytes ({total_size / (1024 * 1024):.2f} MB)**",
        "",
        "## Count by Extension",
        "",
        *_markdown_count_table(extension_counts, "Extension"),
        "",
        "## Count by Parser Type",
        "",
        *_markdown_count_table(parser_counts, "Parser type"),
        "",
        "## Count by Topic Folder",
        "",
        *_markdown_count_table(topic_counts, "Topic"),
        "",
        "## Top 20 Largest Files",
        "",
        "| Relative path | Size (MB) |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| {_escape_table(str(row['relative_path']))} | {float(row['size_mb']):.6f} |"
        for row in largest_rows
    )
    if not largest_rows:
        lines.append("| None | 0 |")

    lines.extend(
        [
            "",
            "## Unsupported File Types",
            "",
            *_markdown_file_list(unsupported_rows),
            "",
            "## Files Needing OCR",
            "",
            *_markdown_file_list(ocr_rows),
            "",
            "## Recommended Ingestion Order",
            "",
        ]
    )
    for priority in ("high", "medium", "low", "skip"):
        priority_rows = [row for row in rows if row["ingest_priority"] == priority]
        lines.append(f"- **{priority.title()}**: {len(priority_rows)} file(s)")

    lines.extend(
        [
            "",
            "## Next Steps",
            "",
            "1. Review topic, parser, priority, unsupported-type, and OCR classifications.",
            "2. Re-run against the intended sample or private mounted path if corrections are needed.",
            "3. Implement parsing and chunking in Milestone 2; do not parse from this inventory output.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _markdown_count_table(counts: Counter[str], label: str) -> list[str]:
    lines = [f"| {label} | Count |", "| --- | ---: |"]
    if not counts:
        return [*lines, "| None | 0 |"]
    lines.extend(
        f"| {_escape_table(value)} | {count} |" for value, count in sorted(counts.items())
    )
    return lines


def _markdown_file_list(rows: Sequence[InventoryRow]) -> list[str]:
    if not rows:
        return ["- None"]
    return [f"- `{row['relative_path']}`" for row in rows]


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|")


def _parse_include_extensions(value: str | None) -> set[str] | None:
    if value is None:
        return None
    return normalize_extensions(value.split(","))


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a metadata-only inventory of an oilfield chemistry document folder."
    )
    parser.add_argument("--data-dir", required=True, help="Directory to scan recursively.")
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory for inventory.csv and inventory_summary.md.",
    )
    parser.add_argument("--max-files", type=int, help="Stop after N files in sorted order.")
    parser.add_argument(
        "--include-ext",
        help="Comma-separated extension filter, for example .pdf,.docx,.xlsx,.csv.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip content checks; this milestone always collects metadata only.",
    )
    return parser


def main() -> int:
    parser = _build_argument_parser()
    args = parser.parse_args()
    try:
        rows = generate_inventory(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            include_extensions=_parse_include_extensions(args.include_ext),
            max_files=args.max_files,
            summary_only=args.summary_only,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    output_dir = Path(args.output_dir).expanduser().resolve()
    print(f"Inventoried {len(rows)} file(s).")
    print(f"CSV: {output_dir / 'inventory.csv'}")
    print(f"Summary: {output_dir / 'inventory_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
