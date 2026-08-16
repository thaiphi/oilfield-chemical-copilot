from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GUIDE_PATH = PROJECT_ROOT / "docs" / "CAPSTONE_REVIEWER_GUIDE.md"
EVIDENCE_PATH = PROJECT_ROOT / "docs" / "CAPSTONE_EVIDENCE.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_reviewer_guide_contains_a_public_reproducible_path() -> None:
    guide = _read(GUIDE_PATH)

    for required_text in (
        "uv sync",
        "data/sample",
        "ollama pull granite4.1:8b",
        "ollama pull granite-embedding:latest",
        "docker compose up -d",
        "monitoring-demo-seed",
        "http://localhost:8501",
        "http://localhost:3000",
        "uv run pytest",
        "git diff --check",
    ):
        assert required_text in guide


def test_reviewer_documents_do_not_include_private_locations_or_secrets() -> None:
    reviewer_text = _read(GUIDE_PATH) + _read(EVIDENCE_PATH)

    for prohibited_text in (
        ".private/",
        "data/private",
        "eval/private",
        "OPENAI_API_KEY=",
        "C:\\Users\\",
    ):
        assert prohibited_text not in reviewer_text


def test_evidence_map_covers_the_course_review_dimensions() -> None:
    evidence = _read(EVIDENCE_PATH)

    for required_text in (
        "Problem description",
        "Retrieval flow",
        "Retrieval evaluation",
        "LLM evaluation",
        "Interface",
        "Ingestion pipeline",
        "Monitoring",
        "Containerization",
        "Reproducibility",
        "Best practices",
        "reranking",
        "query rewriting",
    ):
        assert required_text in evidence


def test_evidence_map_names_the_dashboard_review_surface() -> None:
    evidence = _read(EVIDENCE_PATH)

    for panel_name in (
        "Request volume by outcome",
        "Average response latency by retrieval mode",
        "Response latency minimum and maximum",
        "Outcome mix",
        "Retrieval mode volume",
        "Helpful rate and feedback volume",
    ):
        assert panel_name in evidence
