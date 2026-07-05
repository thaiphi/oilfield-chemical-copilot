"""Document parser scaffolds."""

from oilfield_chemical_copilot.ingest.parsers.docx import parse_docx
from oilfield_chemical_copilot.ingest.parsers.pdf import parse_pdf
from oilfield_chemical_copilot.ingest.parsers.spreadsheet import parse_tabular

__all__ = ["parse_docx", "parse_pdf", "parse_tabular"]
