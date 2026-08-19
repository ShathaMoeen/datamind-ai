"""Page-aware text extraction from stored PDFs."""

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.models.rag import DocumentPage, StoredDocument


class DocumentTextExtractionError(ValueError):
    """Raised when a PDF is encrypted, invalid, or has no extractable text."""


class PDFDocumentLoader:
    """Extract text while preserving source filename and page number."""

    def load(self, document: StoredDocument) -> list[DocumentPage]:
        """Return non-empty text pages ready for chunking."""

        try:
            reader = PdfReader(document.path)
            if reader.is_encrypted:
                raise DocumentTextExtractionError(
                    "Encrypted PDFs are not supported in the local pipeline."
                )
            pages = [
                DocumentPage(
                    document_id=document.document_id,
                    source=document.original_filename,
                    page_number=page_number,
                    text=(page.extract_text() or "").strip(),
                )
                for page_number, page in enumerate(reader.pages, start=1)
            ]
        except (OSError, PdfReadError) as error:
            raise DocumentTextExtractionError("The PDF could not be read.") from error

        extracted = [page for page in pages if page.text]
        if not extracted:
            raise DocumentTextExtractionError(
                "The PDF has no extractable text and may require OCR."
            )
        return extracted
