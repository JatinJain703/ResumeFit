"""Extract plain text from uploaded resume files (PDF, DOCX, TXT)."""
from __future__ import annotations

import io
from pathlib import Path

SUPPORTED = {'.pdf', '.docx', '.txt', '.md'}
MIN_CHARS = 50


class ParseError(ValueError):
    pass


def _pdf(data: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    return '\n'.join(page.extract_text() or '' for page in reader.pages)


def _docx(data: bytes) -> str:
    import docx
    d = docx.Document(io.BytesIO(data))
    parts = [p.text for p in d.paragraphs]
    for table in d.tables:
        for row in table.rows:
            parts.append(' '.join(c.text for c in row.cells))
    return '\n'.join(parts)


def _txt(data: bytes) -> str:
    for enc in ('utf-8', 'utf-16', 'latin-1'):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode('utf-8', errors='replace')


def extract_text(filename: str, data: bytes) -> str:
    ext = Path(filename or '').suffix.lower()
    if ext not in SUPPORTED:
        raise ParseError(f'unsupported file type "{ext or "none"}" (use PDF, DOCX or TXT)')
    try:
        text = {'.pdf': _pdf, '.docx': _docx}.get(ext, _txt)(data)
    except Exception as e:  # corrupt or encrypted files
        raise ParseError(f'could not read file: {e.__class__.__name__}') from e
    if len(text.strip()) < MIN_CHARS:
        hint = ' (scanned/image PDF? OCR is not supported yet)' if ext == '.pdf' else ''
        raise ParseError(f'too little text extracted{hint}')
    return text
