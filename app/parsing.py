"""Extract plain text from uploaded resume files (PDF, DOCX, TXT)."""
from __future__ import annotations

import io
import re
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


# ---------------------------------------------------------------- candidate name
_NOT_NAME = {
    'resume', 'curriculum', 'vitae', 'cv', 'profile', 'summary', 'objective', 'contact', 'experience', 'education',
    'skills', 'professional', 'career', 'personal', 'details', 'information', 'page', 'name', 'address', 'phone',
    'email', 'engineer', 'developer', 'manager', 'analyst', 'consultant', 'designer', 'scientist', 'specialist',
    'administrator', 'architect', 'intern', 'director', 'officer', 'lead', 'senior', 'junior', 'assistant',
    'executive', 'technician', 'associate', 'coordinator', 'representative', 'software', 'data', 'sales', 'project',
    'technical', 'business', 'services', 'service', 'support', 'qualifications', 'accomplishments', 'highlights',
}
_NAME_TOKEN = re.compile(r"^[A-Za-z][A-Za-z.'\-]*$")


def _looks_like_name(line: str) -> str | None:
    line = re.sub(r'^\s*name\s*[:\-]\s*', '', line, flags=re.I).strip(' \t|,•·-')
    if not line or len(line) > 40 or any(c in line for c in '@/:0123456789'):
        return None
    tokens = line.split()
    if not 2 <= len(tokens) <= 4 or not all(_NAME_TOKEN.match(t) for t in tokens):
        return None
    if any(t.lower().strip('.') in _NOT_NAME for t in tokens):
        return None
    if not (line.isupper() or all(t[0].isupper() for t in tokens)):
        return None
    return line.title() if line.isupper() else line


def _name_from_filename(filename: str) -> str:
    stem = Path(filename or '').stem
    pretty = re.sub(r'(?i)\b(resume|cv|curriculum vitae)\b', ' ', re.sub(r'[_\-.]+', ' ', stem))
    pretty = ' '.join(pretty.split())
    if not pretty or pretty.isdigit():
        return stem or filename
    return pretty.title() if pretty.islower() or pretty.isupper() else pretty


def guess_candidate_name(text: str, filename: str = '') -> tuple[str, str]:
    """(name, source): the name from the first lines of the resume, else from the file name.

    A heuristic: the first short line of 2-4 capitalised words that is not a heading or job title.
    source is 'resume' or 'filename'.
    """
    for line in [l for l in str(text).splitlines() if l.strip()][:8]:
        name = _looks_like_name(line)
        if name:
            return name, 'resume'
    return _name_from_filename(filename), 'filename'
