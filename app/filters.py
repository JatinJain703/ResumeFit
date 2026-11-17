"""Recruiter must-have filters: minimum years of experience and required skills.

Filters never change a resume's scores; they only decide who may appear in the shortlist.
"""
from __future__ import annotations

import re

import pandas as pd

from app.explain import _yrs
from app.ml import features as F
from app.ml.pipeline import MatcherPipeline
from app.schemas import Filters, MustHaveCheck


class MustHave:
    """One must-have skill as typed by the recruiter, resolved against the skill catalog.

    "Python", "k8s", "PowerShell" -> catalog skill (aliases and more specific skills count: "Redshift" satisfies
    "AWS"). Anything not in the catalog ("Go", "R", "AWS Redshift") is searched as plain text; names of 1-2
    characters are matched case-sensitively so "R" does not match every "r".
    """

    def __init__(self, pipe: MatcherPipeline, typed: str):
        self.typed = ' '.join(typed.split())
        m = pipe.explain_matcher
        low = self.typed.lower()
        self.canon = m._owner.get(low) or m._owner.get(pipe.clean(self.typed).lower())
        if self.canon is None:
            word = re.escape(self.typed if len(self.typed) <= 2 else low)
            self._rx = re.compile(r'(?<![A-Za-z0-9+#])' + word + r'(?![A-Za-z0-9+#])',
                                  0 if len(self.typed) <= 2 else re.I)

    @property
    def name(self) -> str:
        return self.canon or self.typed

    def check(self, resume_skills: set[str], raw_text: str) -> MustHaveCheck:
        if self.canon is not None:
            _, status, via = F.skill_credit(self.canon, resume_skills)
            return MustHaveCheck(skill=self.canon, found=status == 'matched', via=via if status == 'matched' else None,
                                 matched_as='catalog')
        return MustHaveCheck(skill=self.typed, found=bool(self._rx.search(raw_text)), matched_as='text')


def resolve(pipe: MatcherPipeline, filters: Filters | None) -> tuple[list[MustHave], list[str]]:
    """(must-haves, warnings). Duplicates (same resolved skill) are dropped."""
    if filters is None:
        return [], []
    out, seen, warnings = [], set(), []
    for typed in filters.must_have_skills:
        if not typed.strip():
            continue
        mh = MustHave(pipe, typed)
        if mh.name in seen:
            continue
        seen.add(mh.name)
        out.append(mh)
        if mh.canon is None:
            warnings.append(f'Must-have "{mh.typed}" is not in the skill catalog; it is matched as plain text '
                            f'(exact word or phrase, no synonyms).')
    return out, warnings


def failures(row, checks: list[MustHaveCheck] | None, filters: Filters | None) -> list[str]:
    """Reasons this resume fails the filters (empty = passes)."""
    if filters is None:
        return []
    out = []
    if filters.min_years is not None:
        have = row.years_found
        if have is None or pd.isna(have):
            if filters.unknown_years == 'exclude':
                out.append(f'Filtered out: needs {_yrs(filters.min_years)}+ years; no years or work dates found '
                           f'in the resume')
        elif float(have) < filters.min_years:
            out.append(f'Filtered out: needs {_yrs(filters.min_years)}+ years; resume shows {_yrs(have)}')
    missing = [c.skill for c in checks or [] if not c.found]
    if missing:
        out.append(f'Filtered out: missing must-have skill{"s" if len(missing) > 1 else ""}: {", ".join(missing)}')
    return out


def passed_line(checks: list[MustHaveCheck] | None, filters: Filters | None) -> str | None:
    """Reason line for a resume that passes active filters."""
    if filters is None or not filters.active:
        return None
    parts = []
    if checks:
        parts.append('has all must-haves: ' + ', '.join(f'{c.skill} (via {c.via})' if c.via else c.skill
                                                       for c in checks))
    if filters.min_years is not None:
        parts.append(f'meets the {_yrs(filters.min_years)}+ years filter')
    return ('Passes the filters: ' + '; '.join(parts)) if parts else None
