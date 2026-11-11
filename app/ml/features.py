"""Feature functions shared by training (training/train.py) and the API.

Versioned so the old v2 artifact still scores exactly as in Colab:
  clean   2 = Colab cleaning            3 = + split glued words/numbers
  skills  2 = 217-term list             3 = 394-skill catalog, aliases, longest match, context rules
  exp     2 = max "N years" anywhere    3 = ranges, "N+ / plus / over N", word numbers, work-date ranges
"""
from __future__ import annotations

import re
from datetime import date

from app.ml.skills_catalog import CATALOG_V3, CONTEXT_ALIASES, NEGATIVE_PATTERNS, V2_SKILLS

# ============================================================== cleaning

_CAMEL = re.compile(r'([a-z])([A-Z])')
_WS = re.compile(r'\s+')


def clean_v2(t) -> str:
    t = _CAMEL.sub(r'\1 \2', str(t))
    return _WS.sub(' ', t).strip()


_SHORT_GLUE = {'in', 'on', 'at', 'of', 'to', 'by', 'since', 'from', 'over', 'for', 'with'}
_KEEP_LETTER_DIGIT = {'html', 'ipv', 'win', 'mp', 'utf', 'sha', 'md', 'covid', 'iso', 'web'}
_DIGIT_SUFFIX_KEEP = {'st', 'nd', 'rd', 'th', 'gb', 'mb', 'kb', 'tb', 'ghz', 'mhz', 'am', 'pm', 'k', 'x', 'd',
                      'px', 'mm', 'cm', 'km', 'kg', 'lbs', 'ft', 'hz', 'kw', 'kv', 'v', 'g'}
_GLUE_WORDS = ('with', 'and', 'for', 'from')


def _split_letter_digit(m):
    letters, digit = m.group(1), m.group(2)
    low = letters.lower()
    if low in _KEEP_LETTER_DIGIT or (len(letters) < 4 and low not in _SHORT_GLUE):
        return m.group(0)
    return f'{letters} {digit}'


def _split_digit_letter(m):
    digit, letters = m.group(1), m.group(2)
    return m.group(0) if letters.lower() in _DIGIT_SUFFIX_KEEP else f'{digit} {letters}'


def clean_v3(t, known_words: frozenset | set = frozenset()) -> str:
    """v2 cleaning + split "experience01/2018", "2009to11/2011senior", "summary7+",
    and glued joining words like "engineerwith" (only when the result is two known words)."""
    t = _CAMEL.sub(r'\1 \2', str(t))
    t = re.sub(r'([A-Za-z]{2,})(\d)', _split_letter_digit, t)
    t = re.sub(r'(\d)([A-Za-z]{1,})', _split_digit_letter, t)
    if known_words:
        def glue(m):
            w = m.group(0)
            low = w.lower()
            if low in known_words:
                return w
            for g in _GLUE_WORDS:
                if low.endswith(g) and low[:-len(g)] in known_words:
                    return f'{w[:-len(g)]} {w[-len(g):]}'
            return w
        t = re.sub(r'\b[A-Za-z]{6,}\b', glue, t)
    return _WS.sub(' ', t).strip()


def clean(t, version: int = 2, known_words=frozenset()) -> str:
    return clean_v2(t) if version == 2 else clean_v3(t, known_words)


def build_known_words(texts, min_count: int = 3) -> frozenset:
    from collections import Counter
    c = Counter(w for t in texts for w in re.findall(r'[a-z]+', str(t).lower()))
    return frozenset(w for w, n in c.items() if n >= min_count and len(w) >= 2)


# ============================================================== skills

class SkillMatcher:
    def __init__(self, version: int = 2, catalog: dict | None = None):
        self.version = version
        if version == 2:
            self.skills = list(catalog or V2_SKILLS)
            self._pat = {s: re.compile(r'(?<![a-z0-9])' + re.escape(s) + r'(?![a-z0-9])') for s in self.skills}
            return
        self.catalog = dict(catalog or CATALOG_V3)
        self.skills = sorted(self.catalog)
        self._owner = {a: c for c, al in self.catalog.items() for a in al}
        aliases = sorted(self._owner, key=len, reverse=True)       # longest first -> longest match wins
        self._rx = re.compile(r'(?<![a-z0-9])(' + '|'.join(re.escape(a) for a in aliases) + r')(?![a-z0-9])')
        self._ctx = {a: [(re.compile(r'(?<![a-z0-9])(?:' + '|'.join(re.escape(w) for w in words) + r')(?![a-z0-9])'), c)
                         for words, c in rules] for a, rules in CONTEXT_ALIASES.items()}
        self._ctx_rx = re.compile(r'(?<![a-z0-9])(' + '|'.join(re.escape(a) for a in CONTEXT_ALIASES) + r')(?![a-z0-9])')
        self._neg = re.compile('|'.join(NEGATIVE_PATTERNS))

    def normalize(self, text: str) -> str:
        t = text.lower()
        return self._neg.sub(lambda m: ' ' * len(m.group(0)), t) if self.version != 2 else t

    def spans(self, text: str) -> list[tuple[str, int, int]]:
        """(canonical skill, start, end) for every alias occurrence (v3+ only; context aliases resolved)."""
        t = self.normalize(text)
        out = [(self._owner[m.group(1)], m.start(), m.end()) for m in self._rx.finditer(t)]
        for m in self._ctx_rx.finditer(t):
            for rx, canon in self._ctx[m.group(1)]:
                if rx.search(t):
                    out.append((canon, m.start(), m.end()))
                    break
        return out

    def extract(self, text: str) -> set[str]:
        t = text.lower()
        if self.version == 2:
            return {s for s, p in self._pat.items() if p.search(t)}
        t = self._neg.sub(lambda m: ' ' * len(m.group(0)), t)
        found = {self._owner[m.group(1)] for m in self._rx.finditer(t)}
        for m in set(self._ctx_rx.findall(t)):
            for rx, canon in self._ctx[m]:
                if rx.search(t):
                    found.add(canon)
                    break
        return found


# ============================================================== experience

_YEARS_V2 = re.compile(r'(\d{1,2})\s*\+?\s*(?:years|yrs)')


def years_v2(text: str):
    m = _YEARS_V2.findall(text.lower())
    return max(map(int, m)) if m else None


def exp_score_from(need, have) -> float:
    if need is None:
        return 1.0
    if have is None:
        return 0.5
    return 1.0 if have >= need else have / need


_WORDNUM = {w: i for i, w in enumerate(
    'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen '
    'sixteen seventeen eighteen nineteen twenty'.split())}
_WORDNUM.update({'thirty': 30, 'forty': 40})
_NUM = r'(?:\d{1,2}|' + '|'.join(_WORDNUM) + r')'
_YEARS_V3 = re.compile(
    r'(?P<a>' + _NUM + r')(?:\s*(?:-|–|to)\s*(?P<b>' + _NUM + r'))?\s*(?:\+|plus)?\s*(?:or more\s*)?'
    r'(?:years?|yrs?)\b(?!\s*(?:old|ago|degree|college|university|program|warranty|of age))')


def _num(s):
    return int(s) if s.isdigit() else _WORDNUM[s]


def stated_years(text: str):
    """Largest 'N years' statement, preferring ones followed by 'experience'. Ranges use the low end."""
    t = text.lower()
    exp_hits, all_hits = [], []
    for m in _YEARS_V3.finditer(t):
        n = _num(m.group('a'))
        if not 0 < n <= 45:
            continue
        all_hits.append(n)
        if 'experience' in t[m.end():m.end() + 40]:
            exp_hits.append(n)
    hits = exp_hits or all_hits
    return max(hits) if hits else None


_MON = {m: i for i, m in enumerate('jan feb mar apr may jun jul aug sep oct nov dec'.split(), start=1)}
_DATE = r'(?:(?P<{p}m>\d{{1,2}})\s*/\s*|(?P<{p}mn>jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?,?\s+)?(?P<{p}y>(?:19|20)\d{{2}})'
_RANGE = re.compile(_DATE.format(p='s') + r'\s*(?:-|–|—|to|until|through)\s*(?:' + _DATE.format(p='e') +
                    r'|(?P<cur>current|present|now|today|date))')
_EDU = re.compile(r'universit|college|bachelor|master|degree|school|gpa|diploma|b\.?s\.?c|m\.?b\.?a|graduat')
_YEAR = re.compile(r'(?<!\d)((?:19|20)\d{2})(?!\d)')


def dated_years(text: str):
    """Total years covered by work date ranges like '01/2015 to 06/2018' or 'Jan 2019 - Current'.
    Overlaps are merged; ranges next to education words are ignored. 'Current' ends at the
    latest year written in the resume + 1 (capped at this year), since resumes are not dated."""
    t = text.lower()
    years = [int(y) for y in _YEAR.findall(t) if 1960 <= int(y) <= date.today().year]
    if not years:
        return None
    ref = min(max(years) + 1, date.today().year)
    spans = []
    for m in _RANGE.finditer(t):
        if _EDU.search(t[max(0, m.start() - 50):m.start()]):
            continue
        def point(p, default_month):
            y = int(m.group(f'{p}y'))
            mon = m.group(f'{p}m')
            mon = int(mon) if mon and 1 <= int(mon) <= 12 else _MON.get(m.group(f'{p}mn') or '', default_month)
            return y + (mon - 1) / 12
        start = point('s', 1)
        end = ref + 0.5 if m.group('cur') else point('e', 12)
        if 1960 <= start <= end <= ref + 1 and end - start <= 45:
            spans.append((start, end))
    if not spans:
        return None
    spans.sort()
    total, (cs, ce) = 0.0, spans[0]
    for s, e in spans[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            total += ce - cs
            cs, ce = s, e
    total += ce - cs
    return round(total, 1)


def experience(resume: str, jd: str, version: int = 2) -> dict:
    if version == 2:
        need, have = years_v2(jd), years_v2(resume)
        return {'years_required': need, 'years_found': have, 'years_stated': have, 'years_dated': None,
                'exp_score': exp_score_from(need, have)}
    need = stated_years(jd)
    st, dt = stated_years(resume), dated_years(resume)
    have = max([v for v in (st, dt) if v is not None], default=None)
    return {'years_required': need, 'years_found': have, 'years_stated': st, 'years_dated': dt,
            'exp_score': exp_score_from(need, have)}


# ============================================================== v4: weighted skills with hierarchy

from app.ml.skills_catalog import COMPETENCIES, PARENTS, PREFERRED_CUES, SOFT_SKILLS  # noqa: E402

_PREF_RX = re.compile(PREFERRED_CUES)
_SEGMENT = re.compile(r'(?<=[.;!?•:])\s+|•|\s-\s|\s\*\s')
IMPORTANCE_WEIGHT = {'required': 1.0, 'preferred': 0.5, 'soft': 0.25}
COMPETENCY_FACTOR = 0.5          # broad competencies count half of a concrete technology


def _ancestors(skill: str) -> set[str]:
    out, todo = set(), list(PARENTS.get(skill, []))
    while todo:
        p = todo.pop()
        if p not in out:
            out.add(p)
            todo += PARENTS.get(p, [])
    return out


_CHILDREN: dict[str, set[str]] = {}
for _c in list(PARENTS):
    for _p in _ancestors(_c):
        _CHILDREN.setdefault(_p, set()).add(_c)          # all descendants


# Only these families have interchangeable members (MySQL ~ PostgreSQL). Cloud providers, ERPs etc. are not.
_SUBSTITUTABLE = {'sql', 'nosql', 'data visualization'}
_PREF_HEADER = re.compile(r'\b(?:preferred|desired|nice[\s-]to[\s-]have|bonus|additional)\s+(?:qualifications|skills|'
                          r'requirements|experience|points)\b|\bnice[\s-]to[\s-]have\s*:|\bpluses\s*:')
_REQ_HEADER = re.compile(r'\b(?:required|minimum|basic|must[\s-]have)\s+(?:qualifications|skills|requirements|experience)\b'
                         r'|\brequirements\s*:|\bresponsibilities\s*:|\bmust[\s-]haves?\s*:|\bwhat you\W*ll do\b')
_BOUNDARY = re.compile(r'[.;!?•]\s')


def jd_skill_importance(jd_text: str, matcher: 'SkillMatcher') -> dict[str, tuple[str, float]]:
    """skill -> (importance, weight) for every skill found in the JD.
    A mention is 'preferred' if it sits under a 'Preferred qualifications'-style header, or a cue such as
    'is a plus', 'nice to have', 'preferred', 'familiarity with' is within a few words of it in the same sentence.
    A skill is required if any mention is not preferred."""
    found = matcher.extract(jd_text)
    t = matcher.normalize(jd_text)
    marks = sorted([(m.start(), 'pref') for m in _PREF_HEADER.finditer(t)] +
                   [(m.start(), 'req') for m in _REQ_HEADER.finditer(t)])
    cues = [m.start() for m in _PREF_RX.finditer(t)]

    def preferred_at(pos: int, end: int) -> bool:
        section = 'req'
        for mpos, kind in marks:
            if mpos > pos:
                break
            section = kind
        if section == 'pref':
            return True
        for c in cues:
            lo, hi = (c, pos) if c < pos else (end, c)
            gap = t[lo:hi]
            if len(gap.split()) <= (5 if c < pos else 10) and not _BOUNDARY.search(gap):
                return True
        return False

    required: dict[str, bool] = {}
    for canon, start, end in matcher.spans(jd_text):
        required[canon] = required.get(canon, False) or not preferred_at(start, end)
    out = {}
    for sk in found:
        imp = 'soft' if sk in SOFT_SKILLS else ('required' if required.get(sk, True) else 'preferred')
        w = IMPORTANCE_WEIGHT[imp]
        if sk in COMPETENCIES and imp != 'soft':
            w *= COMPETENCY_FACTOR
        if _CHILDREN.get(sk, set()) & found:      # JD also lists specific children (AWS + Redshift): don't double count
            w *= 0.5
        out[sk] = (imp, w)
    return out


def skill_kind(skill: str) -> str:
    return 'soft' if skill in SOFT_SKILLS else ('competency' if skill in COMPETENCIES else 'technology')


def skill_credit(skill: str, resume_skills: set[str]) -> tuple[float, str, str | None]:
    """(credit, status, via). Evidence only flows upward:
      exact mention                      -> 1.0 matched
      a more specific skill in resume    -> 1.0 matched   (Redshift => AWS, Airflow => data engineering)
      interchangeable sibling            -> 0.5 related   (MySQL for PostgreSQL; only for _SUBSTITUTABLE families)
    A broader skill never implies a specific one: "data engineering" does not imply Airflow, Azure does not imply AWS."""
    if skill in resume_skills:
        return 1.0, 'matched', None
    kids = _CHILDREN.get(skill, set()) & resume_skills
    if kids:
        return 1.0, 'matched', sorted(kids)[0]
    parents = set(PARENTS.get(skill, []))
    rel = {sib for p in parents & _SUBSTITUTABLE for sib in _CHILDREN.get(p, ())
           if sib != skill and sib in resume_skills and p in PARENTS.get(sib, [])}
    if rel:
        return 0.5, 'related', sorted(rel)[0]
    return 0.0, 'missing', None


def weighted_skill_features(jd_imp: dict, resume_skills: set[str]) -> dict:
    details, tot, got = [], {'all': 0.0, 'required': 0.0, 'preferred': 0.0}, {'all': 0.0, 'required': 0.0, 'preferred': 0.0}
    for s, (imp, w) in sorted(jd_imp.items(), key=lambda kv: (-kv[1][1], kv[0])):
        credit, status, via = skill_credit(s, resume_skills)
        details.append({'skill': s, 'importance': imp, 'kind': skill_kind(s), 'weight': w, 'status': status, 'via': via})
        tot['all'] += w
        got['all'] += w * credit
        if imp in tot:
            tot[imp] += w
            got[imp] += w * credit
    cov = {k: (got[k] / tot[k] if tot[k] else 0.0) for k in tot}
    core = [d for d in details if d['importance'] == 'required' and d['kind'] == 'technology']
    return {'w_skill_score': cov['all'], 'req_coverage': cov['required'], 'pref_coverage': cov['preferred'],
            'n_required': sum(d['importance'] == 'required' for d in details),
            'n_core': len(core), 'n_core_matched': sum(d['status'] == 'matched' for d in core),
            'skill_details': details}


_JD_FOOTER = re.compile(r'recruiter\s+(?:details|name|contact)|recruiter\s*:|equal\s+(?:employment\s+)?opportunity\s+employer|'
                        r'\beeo\b|\beoe\b|to\s+know\s+more\s+about|please\s+visit\s+www|for\s+more\s+information')


def strip_jd_footer(jd: str) -> str:
    """Drop agency/legal boilerplate (recruiter contact, EEO statement, 'please visit www...') at the end of a JD,
    so e.g. 'Recruiter Details' does not make 'recruiting' a required skill. Only markers in the second half count."""
    t = jd.lower()
    for m in _JD_FOOTER.finditer(t):
        if m.start() >= len(t) * 0.5:
            return jd[:m.start()].rstrip()
    return jd
