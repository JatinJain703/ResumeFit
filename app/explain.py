"""Turn model inputs into plain-English reasons.

These reasons describe the signals the model receives (skills, experience,
similarity relative to the other applicants). They are not a trace of the
Random Forest's internal decision path.
"""
from __future__ import annotations

import pandas as pd

MAX_LIST = 6


def _ordinal(n: int) -> str:
    suf = 'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f'{n}{suf}'


def _fmt_list(items: list[str]) -> str:
    items = list(items)
    if len(items) <= MAX_LIST:
        return ', '.join(items)
    return ', '.join(items[:MAX_LIST]) + f' (+{len(items) - MAX_LIST} more)'


def add_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Position (1 = best) of each resume among this batch on each signal."""
    df = df.copy()
    skill_col = 'w_skill_score' if 'w_skill_score' in df else 'skill_score'
    for col, out in [(skill_col, 'skill_pos'), ('bert_score', 'semantic_pos'),
                     ('tfidf_score', 'keyword_pos')]:
        df[out] = df[col].rank(ascending=False, method='min').astype(int)
    return df


def _yrs(v: float) -> str:
    v = round(float(v), 1)
    return f'{int(v)}' if v.is_integer() else f'{v}'


def _experience_line(row) -> str | None:
    need, have = row.years_required, row.years_found
    if need is None or pd.isna(need):
        return None
    if have is None or pd.isna(have):
        return f'Job asks for {_yrs(need)}+ years; no years of experience or work dates found in the resume'
    dated = getattr(row, 'years_dated', None)
    how = 'from work dates' if dated is not None and not pd.isna(dated) and float(dated) == float(have) else 'stated'
    if have >= need:
        return f'Meets the experience requirement ({_yrs(have)} years {how}, {_yrs(need)} required)'
    return f'Below the experience requirement ({_yrs(have)} years {how}, {_yrs(need)} required)'


def _name(d) -> str:
    if d['status'] == 'related':
        return f"{d['skill']} (related: {d['via']})"
    return f"{d['skill']} (via {d['via']})" if d['via'] else d['skill']


def _split(row):
    det = getattr(row, 'skill_details', None)
    if det is None:
        return None
    req = [d for d in det if d['importance'] == 'required']
    return {'tech': [d for d in req if d['kind'] == 'technology'],
            'comp': [d for d in req if d['kind'] == 'competency'],
            'preferred': [d for d in det if d['importance'] == 'preferred'],
            'soft': [d for d in det if d['importance'] == 'soft']}


def _coverage_line(title: str, items, extra: str = '') -> str | None:
    if not items:
        return None
    have = [d for d in items if d['status'] == 'matched']
    line = f'{title}: {len(have)} of {len(items)}'
    if have:
        line += f' ({_fmt_list([_name(d) for d in have])})'
    return line + extra


def _weighted_selected(row, n: int, g) -> list[str]:
    out = [_coverage_line('Core technologies', g['tech'],
                          f' - {_ordinal(row.skill_pos)} of {n} applicants on weighted skill coverage'),
           _coverage_line('Competencies', g['comp']),
           _coverage_line('Preferred skills', g['preferred'])]
    rel = [d for d in g['tech'] + g['comp'] + g['preferred'] if d['status'] == 'related']
    if rel:
        out.append(f'Related (partial credit): {_fmt_list([_name(d) for d in rel])}')
    return [o for o in out if o]


def _weighted_gaps(g) -> list[str]:
    out = []
    for title, key in (('core technologies', 'tech'), ('competencies', 'comp'), ('preferred', 'preferred')):
        miss = [d['skill'] for d in g[key] if d['status'] == 'missing']
        if miss:
            out.append(f'missing {title}: {_fmt_list(miss)}')
    return out


def _cap_line(row) -> str | None:
    if getattr(row, 'label_capped', False):
        return (f'Label capped at Potential Fit: none of the {row.n_core} core technologies were found in the resume '
                f'(the model alone said Good Fit)')
    return None


def why_selected(row, n: int) -> list[str]:
    out = []
    g = _split(row)
    if g is not None and (g['tech'] or g['comp'] or g['preferred']):
        out += _weighted_selected(row, n, g)
    elif row.n_jd_skills:
        line = f'Has {row.n_matched} of {row.n_jd_skills} required skills'
        if row.matched_skills:
            line += f': {_fmt_list(row.matched_skills)}'
        out.append(line + f' ({_ordinal(row.skill_pos)} of {n} applicants on skill coverage)')
    else:
        out.append('Job description lists no skills from the skill catalog; ranking relies on text similarity')
    top = max(1, round(n * 0.25))
    if row.semantic_pos <= top:
        out.append(f'Resume content is closely related to the job in meaning '
                   f'({_ordinal(row.semantic_pos)} of {n} on semantic similarity)')
    if row.keyword_pos <= top:
        out.append(f'Uses much of the job\'s specific vocabulary '
                   f'({_ordinal(row.keyword_pos)} of {n} on keyword similarity)')
    exp = _experience_line(row)
    if exp:
        out.append(exp)
    if g is not None:
        gaps = _weighted_gaps(g)
        if gaps:
            out.append('Gaps to check in interview: ' + '; '.join(gaps))
    elif row.missing_skills:
        out.append(f'Gaps to check in interview: {_fmt_list(row.missing_skills)}')
    cap = _cap_line(row)
    if cap:
        out.append(cap)
    return out


def why_not_selected(row, n: int, k: int) -> list[str]:
    out = []
    g = _split(row)
    if g is not None and (g['tech'] or g['comp'] or g['preferred']):
        for title, key in (('core technologies', 'tech'), ('competencies', 'comp'), ('preferred skills', 'preferred')):
            miss = [d['skill'] for d in g[key] if d['status'] == 'missing']
            if miss:
                out.append(f'Missing {len(miss)} of {len(g[key])} {title}: {_fmt_list(miss)}')
    elif row.missing_skills:
        out.append(f'Missing {len(row.missing_skills)} of {row.n_jd_skills} required skills: '
                   f'{_fmt_list(row.missing_skills)}')
    half = n / 2
    if row.semantic_pos > half:
        out.append(f'Content is less related to the job than most applicants '
                   f'({_ordinal(row.semantic_pos)} of {n} on semantic similarity)')
    if row.keyword_pos > half:
        out.append(f'Shares little of the job\'s specific vocabulary '
                   f'({_ordinal(row.keyword_pos)} of {n} on keyword similarity)')
    exp = _experience_line(row)
    if exp and not exp.startswith('Meets'):
        out.append(exp)
    cap = _cap_line(row)
    if cap:
        out.append(cap)
    by = 'the formula score' if getattr(row, 'ranked_by', 'model') == 'formula' else 'the model'
    out.append(f'Overall, ranked below the top {k} by {by}')
    return out
