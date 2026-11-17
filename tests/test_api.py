import io

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope='module')
def client():
    with TestClient(app) as c:
        yield c


def test_health_and_model_info(client):
    assert client.get('/health').json()['status'] == 'ok'
    info = client.get('/model-info').json()
    assert info['final_model']
    assert info['n_stage2_features'] in (26, 32)          # v2 / v3 feature sets
    assert len(info['test_set_results']) == 7


def test_rank_json(client, big_job):
    jd, cands = big_job
    body = {'job_description': jd, 'top_k': 10,
            'resumes': [{'id': f'cand_{i}', 'text': t} for i, t in enumerate(cands.resume_text)]}
    r = client.post('/rank', json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    n = len(cands)
    assert out['summary']['scored'] == n
    assert len(out['top_k']) == 10 and len(out['not_selected']) == n - 10
    ranks = [c['rank'] for c in out['top_k'] + out['not_selected']]
    assert ranks == list(range(1, n + 1))
    scores = [c['rank_score'] for c in out['top_k'] + out['not_selected']]
    assert scores == sorted(scores, reverse=True)
    assert all(c['reasons'] for c in out['top_k'] + out['not_selected'])
    assert out['not_selected'][0]['reasons'][-1].startswith('Overall, ranked below the top')
    assert sum(out['summary']['label_counts'].values()) == n


def test_rank_upload_mixed_files(client, big_job):
    import docx
    jd, cands = big_job
    files = []
    for i, t in enumerate(cands.resume_text.head(11)):
        if i % 2:
            d = docx.Document(); d.add_paragraph(t); buf = io.BytesIO(); d.save(buf)
            files.append(('files', (f'r{i}.docx', buf.getvalue(),
                                    'application/vnd.openxmlformats-officedocument.wordprocessingml.document')))
        else:
            files.append(('files', (f'r{i}.txt', t.encode(), 'text/plain')))
    files.append(('files', ('photo.png', b'\x89PNG....', 'image/png')))
    files.append(('files', ('blank.txt', b'   ', 'text/plain')))
    r = client.post('/rank/upload', data={'job_description': jd, 'top_k': '5'}, files=files)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out['summary'] == {**out['summary'], 'total_received': 13, 'scored': 11, 'skipped': 2, 'top_k': 5}
    assert {s['resume_id'] for s in out['skipped']} == {'photo.png', 'blank.txt'}


def test_small_batch_warns(client, big_job):
    jd, cands = big_job
    r = client.post('/rank', json={'job_description': jd, 'top_k': 5,
                                   'resumes': [{'text': t} for t in cands.resume_text.head(3)]})
    out = r.json()
    assert len(out['top_k']) == 3
    assert any('less reliable' in w for w in out['warnings'])
    assert any('larger than' in w for w in out['warnings'])


def test_validation_errors(client):
    assert client.post('/rank', json={'job_description': 'x', 'resumes': []}).status_code == 422
    assert client.post('/rank', json={'job_description': 'x', 'resumes': [{'text': '  '}]}).status_code == 422


def test_cors_preflight(client):
    r = client.options('/rank', headers={'Origin': 'http://localhost:5173', 'Access-Control-Request-Method': 'POST'})
    assert r.status_code == 200
    assert r.headers['access-control-allow-origin'] == 'http://localhost:5173'


def test_rank_upload_jd_as_file_with_names_and_indices(client, big_job):
    jd, cands = big_job
    texts = cands.resume_text.head(10).tolist()
    texts[3] = 'PRIYA SHARMA\n\n' + texts[3]
    files = [('files', ('bad.png', b'\x89PNG....', 'image/png'))]          # skipped -> indices must still line up
    files += [('files', (f'cand_{i}.txt', t.encode(), 'text/plain')) for i, t in enumerate(texts)]
    r = client.post('/rank/upload', data={'top_k': '10'}, files=files + [('jd_file', ('jd.txt', jd.encode(), 'text/plain'))])
    assert r.status_code == 200, r.text
    cands_out = r.json()['top_k']
    for c in cands_out:
        assert files[c['upload_index']][1][0] == c['resume_id']
    by_id = {c['resume_id']: c for c in cands_out}
    assert by_id['cand_3.txt']['candidate_name'] == 'Priya Sharma'
    assert by_id['cand_3.txt']['name_source'] == 'resume'
    # no JD at all -> 422
    assert client.post('/rank/upload', files=files[1:3]).status_code == 422


def test_guess_candidate_name():
    from app.parsing import guess_candidate_name as g
    assert g('JOHN A. SMITH\nSoftware Engineer\njohn@x.com', 'a.pdf') == ('John A. Smith', 'resume')
    assert g('Curriculum Vitae\nName: Ravi Kumar\n+91 99', 'x.pdf') == ('Ravi Kumar', 'resume')
    assert g('Software Engineer\nSummary\nExperienced', 'jatin_jain_resume.pdf') == ('Jatin Jain', 'filename')


# ---------------------------------------------------------------- candidate mode

def test_fit_check_json(client, big_job):
    jd, cands = big_job
    r = client.post('/fit-check', json={'job_description': jd, 'resume_text': cands.resume_text.iloc[0]})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out['label'] in ('No Fit', 'Potential Fit', 'Good Fit')
    assert 0 <= out['match_score'] <= 100
    cmp = out['comparison']
    assert cmp['pool_size'] == 24 and 1 <= cmp['rank_in_pool'] <= 25 and 0 <= cmp['percentile'] <= 100
    gap = out['skills']
    n_skills = len(gap['matched']) + len(gap['related']) + len(gap['missing'])
    assert n_skills == len(out['job']['skills_required'])
    assert all(d['status'] == 'missing' for d in gap['missing'])
    assert out['suggestions'] and out['strengths']
    order = {'high': 0, 'medium': 1, 'low': 2}
    pr = [order[s['priority']] for s in out['suggestions']]
    assert pr == sorted(pr)
    # every missing core technology gets a suggestion
    core_missing = [d['skill'] for d in gap['missing'] if d['importance'] == 'required' and d['kind'] == 'technology']
    text = ' '.join(s['text'] for s in out['suggestions'])
    assert all(s in text for s in core_missing)
    # keywords are not already in the resume
    res = cands.resume_text.iloc[0].lower()
    assert all(k not in res for k in out['missing_keywords'] if ' ' not in k)


def test_fit_check_is_stable_and_excludes_itself(client, big_job):
    """The same resume always gets the same answer, and a dataset resume is not compared with its own copy."""
    jd, cands = big_job
    body = {'job_description': jd, 'resume_text': cands.resume_text.iloc[1]}
    a, b = client.post('/fit-check', json=body).json(), client.post('/fit-check', json=body).json()
    assert a == b
    assert a['comparison']['pool_size'] == 24


def test_fit_check_upload(client, big_job):
    jd, cands = big_job
    resume = 'Jane Q Doe\n' + cands.resume_text.iloc[2]
    r = client.post('/fit-check/upload', files={'resume': ('jane_cv.txt', resume.encode(), 'text/plain'),
                                                'jd_file': ('job.txt', jd.encode(), 'text/plain')})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out['candidate_name'] == 'Jane Q Doe' and out['name_source'] == 'resume'


def test_fit_check_errors(client, big_job):
    jd, cands = big_job
    r = client.post('/fit-check/upload', files={'resume': ('a.txt', cands.resume_text.iloc[0].encode(), 'text/plain')})
    assert r.status_code == 422 and 'job description' in r.json()['detail']
    r = client.post('/fit-check/upload', data={'job_description': jd},
                    files={'resume': ('photo.png', b'\x89PNG....', 'image/png')})
    assert r.status_code == 422 and r.json()['detail'].startswith('resume file:')
    r = client.post('/fit-check', json={'job_description': jd, 'resume_text': 'Python developer, 3 years.'})
    assert r.status_code == 422 and 'too little text' in r.json()['detail']


# ---------------------------------------------------------------- recruiter weights, filters, /rerank

def _all(out):
    return out['top_k'] + out['not_selected'] + out['filtered_out']


def test_custom_weights_order_by_formula(client, big_job):
    jd, cands = big_job
    body = {'job_description': jd, 'top_k': 5, 'weights': {'skills': 100, 'experience': 0, 'semantic': 0, 'keyword': 0},
            'resumes': [{'id': f'c{i}', 'text': t} for i, t in enumerate(cands.resume_text)]}
    out = client.post('/rank', json=body).json()
    s = out['summary']
    assert s['ranked_by'] == 'formula'                          # implied by custom weights
    assert s['weights'] == {'skills': 1.0, 'experience': 0.0, 'semantic': 0.0, 'keyword': 0.0}
    ranked = out['top_k'] + out['not_selected']
    skills = [c['score_breakdown']['skills'] for c in ranked]
    assert skills == sorted(skills, reverse=True)
    assert all(abs(c['rank_score'] - c['score_breakdown']['skills']) < 1e-3 for c in ranked)
    # explicit ranking=model keeps the model order and warns
    out2 = client.post('/rank', json={**body, 'ranking': 'model'}).json()
    assert out2['summary']['ranked_by'] == 'model'
    assert any('Custom weights change formula_score only' in w for w in out2['warnings'])
    # default request: model ranking with default weights
    plain = client.post('/rank', json={k: v for k, v in body.items() if k != 'weights'}).json()
    assert plain['summary']['ranked_by'] == 'model'
    assert plain['summary']['weights'] == {'skills': 0.4, 'experience': 0.25, 'semantic': 0.2, 'keyword': 0.15}


def test_invalid_weights(client, big_job):
    jd, cands = big_job
    body = {'job_description': jd, 'resumes': [{'text': t} for t in cands.resume_text.head(10)],
            'weights': {'skills': 0, 'experience': 0, 'semantic': 0, 'keyword': 0}}
    r = client.post('/rank', json=body)
    assert r.status_code == 422 and 'above 0' in r.json()['detail']
    body['weights'] = {'skills': -1}
    assert client.post('/rank', json=body).status_code == 422


def test_rerank_matches_fresh_request_and_is_fast(client, big_job):
    import time
    jd, cands = big_job
    resumes = [{'id': f'c{i}', 'text': t} for i, t in enumerate(cands.resume_text)]
    first = client.post('/rank', json={'job_description': jd, 'resumes': resumes, 'top_k': 10}).json()
    sid = first['session_id']
    assert sid
    opts = {'top_k': 7, 'weights': {'skills': 1, 'experience': 3, 'semantic': 1, 'keyword': 0},
            'filters': {'min_years': 3, 'must_have_skills': ['sql']}}
    t = time.time()
    re = client.post('/rerank', json={'session_id': sid, **opts})
    took = time.time() - t
    assert re.status_code == 200, re.text
    fresh = client.post('/rank', json={'job_description': jd, 'resumes': resumes, **opts}).json()
    strip = lambda o: [(c['resume_id'], c['rank'], c['rank_score'], c['passed_filters']) for c in _all(o)]
    assert strip(re.json()) == strip(fresh)
    assert re.json()['session_id'] == sid
    assert took < 2.0, took                                    # no BERT / model calls
    # rerank back to defaults reproduces the first answer
    back = client.post('/rerank', json={'session_id': sid, 'top_k': 10}).json()
    assert strip(back) == strip(first)
    assert client.post('/rerank', json={'session_id': 'nope'}).status_code == 404


def test_filters_must_have_and_min_years(client, big_job):
    jd, cands = big_job
    resumes = [{'id': f'c{i}', 'text': t} for i, t in enumerate(cands.resume_text)]
    out = client.post('/rank', json={'job_description': jd, 'resumes': resumes, 'top_k': 5,
                                     'filters': {'must_have_skills': ['SQL', 'k8s', 'sql'], 'min_years': 2,
                                                 'unknown_years': 'exclude'}}).json()
    n = len(cands)
    assert out['summary']['filtered_out'] == len(out['filtered_out'])
    assert len(_all(out)) == n
    assert [c['rank'] for c in _all(out)] == list(range(1, n + 1))
    kept = out['top_k'] + out['not_selected']
    for c in kept:
        assert c['passed_filters'] and [m['skill'] for m in c['must_have']] == ['sql', 'kubernetes']
        assert all(m['found'] for m in c['must_have'])
        assert c['experience']['found_years'] is not None and c['experience']['found_years'] >= 2
        assert c['reasons'][-1].startswith('Passes the filters')
    for c in out['filtered_out']:
        assert not c['passed_filters'] and c['reasons'][0].startswith('Filtered out')
        assert (not all(m['found'] for m in c['must_have']) or c['experience']['found_years'] is None
                or c['experience']['found_years'] < 2)
    scores = [c['rank_score'] for c in out['filtered_out']]
    assert scores == sorted(scores, reverse=True)


def test_rank_upload_form_options(client, big_job):
    jd, cands = big_job
    files = [('files', (f'r{i}.txt', t.encode(), 'text/plain')) for i, t in enumerate(cands.resume_text.head(12))]
    r = client.post('/rank/upload', files=files,
                    data={'job_description': jd, 'top_k': '3', 'weight_skills': '80', 'weight_keyword': '20',
                          'weight_experience': '0', 'weight_semantic': '0', 'must_have': 'zzzunknownskill'})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out['summary']['ranked_by'] == 'formula' and out['summary']['weights']['skills'] == 0.8
    assert len(out['filtered_out']) == 12 and out['top_k'] == [] and out['summary']['shortlist_cutoff'] is None
    assert any('not in the skill catalog' in w for w in out['warnings'])
    assert out['filtered_out'][0]['must_have'][0]['matched_as'] == 'text'


def test_must_have_resolution(pipe):
    from app.filters import MustHave
    k8s = MustHave(pipe, 'k8s')
    assert k8s.canon == 'kubernetes'
    aws = MustHave(pipe, 'AWS')
    chk = aws.check({'redshift'}, 'built warehouse on redshift')
    assert chk.found and chk.via == 'redshift' and chk.matched_as == 'catalog'
    assert not aws.check({'azure'}, 'azure').found               # broader / sibling skills don't count
    r = MustHave(pipe, 'R')
    assert r.canon is None
    assert r.check(set(), 'Statistics in R and Python').found
    assert not r.check(set(), 'great work for our clients').found
