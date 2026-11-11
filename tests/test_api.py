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
