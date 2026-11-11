"""Unit tests for the v3 feature functions (cleaning, skills, experience)."""
from app.ml.features import SkillMatcher, clean_v2, clean_v3, dated_years, stated_years

KNOWN = frozenset({'data', 'engineering', 'engineer', 'warehousing', 'teradata', 'experience', 'demand',
                   'understand', 'command', 'python'})


def test_clean_v3_splits_glued_text():
    assert clean_v3('Data Engineeringwith experience over8 years', KNOWN) == 'Data Engineering with experience over 8 years'
    assert clean_v3('Experience01/2018to current', KNOWN) == 'Experience 01/2018 to current'
    assert clean_v3('engineering.12/2009to11/2011senior', KNOWN) == 'engineering.12/2009 to 11/2011 senior'
    assert clean_v3('Teradataand DB2', KNOWN) == 'Teradata and DB2'


def test_clean_v3_keeps_real_words_and_tokens():
    assert clean_v3('demand understand command HTML5 CSS3 1st 4GB EC2 S3', KNOWN) == \
        'demand understand command HTML5 CSS3 1st 4GB EC2 S3'


def test_clean_v3_is_superset_of_v2():
    s = 'SummaryHighly motivated   engineer'
    assert clean_v3(s) == clean_v2(s) == 'Summary Highly motivated engineer'


def test_skills_longest_match_and_aliases():
    m = SkillMatcher(3)
    found = m.extract('Built a data warehouse on Postgres, k8s, AWS EMR with Hadoop, DBT and Redshift')
    assert {'data warehouse', 'postgresql', 'kubernetes', 'aws emr', 'dbt', 'redshift'} <= found
    assert 'warehouse' not in found          # only part of "data warehouse"


def test_skills_context_and_negatives():
    m = SkillMatcher(3)
    medical = m.extract('Patient care, charting in EMR and Epic. I excel at teamwork with the rest of the team')
    assert {'ehr', 'epic', 'patient care', 'teamwork'} <= medical
    assert 'aws emr' not in medical and 'excel' not in medical and 'rest' not in medical
    assert 'epic' not in m.extract('an epic product launch')       # no medical context
    assert 'spring' not in m.extract('graduated Spring 2015')
    assert 'excel' in m.extract('Advanced MS Excel and pivot tables')


def test_stated_years():
    assert stated_years('8 to 10 years of experience working in data') == 8
    assert stated_years('10 plus years of customer service') == 10
    assert stated_years('minimum five years experience. The company has 50 years of history') == 5
    assert stated_years('4-year degree; 3+ yrs experience in SQL') == 3
    assert stated_years('four year college') is None


def test_dated_years_merges_and_skips_education():
    text = ('Data Engineer 01/2015 to 06/2016 Acme; Senior Engineer 07/2016 - Current. '
            'Bachelor of Science 2008 - 2012 University. Certified 2019')
    assert dated_years(text) == 5.4          # 1.4 + (2016.5 -> 2020.5); education range ignored
    assert dated_years('Jan 2010 to Dec 2012 analyst, March 2011 - May 2014 lead') == 4.3   # overlap merged
    assert dated_years('no dates here') is None


def test_required_vs_preferred_and_hierarchy():
    from app.ml.features import jd_skill_importance, weighted_skill_features
    from app.ml.skills_catalog import CATALOG_V4
    m = SkillMatcher(3, CATALOG_V4)
    jd = ('Requirements: Python, SQL, AWS (Redshift, EC2). Strong communication. '
          'Preferred Qualifications: Kafka. Experience with DBT is a plus.')
    imp = jd_skill_importance(jd, m)
    assert imp['python'] == ('required', 1.0)
    assert imp['kafka'][0] == 'preferred' and imp['dbt'][0] == 'preferred'
    assert imp['communication'][0] == 'soft'
    assert imp['aws'] == ('required', 0.5)            # children (redshift, ec2) also listed -> not double counted
    f = weighted_skill_features(imp, m.extract('Python, MySQL, Amazon Redshift'))
    det = {d['skill']: (d['status'], d['via']) for d in f['skill_details']}
    assert det['redshift'] == ('matched', None)
    assert det['aws'] == ('matched', 'redshift')      # child is evidence of the parent
    assert det['sql'] == ('matched', 'mysql')
    assert det['ec2'] == ('missing', None)            # AWS services are not interchangeable
    assert f['pref_coverage'] == 0.0 and 0 < f['req_coverage'] < 1     # no kafka/dbt; ec2 missing


def test_evidence_only_flows_upward():
    from app.ml.features import skill_credit
    assert skill_credit('postgresql', {'mysql'}) == (0.5, 'related', 'mysql')   # interchangeable SQL dialects
    assert skill_credit('aws', {'redshift'}) == (1.0, 'matched', 'redshift')    # specific => broad
    assert skill_credit('airflow', {'data engineering'})[1] == 'missing'        # broad never implies a tool
    assert skill_credit('dbt', {'airflow', 'data engineering'})[1] == 'missing'
    assert skill_credit('aws', {'azure'})[1] == 'missing'                       # providers are not interchangeable
    assert skill_credit('ec2', {'redshift'})[1] == 'missing'


def test_core_guard_and_formula_ranking():
    import pandas as pd
    from app.ml.pipeline import finalize
    df = pd.DataFrame({
        'job_description_text': ['jd'] * 3, 'bert_score': [0.6, 0.4, 0.5], 'tfidf_score': [0.1, 0.05, 0.2],
        'w_skill_score': [0.0, 0.9, 0.5], 'exp_score': [1.0, 1.0, 0.5], 'rank_score': [3.0, 1.0, 2.0],
        'label': ['Good Fit', 'Good Fit', 'Potential Fit'], 'confidence': [0.7, 0.6, 0.5],
        'p_potential_fit': [0.2, 0.3, 0.5], 'n_core': [5, 5, 5], 'n_core_matched': [0, 4, 2]})
    m = finalize(df, 'model', core_guard=True)
    assert list(m.label) == ['Potential Fit', 'Good Fit', 'Potential Fit'] and m.label_capped.tolist() == [True, False, False]
    assert m.rank_score.tolist() == [3.0, 1.0, 2.0]                     # guard never changes the order
    f = finalize(df, 'formula', core_guard=False)
    assert (f.rank_score == f.formula_score).all() and f.rank_score.idxmax() == 1
    assert f.label.tolist() == df.label.tolist()


def test_jd_footer_is_ignored_for_skills():
    from app.ml.features import strip_jd_footer
    jd = ('Developer with Python, Ansible and Jenkins. ' * 5 +
          'About US Tech Solutions, please visit www.example.com. Recruiter Details: Name: A. Person')
    assert 'Recruiter' not in strip_jd_footer(jd) and 'Jenkins' in strip_jd_footer(jd)
    early = 'Equal Opportunity Employer is mentioned early. ' + 'Python developer needed. ' * 10
    assert strip_jd_footer(early) == early          # marker in the first half: keep everything
