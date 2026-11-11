"""Skill catalog v3: canonical skill -> aliases (all lowercase, matched on word boundaries).

Matching is longest-first and non-overlapping, so "data warehouse" is not also
counted as "warehouse". A canonical name is always its own alias.
"""

# v2 catalog (kept so v2 artifacts still match exactly)
V2_MULTI = ["machine learning", "deep learning", "computer vision", "data analysis", "data visualization", "power bi", "sql server", "a/b testing", "data warehouse",
            "project management", "program management", "stakeholder management", "risk management", "six sigma", "financial analysis", "financial reporting", "accounts payable",
            "accounts receivable", "variance analysis", "lead generation", "cold calling", "account management", "business development", "customer service", "client relations",
            "electrical engineering", "mechanical engineering", "civil engineering", "power distribution", "site supervision", "safety compliance", "quality control",
            "penetration testing", "vulnerability assessment", "network security", "incident response", "iso 27001", "active directory", "windows server", "help desk",
            "employee relations", "benefits administration", "performance management", "training and development", "talent acquisition", "google analytics", "content marketing",
            "social media", "email marketing", "brand management", "patient care", "medical terminology", "problem solving", "time management", "attention to detail", "team building",
            "public speaking", "inventory control", "supply chain", "lean manufacturing", "quality assurance", "iso 9001", "root cause analysis", "curriculum development",
            "lesson planning", "classroom management", "legal research", "oracle financials"]
V2_SINGLE = """python java sql c++ c# javascript typescript react angular vue node.js django flask spring .net html css php ruby scala kotlin swift aws azure gcp docker kubernetes terraform jenkins git linux unix bash powershell ci/cd devops microservices rest graphql nlp tensorflow pytorch scikit-learn pandas numpy spark hadoop kafka airflow etl tableau excel looker statistics mysql postgresql oracle mongodb redis snowflake agile scrum kanban jira confluence pmp lean budgeting forecasting gaap accounting payroll auditing tax reconciliation quickbooks sap salesforce crm negotiation upselling retail merchandising pos autocad solidworks revit plc scada hvac blueprints schematics nec osha construction estimating scheduling inspection carpentry welding plumbing cybersecurity siem firewall nist encryption networking tcp/ip dns dhcp vpn cisco routing switching vmware troubleshooting recruiting onboarding marketing seo sem copywriting photoshop illustrator nursing hipaa emr epic phlebotomy cpr clinical leadership communication teamwork mentoring logistics procurement warehouse forklift manufacturing cnc machining assembly kaizen 5s teaching contracts compliance regulatory""".split()
V2_SKILLS = sorted(set(V2_SINGLE + V2_MULTI))

# Aliases for existing skills
_ALIASES = {
    'javascript': ['js', 'java script', 'ecmascript'],
    'typescript': ['type script'],
    'node.js': ['nodejs', 'node js'],
    'react': ['reactjs', 'react.js', 'react js'],
    'angular': ['angularjs', 'angular.js', 'angular js'],
    'vue': ['vuejs', 'vue.js'],
    '.net': ['dotnet', 'asp.net', '.net core', 'net core'],
    'c#': ['c sharp'],
    'c++': ['cpp'],
    'postgresql': ['postgres', 'postgre sql'],
    'mongodb': ['mongo db', 'mongo'],
    'kubernetes': ['k8s'],
    'aws': ['amazon web services'],
    'gcp': ['google cloud', 'google cloud platform'],
    'azure': ['microsoft azure'],
    'ci/cd': ['ci cd', 'continuous integration', 'continuous delivery', 'continuous deployment'],
    'rest': ['restful', 'rest api', 'rest apis'],
    'machine learning': ['ml models', 'machine-learning'],
    'nlp': ['natural language processing'],
    'scikit-learn': ['sklearn', 'scikit learn'],
    'power bi': ['powerbi', 'power-bi'],
    'excel': ['ms excel', 'microsoft excel', 'advanced excel'],
    'sql server': ['mssql', 'ms sql', 'microsoft sql server'],
    'data warehouse': ['data warehousing', 'datawarehouse', 'data warehouses', 'edw', 'enterprise data warehouse'],
    'data analysis': ['data analytics', 'analyzing data', 'analysing data'],
    'data visualization': ['data visualisation', 'dashboards', 'dashboarding'],
    'etl': ['elt', 'extract transform load', 'etl pipelines'],
    'a/b testing': ['ab testing', 'a/b tests', 'split testing'],
    'project management': ['project manager', 'managing projects'],
    'customer service': ['customer support', 'client service', 'customer care'],
    'quality assurance': ['qa testing', 'software quality assurance'],
    'accounts payable': ['a/p'],
    'accounts receivable': ['a/r'],
    'cybersecurity': ['cyber security', 'information security', 'infosec'],
    'penetration testing': ['pen testing', 'pentesting', 'ethical hacking'],
    'tcp/ip': ['tcp ip'],
    'autocad': ['auto cad'],
    'hvac': ['heating ventilation'],
    'quickbooks': ['quick books'],
    'salesforce': ['sfdc', 'salesforce.com'],
    'seo': ['search engine optimization'],
    'sem': ['search engine marketing'],
    'crm': ['customer relationship management'],
    'cpr': ['bls', 'basic life support'],
    'git': ['github', 'gitlab', 'bitbucket'],
    'statistics': ['statistical analysis', 'statistical modeling'],
    'deep learning': ['neural networks'],
    'social media': ['social media marketing'],
    'six sigma': ['lean six sigma', 'six-sigma'],
    'pmp': ['project management professional'],
    'scrum': ['scrum master'],
    'gaap': ['us gaap'],
    'payroll': ['payroll processing'],
    'onboarding': ['on-boarding', 'on boarding'],
    'recruiting': ['recruitment', 'recruiter'],
    'help desk': ['helpdesk', 'service desk'],
    'troubleshooting': ['troubleshoot', 'trouble shooting'],
    'spring': ['spring boot', 'springboot', 'spring framework', 'spring mvc'],
}

# New skills (canonical: aliases)
_NEW = {
    # data engineering / analytics
    'dbt': ['data build tool'],
    'redshift': ['amazon redshift', 'aws redshift'],
    'bigquery': ['big query', 'google bigquery'],
    'databricks': [],
    'informatica': ['informatica powercenter'],
    'talend': [],
    'ssis': [], 'ssrs': [], 'ssas': [],
    't-sql': ['tsql', 'transact-sql'],
    'pl/sql': ['plsql', 'pl sql'],
    'nosql': ['no sql', 'no-sql'],
    'cassandra': [], 'dynamodb': ['dynamo db'], 'elasticsearch': ['elastic search', 'elk'],
    'teradata': [], 'hive': ['apache hive'], 'sqoop': [], 'pig': ['apache pig'], 'hbase': [],
    'pyspark': ['py spark'], 'big data': ['bigdata'],
    'data modeling': ['data modelling', 'data model', 'data models'],
    'dimensional modeling': ['dimensional modelling', 'star schema', 'snowflake schema', 'kimball'],
    'data pipelines': ['data pipeline', 'etl pipeline'],
    'data governance': ['data quality', 'master data management', 'mdm'],
    'data mining': [],
    'predictive modeling': ['predictive analytics', 'predictive models'],
    'r language': ['r programming', 'rstudio', 'r studio'],
    'matlab': [], 'sas': ['sas programming'], 'spss': [],
    'qlik': ['qlikview', 'qlik sense'], 'cognos': [], 'alteryx': [], 'microstrategy': [],
    'stream processing': ['streaming data', 'kinesis', 'spark streaming', 'flink'],
    # cloud / aws services / devops
    'ec2': ['amazon ec2', 'aws ec2'],
    's3': ['amazon s3', 'aws s3'],
    'rds': ['amazon rds', 'aws rds'],
    'aws lambda': ['lambda functions'],
    'aws emr': ['amazon emr', 'elastic mapreduce'],
    'aws glue': ['glue jobs'],
    'cloudformation': ['cloud formation'],
    'ansible': [], 'puppet': [],
    'openshift': [], 'helm charts': [],
    'prometheus': [], 'grafana': [], 'splunk': [], 'datadog': [], 'nagios': [],
    'maven': [], 'gradle': [],
    'serverless': [],
    'cloud computing': ['cloud infrastructure', 'cloud services'],
    # software / web / mobile
    'golang': ['go language'], 'rust': [], 'perl': [], 'shell scripting': ['shell script', 'shell scripts'],
    'html5': [], 'css3': [], 'jquery': [], 'bootstrap': [], 'redux': [], 'next.js': ['nextjs'],
    'express.js': ['expressjs'], 'hibernate': [], 'j2ee': ['java ee', 'jee'],
    'android': ['android development'], 'ios': ['ios development'], 'flutter': [], 'react native': [],
    'object oriented programming': ['oop', 'object-oriented', 'object oriented'],
    'design patterns': [], 'data structures': ['algorithms'],
    'api development': ['api design', 'web services', 'soap'],
    'unit testing': ['junit', 'pytest', 'testng', 'unit tests'],
    'test automation': ['automation testing', 'automated testing', 'selenium', 'cucumber'],
    'manual testing': ['functional testing', 'regression testing'],
    'performance testing': ['load testing', 'jmeter', 'loadrunner'],
    'sdlc': ['software development life cycle'],
    'uml': [], 'ux design': ['user experience', 'ui/ux', 'ux/ui', 'ui design'],
    'figma': [],
    'mainframe': ['cobol', 'jcl', 'mainframes'],
    'servicenow': ['service now'],
    'sharepoint': ['share point'],
    # security / networking / IT
    'identity and access management': ['iam', 'access management', 'okta', 'sso', 'single sign-on'],
    'soc': ['security operations center', 'security operations'],
    'risk assessment': ['risk assessments'],
    'cissp': [], 'comptia': ['security+', 'network+', 'a+ certification'],
    'ccna': [], 'ccnp': [], 'lan/wan': ['lan', 'wan'], 'wireless': ['wifi', 'wi-fi', 'wlan'],
    'itil': [], 'technical support': ['desktop support', 'it support'],
    'office 365': ['microsoft 365', 'o365', 'exchange server'],
    'ms office': ['microsoft office', 'ms word', 'microsoft word', 'outlook', 'powerpoint', 'power point'],
    'erp': ['erp systems'], 'oracle ebs': ['oracle e-business'], 'workday': [], 'peoplesoft': [],
    'netsuite': [], 'dynamics 365': ['microsoft dynamics'],
    # business / finance
    'business analysis': ['business analyst', 'requirements gathering', 'requirement gathering', 'brd'],
    'process improvement': ['continuous improvement', 'process optimization'],
    'change management': [],
    'vendor management': ['supplier management'],
    'contract negotiation': ['contract management'],
    'strategic planning': ['strategy development'],
    'operations management': ['operations manager'],
    'general ledger': ['gl'], 'financial modeling': ['financial modelling'],
    'cpa': ['certified public accountant'], 'cost accounting': [], 'invoicing': ['billing'],
    'bookkeeping': [], 'month-end close': ['month end close', 'month-end', 'year-end close'],
    'sox': ['sarbanes-oxley', 'sarbanes oxley'],
    'underwriting': [], 'loan processing': ['mortgage'], 'banking': [],
    'insurance claims': ['claims processing'],
    # sales / marketing
    'b2b sales': ['b2b'], 'inside sales': [], 'sales management': ['sales manager'],
    'market research': [], 'digital marketing': ['online marketing'],
    'google ads': ['adwords', 'google adwords', 'ppc', 'pay per click'],
    'hubspot': [], 'marketo': [], 'event planning': ['event management'],
    'public relations': ['pr campaigns'],
    # HR / admin
    'hris': [], 'compensation and benefits': [],
    'data entry': [], 'scheduling appointments': ['appointment scheduling'],
    'administrative support': ['office administration', 'clerical'],
    'bilingual': ['spanish', 'fluent in spanish'],
    # healthcare
    'ehr': ['electronic health records', 'electronic medical records'],
    'medical billing': ['medical coding', 'icd-10', 'cpt coding'],
    'pharmacy': ['pharmacy technician'], 'vital signs': [], 'acls': [], 'rn': ['registered nurse'],
    # engineering / trades / manufacturing
    'pcb design': ['pcb'], 'embedded systems': ['firmware', 'microcontrollers'],
    'circuit design': ['circuit analysis'], 'matlab simulink': ['simulink'],
    'labview': [], 'catia': [], 'creo': [], 'ansys': [], 'gd&t': ['gd & t'],
    'project engineering': ['project engineer'], 'commissioning': [],
    'preventive maintenance': ['preventative maintenance'], 'electrical wiring': ['wiring'],
    'cad': ['computer aided design'], 'six sigma green belt': ['green belt'],
    'fmea': [], 'iso 13485': [], 'gmp': ['good manufacturing practices', 'cgmp'],
    'osha 30': ['osha 10'],
    # logistics
    'shipping and receiving': ['shipping', 'receiving'], 'inventory management': ['inventory'],
    'fleet management': [], 'dispatch': ['dispatching'], 'purchasing': ['purchase orders'],
    # education / legal
    'tutoring': ['tutor'], 'special education': [], 'litigation': [], 'paralegal': [],
    'contract drafting': ['legal drafting'],
}

# Ambiguous aliases resolved by context words found anywhere in the same document.
# alias -> list of (context words, canonical). First matching rule wins; no match -> skipped.
CONTEXT_ALIASES = {
    'emr': [(['aws', 'amazon', 'hadoop', 'spark', 'hive', 'ec2', 's3', 'redshift', 'big data', 'pyspark'], 'aws emr'),
            (['patient', 'clinical', 'hospital', 'nurse', 'nursing', 'medical', 'hipaa', 'physician', 'clinic'], 'ehr')],
    'epic': [(['patient', 'clinical', 'hospital', 'nurse', 'nursing', 'medical', 'hipaa', 'physician', 'clinic'], 'epic')],
}

# Phrases that look like a skill but are not (regex on lowercased text).
NEGATIVE_PATTERNS = [
    r'\bexcel(?:s|led)?\s+(?:at|in)\b',          # "excel at analysing"
    r'\b(?:the\s+)?rest\s+of\b',               # "the rest of the team"
    r'\bspring\s+(?:\d{4}|semester|break|term)\b',  # seasons
    r'\blean\s+(?:on|into)\b',
]


def build_catalog() -> dict[str, list[str]]:
    cat: dict[str, list[str]] = {s: [s] for s in V2_SKILLS if s not in ('emr', 'epic')}
    for canon, aliases in _ALIASES.items():
        cat.setdefault(canon, [canon])
        cat[canon] += [a for a in aliases if a not in cat[canon]]
    for canon, aliases in _NEW.items():
        cat.setdefault(canon, [canon])
        cat[canon] += [a for a in aliases if a not in cat[canon]]
    for rules in CONTEXT_ALIASES.values():
        for _, canon in rules:
            cat.setdefault(canon, [] if canon in CONTEXT_ALIASES else [canon])
    # an alias must belong to exactly one canonical skill
    owner: dict[str, str] = {}
    for canon, aliases in cat.items():
        for a in aliases:
            assert a not in owner or owner[a] == canon, f'alias {a!r} in {owner.get(a)!r} and {canon!r}'
            owner[a] = canon
    return cat


CATALOG_V3 = build_catalog()


# ============================================================== v4: hierarchy + importance
# Extra skills/aliases for v4 (v3 artifacts store their own catalog, so this does not change them)
_V4_NEW = {
    'data engineering': ['data engineer', 'data engineers'],
    'data warehouse': ['dwh'],
    'airflow': ['apache airflow'],
    'kafka': ['apache kafka'],
    'spark': ['apache spark'],
    'ci/cd': ['cicd', 'ci-cd'],
}

# child -> parents. A resume showing a child is evidence of the parent ("Redshift" => AWS experience).
PARENTS = {}
for parent, children in {
    'aws': ['ec2', 's3', 'rds', 'redshift', 'aws emr', 'aws lambda', 'aws glue', 'dynamodb', 'cloudformation'],
    'gcp': ['bigquery'],
    'azure': ['dynamics 365'],
    'cloud computing': ['aws', 'azure', 'gcp', 'serverless'],
    'data engineering': ['etl', 'data pipelines', 'airflow', 'data warehouse', 'dbt', 'informatica', 'talend',
                         'ssis', 'aws glue', 'stream processing', 'big data'],
    'big data': ['hadoop', 'spark', 'pyspark', 'hive', 'hbase', 'pig', 'sqoop', 'kafka', 'aws emr', 'databricks'],
    'data modeling': ['dimensional modeling'],
    'data warehouse': ['dimensional modeling', 'snowflake', 'redshift', 'bigquery', 'teradata'],
    'sql': ['mysql', 'postgresql', 'sql server', 'oracle', 't-sql', 'pl/sql', 'teradata', 'snowflake'],
    'nosql': ['mongodb', 'cassandra', 'dynamodb', 'hbase', 'redis', 'elasticsearch'],
    'python': ['pandas', 'numpy', 'pyspark', 'django', 'flask', 'scikit-learn'],
    'java': ['spring', 'hibernate', 'j2ee'],
    'javascript': ['react', 'angular', 'vue', 'node.js', 'jquery', 'typescript', 'express.js', 'next.js'],
    'machine learning': ['deep learning', 'tensorflow', 'pytorch', 'scikit-learn', 'nlp', 'computer vision',
                         'predictive modeling'],
    'devops': ['docker', 'kubernetes', 'jenkins', 'ci/cd', 'terraform', 'ansible', 'puppet', 'helm charts'],
    'data visualization': ['tableau', 'power bi', 'looker', 'qlik', 'cognos', 'microstrategy'],
    'accounting': ['accounts payable', 'accounts receivable', 'general ledger', 'bookkeeping', 'reconciliation',
                   'gaap', 'month-end close', 'cost accounting'],
    'ehr': ['epic'],
    'networking': ['tcp/ip', 'dns', 'dhcp', 'vpn', 'routing', 'switching', 'lan/wan', 'ccna', 'ccnp'],
    'cybersecurity': ['siem', 'firewall', 'penetration testing', 'vulnerability assessment', 'incident response',
                      'network security', 'identity and access management', 'cissp'],
    'quality assurance': ['test automation', 'manual testing', 'performance testing', 'unit testing'],
    'erp': ['sap', 'oracle ebs', 'workday', 'peoplesoft', 'netsuite', 'dynamics 365'],
    'crm': ['salesforce', 'hubspot'],
    'digital marketing': ['seo', 'sem', 'google ads', 'email marketing', 'social media', 'content marketing',
                          'google analytics', 'marketo'],
}.items():
    for c in children:
        PARENTS.setdefault(c, []).append(parent)

SOFT_SKILLS = {'communication', 'leadership', 'teamwork', 'problem solving', 'time management',
               'attention to detail', 'team building', 'public speaking', 'mentoring', 'bilingual'}

# A JD sentence containing one of these marks its skills as preferred (not required).
PREFERRED_CUES = (r'\b(?:preferred|preferably|nice[\s-]to[\s-]have|good[\s-]to[\s-]have|(?:is\s+)?an?\s+plus|'
                  r'is\s+plus|bonus|desir(?:ed|able)|an\s+advantage|advantageous|optional|familiarity\s+with|'
                  r'exposure\s+to)\b')


def build_catalog_v4() -> dict[str, list[str]]:
    cat = {k: list(v) for k, v in build_catalog().items()}
    for canon, aliases in _V4_NEW.items():
        cat.setdefault(canon, [canon])
        cat[canon] += [a for a in aliases if a not in cat[canon]]
    owner = {}
    for canon, aliases in cat.items():
        for a in aliases:
            assert a not in owner or owner[a] == canon, f'alias {a!r} in {owner.get(a)!r} and {canon!r}'
            owner[a] = canon
    missing = {s for s in PARENTS} | {p for ps in PARENTS.values() for p in ps}
    assert missing <= set(cat), f'hierarchy names not in catalog: {missing - set(cat)}'
    return cat


CATALOG_V4 = build_catalog_v4()

# Broad competencies (a field or practice) vs. concrete technologies/tools. A JD's competencies are
# weighted lower and are never used to infer a specific tool (no Airflow from "data engineering").
COMPETENCIES = {
    'data engineering', 'big data', 'data modeling', 'dimensional modeling', 'data warehouse', 'data pipelines', 'etl', 'data governance',
    'data analysis', 'data visualization', 'data mining', 'predictive modeling', 'statistics', 'machine learning',
    'deep learning', 'nlp', 'computer vision', 'cloud computing', 'devops', 'ci/cd', 'microservices', 'serverless',
    'stream processing', 'sdlc', 'object oriented programming', 'design patterns', 'data structures',
    'api development', 'networking', 'cybersecurity', 'network security', 'quality assurance', 'test automation',
    'manual testing', 'performance testing', 'unit testing', 'project management', 'program management',
    'business analysis', 'process improvement', 'change management', 'risk management', 'accounting',
    'financial analysis', 'financial reporting', 'digital marketing', 'marketing', 'sales', 'erp', 'crm',
    'technical support', 'cloud services', 'electrical engineering', 'mechanical engineering', 'civil engineering',
    'manufacturing', 'construction', 'logistics', 'supply chain', 'inventory management', 'agile',
}
