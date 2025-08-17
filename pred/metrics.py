import re
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import jaccard_score
def multiple_choice_answer(response):
    response = response.replace('*', '')
    match = re.search(r"[Tt]he correct answer is .*([A-D])", response)
    if match:
        return match.group(1)  
    else:
        return None

def extract_case_answer(response):
    response = response.replace('*', '')
    match = re.search(r'[Tt]he correct answer is.*?(CASE_\d+)', response, re.IGNORECASE)
    return match.group(1) if match else None
    return None

def extract_law_answer(response):
    response = response.replace('*', '')
    match = re.search(r'[Tt]he correct answer is.*?(LAW_\d+)', response, re.IGNORECASE | re.DOTALL)
    return match.group(1) if match else None

def extract_trend_analysis_answer(response):
    match = re.search(r"[Tt]he correct answer is\s*(.*)", response)
    if not match:
        return []
    answer_text = match.group(1)
    pattern = re.compile(r'\b\d{4}-\d{4}\b')
    answer_text = pattern.findall(answer_text)
    return answer_text


def extract_finance_answer(response):
    match = re.search(r"[Tt]he correct answer is+(.*)", response)
    if not match:
        return []
    answer_text = match.group(1)
    pattern = re.compile(r'\d{1,6}(?:,\d{3})*(?:\.\d+)?(?:[M%])?')
    answer_text = pattern.findall(answer_text)
    return answer_text

def extract_version_control_answer(response):
    matches = re.findall(r'[\w/]+\.py', response)
    return list(set(matches))


def compute_jaccard_score(list1, list2):

    if not list1 or not list2:
        return 0.0
    mlb = MultiLabelBinarizer()
    binarized = mlb.fit_transform([list1, list2])
    score = jaccard_score(binarized[0], binarized[1]) * 100
    return round(score, 2)

def normalize_number(s):
    if isinstance(s, list):
        s = s[0] if s else ''
    if not s:
        return None
    try:
        s = s.replace(',', '').replace('$', '').strip()
        multiplier = 1.0
        if s.endswith('M'):
            s = s[:-1]
        elif s.endswith('%'):
            s = s[:-1]
            multiplier = 1
        return float(s) * multiplier
    except ValueError:
        return None
    
def safe_get_first(lst):
    return lst[0] if isinstance(lst, list) and lst else None

extractor_map = {
    'Legal Case Retrieval': extract_case_answer,
    'Legal Article Extraction': extract_law_answer,
    'Trend Analysis': lambda r: safe_get_first(extract_trend_analysis_answer(r)),
    'Metric Calculation': lambda r: safe_get_first(extract_finance_answer(r)),
    'Cross-Company Comparison': lambda r: safe_get_first(extract_finance_answer(r)),
    'Version Control': extract_version_control_answer,
}