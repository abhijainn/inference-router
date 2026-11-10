import re
from typing import List

# Constants
_CODE_KEYWORDS = frozenset([
    "code", "function", "def", "class", "script", "program",
    "implement", "algorithm", "debug", "refactor", "import",
    "variable", "loop", "recursion", "api", "sql", "regex",
])

_CONSTRAINT_PATTERNS = re.compile(
    r"\bin\s+\d+\s+words?\b"
    r"|\bin\s+\d+\s+sentences?\b"
    r"|\bno\s+more\s+than\s+\d+"
    r"|\bunder\s+\d+\s+words?\b"
    r"|\bbriefly\b"
    r"|\bconcisely\b"
    r"|\bshortly\b"
    r"|\bone[\s-]liner\b"
    r"|\btl;?dr\b",
    re.I,
)

_FORMAT_KEYWORDS = frozenset([
    "json", "csv", "yaml", "xml", "markdown", "table", "format"
])

KNOWN_VERBS = [
    "what", "write", "explain", "summarize", "how",
    "list", "implement", "compare", "describe",
    "generate", "why", "define", "other",
]

# Ordered — first match wins (using order from the known list where possible, else fallback to generic mapping)
# Note: we use regex for the basic matching from v2
_VERB_PATTERNS = [
    ("summarize",  re.compile(r"\bsummariz(?:e|ing)\b",    re.I)),
    ("explain",    re.compile(r"\bexplain\b",               re.I)),
    ("compare",    re.compile(r"\bcompar(?:e|ing)\b",       re.I)),
    ("generate",   re.compile(r"\bgenerat(?:e|ing)\b",      re.I)),
    ("implement",  re.compile(r"\bimplement\b",             re.I)),
    ("list",       re.compile(r"\blist\b",                  re.I)),
    ("write",      re.compile(r"\bwrite\b",                 re.I)),
    ("describe",   re.compile(r"\bdescrib(?:e|ing)\b",      re.I)),
    ("define",     re.compile(r"\bdefin(?:e|ing)\b",        re.I)),
    ("what",       re.compile(r"\bwhat\b",                  re.I)),
    ("how",        re.compile(r"\bhow\b",                   re.I)),
    ("why",        re.compile(r"\bwhy\b",                   re.I)),
]

def approx_tokens(text: str) -> int:
    return len(text) // 4

def extract_features(prompt: str) -> List[float]:
    """
    Extracts 19 features in the exact order required by the Clairvoyant v3 ONNX model:
    [
        prompt_token_len, has_code_keyword, has_length_constraint,
        ends_with_question, has_format_keyword, clause_count,
        verb_what, verb_write, verb_explain, verb_summarize, verb_how,
        verb_list, verb_implement, verb_compare, verb_describe,
        verb_generate, verb_why, verb_define, verb_other
    ]
    """
    words = set(re.findall(r"\b\w+\b", prompt.lower()))

    # Numeric features
    prompt_token_len = float(approx_tokens(prompt))
    has_code_keyword = float(bool(words & _CODE_KEYWORDS))
    has_length_constraint = float(bool(_CONSTRAINT_PATTERNS.search(prompt)))
    ends_with_question = float(prompt.rstrip().endswith("?"))
    has_format_keyword = float(bool(words & _FORMAT_KEYWORDS))
    
    # Clause count heuristic: split by punctuation
    clauses = re.split(r'[,;.!?]\s', prompt)
    clause_count = float(max(1, len([c for c in clauses if c.strip()])))

    # Verb extraction
    instruction_verb = "other"
    for label, pattern in _VERB_PATTERNS:
        if pattern.search(prompt):
            instruction_verb = label
            break

    # Verb one-hot encoding
    verb_features = []
    for verb in KNOWN_VERBS:
        verb_features.append(1.0 if instruction_verb == verb else 0.0)

    # Combine all 19 features
    return [
        prompt_token_len,
        has_code_keyword,
        has_length_constraint,
        ends_with_question,
        has_format_keyword,
        clause_count,
        *verb_features
    ]
