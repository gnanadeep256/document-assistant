import re
from enum import Enum, auto


class QueryIntent(Enum):
    DEFINITION = auto()
    SECTION = auto()
    EXPLANATION = auto()
    COMPARISON = auto()
    WHY = auto()
    FACT = auto()
    GENERAL = auto()
    DOCUMENT_SUMMARY = auto()


# ---------------- Pattern definitions ---------------- #

SECTION_PATTERN = re.compile(r"\bsection\s+\d+(\.\d+)*|\b\d+(\.\d+)+\b", re.I)

DOCUMENT_SUMMARY_KEYWORDS = [
    "summarize",
    "summary",
    "overview",
    "abstract",
    "what does this paper propose",
    "main contribution",
    "key idea of this paper",
]

DEFINITION_PATTERNS = [
    r"\bdefine\b",
    r"\bwhat is\b",
    r"\bwhat are\b",
    r"\bmeaning of\b",
]

EXPLANATION_PATTERNS = [
    r"\bexplain\b",
    r"\bdescribe\b",
    r"\bhow does\b",
    r"\bhow do\b",
]

COMPARISON_PATTERNS = [
    r"\bcompare\b",
    r"\bdifference\b",
    r"\bvs\b",
    r"\bversus\b",
]

WHY_PATTERNS = [
    r"\bwhy\b",
    r"\breason\b",
]

FACT_PATTERNS = [
    r"\bhow many\b",
    r"\bhow much\b",
    r"\bwhat score\b",
    r"\bwhat value\b",
    r"\bwhat rate\b",
]


def _matches_any(patterns, text: str) -> bool:
    return any(re.search(p, text) for p in patterns)


def detect_intent(query: str) -> QueryIntent:
    q = query.lower().strip()

    if SECTION_PATTERN.search(q):
        return QueryIntent.SECTION

    if any(k in q for k in DOCUMENT_SUMMARY_KEYWORDS):
        return QueryIntent.DOCUMENT_SUMMARY

    if _matches_any(COMPARISON_PATTERNS, q):
        return QueryIntent.COMPARISON

    if _matches_any(WHY_PATTERNS, q):
        return QueryIntent.WHY

    if _matches_any(DEFINITION_PATTERNS, q):
        return QueryIntent.DEFINITION

    if _matches_any(FACT_PATTERNS, q):
        return QueryIntent.FACT

    if _matches_any(EXPLANATION_PATTERNS, q):
        return QueryIntent.EXPLANATION

    return QueryIntent.GENERAL
