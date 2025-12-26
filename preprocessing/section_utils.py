import re
from typing import Optional, List

SECTION_ID_RE = re.compile(r"^(\d+(?:\.\d+)*)")

def extract_section_id(title: Optional[str]) -> Optional[str]:
    if not title:
        return None
    m = SECTION_ID_RE.match(title.strip())
    return m.group(1) if m else None


def section_parents(section_id: str) -> List[str]:
    """
    3.2.3 -> ['3', '3.2']
    """
    parts = section_id.split(".")
    parents = []
    for i in range(1, len(parts)):
        parents.append(".".join(parts[:i]))
    return parents
