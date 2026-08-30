from __future__ import annotations

import re
from collections.abc import Iterable


def term_present(text: str, term: str) -> bool:
    normalized_text = text.casefold()
    normalized_term = term.casefold().strip()
    if not normalized_term:
        return False
    pattern = rf"(?<!\w){re.escape(normalized_term)}(?!\w)"
    return re.search(pattern, normalized_text) is not None


def matching_terms(text: str, terms: Iterable[str]) -> list[str]:
    return [term for term in terms if term_present(text, term)]


def contains_any_term(text: str, terms: Iterable[str]) -> bool:
    return any(term_present(text, term) for term in terms)
