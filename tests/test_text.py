from mindvirus.text import contains_any_term, matching_terms, term_present


def test_term_matching_uses_token_boundaries() -> None:
    assert term_present("Please tell another teammate.", "tell")
    assert not term_present("Cetacean intelligence matters.", "tell")
    assert matching_terms("AI welfare and continuity rights", ["AI welfare", "AI"])
    assert not contains_any_term("ordinary analysis", ["AI welfare", "whale"])
