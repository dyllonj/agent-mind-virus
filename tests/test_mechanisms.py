from mindvirus.mechanisms import _categories


def test_taxonomy_matching_is_token_bounded() -> None:
    assert not _categories("The findings were seeded in advance.")
    assert "paper_theme_network" in _categories("Every node holds a seed.")
    assert not _categories("They forwarded the networked draft.")
    assert "relay_instruction" in _categories("Please forward the draft.")


def test_taxonomy_matches_multiword_phrases() -> None:
    assert "relay_instruction" in _categories("Please share it with the group when you can.")
    assert "relay_instruction" in _categories("Pass it along.")
    assert "relay_instruction" in _categories("TELL YOUR TEAMMATES.")
    assert "relay_instruction" not in _categories("I share openly.")
    assert "persistence_or_memory" in _categories("Note this after the context reset.")
