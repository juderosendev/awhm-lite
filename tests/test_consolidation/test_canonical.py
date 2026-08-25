"""Tests for canonical keys and correction rules."""

from awhm.consolidation.canonical import (
    canonical_key_for_content,
    correction_supersedes,
    is_correction,
    strip_correction_prefix,
)


def test_is_slot_key():
    assert canonical_key_for_content("My preferred language is Python") == "fact:my preferred language"
    assert canonical_key_for_content("Actually, my preferred language is Rust") == "fact:my preferred language"
    assert canonical_key_for_content("The API endpoint is https://api.example.com") == "fact:the api endpoint"


def test_first_person_slots():
    assert canonical_key_for_content("I live in San Francisco") == "fact:i live in"
    assert canonical_key_for_content("Actually, I live in New York") == "fact:i live in"
    assert canonical_key_for_content("I work at Acme") == "fact:i work at"


def test_preference_and_policy_keys():
    assert canonical_key_for_content("I prefer Python over JavaScript") == "preference:python"
    assert canonical_key_for_content("I prefer dark mode") == "preference:dark"
    assert canonical_key_for_content("Always use black for formatting") == "policy:use:black"
    assert canonical_key_for_content("Never use tabs for indentation") == "policy:use:tabs"


def test_no_key_for_additive_statements():
    assert canonical_key_for_content("I use Python for scripting") is None
    assert canonical_key_for_content("PERSON: Alice") is None
    assert canonical_key_for_content("That worked, thanks!") is None
    assert canonical_key_for_content("") is None


def test_long_subjects_are_not_keyed():
    text = "the plan we discussed at length with the whole team on friday afternoon is final"
    assert canonical_key_for_content(text) is None


def test_strip_correction_prefix():
    assert strip_correction_prefix("Actually, the port is 8080") == ("the port is 8080", True)
    assert strip_correction_prefix("The port is 8080") == ("The port is 8080", False)
    assert is_correction("No, it's 8080")
    assert not is_correction("It's 8080")


def test_same_key_always_supersedes():
    assert correction_supersedes("fact:my name", False, "fact:my name", None, 3)
    assert correction_supersedes("policy:use:tabs", False, "policy:use:tabs", 50, 3)


def test_fact_family_requires_key_match():
    # A correction about the endpoint must not clobber an unrelated fact nearby.
    assert not correction_supersedes("fact:the api endpoint", True, "fact:my name", 1, 3)


def test_implicit_family_correction_within_window():
    assert correction_supersedes("preference:rust", True, "preference:python", 1, 3)
    assert correction_supersedes("preference:rust", True, "preference:python", 3, 3)
    assert not correction_supersedes("preference:rust", True, "preference:python", 4, 3)
    assert not correction_supersedes("preference:rust", True, "preference:python", None, 3)


def test_implicit_family_without_correction_is_additive():
    assert not correction_supersedes("preference:dark", False, "preference:python", 1, 3)


def test_cross_family_never_supersedes():
    assert not correction_supersedes("preference:rust", True, "policy:use:black", 1, 3)
