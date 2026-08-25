"""Tests for session buffer pattern matching."""

from awhm.session_buffer.patterns import match_message
from awhm.types import BufferEntryType


def test_correction_actually():
    matches = match_message("Actually, the port is 8080")
    types = [m.type for m in matches]
    assert BufferEntryType.CORRECTION in types


def test_correction_no_its():
    matches = match_message("No, it's Python 3.11")
    types = [m.type for m in matches]
    assert BufferEntryType.CORRECTION in types


def test_preference_i_prefer():
    matches = match_message("I prefer using TypeScript over JavaScript")
    types = [m.type for m in matches]
    assert BufferEntryType.PREFERENCE in types


def test_preference_always_use():
    matches = match_message("Always use black for formatting")
    types = [m.type for m in matches]
    assert BufferEntryType.PREFERENCE in types


def test_preference_never_use():
    matches = match_message("Never use tabs for indentation")
    types = [m.type for m in matches]
    assert BufferEntryType.PREFERENCE in types


def test_fact_endpoint():
    matches = match_message("The API endpoint is https://api.example.com/v2")
    types = [m.type for m in matches]
    assert BufferEntryType.FACT in types


def test_fact_my_name():
    matches = match_message("My name is Alice")
    types = [m.type for m in matches]
    assert BufferEntryType.FACT in types


def test_outcome_worked():
    matches = match_message("That worked, thanks!")
    types = [m.type for m in matches]
    assert BufferEntryType.OUTCOME in types


def test_outcome_failed():
    matches = match_message("That failed with an error")
    types = [m.type for m in matches]
    assert BufferEntryType.OUTCOME in types


def test_no_match():
    matches = match_message("Can you help me write a function?")
    assert len(matches) == 0
