"""Tests for temporal validity inference."""

from awhm import AWHMSession
from awhm.config import AWHMConfig
from awhm.consolidation.temporal import TemporalParser, validity_from_context
from awhm.types import Role


def test_validity_from_context_markers():
    parser = TemporalParser()
    text = "From 2026-03-01 the API endpoint is https://api.v2.example.com"
    dates = parser.extract_dates(text)
    valid_from, valid_to = validity_from_context(text, dates)
    assert valid_from and valid_from.startswith("2026-03-01")
    assert valid_to is None

    text = "The office is closed until 2026-04-10"
    valid_from, valid_to = validity_from_context(text, parser.extract_dates(text))
    assert valid_from is None
    assert valid_to and valid_to.startswith("2026-04-10")


def test_dates_in_passing_leave_window_open():
    parser = TemporalParser()
    text = "The deploy on 2026-02-20 went fine, the port is 8080"
    assert validity_from_context(text, parser.extract_dates(text)) == (None, None)


def test_dates_enrich_statements_instead_of_becoming_nodes(tmp_path):
    config = AWHMConfig(data_dir=str(tmp_path / "awhm"), cold_start_session_count=0)
    with AWHMSession.start_session(config, session_id="s1", use_mock_embeddings=True) as s:
        s.log_message(Role.USER, "From 2026-03-01 the API endpoint is https://api.v2.example.com")
        s.consolidate_current()
        nodes = s.graph.all_nodes()
        assert not any(n.content.startswith("Date reference") for n in nodes)
        fact = next(n for n in nodes if "endpoint" in n.content)
        assert fact.valid_from.startswith("2026-03-01")
        assert fact.mentioned_dates and fact.mentioned_dates[0].startswith("2026-03-01")

        assert s.query("API endpoint", as_of="2026-02-01") == []
        assert any("v2" in r.content for r in s.query("API endpoint", as_of="2026-03-15"))
