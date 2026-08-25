"""Tests for Stage 1 consolidation pipeline."""

from awhm.consolidation.entity_linking import LinkCandidate
from awhm.consolidation.ner import ExtractedEntity
from awhm.consolidation.pipeline import Stage1Pipeline
from awhm.graph.memory_graph import MemoryGraph
from awhm.graph.models import MemoryNode
from awhm.raw_log.logger import RawLogger
from awhm.types import NodeType, Role


def test_consolidate_session(config, mock_embedding):
    # Create a session with some content
    logger = RawLogger(config, "test-session")
    logger.log(Role.USER, "My name is Alice")
    logger.log(Role.ASSISTANT, "Hello Alice!")
    logger.log(Role.USER, "I prefer Python over JavaScript")
    logger.log(Role.USER, "The API endpoint is https://api.example.com")

    graph = MemoryGraph()
    pipeline = Stage1Pipeline(config, graph, mock_embedding)
    count = pipeline.consolidate_session("test-session")

    assert count > 0
    assert graph.node_count() > 0


def test_consolidate_all_pending(config, mock_embedding):
    # Create two sessions
    for sid in ["s1", "s2"]:
        logger = RawLogger(config, sid)
        logger.log(Role.USER, f"Session {sid}: I prefer dark mode")

    graph = MemoryGraph()
    pipeline = Stage1Pipeline(config, graph, mock_embedding)
    results = pipeline.consolidate_all_pending()

    assert len(results) == 2


def test_consolidate_idempotent(config, mock_embedding):
    logger = RawLogger(config, "s1")
    logger.log(Role.USER, "I prefer dark mode")

    graph = MemoryGraph()
    pipeline = Stage1Pipeline(config, graph, mock_embedding)
    pipeline.consolidate_session("s1")
    count_after_first = graph.node_count()

    # Consolidating again should not add more nodes
    results = pipeline.consolidate_all_pending()
    assert len(results) == 0
    assert graph.node_count() == count_after_first


def test_consolidate_session_incremental(config, mock_embedding):
    logger = RawLogger(config, "s1")
    logger.log(Role.USER, "I prefer dark mode")

    graph = MemoryGraph()
    pipeline = Stage1Pipeline(config, graph, mock_embedding)
    first_count = pipeline.consolidate_session("s1")
    assert first_count > 0

    # Append a new message after initial consolidation.
    logger.log(Role.USER, "Actually, I prefer light mode")

    second_count = pipeline.consolidate_session("s1")
    assert second_count > 0


def test_association_edges_only_from_relevant_entity_nodes(
    config,
    mock_embedding,
    monkeypatch,
):
    logger = RawLogger(config, "s1")
    logger.log(Role.USER, "I prefer dark mode")

    graph = MemoryGraph()
    existing_emb = mock_embedding.encode(["PERSON: Alice"])[0].tolist()
    graph.add_node(
        MemoryNode(
            id="existing",
            type=NodeType.SEMANTIC.value,
            content="PERSON: Alice",
            embedding=existing_emb,
            source_sessions=["seed"],
        ),
    )

    pipeline = Stage1Pipeline(config, graph, mock_embedding)

    monkeypatch.setattr(
        pipeline.ner,
        "extract_from_messages",
        lambda messages, message_offset=0: [
            ExtractedEntity(
                text="Alice",
                label="PERSON",
                start=0,
                end=5,
                message_index=message_offset,
            ),
        ],
    )
    monkeypatch.setattr(
        "awhm.consolidation.pipeline.find_links",
        lambda entity_texts, graph, embedding_service, threshold=0.85, entity_labels=None: [
            LinkCandidate(
                entity_text="Alice",
                matched_node_id="existing",
                matched_content="PERSON: Alice",
                string_sim=1.0,
                embedding_sim=0.95,
            ),
        ],
    )

    pipeline.consolidate_session("s1")

    # Entity node is skipped because it linked to existing; no association edge
    # should be created from unrelated session nodes.
    wrong_edges = [
        e for e in graph.edges
        if e.target == "existing" and e.type == "association"
    ]
    assert wrong_edges == []


def _spacy_model_available() -> bool:
    try:
        import spacy

        spacy.load("en_core_web_sm")
        return True
    except Exception:
        return False


import pytest  # noqa: E402


@pytest.mark.skipif(not _spacy_model_available(), reason="spaCy model not installed")
def test_ner_label_filter_drops_numeric_entities(config, mock_embedding):
    logger = RawLogger(config, "s1")
    logger.log(Role.USER, "Alice works at Google in London and bought 3 laptops")

    graph = MemoryGraph()
    Stage1Pipeline(config, graph, mock_embedding).consolidate_session("s1")
    contents = {n.content for n in graph.all_nodes() if n.entity_type}
    assert any(c.startswith("PERSON:") for c in contents)
    assert any(c.startswith("ORG:") for c in contents)
    assert not any(c.startswith("CARDINAL:") for c in contents)
