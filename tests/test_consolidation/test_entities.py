"""Tests for entity resolution."""

from awhm import AWHMSession
from awhm.config import AWHMConfig
from awhm.consolidation.entities import EntityResolver, normalize_entity
from awhm.consolidation.ner import ExtractedEntity
from awhm.graph.memory_graph import MemoryGraph
from awhm.graph.models import MemoryNode
from awhm.types import NodeType, Role


def test_normalize_entity_forms():
    assert normalize_entity("Acme Holdings Ltd") == "acme"
    assert normalize_entity("acme.com") == "acme"
    assert normalize_entity("https://www.acme-corp.com/about") == "acme corp"
    assert normalize_entity("Acme's") == "acme"
    assert normalize_entity("John Smith") == "john smith"


def _entity(id_, label, text, emb):
    return MemoryNode(id=id_, type=NodeType.SEMANTIC.value, content=f"{label}: {text}",
                      embedding=emb.tolist(), entity_type=label)


def test_resolve_by_alias_and_containment(config, mock_embedding):
    g = MemoryGraph()
    embs = mock_embedding.encode(["ORG: Acme", "PERSON: John Smith", "PERSON: John Doe"])
    g.add_node(_entity("acme", "ORG", "Acme", embs[0]))
    g.add_node(_entity("smith", "PERSON", "John Smith", embs[1]))
    g.add_node(_entity("doe", "PERSON", "John Doe", embs[2]))
    resolver = EntityResolver(g, mock_embedding, config)

    assert resolver.resolve("acme.com", "ORG").id == "acme"
    assert resolver.resolve("Acme Holdings Ltd", "ORG").id == "acme"
    assert resolver.resolve("ACME", None).id == "acme"
    # Containment must be unambiguous: "John" could be either person.
    assert resolver.resolve("John", "PERSON") is None
    assert resolver.resolve("Smith", "PERSON") is None  # first token differs
    # Type mismatch blocks a match
    assert resolver.resolve("Acme", "PERSON") is None
    assert resolver.resolve("Globex", "ORG") is None


def test_register_alias_makes_future_lookups_exact(config, mock_embedding):
    g = MemoryGraph()
    g.add_node(_entity("acme", "ORG", "Acme", mock_embedding.encode(["ORG: Acme"])[0]))
    resolver = EntityResolver(g, mock_embedding, config)
    node = g.get_node("acme")
    assert resolver.register_alias(node, "Acme Holdings") is True
    assert resolver.register_alias(node, "Acme Holdings") is False
    assert "Acme Holdings" in node.aliases
    assert resolver.resolve("Acme Holdings", "ORG").id == "acme"


def test_mentions_finds_entities_in_statements(config, mock_embedding):
    g = MemoryGraph()
    g.add_node(_entity("acme", "ORG", "Acme", mock_embedding.encode(["ORG: Acme"])[0]))
    resolver = EntityResolver(g, mock_embedding, config)
    assert [n.id for n in resolver.mentions("The Acme contract is signed")] == ["acme"]
    assert resolver.mentions("The contract is signed") == []


def test_pipeline_merges_entity_mentions_and_links_facts(tmp_path, monkeypatch):
    config = AWHMConfig(data_dir=str(tmp_path / "awhm"), cold_start_session_count=0)
    with AWHMSession.start_session(config, session_id="s1", use_mock_embeddings=True) as s:
        from awhm.consolidation.pipeline import Stage1Pipeline

        pipeline = Stage1Pipeline(config, s.graph, s.embedding)
        fake = {
            0: [ExtractedEntity(text="Acme", label="ORG", start=0, end=4, message_index=0)],
            1: [ExtractedEntity(text="Acme Holdings", label="ORG", start=4, end=17, message_index=1)],
        }

        def extract(messages, message_offset=0):
            out = []
            for i, _ in enumerate(messages):
                out.extend(fake.get(message_offset + i, []))
            return out

        monkeypatch.setattr(pipeline.ner, "extract_from_messages", extract)

        s.log_message(Role.USER, "My biggest client is Acme")
        s.log_message(Role.USER, "The Acme Holdings contract is signed")
        pipeline.consolidate_session("s1")

        entities = [n for n in s.graph.all_nodes() if n.entity_type == "ORG"]
        assert len(entities) == 1
        acme = entities[0]
        assert "Acme Holdings" in acme.aliases
        assert {r["message_index"] for r in acme.source_refs} == {0, 1}

        linked = {e.source for e in s.graph.get_edges_for_node(acme.id)}
        facts = {n.id for n in s.graph.all_nodes() if "acme" in n.content.lower() and n.id != acme.id}
        assert facts and facts <= linked

        # Walking from the entity reaches its facts (clear the buffer so the
        # graph path, not the session buffer, answers)
        s.buffer.clear()
        results = s.query("Acme")
        assert any(r.node_id == acme.id for r in results)
        assert any(r.node_id in facts for r in results)
