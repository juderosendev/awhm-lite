"""Enums for AWHM Lite."""

from enum import Enum


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


class NodeType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class NodeStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"


class EdgeType(str, Enum):
    TEMPORAL = "temporal"
    ABSTRACTION = "abstraction"
    ASSOCIATION = "association"


class BufferEntryType(str, Enum):
    CORRECTION = "correction"
    PREFERENCE = "preference"
    FACT = "fact"
    OUTCOME = "outcome"
