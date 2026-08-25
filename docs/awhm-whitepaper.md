# Activation-Weighted Hierarchical Memory: An External Memory Architecture for Large Language Models

**Author:** Jude Rosen

**Date:** February 2026

---

## Abstract

Large language models exhibit a fundamental limitation: complete amnesia across conversational sessions. Existing approaches to external memory — flat-file storage, retrieval-augmented generation, and agent-managed memory systems — each suffer from critical shortcomings ranging from signal-to-noise degradation to lossy compression to unreliable self-management. We present *Activation-Weighted Hierarchical Memory* (AWHM), an external memory architecture that requires no model retraining and integrates three memory types — episodic, semantic, and procedural — within a unified graph-tree hybrid structure deployable across three modular profiles of increasing capability. Our system introduces a multi-dimensional strength scoring model governed by power-law decay, access-driven reinforcement, and conditional recency dampening with a floor guarantee; a spreading activation retrieval mechanism with symmetric sender-receiver normalisation, formally defined lateral inhibition with composite structural proximity, and similarity-gated strength ranking that eliminates query-independent bias; and a two-stage offline consolidation pipeline — symbolic pre-processing followed by LLM refinement — that amortises all LLM inference costs away from the interaction path while reducing consolidation token consumption by a domain-dependent factor (approximately 35% for nuance-heavy conversations to 80% for task-oriented technical interactions). Retrieval operates entirely algorithmically at inference time, fusing semantic similarity, graph-propagated activation, and query-conditioned strength scores into a single ranking function. Structured memories are modelled as stochastic projections over append-only raw logs — recoverable with probability approaching 1 over successive extraction passes rather than deterministically rebuildable — ensuring that consolidation quality improves monotonically in expectation. A session buffer provides intra-session memory continuity with explicitly characterised extraction ceilings, while a cold-start bootstrapping protocol — combining seed import with hybrid raw-log fallback — ensures the system delivers value from the first session. We describe a proactive surfacing mechanism with formal context-budget bounds and anchoring-bias mitigation, an embedding lifecycle management protocol for long-horizon deployment, temporal context tagging for semantic drift resistance, dual-register encoding to bridge the gap between dense storage and colloquial queries, and a three-regime phase model characterising the density conditions under which associative retrieval provides maximal advantage. We propose an eight-metric evaluation framework emphasising longitudinal performance over hundreds of sessions — the critical regime that no existing benchmark adequately captures.

---

## 1. Introduction

Every interaction with a large language model begins from a blank slate. Despite remarkable advances in reasoning, generation, and tool use, LLMs retain no information between sessions unless external mechanisms intervene. This *conversational amnesia* represents a fundamental barrier to deploying LLMs as persistent collaborators, personal assistants, or long-term knowledge workers.

The problem is not merely one of convenience. Without memory, LLMs cannot accumulate expertise about a user's preferences, cannot learn from past mistakes, cannot build on prior conversations, and cannot maintain coherent long-term projects. Users must repeatedly re-establish context, re-explain constraints, and re-correct errors — a cognitive tax that scales linearly with relationship duration.

Several approaches have emerged to address this limitation, yet each exhibits fundamental architectural flaws:

- **Flat-file memory** (e.g., appending to a text file) drowns signal in noise as the file grows, offering no principled mechanism for prioritisation or decay.
- **Retrieval-augmented generation (RAG)** conflates semantic similarity with relevance, retrieving passages that *sound related* rather than passages that *are useful*, and lacks mechanisms for multi-hop reasoning across memory fragments.
- **Mem0** achieves compactness through lossy compression, discarding contextual detail that may later prove essential.
- **MemGPT** delegates memory management to the LLM itself, introducing unreliable self-management where the model must simultaneously reason about the task and curate its own memory.
- **Zep** incurs excessive token costs by surfacing large memory payloads during interaction — with injection scaling linearly with memory volume and no hard cap — eroding the context budget available for actual reasoning.

No existing system integrates episodic, semantic, and procedural memory types with proper decay and reinforcement dynamics, nor does any cleanly separate the cost of memory formation from the cost of memory retrieval.

### Contributions

We make the following contributions:

1. **A three-tier storage architecture** that separates append-only raw logs (permanent ground truth, captured in real time via middleware-level logging), structured memories (stochastic projections over those logs, recoverable with probability approaching 1 over successive passes), and a session buffer (intra-session working memory with explicitly characterised extraction ceilings), enabling closed-loop consolidation where extraction quality improves over time and never degrades.

2. **A graph-tree hybrid memory representation** that unifies episodic, semantic, and procedural memory types under a hierarchical index navigable via targeted tool calls, with progressive abstraction that bounds the active graph regardless of total memory volume, temporal context tagging for semantic drift resistance, and dual-register encoding to bridge the gap between dense storage and colloquial queries.

3. **A multi-dimensional strength scoring model** combining salience, specificity, utility, access frequency, and recency, governed by power-law decay consistent with human forgetting curves, with conditional recency dampening (replacing the prior gating formulation) and adaptive heuristics for the most sensitive parameters.

4. **A spreading activation retrieval engine** that operates with zero LLM calls, employing symmetric sender-receiver normalisation, formally defined lateral inhibition via composite structural proximity, and similarity-gated strength ranking to produce a three-signal relevance function superior to cosine-similarity baselines.

5. **A two-stage hybrid consolidation pipeline** — symbolic pre-processing for a domain-dependent fraction of extractions (approximately 35–80%), followed by LLM refinement for ambiguous cases — that amortises all LLM costs to offline batch processing, with a taxonomy-based parametric cost model, specialist sub-agent fan-out, graph-state tagging for ordering-aware re-extraction, and closed-loop re-extraction from raw logs.

6. **A proactive surfacing mechanism** that injects memories at natural reasoning breakpoints with formal context-budget bounds ($\leq 8\%$ of context window), anchoring-bias mitigation through relevance-threshold gating and epistemic framing, and bounded latency guarantees.

7. **A cold-start bootstrapping protocol** combining seed import from existing documents with a hybrid raw-log fallback — framed as the explore phase of an adaptive system — that ensures the system delivers value from the first session, before the memory graph has densified.

8. **A session buffer** that provides intra-session memory continuity — capturing corrections, preferences, and facts from the current interaction via pattern matching with an explicitly characterised quality ceiling (approximately 60–70% of explicit signals, <20% of implicit signals) — without waiting for offline consolidation.

9. **An embedding lifecycle management protocol** with versioned embedding storage, lazy re-embedding on model migration, and compatibility testing, ensuring the system remains operational across embedding model upgrades over multi-year deployments.

10. **Versioned graph snapshots and graph health monitoring** enabling rollback from corrupted consolidation passes and automated detection of graph drift, with targeted cleanup integrated into the consolidation cycle.

11. **An eight-metric evaluation framework** designed for longitudinal assessment over hundreds and thousands of sessions, anchored to established benchmarks including LoCoMo, LongMemEval, and MemoryBench.

12. **A modular deployment profile system** (Lite, Standard, Full) that demonstrates the architecture's decomposability and provides an incremental adoption path from zero-LLM-cost operation to the complete associative architecture.

The remainder of this paper is organised as follows. Section 2 surveys related work. Section 3 presents the system architecture. Section 4 describes the memory representation. Section 5 details the strength scoring model. Section 6 specifies the retrieval engine. Section 7 covers the consolidation pipeline. Section 8 describes proactive surfacing. Section 9 addresses consistency and maintenance. Section 10 analyses the cost model. Section 11 proposes the evaluation framework. Section 12 discusses limitations and future directions. Section 13 concludes.

---

## 2. Related Work

We categorise existing approaches to LLM memory along two axes: *storage paradigm* (how memories are represented) and *management paradigm* (who decides what to remember and when to retrieve it).

### 2.1 Flat-File Approaches

The simplest external memory strategy appends conversational summaries or key-value pairs to a persistent text file, loaded into the context window at session start. While straightforward to implement, this approach exhibits O(n) degradation: as the file grows, signal is progressively buried in noise, the context budget is consumed by low-relevance material, and no mechanism exists for decay, prioritisation, or associative retrieval. The approach implicitly treats all memories as equally important and equally current — assumptions that hold only in the shortest deployments.

### 2.2 Retrieval-Augmented Generation

RAG systems (Lewis et al., 2020) store memories as embedded text chunks in a vector database and retrieve the top-k most similar chunks at query time. While RAG introduces relevance-based filtering, it conflates *semantic similarity* with *contextual relevance*. A memory that is lexically similar to the current query is not necessarily the memory the model needs — and the memory the model needs may be connected only through a chain of associations invisible to cosine similarity. RAG also lacks temporal reasoning (it cannot distinguish current from superseded information), strength-based decay (all memories are equally retrievable regardless of their importance or age), and multi-hop retrieval (it cannot follow associative chains across memory fragments).

### 2.3 Lossy Compression Systems

Mem0 and similar systems address the scaling problem by compressing memories into compact representations. While this controls context consumption, the compression is inherently lossy: contextual detail, hedging, and nuance are discarded during summarisation. This creates an irrecoverable information deficit — when the discarded detail later proves relevant, no mechanism exists to recover it. The system's memory quality has a ceiling but no floor.

### 2.4 Self-Managed Memory

MemGPT (Packer et al., 2023) treats memory management as a subtask delegated to the LLM itself, allowing the model to page information in and out of its context window. This approach introduces a fundamental reliability problem: the model must simultaneously reason about the primary task and curate its own memory, using the same context budget for both. Memory management competes with task performance for cognitive resources, and the model's memory decisions are subject to the same failure modes (hallucination, omission, inconsistency) as its primary outputs. Empirically, self-managed memory systems exhibit unpredictable memory curation quality.

### 2.5 High-Token-Cost Systems

Zep and comparable platforms surface rich memory context during interactions but at substantial token cost. By injecting large memory payloads into the context window — with injection volume scaling linearly with memory volume and no formal budget cap — these systems erode the budget available for reasoning about the actual task. The cost scales with memory volume, creating a tension between memory richness and reasoning capacity that worsens over time.

### 2.6 Graph-Based and Neuroscience-Inspired Systems

Recent work has explored graph-structured memory and neuroscience-inspired retrieval mechanisms for LLMs, representing the closest prior art to our approach.

**SYNAPSE** (Jiang et al., 2026) models LLM memory as a dynamic graph with episodic-semantic integration, employing spreading activation inspired by cognitive science and a triple hybrid retrieval strategy. Evaluated on the LoCoMo benchmark (Maharana et al., 2024), SYNAPSE demonstrates the viability of graph-based associative retrieval for conversational memory. However, it lacks multi-dimensional strength scoring with decay dynamics, progressive abstraction for graph boundedness, and the strict offline/online cost separation that is central to our architecture.

**HippoRAG** (Gutiérrez et al., 2024) draws on hippocampal indexing theory to construct a knowledge graph from passages, using Personalised PageRank for retrieval. Accepted at NeurIPS 2024 with a follow-up at ICML 2025, HippoRAG shares our neuroscience-inspired motivation but employs a fundamentally different retrieval mechanism (PageRank vs. spreading activation with symmetric normalisation and lateral inhibition) and does not address memory decay, reinforcement, or offline consolidation.

**HGMem** (Zhou et al., 2025) introduces hypergraph-based memory for multi-step RAG, where hyperedges represent higher-order relations among memory units. While hypergraph structures offer expressive advantages for complex relational patterns, we achieve comparable expressiveness through reified event nodes within a standard directed graph — avoiding the storage overhead and implementation complexity of native hypergraph operations while maintaining compatibility with established graph database tooling.

### 2.7 Positioning

Our system addresses the limitations identified above through architectural separation of concerns. By moving all LLM-dependent processing to offline consolidation, we eliminate the token cost of memory formation from the interaction path. By maintaining append-only raw logs as permanent ground truth, we bound information loss to the active graph layer — raw source material is always available for re-extraction, making lossiness controlled and systematically reducible rather than irrecoverable. By employing graph-based spreading activation with symmetric normalisation rather than flat vector similarity, we support multi-hop and associative retrieval. By implementing principled strength scoring with decay dynamics, we provide automatic prioritisation without self-management. Table 1 summarises the comparison.

| System | Storage | Management | Multi-hop | Decay | Cost Model | Ground Truth |
|--------|---------|------------|-----------|-------|------------|--------------|
| Flat file | Append-only text | Manual | No | No | O(n) context | Yes |
| RAG | Vector DB | Algorithmic | No | No | Per-query embedding | No |
| Mem0 | Compressed KV | Automated | No | No | Per-extraction LLM | No |
| MemGPT | Paged context | Self-managed | Limited | No | Per-turn LLM | No |
| Zep | Rich context (uncapped) | Automated | Limited | No | High per-turn | Partial |
| SYNAPSE | Dynamic graph | Algorithmic | Yes | No | Per-query | No |
| HippoRAG | Knowledge graph | Algorithmic (PPR) | Yes | No | Per-query + build | No |
| **AWHM (ours)** | **Graph + raw logs** | **Offline consolidation** | **Yes** | **Yes** | **Amortised offline** | **Yes** |

---

## 3. System Overview

AWHM is organised around a strict temporal separation between *interaction-time* operations and *offline (sleep-time)* operations. This separation is the primary architectural decision: it ensures that the user-facing agent never pays the latency or token cost of memory formation, while the consolidation agent operates without latency constraints and can leverage stronger models.

### 3.1 Architecture

```
 INTERACTION TIME                          OFFLINE (SLEEP-TIME)
 ─────────────────────────────────────     ─────────────────────────────────
                                           ┌───────────────────────────────┐
 ┌─────────────────────┐                   │    CONSOLIDATION AGENT       │
 │    PRIMARY AGENT     │  real-time log   │    (two-stage hybrid)        │
 │   (user-facing LLM)  │────────────────► │                               │
 └──────┬──────┬────────┘  (middleware,    │  Stage 1: symbolic extraction │
        │      │            no LLM)        │  (NER, rules, entity linking) │
        │      │                           │  Stage 2: LLM refinement     │
        │      │ navigates index           │  (ambiguous cases only)      │
        │      │ via tool calls            │                               │
        │      ▼                           │  • dense + natural-query     │
        │  ┌────────────────┐              │    dual-register encoding    │
        │  │ HIERARCHICAL   │              │  • assign strength scores    │
        │  │ INDEX          │              │  • build graph edges         │
        │  │                │              │  • embed versioning + tags   │
        │  │ top-level map  │              │  • reorganise tree index     │
        │  │ fits in small  │              │  • health check + snapshot   │
        │  │ context slice  │              │  (stronger model, no latency │
        │  └───────┬────────┘              │   constraint, batchable)     │
        │          │                       └──────────────┬────────────────┘
        │          ▼                                      │
        │  ┌──────────────────────────────────────────┐   │ writes
  memory│  │           MEMORY GRAPH                   │◄──┘
  clouds│  │                                          │
  inject│  │   ◉ episodic ──── ◇ semantic             │
  into  │  │   │                │                     │
  reasoning │   temporal    abstraction               │
  (≤8%  │  │   │                │                     │
  context│  │   ◉ ──association──▣ procedural          │
  budget)│  │                                          │
        │  │   STRENGTH per node:                     │
        │  │   S = f(salience, specificity, utility,    │
        │  │         frequency, recency[dampened])     │
        │  │   decay: power-law  │  reinforce: on use │
        │  └──────────┬───────────────────────────────┘
        │             │
        ▼             ▼
 ┌────────────────────────────────┐    ┌──────────────────────────────┐
 │      RETRIEVAL ENGINE          │    │        RAW LOGS              │
 │                                │    │  (append-only, permanent,    │
 │  0. check: session buffer      │    │   complete ground truth,     │
 │  1. anchor: BM25 + embedding   │    │   real-time middleware       │
 │     (against natural-query     │    │   capture — every message    │
 │      register for embeddings)  │    │   logged as it occurs)       │
 │  2. propagate: spreading       │    └──────────────────────────────┘
 │     activation w/ symmetric    │
 │     norm + lateral inhibition  │    ┌──────────────────────────────┐
 │  3. rank: similarity +         │    │      SESSION BUFFER          │
 │     activation + sim-gated     │    │  (intra-session working      │
 │     strength                   │    │   memory: corrections,       │
 │                                │    │   preferences, facts —       │
 │  → top memories surfaced as    │    │   ~60-70% explicit capture,  │
 │    "memory clouds" with        │    │   <20% implicit signals —    │
 │    epistemic framing           │    │   checked before main graph, │
 │                                │    │   queued for consolidation   │
 │  no LLM calls (all algorithmic)│    │   on session end)            │
 └────────────────────────────────┘    └──────────────────────────────┘
```

*Figure 1. System architecture showing the separation between interaction-time retrieval (left) and offline sleep-time consolidation (right). The primary agent logs all I/O to append-only raw logs via middleware-level capture (no LLM involvement). The session buffer provides intra-session working memory with explicitly characterised extraction ceilings, checked before the main graph during retrieval. The consolidation agent processes raw logs through a two-stage hybrid pipeline (symbolic pre-processing, then LLM refinement for ambiguous cases) into structured memories with dual-register encoding in the memory graph. The retrieval engine operates with zero LLM calls, using symmetric normalisation and similarity-gated strength ranking. Memory injection is hard-capped at $\rho$ of the context window (default 8%).*

### 3.2 Three-Tier Storage

The storage layer comprises three tiers with distinct durability, mutability, and latency characteristics:

**Raw logs** constitute the first tier: an append-only, permanent record of all interactions. Every message — user input, agent response, tool call, tool result — is appended to the log *as it occurs* via middleware-level capture, independent of the LLM. This real-time logging is critical: because each interaction is captured the moment it happens, context-window compaction (which discards older messages to stay within token limits) has no effect on the raw log — the log is always ahead of compaction by definition. Raw logs serve as the complete ground truth of the system and function analogously to a write-ahead log in database systems. They are never modified (except for targeted hard deletion under privacy compliance; see Section 9) and can always be reprocessed. Each log entry carries a timestamp, session identifier, message role, and structural metadata (e.g., whether a tool call succeeded or failed).

**Structured memories** constitute the second tier: extracted, encoded, and organised representations that live as nodes in the memory graph (Section 4). These are *stochastic projections* over the raw logs — derived representations analogous to materialised views in databases, though non-deterministic in extraction. Unlike deterministic materialised views, each extraction pass is a sampling from the space of valid interpretations. The system's ground-truth guarantee is therefore not "we can recover the same graph" but rather: **(a)** any factual content expressible from the raw logs can be recovered with probability approaching 1 over repeated extraction passes; **(b)** the raw logs bound the information-theoretic ceiling — no extraction can hallucinate content absent from the logs; and **(c)** successive re-extractions with improving models monotonically increase expected recall against the log content. This relationship is critical: it means consolidation quality has a floor but no ceiling. As models and extraction prompts improve, old logs can be progressively re-processed, and any information the system initially missed can be recovered later.

**The session buffer** constitutes the third tier: a session-scoped working memory that provides intra-session continuity. Without this tier, the system would have no access to information from the current session until the next offline consolidation pass — a user correction at minute 5 could be repeated as an error at minute 20. The session buffer stores tagged items (corrections, preferences, facts, task outcomes) extracted from the current interaction via lightweight pattern matching (no LLM calls). During retrieval, the session buffer is checked *before* the main memory graph, ensuring that intra-session information takes precedence. On session end, buffer contents are queued for full consolidation into the structured memory graph. The buffer is periodically persisted to a write-ahead log for crash recovery.

The session buffer's pattern-based extraction operates at a well-defined quality ceiling. Explicit signals — corrections using the pattern "actually, X is Y," stated preferences using "I prefer X," and factual declarations — are captured reliably by rule-based matching. However, implicit corrections (the user silently switches approaches), hedged preferences ("I guess X might work better"), contextual nuances, and tone-level signals fall below the pattern-matching floor. We estimate the session buffer captures approximately 60–70% of explicitly stated information and less than 20% of implicit signals.

This is an acceptable trade-off: the session buffer's purpose is *intra-session continuity*, not comprehensive extraction. Implicit signals are captured in the raw logs and recovered during full offline consolidation, which has access to LLM-level understanding (Stage 2). The session buffer provides a fast, zero-cost safety net for the most unambiguous signals; completeness is the consolidation agent's responsibility.

### 3.3 Deployment Profiles

AWHM is designed as a modular architecture where components can be enabled incrementally. We define three deployment profiles that represent increasing levels of capability and complexity:

| Component | **Lite** | **Standard** | **Full** |
|---|---|---|---|
| Raw logs | ✓ | ✓ | ✓ |
| Session buffer | ✓ | ✓ | ✓ |
| Two-stage consolidation | Stage 1 only | Both stages | Both stages |
| Graph structure | Flat (no hierarchy) | Hierarchical index | Full hierarchy + bridge nodes |
| Strength scoring | Recency + frequency only | All 5 dimensions | All 5 + adaptive tuning |
| Retrieval | BM25 + embedding | + spreading activation | + proactive surfacing |
| Maintenance | Manual snapshots | Auto snapshots + rollback | + health monitoring + defrag |
| Progressive abstraction | None | Basic absorption | Full absorption + cold storage |

**Lite** is deployable immediately with zero LLM consolidation costs and provides value comparable to a well-designed RAG system with temporal awareness. **Standard** adds the associative retrieval and multi-dimensional scoring that differentiate AWHM. **Full** enables the complete architecture described in this paper.

This decomposition also defines the ablation path for evaluation (Section 11.6): each profile transition enables measurement of the marginal contribution of the added components. It demonstrates that the system does not require all components active simultaneously to be useful, and that each component layer has independently testable value.

---

## 4. Memory Representation

### 4.1 Graph Structure

Memories are represented as nodes in a directed graph $G = (V, E)$, where each node $v \in V$ carries a structured payload and each edge $e \in E$ encodes a typed relationship. We define three node types:

- **Episodic nodes** ($v_{\text{ep}}$): Representations of specific events, interactions, or experiences. These are the most granular memory units, capturing what happened, when, with whom, and in what context.

- **Semantic nodes** ($v_{\text{sem}}$): Abstracted concepts, facts, and generalised knowledge distilled from one or more episodic nodes. These capture *what is known* independent of any single episode.

- **Procedural nodes** ($v_{\text{proc}}$): Learned patterns, skills, workflows, and behavioural heuristics. These capture *how to do things* — recurring strategies, preferred approaches, and operational knowledge.

Each node carries a structured payload:

$$v = (\text{payload}_{\text{dense}}, \text{payload}_{\text{natural}}, \mathbf{e}_v, m_v, S(v), \tau_v)$$

where $\text{payload}_{\text{dense}}$ is the dense canonical encoding (precise, terse, technical — optimised for token efficiency), $\text{payload}_{\text{natural}}$ is the natural query form (a colloquial paraphrase used exclusively for embedding-based similarity matching), $\mathbf{e}_v$ is the dense embedding vector, $m_v$ is the embedding model version tag, $S(v)$ is the composite strength score, and $\tau_v = [t_{\text{created}}, t_{\text{last\_relevant}}]$ defines the temporal validity window of the memory.

The dual-register encoding addresses a fundamental retrieval mismatch: memories encoded in maximally dense technical language occupy a different region of embedding space than the colloquial queries users naturally produce. By maintaining both representations, BM25 matching runs against the dense canonical form (keyword matching is less sensitive to register mismatch), while embedding similarity $\text{sim}(v, q)$ is computed against the natural query form, reducing the style gap between stored memories and user queries. The memory surfaced into the context window uses the dense canonical form, preserving token efficiency. The cost is one additional short text generation per memory during Stage 2 consolidation (estimated: 20–40 tokens per memory) and doubled embedding storage, which is negligible (a few KB per node).

Edges are typed into three categories:

- **Temporal edges** connect episodic nodes in chronological sequence, encoding the narrative structure of interactions over time. These edges carry a timestamp and are subject to exponential decay during activation propagation (Section 6). Additionally, temporal edges connecting nodes from distant temporal validity windows receive reduced weight, reducing the chance of semantically drifted terms activating stale associations.

- **Abstraction edges** connect episodic nodes to the semantic nodes that generalise over them, encoding the is-instance-of relationship between specific events and the concepts they instantiate.

- **Association edges** connect any pair of nodes that are meaningfully related, encoding lateral connections that enable multi-hop retrieval. Association edges between episodic and procedural nodes are particularly important, as they link specific experiences to the skills or patterns they exemplify.

- **Semantic version edges** connect nodes representing the same entity or concept whose meaning has evolved over time. When the consolidation agent detects that a term's usage has shifted (identifiable via embedding drift of new extractions containing the term compared to existing nodes), it creates a semantic version edge linking the old and new meanings. This is similar to the supersession protocol (Section 9.1) but for meaning evolution rather than factual correction.

### 4.2 Temporal Context Tagging

Terms and entities can undergo semantic drift over long time horizons — "Project Alpha" might refer to one initiative in session 1 and a different initiative in session 200. The embedding space alone does not capture this evolution. We address this through temporal context tags on each memory node.

**Temporal disambiguation during retrieval.** When a query matches multiple memories sharing the same key entity but from different temporal windows, the system applies temporal disambiguation:

1. **Active-window preference.** Memories whose $\tau_v$ window includes the current time (or recent sessions) are preferred over memories from expired windows.
2. **Explicit temporal scoping.** During entity linking in Stage 1, candidate matches are filtered by temporal proximity. A new mention of "Project Alpha" is more likely to link to the *recent* "Project Alpha" node than one from 6 months ago, unless the user explicitly references the historical context.
3. **Semantic versioning.** When embedding drift is detected for a term, a semantic version edge (Section 4.1) links the old and new meanings, enabling explicit navigation of meaning evolution via temporal queries.

### 4.3 Hierarchical Index

The memory graph is organised under a hierarchical tree index $\mathcal{T}$ whose top-level branches consist of compressed, dense summaries that fit within a small fraction of the context window. This design enables the primary agent to navigate the memory space efficiently: it reads the top-level index to identify relevant regions, then issues targeted tool calls to drill into specific branches, rather than receiving a massive memory dump.

The tree routes between storage tiers (active graph and cold storage) and scopes spreading activation: propagation is bounded by the tree branch the query enters, not the full graph. This ensures retrieval cost scales with the relevant subgraph, not total memory volume.

### 4.4 Progressive Abstraction and Cold Storage

The graph maintains bounded size through two mechanisms inspired by memory consolidation in cognitive neuroscience:

**Progressive abstraction.** Episodic nodes whose strength scores (Section 5) decay below a configurable threshold $\theta_{\text{abs}}$ are absorbed into their parent semantic nodes. Their salient information is merged into the semantic summary and their edges are transferred to the absorbing node. This is inspired by hippocampal-to-neocortical consolidation in the brain, where specific episodic memories are gradually transformed into generalised semantic knowledge — adopted as a design heuristic rather than a biological derivation, and subject to empirical validation of the specific threshold and merging parameters.

**Cold storage.** Nodes whose strength falls below a dormancy threshold $\theta_{\text{cold}}$ (where $\theta_{\text{cold}} < \theta_{\text{abs}}$) are moved to cold storage. These nodes remain indexed by the hierarchical tree and are retrievable by specific query, but are excluded from spreading activation to keep the active subgraph tractable.

Together, these mechanisms bound the active graph to approximately 10,000–50,000 nodes regardless of total memory volume. Critically, these bounds serve a dual purpose: they ensure retrieval performance degrades gracefully rather than linearly with memory accumulation, *and* they maintain the graph in the density regime where associative retrieval provides maximal advantage (see Section 9.4).

---

## 5. Strength Scoring

### 5.1 Multi-Dimensional Model

Each memory node $v$ carries a composite strength score $S(v)$ computed from five dimensions:

$$S(v) = \sum_{d \in \mathcal{D}} w_d \cdot s_d(v)$$

where $\mathcal{D} = \{\text{sal}, \text{spec}, \text{util}, \text{freq}, \text{rec}\}$ denotes the set of scoring dimensions, $s_d(v) \in [0, 1]$ is the normalised score for dimension $d$, and $w_d$ is the learnable weight for that dimension. The five dimensions are:

1. **Salience** ($s_{\text{sal}}$): The consequence magnitude at formation — corrections, errors, explicit user emphasis, expectation violations. This is the only dimension that requires LLM judgment, and it is assessed during offline consolidation via a structured rubric applied by the consolidation agent.

2. **Specificity** ($s_{\text{spec}}$): The entity density and concreteness of the memory content. This is computable algorithmically without an LLM call: memories containing named projects, concrete values, and causal chains score higher than vague abstractions. Formally, specificity can be approximated as a function of named entity count, numeric literal count, and syntactic concreteness markers.

3. **Utility** ($s_{\text{util}}$): A retroactively updated score reflecting whether the memory has contributed to useful outcomes. When a memory is surfaced and the resulting interaction is successful (as measured by downstream signals), the utility score is incremented. To counteract the bootstrapping problem — where memories that are retrieved early accumulate utility and dominate future retrieval while equally valuable memories that are never retrieved cannot prove their worth — utility includes an exploration bonus inspired by multi-armed bandit theory: $s_{\text{util}}'(v) = s_{\text{util}}(v) + c\sqrt{\ln(N) / n_v}$, where $N$ is the total query count (windowed to a rolling horizon to prevent unbounded growth), $n_v$ is the number of times memory $v$ has been retrieved, and $c$ is a tunable exploration coefficient. This bonus is largest for under-retrieved memories and diminishes naturally as a memory accumulates retrieval opportunities.

4. **Access frequency** ($s_{\text{freq}}$): A count of how often the memory has been retrieved, normalised against the population distribution.

5. **Recency** ($s_{\text{rec}}$): A timestamp-derived score reflecting how recently the memory was last accessed or reinforced.

### 5.2 Weighting and Recency Dampening

The dimension weights $w_d$ are not co-equal. Utility and frequency are weighted higher by default, as they represent the most directly measurable indicators of memory value. Recency functions as a *dampened corroborative signal* — evidence that contributes meaningfully to strength but is prevented from dominating through a conditional dampening mechanism with a floor guarantee:

$$s_{\text{rec}}'(v) = s_{\text{rec}}(v) \cdot \left(\phi + (1 - \phi) \cdot \left(1 - \bar{s}_{\text{other}}(v)\right)\right)$$

where $\bar{s}_{\text{other}}(v) = \frac{1}{|\mathcal{D} \setminus \{\text{rec}\}|} \sum_{d \neq \text{rec}} s_d(v)$ is the mean of the non-recency dimensions and $\phi \in [0, 1]$ is the recency floor parameter (default: $\phi = 0.3$).

This formulation achieves three properties simultaneously:

1. **Floor guarantee.** Even when all other dimensions are maximal, recency retains at least $\phi \cdot s_{\text{rec}}$ of its original value. A high-salience memory discussed yesterday still receives meaningful recency contribution.

2. **Mean-based dampening.** Using the mean of other dimensions (rather than the max) prevents a single strong dimension from silencing recency. A memory with $s_{\text{sal}} = 0.9$ but $s_{\text{spec}} = 0.1, s_{\text{util}} = 0.1, s_{\text{freq}} = 0.1$ retains $0.3 + 0.7 \cdot (1 - 0.3) = 0.79$ of its recency value — dampened slightly but not silenced.

3. **Low-value recent memories benefit fully.** When $\bar{s}_{\text{other}} \approx 0$, the formula reduces to $s_{\text{rec}}' = s_{\text{rec}}$ — recency contributes fully, which is appropriate because recency is the *only* reason to surface this memory.

Weights are tunable via ablation against retrieval performance (Section 11).

### 5.3 Decay Dynamics

Strength evolves over time according to a power-law decay function, adopted as an initial heuristic inspired by the Ebbinghaus forgetting curve and to be validated via ablation against alternative decay functions (exponential, step-function, utility-weighted):

$$S(v, t) = S_0(v) \cdot (1 + \beta \cdot \Delta t)^{-\alpha}$$

where $S_0(v)$ is the initial strength at formation or last reinforcement, $\Delta t$ is the elapsed time since last reinforcement, $\alpha > 0$ controls the decay rate, and $\beta > 0$ is a scaling constant. Power-law decay (rather than exponential) produces a long tail: memories fade quickly at first but asymptotically approach zero without ever reaching it.

### 5.4 Reinforcement

When a memory is accessed — either through explicit retrieval or through spreading activation that reaches the node — its strength is reinforced:

$$S_0(v) \leftarrow S(v, t_{\text{access}}) + \delta_{\text{reinforce}}$$

where $\delta_{\text{reinforce}}$ is scaled by the nature of the access (direct retrieval reinforces more than passive activation). This mechanism ensures that useful memories are maintained against decay through natural usage patterns.

Critically, decayed memories are never deleted. They simply stop surfacing in standard retrieval because their strength falls below the activation threshold. They remain in the graph and can be recovered by specific query or re-extraction from raw logs.

### 5.5 Adaptive Weight Tuning

The dimension weights $w_d$, decay parameters ($\alpha$, $\beta$), recency floor ($\phi$), and key thresholds ($\theta_{\text{abs}}$, $\theta_{\text{cold}}$) are shipped with empirically derived defaults suitable for most deployments. For the most sensitive parameters, simple adaptive heuristics adjust values based on observed system behaviour without requiring manual intervention:

- **Decay rate adaptation:** If the active graph growth rate consistently outpaces retrieval utilisation (many nodes accumulate that are never retrieved), the decay rate $\alpha$ is incrementally increased to prune low-value memories more aggressively.
- **Lateral inhibition adaptation:** If retrieval consistently returns max-$k$ results with low diversity (high pairwise similarity among returned memories), the inhibition coefficient $\gamma$ is increased to enforce sharper competition.
- **Absorption threshold adaptation:** If consolidation-pass duration exceeds a time budget, the absorption threshold $\theta_{\text{abs}}$ is raised to merge episodic nodes more aggressively, reducing graph size.
- **Recency floor adaptation:** If retrieval results are over-populated with very recent but low-relevance memories, $\phi$ is decreased. If users frequently ask about recent interactions and the system underperforms on these queries, $\phi$ is increased.

These heuristics operate on simple feedback rules (compare a metric against a threshold, adjust a parameter by a step size) and introduce no additional LLM calls. Full Bayesian optimisation of the parameter space is noted as a future direction for deployments requiring zero-touch operation (Section 12).

### 5.6 Cost Efficiency

Only initial salience scoring requires LLM judgment, assessed during offline consolidation. All other scoring — access counts, recency timestamps, utility updates, decay curves, adaptive tuning, and aggregate strength computation — is performed mathematically by the system with zero LLM calls.

---

## 6. Retrieval

The retrieval engine operates with zero LLM calls at inference time. It implements a three-stage spreading activation pipeline that replaces crude top-k cosine similarity with a multi-signal ranking function capable of handling ambiguous queries, multi-hop reasoning, and temporal context.

### 6.1 Anchor Identification

When a query $q$ arrives, anchor nodes are identified via a dual-trigger mechanism:

- **Lexical matching (BM25):** Identifies nodes with high term-overlap with the query, computed against the dense canonical form of each node. Captures exact-match and keyword-level relevance.
- **Dense embedding similarity:** Identifies nodes with high semantic similarity to the query in embedding space, computed against the *natural query form* of each node (Section 4.1). This reduces the register mismatch between colloquial user queries and technical memory encodings.

The union of nodes exceeding threshold scores on either trigger forms the initial anchor set $A_0 \subseteq V$.

### 6.2 Activation Propagation

From the anchor set, activation propagates outward through graph edges according to a symmetrically normalised formula:

$$a^{(t+1)}(v) = \frac{1}{\sqrt{|\mathcal{N}_{\text{in}}(v)|}} \sum_{u \in \mathcal{N}(v)} \frac{w_{u \to v}}{\sqrt{|\mathcal{N}_{\text{out}}(u)|}} \cdot a^{(t)}(u)$$

where $a^{(t)}(v)$ is the activation of node $v$ at iteration $t$, $\mathcal{N}(v)$ denotes the neighbours of $v$, $w_{u \to v}$ is the edge weight from $u$ to $v$, $|\mathcal{N}_{\text{out}}(u)|$ is the out-degree of the sender, and $|\mathcal{N}_{\text{in}}(v)|$ is the in-degree of the receiver.

The symmetric normalisation addresses both sides of the topological bias problem. The sender-side term ($\frac{1}{\sqrt{|\mathcal{N}_{\text{out}}(u)|}}$) implements **fan-effect normalisation**: high-degree hub nodes distribute their activation across more neighbours, preventing them from monopolising the activation signal. The receiver-side term ($\frac{1}{\sqrt{|\mathcal{N}_{\text{in}}(v)|}}$) prevents high-in-degree sink nodes from accumulating disproportionate activation simply by virtue of having many incoming edges. This is analogous to the symmetric normalisation used in Graph Convolutional Networks (Kipf & Welling, 2017), ensuring that activation magnitude is governed by relevance rather than topology.

**Temporal decay on edges.** For temporal edges, the weight $w_{u \to v}$ is modulated by exponential decay over the age of the edge:

$$w_{u \to v}^{\text{temporal}} = w_0 \cdot e^{-\lambda \cdot \text{age}(u, v)}$$

where $\lambda$ controls the rate at which old temporal connections attenuate.

**Lateral inhibition.** After each propagation iteration, a lateral inhibition step enforces result diversity: the top-$k$ most activated nodes suppress the activation of competing nodes within the same structural neighbourhood. With symmetric normalisation handling topological bias structurally, lateral inhibition's role is cleanly separated as an enforcer of result diversity rather than a corrector of topological artifacts. Formally:

$$a^{(t)}(v) \leftarrow a^{(t)}(v) - \gamma_q \cdot \sum_{u \in \text{top-}k} a^{(t)}(u) \cdot \text{overlap}(v, u)$$

where $\gamma_q$ is the adaptive inhibition coefficient and $\text{overlap}(v, u)$ is a composite structural proximity function defined as:

$$\text{overlap}(v, u) = \mu \cdot J(v, u) + (1 - \mu) \cdot B(v, u)$$

The first component is the **Jaccard neighbourhood overlap**:

$$J(v, u) = \frac{|\mathcal{N}(v) \cap \mathcal{N}(u)|}{|\mathcal{N}(v) \cup \mathcal{N}(u)|}$$

which measures local structural similarity — nodes that share many neighbours are in the same "neighbourhood" and should compete. This is $O(d)$ where $d$ is the average degree, cheap enough for real-time retrieval.

The second component is **branch co-membership**:

$$B(v, u) = \begin{cases} 1 & \text{if } v \text{ and } u \text{ are in the same tree branch} \\ \delta_{\text{bridge}} & \text{if either is a bridge node in the other's branch} \\ 0 & \text{otherwise} \end{cases}$$

where $\delta_{\text{bridge}} \in (0, 1)$ is a discount factor for bridge-node adjacency (default: 0.3).

This composite definition ensures that inhibition respects both local and hierarchical structure: nodes in the same neighbourhood and tree branch compete strongly, while nodes in different branches compete weakly or not at all — aligning with the multi-intent query handling where reduced inhibition preserves cross-branch diversity. The recommended default is $\mu = 0.6$, favouring local structural overlap over branch membership since branch assignment can be imperfect.

Embedding similarity is intentionally excluded from the overlap function because it is already captured by the similarity term in the ranking function. Using it again in inhibition would double-penalise semantically similar nodes, creating excessive suppression of genuinely related memories. Structural proximity is the right signal for inhibition because it identifies *redundant retrieval pathways*, not redundant content.

The inhibition coefficient adapts to the inferred intent structure of the query: the anchor set $A_0$ is clustered by embedding distance, and $\gamma_q$ is scaled inversely with the number of distinct clusters detected. Single-intent queries (one tight anchor cluster) receive full inhibition for sharp focus; multi-intent queries (multiple distinct anchor clusters) receive reduced inhibition to preserve diverse results across topics. With receiver-side normalisation in place, the default $\gamma_q$ can be set lower than would otherwise be necessary, since inhibition no longer needs to compensate for sink accumulation.

**Convergence.** The propagation equation with symmetric normalisation can be expressed in matrix form as:

$$\mathbf{a}^{(t+1)} = \mathbf{D}_{\text{in}}^{-1/2} \mathbf{W} \mathbf{D}_{\text{out}}^{-1/2} \mathbf{a}^{(t)}$$

where $\mathbf{W}$ is the weighted adjacency matrix and $\mathbf{D}_{\text{in}}, \mathbf{D}_{\text{out}}$ are the in-degree and out-degree diagonal matrices. The spectral radius of the normalised propagation matrix $\hat{\mathbf{W}} = \mathbf{D}_{\text{in}}^{-1/2} \mathbf{W} \mathbf{D}_{\text{out}}^{-1/2}$ governs convergence rate. For graphs with bounded degree and edge weights below unity, $\rho(\hat{\mathbf{W}}) < 1$, guaranteeing geometric convergence. Under typical AWHM graph parameters (mean degree 5–15, edge weights in $[0, 1]$, 10K–50K active nodes), activation concentrates on the associatively relevant subgraph within 3–5 iterations. However, convergence speed is topology-dependent: sparse chain-like subgraphs may require 5–7 iterations, while densely connected clusters converge in 2–3. The system uses a convergence criterion ($\|\mathbf{a}^{(t+1)} - \mathbf{a}^{(t)}\|_\infty < \epsilon$) rather than a fixed iteration count, with a maximum iteration cap of $T_{\max} = 7$ to bound worst-case retrieval latency.

**Subgraph scoping.** Spreading activation is primarily bounded by the tree branch that the query enters via the hierarchical index (Section 4.3), ensuring that retrieval cost scales with the relevant subgraph size rather than total memory volume. However, to prevent hard retrieval boundaries that would block cross-domain associations, a configurable fraction (10–20%) of activation signal leaks across branch boundaries at each propagation step. This cross-branch leakage is budget-capped to prevent degeneration into full-graph traversal. Additionally, the consolidation agent maintains *bridge nodes* — memories that genuinely belong in multiple branches — indexed in each relevant branch, providing explicit cross-domain pathways without relying solely on leakage.

### 6.3 Ranking

The final relevance score for each candidate node fuses three signals, with a similarity-gated strength term that prevents query-independent bias:

$$R(v, q) = w_{\text{sim}} \cdot \text{sim}(v, q) + w_{\text{act}} \cdot a^{(T)}(v) + w_{\text{str}} \cdot S(v) \cdot \text{sim}(v, q)^{\eta}$$

where $\text{sim}(v, q)$ is the embedding-based semantic similarity between the memory's natural query form and the query (cheap to compute), $a^{(T)}(v)$ is the converged activation level from graph propagation (captures associative and multi-hop relevance), $S(v)$ is the precomputed strength score (free to look up), $\eta \in (0, 1]$ controls the gating severity (default: $\eta = 0.5$), and the $w$ terms are tunable weights.

The similarity-gating term $\text{sim}(v, q)^{\eta}$ on the strength signal eliminates the "celebrity memory" effect where high-strength memories dominate retrieval results regardless of their relevance to the current query. A memory with $S(v) = 0.95$ but $\text{sim}(v, q) = 0.2$ has its strength contribution reduced by a factor of $0.2^{0.5} \approx 0.45$. A memory with $S(v) = 0.5$ and $\text{sim}(v, q) = 0.9$ retains $0.9^{0.5} \approx 0.95$ of its strength contribution. The square root ($\eta = 0.5$) provides a soft gate: a linear gate ($\eta = 1$) would be too aggressive, essentially making strength a multiplier on similarity and undermining the additive fusion design.

The first two terms remain ungated — a high-similarity, low-strength memory can still rank well through similarity alone, and a well-connected memory can rank well through activation alone. Only the strength bonus is conditioned, preventing query-independent bias while preserving the additive fusion benefit.

We use additive fusion (with gated strength) rather than multiplicative fusion to prevent any single weak signal from zeroing out the relevance score. Under a multiplicative formulation, a memory with high similarity and high strength but low activation (e.g., a recently consolidated node with few graph edges) would score near-zero — precisely the failure mode spreading activation is intended to prevent. The anchor identification gate (Section 6.1) already ensures that only memories with sufficient lexical or semantic relevance enter the candidate pool, preventing noise from high-strength-only memories.

This three-signal fusion function captures dimensions of relevance that no single signal addresses alone: semantic similarity identifies topical relevance, activation level captures associative and structural relevance, and query-conditioned strength ensures that important, well-maintained memories surface preferentially *when they are relevant*.

---

## 7. Consolidation

### 7.1 Two-Stage Hybrid Consolidation

Memory extraction occurs offline — at session end, context limit, or periodic intervals — via a dedicated *consolidation agent*, following the sleep-time compute pattern described by Letta. The consolidation pipeline is split into two stages that exploit the observation that a domain-dependent fraction of extraction work is deterministic and does not require LLM judgment.

**Stage 1 — Symbolic Pre-Processing (no LLM calls).** A fast, cheap pipeline handles the deterministic portion of extraction using established NLP tools:

- **Named entity recognition (NER):** Identifies people, projects, organisations, locations, and other named entities mentioned in the raw logs.
- **Temporal parsing:** Extracts dates, times, durations, and relative temporal references ("last Tuesday," "next sprint") and resolves them to absolute timestamps.
- **Rule-based pattern extraction:** Detects explicit corrections ("actually, X is Y"), stated preferences ("I prefer X over Y"), factual declarations ("the API endpoint is X"), and task outcomes (tool call success/failure sequences).
- **Entity linking:** Matches extracted entities to existing nodes in the memory graph, identifying which memories a new extraction relates to. Candidate matches are filtered by temporal proximity (Section 4.2) to reduce false links across semantically drifted terms.
- **Embedding-based similarity:** Computes dense embeddings for candidate extractions and flags near-duplicates of existing memories.

Stage 1 outputs candidate memory nodes tagged with confidence scores. High-confidence candidates (above a configurable threshold $\theta_{\text{conf}}$) — typically entities, explicit corrections, and stated preferences — can be committed directly to the graph without LLM involvement.

**Stage 2 — LLM Refinement (Opus-class model, offline).** A single consolidation-model call processes the remaining material that Stage 1 could not resolve with high confidence:

- **Salience scoring:** Assessing consequence magnitude for ambiguous extractions via structured rubric.
- **Dual-register encoding:** Representing complex memories in both maximally dense language (precise, terse, technical for token efficiency) and natural query form (colloquial paraphrase for embedding-based retrieval matching). The natural query form is generated during Stage 2 at trivial additional cost (estimated: 20–40 tokens per memory), since the LLM is already processing the memory.
- **Contradiction resolution:** Evaluating conflicts between new extractions and existing memories when the evidence is ambiguous (see Section 9).
- **Non-obvious edge construction:** Identifying association edges that require semantic understanding beyond entity co-occurrence — e.g., linking a debugging session to an architectural decision made weeks earlier.

Candidates below a lower confidence threshold $\theta_{\text{review}}$ are flagged for human review or deferred to the next consolidation pass with a re-extraction note. The primary agent uses whatever model the user selects; the cost of consolidation is decoupled from the cost of interaction.

**Taxonomy-based cost model.** The fraction of extraction handled by Stage 1 varies by conversation type. Rather than asserting a single percentage, we define a conversation content taxonomy with per-category Stage 1 extraction rates:

| Extraction Category | Stage 1 Capable? | Estimated Stage 1 Rate | Examples |
|---|---|---|---|
| Named entities | Yes | ~95% | People, projects, tools, URLs |
| Explicit preferences | Yes | ~90% | "I prefer X over Y" |
| Explicit corrections | Yes | ~85% | "Actually, X is Y" |
| Temporal facts | Yes | ~90% | Dates, deadlines, durations |
| Task outcomes | Yes | ~95% | Tool call success/failure |
| Implicit preferences | No | ~15% | Hedged statements, tone signals |
| Nuanced judgments | No | ~10% | "That approach felt fragile" |
| Complex relationships | No | ~5% | Non-obvious causal links |
| Code semantics | Partial | ~40% | AST-parseable vs. intent-laden |
| Multi-turn reasoning chains | No | ~10% | Conclusions spanning messages |

Let $f_i$ be the fraction of conversation content in category $i$ and $r_i$ be the Stage 1 extraction rate for that category. The overall Stage 1 rate is:

$$R_{\text{Stage1}} = \sum_i f_i \cdot r_i$$

For a **task-oriented technical conversation** (heavy in entities, explicit corrections, tool calls): $R_{\text{Stage1}} \approx 70\text{-}80\%$. For a **nuance-heavy advisory conversation** (heavy in implicit preferences, hedged judgments, complex reasoning): $R_{\text{Stage1}} \approx 35\text{-}50\%$. For a **code-heavy development session**: $R_{\text{Stage1}} \approx 55\text{-}65\%$.

This taxonomy enables deployers to estimate their own cost profile based on their conversational domain, rather than relying on a single aggregate figure.

Following both stages, the consolidation pipeline performs:

1. **Strength scoring:** Initial salience from Stage 2; specificity computed algorithmically from Stage 1 outputs.
2. **Edge construction:** Temporal edges from Stage 1 timestamps; abstraction and association edges from Stage 2; semantic version edges where drift is detected.
3. **Index reorganisation:** Updating the hierarchical tree to accommodate new nodes and rebalancing branches as needed.
4. **Graph health check:** Running health metrics and snapshotting (see Section 9.5–9.6).

### 7.2 Closed-Loop Extraction

Consolidation is closed-loop, not fire-and-forget. Because raw logs function as a write-ahead log and structured memories function as derived views, the system can always detect and correct extraction gaps:

- **Retrieval failure signals:** When a query that should match existing knowledge (based on raw log content) returns no results, this signals an extraction gap. The system triggers targeted re-extraction from the relevant raw log segments.
- **Confidence scoring:** The consolidation agent assigns confidence scores to each extraction. Low-confidence memories are staged for secondary review in subsequent consolidation passes.
- **Progressive re-processing:** As models and extraction prompts improve over time, old logs can be re-processed to yield higher-quality derived views.
- **Proactive re-extraction:** Rather than relying solely on retrieval failure signals to detect extraction gaps, the consolidation agent runs re-extraction on a rotating schedule — revisiting previously absorbed episodes and prioritising log segments originally consolidated with older or weaker models. This makes recovery of initially-missed details systematic rather than dependent on the right question being asked.

Consolidation quality has a floor (initial extraction) but no hard ceiling, though it is bounded by the information content of the raw logs themselves — details the user thought but never expressed cannot be extracted regardless of model capability.

### 7.3 Consolidation Ordering and Graph-State Tagging

Entity linking quality during Stage 1 depends on the current state of the graph: the same raw-log segment processed when the graph is sparse (few nodes for linking targets) will produce fewer entity links than when processed against a dense graph. This creates an ordering effect: early sessions are consolidated with less context than later sessions.

We address this through two mechanisms:

1. **Proactive re-extraction** (Section 7.2) revisits previously consolidated log segments on a rotating schedule. When early-session logs are re-processed against the now-denser graph, Stage 1 entity linking identifies connections that were invisible during the first pass. This means ordering effects are *transient* — they are progressively corrected.

2. **Consolidation-time graph state tagging.** Each extraction records the graph snapshot version at the time of consolidation. Re-extraction is prioritised for extractions made against sparse graph states (low snapshot version numbers), since these are most likely to benefit from re-processing against the current, richer graph.

### 7.4 Specialist Sub-Agents

For throughput, the consolidation agent can fan out into coordinated specialist sub-agents, each handling a distinct consolidation operation concurrently:

- **Extraction agents:** Process raw log segments into candidate memories.
- **Contradiction detection agents:** Identify conflicts between new and existing memories (see Section 9).
- **Defragmentation agents:** Merge duplicate semantic nodes, split bloated nodes, and clean up organisational drift.
- **Tree rebalancing agents:** Restructure the hierarchical index for optimal navigability.

Orchestration can employ coordinated agent teams with shared task boards and dependency tracking, or simply spawn independent sub-agents that report back to the consolidation lead. Results are merged after all sub-agents complete.

Parallel fan-out should be used judiciously: token cost scales linearly with agent count, so fan-out is reserved for large log volumes where the speedup justifies the overhead.

### 7.5 Defragmentation

Defragmentation is a named consolidation operation that addresses organisational drift over long time horizons. It comprises three sub-operations:

1. **Merging:** Duplicate or near-duplicate semantic nodes are identified and consolidated into a single node, with edges merged and strength scores combined.
2. **Splitting:** Bloated nodes that have absorbed too many episodic memories are split into more granular semantic categories.
3. **Rebalancing:** The tree hierarchy is restructured to maintain balanced branch depths and prevent pathological navigation patterns.

### 7.6 Embedding Lifecycle Management

The retrieval pipeline's dependence on dense embeddings creates a versioning requirement for any system designed to operate over months and years. We address this through an explicit embedding lifecycle protocol.

**Versioned embedding storage.** Each memory node stores its embedding alongside a model version tag: $(v, \mathbf{e}_v, m_v)$ where $m_v$ identifies the embedding model used to generate $\mathbf{e}_v$.

**Lazy re-embedding on model migration.** When the embedding model is updated from version $m_{\text{old}}$ to $m_{\text{new}}$:

1. New memories are embedded with $m_{\text{new}}$ immediately.
2. Existing memories are re-embedded **lazily during consolidation passes**, prioritised by strength score (high-strength memories first, since they are most likely to appear in retrieval results).
3. During the transition period, query embeddings are computed with $m_{\text{new}}$, and similarity is computed only against memories already migrated to $m_{\text{new}}$, falling back to BM25 lexical matching for un-migrated memories.

**Migration budget.** Re-embedding is rate-limited to a configurable token/compute budget per consolidation pass. For a typical 30K-node active graph with 512-dimensional embeddings, full re-embedding requires approximately 15M embedding tokens — achievable in 2–3 consolidation passes at standard embedding API rates.

**Compatibility testing.** Before committing to a migration, a sample of high-frequency retrieval queries is evaluated against both the old and new embeddings. If recall degrades by more than a configurable threshold (default: 5%), the migration is aborted and flagged for review.

---

## 8. Proactive Surfacing

Rather than requiring the LLM to explicitly query its memory, AWHM surfaces relevant memories at natural reasoning breakpoints. This mechanism approximates associative human recall — one thought triggers a memory, which shapes the next thought — while keeping latency costs explicit and bounded.

### 8.1 Context Budget Allocation

Memory injection is governed by a formal context budget to prevent the context-window erosion that characterises high-token-cost systems like Zep:

$$\text{tokens}_{\text{memory}} \leq \rho \cdot C_{\text{total}}$$

where $C_{\text{total}}$ is the total context window size and $\rho$ is the **memory budget ratio** (default: $\rho = 0.08$, i.e., memory injection never exceeds 8% of the context window).

Within this budget:

- Tier 1 (pre-fetch) receives up to $0.6\rho \cdot C_{\text{total}}$
- Tier 2 (tool-boundary refresh) receives up to $0.3\rho \cdot C_{\text{total}}$, and refreshed memories *replace* pre-fetched memories rather than appending
- Tier 3 fallback uses the existing pre-fetched set (no additional tokens)

For a 200K-token context window with $\rho = 0.08$: the total memory budget is 16K tokens. This is a hard cap, enforced by truncating the ranked memory list at the budget boundary. This quantitatively distinguishes AWHM from systems like Zep, whose injection scales linearly with memory volume (no cap): AWHM's injection is bounded by $\rho$ regardless of total memory size.

### 8.2 Three-Tier Injection Strategy

We define three injection tiers, each triggered at a different point in the reasoning cycle:

**Tier 1: Pre-fetch on initial prompt.** When a user query arrives, the full retrieval pipeline (Section 6) executes immediately, and a baseline memory set is injected before reasoning begins. This ensures the model has access to relevant historical context from the first token of its response.

**Tier 2: Refresh at tool-call boundaries.** Agent loops already pause at tool-call boundaries while awaiting tool results. After each tool result returns, a lightweight retrieval update runs against the accumulated context (original query plus reasoning so far plus tool results). This captures memories that become relevant only as the reasoning trajectory unfolds — associations that were not predictable from the initial query alone.

**Tier 3: Bounded latency budget.** Each retrieval invocation must complete within a fixed latency window (e.g., 200ms). If retrieval cannot complete within this budget — due to graph size, query complexity, or system load — the system falls back to the most recent pre-fetched memory set rather than blocking the reasoning pipeline.

### 8.3 Memory Clouds and Anchoring Bias Mitigation

Surfaced memories are injected as *memory clouds*: compact, contextualised packets inserted between reasoning steps. The LLM incorporates or ignores surfaced memories naturally based on their relevance to the current reasoning trajectory.

Injected memory clouds carry an inherent risk of anchoring bias: the LLM may weight surfaced memories disproportionately simply because they appear in context, even when they are only marginally relevant to the current reasoning step. We mitigate this through three mechanisms:

1. **Relevance-threshold gating.** Memories below a minimum relevance score $R_{\min}$ are excluded from injection even if they fall within the token budget. This prevents marginally relevant memories from entering the context.

2. **Epistemic framing.** Memory clouds are injected with explicit epistemic markers (e.g., "The following memories *may* be relevant to the current query. Use or disregard as appropriate.") rather than presented as authoritative context. Research on LLM instruction-following suggests that epistemic hedging reduces anchoring effects.

3. **Injection placement.** Memories are injected as a clearly delineated block before each reasoning step, rather than interleaved within the LLM's own reasoning tokens. This structural separation reduces the likelihood that the model treats injected memories as its own prior reasoning.

This non-prescriptive injection avoids forcing the model to use every surfaced memory, preserving its autonomy to judge relevance in context.

---

## 9. Consistency and Maintenance

### 9.1 Contradiction Detection and Supersession

When the consolidation agent detects a new memory that contradicts an existing one — identified through semantic similarity combined with opposing content — resolution is evidence-weighted rather than purely recency-based. The system evaluates three signals before determining the resolution path:

1. **Confidence comparison:** The confidence scores assigned during consolidation for both the new and existing memories.
2. **Corroboration check:** Whether the new memory contradicts a single existing memory or multiple corroborating memories. Contradicting many established, mutually-reinforcing memories is a red flag against the new memory, not a trigger for mass supersession.
3. **Stakes assessment:** The salience score of the existing memory. High-salience existing memories warrant more scrutiny before supersession.

For low-stakes contradictions with comparable confidence and no corroboration asymmetry, the newer memory supersedes by default. For high-stakes contradictions — where the existing memory has high salience, or multiple corroborating memories are contradicted — the conflict is flagged for user confirmation rather than auto-resolved. In all cases, the supersession protocol applies:

1. The old node receives a `superseded_by` edge pointing to the new node.
2. The old node's strength score undergoes a decay proportional to the confidence differential, removing it from standard retrieval results.
3. The old node is *not deleted*, preserving the record that knowledge changed and enabling temporal queries of the form "what did we believe about X before Y?"

This protocol avoids the epistemological trap of assuming recency equals accuracy, while maintaining a complete audit trail of knowledge evolution.

### 9.2 Explicit Corrections

When a user provides an explicit correction ("actually, X is Y"), the system triggers immediate re-scoring rather than waiting for the next consolidation cycle. The corrected information receives elevated salience, the superseded memory receives sharp strength decay, and the `superseded_by` edge is created in real time. This ensures that corrections take effect within the current session rather than after an offline consolidation pass.

### 9.3 Hard Deletion

For privacy compliance and user autonomy, targeted hard deletion is supported as a mechanism parallel to (and independent of) organic decay:

- Specified memory nodes are purged from the graph.
- Corresponding raw log segments are deleted from the append-only log.
- Graph edges incident on deleted nodes are severed.
- Tree references to deleted nodes are removed.

Hard deletion is architecturally distinct from strength decay. Decay is an organic process that governs retrieval priority; deletion is an administrative operation that removes information from the system entirely. The two mechanisms serve different purposes and operate on independent paths.

### 9.4 Cold Start and Bootstrapping

Cold start — the period before the memory graph has densified enough for spreading activation to outperform simpler retrieval methods — is a critical adoption barrier. A system that provides no value in the first several sessions will be abandoned before its associative advantages emerge. AWHM addresses this through three mechanisms:

**Seed import.** Users can ingest existing documents, past chat exports, preference files, knowledge bases, or prior conversation histories. These seed materials are processed through the standard two-stage consolidation pipeline — no special ingestion mechanism is required. Materials enter as raw logs and are extracted, scored, and integrated normally. This allows the memory graph to begin with a non-trivial initial population.

**Hybrid raw-log fallback.** During the bootstrap period (approximately the first 10 sessions, or until graph density crosses a configurable threshold), the retrieval engine runs a parallel BM25 search directly over raw logs alongside the standard graph-based retrieval pipeline. Results from both paths are merged and ranked together. This ensures the system delivers value from the first session: even before the graph has enough nodes and edges for spreading activation to be effective, direct keyword search over raw transcripts can surface relevant prior context. As the graph densifies, the raw-log fallback contributes diminishing marginal returns and is phased out automatically when graph-based retrieval recall exceeds the fallback's contribution (measured as the fraction of returned results originating from each path).

The hybrid raw-log fallback during cold start is not a concession that the associative architecture fails — it is a *bootstrapping mechanism* analogous to the explore phase in multi-armed bandit problems. The system begins with a simpler strategy (BM25 over raw logs) and transitions to the richer strategy (graph-based spreading activation) as sufficient data accumulates to make the richer strategy outperform the simpler one. The transition is measured and automatic, not heuristic. This is standard practice in adaptive systems and does not undermine the value of the steady-state architecture.

**Organic growth and density regimes.** Beyond the bootstrap period, subsequent sessions and consolidation passes incrementally expand and enrich the memory graph. The associative advantage of spreading activation exhibits three density regimes:

**Sparse regime** (< ~2,000 active nodes, mean degree < 3): Graph is too sparse for multi-hop retrieval to reliably find associative paths. Performance is comparable to or worse than direct embedding similarity. The hybrid raw-log fallback operates in this regime.

**Growth regime** (~2,000–30,000 active nodes, mean degree 5–15): Each new node and edge creates new associative pathways that benefit retrieval for *existing* nodes, not just the new one. The marginal value of each additional node is increasing — this is the super-linear regime, and the design target for AWHM.

**Saturation regime** (> ~30,000 active nodes or mean degree > 20): Dense connectivity causes activation to spread broadly, reducing signal-to-noise ratio. Lateral inhibition partially compensates, but the marginal value of additional nodes diminishes. Beyond a topology-dependent threshold, further densification actively degrades retrieval precision.

Progressive abstraction and cold storage (Section 4.4) are designed to keep the active graph in the growth regime by absorbing old episodic nodes and archiving low-strength nodes, maintaining the density range where associative retrieval provides maximal advantage. This converts the graph-bounding mechanisms from a purely performance concern into a *retrieval quality* concern: they keep the graph in its optimal density regime.

### 9.5 Versioned Snapshots and Rollback

The memory graph is a mutable data structure modified by every consolidation pass. Consolidation bugs, corrupted extractions, or incorrect merges can degrade graph quality in ways that are difficult to detect immediately. Versioned snapshots provide a safety net.

**Snapshot schedule.** Full graph snapshots are taken at configurable intervals: by default, before and after each consolidation pass, plus a daily snapshot. For large graphs, incremental (delta) snapshots reduce storage cost by recording only the changes since the last full snapshot.

**Retention policy.** Snapshot history is bounded to prevent unbounded storage growth — by default, retaining the last 7 daily snapshots and the last 4 weekly snapshots. Older snapshots are pruned automatically. Note that versioned snapshots should be understood as preserving *specific graph states*, not as redundant backups of a deterministic function — since extraction is non-deterministic, no two consolidation passes will produce the same graph from the same raw logs.

**Rollback triggers.** Rollback can be initiated manually by user command, or automatically if graph health metrics (Section 9.6) indicate degradation post-consolidation. Automated rollback restores the pre-consolidation snapshot and flags the failed consolidation pass for investigation.

**Implementation.** Snapshots are stored as serialised graph state (nodes, edges, strength scores, tree structure). The storage cost is proportional to active graph size — for a typical 10K–50K node graph, individual snapshots are small (single-digit megabytes).

### 9.6 Graph Health Monitoring

To detect graph drift and degradation before they impact retrieval quality, the consolidation pipeline includes a health-check phase that tracks four key metrics:

1. **Orphaned node count:** Nodes with no edges, indicating extraction or edge-construction failures. Orphaned nodes are either re-linked (if Stage 1 entity linking identifies a match) or flagged for re-extraction.
2. **Contradiction density:** The number of conflicting active memories per topic cluster, normalised against cluster size. Sustained high contradiction density indicates extraction inconsistency.
3. **Retrieval hit rate:** The fraction of queries returning at least one result above the relevance threshold. Sustained low hit rate signals extraction gaps — the graph is missing memories that the raw logs contain.
4. **Edge staleness distribution:** The fraction of edges not traversed during any retrieval in the last $N$ consolidation cycles. High staleness indicates structural drift — edges that once represented meaningful associations but no longer connect to active retrieval paths.

Thresholds on each metric trigger targeted cleanup during the same consolidation pass: re-linking orphans, resolving contradictions, triggering re-extraction from raw logs for low-hit-rate topic regions, and pruning stale edges. Metrics are logged for trend analysis; sustained degradation across multiple cycles triggers broader re-extraction and, if snapshots are available, comparison against prior graph states to identify the point of divergence.

This health monitoring is integrated into the consolidation cycle — it is not a separate daemon or background process. It runs at the cost of a few additional graph queries per consolidation pass.

---

## 10. Cost Analysis

A primary design goal of AWHM is sub-linear cost scaling with memory size. We achieve this through strict separation of LLM-dependent and LLM-independent operations.

### 10.1 LLM Calls: Offline Only, Two-Stage Reduction

LLM inference occurs exclusively during offline consolidation, and only for the subset of extractions that require it. The two-stage hybrid pipeline (Section 7.1) routes a domain-dependent fraction of extraction through Stage 1 (symbolic pre-processing, zero LLM calls) and reserves Stage 2 (LLM refinement) for the remainder. Based on the conversation content taxonomy (Section 7.1), the Stage 1 rate ranges from approximately 35% for nuance-heavy advisory conversations to 80% for task-oriented technical interactions, with most mixed-domain deployments falling in the 55–70% range.

Remaining LLM calls comprise:

- **Ambiguous extraction refinement** from Stage 2 (batchable, async).
- **Dual-register encoding** — generating natural query forms alongside dense canonical forms (trivial additional cost per memory during Stage 2).
- **Salience scoring** for non-trivial memories via structured rubric (performed once per memory at creation).
- **Non-obvious edge construction** requiring semantic understanding beyond entity co-occurrence.
- **Index reorganisation** (periodic, not per-query).

All consolidation LLM calls are batchable and asynchronous. Their cost is amortised across the memories produced and is independent of the number of retrieval queries served.

### 10.2 Retrieval: Entirely Algorithmic

Retrieval incurs zero LLM calls. The cost components are:

| Operation | Cost | Frequency |
|-----------|------|-----------|
| Session buffer check | Free (in-memory lookup) | Per query |
| Embedding computation | Cheap (embedding model, against natural query form) | Per query |
| BM25 scoring | Free (inverted index, against dense canonical form) | Per query |
| Raw-log fallback (bootstrap only) | Free (BM25 over logs) | Per query (first ~10 sessions) |
| Spreading activation | Free (graph traversal with symmetric normalisation) | Per query |
| Strength lookup | Free (precomputed) | Per query |
| Score fusion (with similarity gating) | Free (arithmetic) | Per query |

### 10.3 Context Efficiency

The hierarchical index enables the primary agent to navigate the memory space via a few targeted tool calls rather than receiving a massive memory dump. Dense encoding minimises tokens per memory node, while the formal context budget ($\rho \leq 0.08$ of the context window) provides a hard cap on memory injection. Together, these mechanisms ensure that memory consumes a small, bounded fraction of the context window regardless of total memory volume.

### 10.4 Scaling Argument

Cost scales sub-linearly with memory size for two reasons:

1. **Retrieval cost is O(subgraph)**, not O(total graph), because spreading activation is scoped to the tree branch entered by the query.
2. **Consolidation cost is O(new logs)**, not O(total logs), because only new raw log segments are processed in each consolidation pass (with optional re-processing of old segments for quality improvement). The two-stage pipeline further reduces the LLM-dependent portion of this cost to the Stage 2 fraction of new log volume.

**Concrete scaling bounds.** For a target active graph of 30K nodes with mean degree 10, the graph contains approximately 300K edges. Spreading activation scoped to a typical subgraph of 2,000–5,000 nodes (one tree branch) traverses 20K–50K edges per query — completing in under 50ms on commodity hardware with in-memory graph storage. The 10K–50K node bound is maintained by progressive abstraction (absorbing ~5–15% of episodic nodes per consolidation pass) and cold storage (archiving ~2–5% per pass). For a user generating approximately 50–100 new memories per session with daily sessions, the inflow of ~50–100 nodes/day is balanced by absorption and archival outflows within 200–500 sessions, at which point the active graph reaches equilibrium size.

Additional storage costs are bounded: the session buffer is negligible (in-memory, cleared per session), raw logs grow linearly with interaction volume but are cheap text storage, graph snapshots are bounded by the retention policy (Section 9.5), and the dual-register encoding doubles text storage per node while remaining negligible relative to embedding storage.

---

## 11. Evaluation Framework

We propose eight metrics to evaluate AWHM, with particular emphasis on longitudinal performance — the regime where existing benchmarks are weakest. Established benchmarks including LoCoMo (Maharana et al., 2024; ACL 2024), LongMemEval (Wu et al., 2024; ICLR 2025), and MemoryBench (arXiv 2510.17281) provide useful starting points for component-level validation.

### 11.1 Retrieval Precision and Recall at k

Standard information retrieval metrics computed against human-annotated ground truth derived from longitudinal sessions. Annotators identify which memories *should* have been retrieved for each query, enabling precision@k and recall@k computation. This measures the system's ability to surface relevant memories and avoid irrelevant ones.

### 11.2 Multi-Hop Accuracy

Accuracy on questions that require chaining multiple memories to produce a correct answer. These questions are specifically designed so that no single memory node contains the answer — the system must traverse association edges to connect disparate facts. This is the metric where spreading activation is expected to demonstrate its strongest advantage over cosine-similarity baselines.

### 11.3 Temporal Accuracy

Accuracy on queries of the form "what was X at time T?" posed after X has been updated one or more times. This tests both the contradiction handling protocol (Section 9.1) and the temporal context tagging (Section 4.2): the system must retrieve the historically correct value rather than the current value, navigating the `superseded_by` and semantic version edge chains.

### 11.4 Consolidation Fidelity

A comparison of machine-extracted memories against human-extracted ground truth from the same raw logs. Human annotators independently extract structured memories from raw session logs; the system's extractions are compared for completeness, accuracy, and density. This measures the quality of the consolidation pipeline in isolation, including the dual-register encoding quality.

### 11.5 Longitudinal Degradation

The critical metric: whether retrieval quality (precision, recall, multi-hop accuracy) holds stable over 100+ and 1,000+ sessions. This is measured as the slope of retrieval quality over session count. A well-functioning system should exhibit zero or positive slope (quality improves as more memories create richer associative structure in the growth regime); a degrading system exhibits negative slope. No current benchmark adequately captures this regime, making it the most important — and most difficult — evaluation axis.

### 11.6 Component Ablation

Systematic removal of individual system components to validate each one's contribution. The deployment profiles (Section 3.3) define the primary ablation path — each profile transition enables measurement of marginal contribution:

| Ablation | Expected Impact |
|----------|----------------|
| Remove spreading activation (cosine-only retrieval) | Degraded multi-hop accuracy |
| Remove strength scoring (uniform weighting) | Increased noise in retrieval, degraded precision |
| Remove similarity-gating on strength ($\eta = 0$) | Celebrity memory effect, reduced precision for off-topic queries |
| Remove receiver-side normalisation | Sink-node accumulation, topology-driven bias |
| Remove hierarchical index (flat graph) | Increased retrieval latency, degraded scaling |
| Remove proactive surfacing (explicit query only) | Reduced recall for associatively relevant memories |
| Remove dual-register encoding (dense-only similarity) | Reduced recall for colloquially phrased queries |
| Remove recency dampening floor ($\phi = 0$) | Recency oversuppression for high-value recent memories |

### 11.7 Cold-Start Convergence

The number of sessions required for graph-based retrieval to match or exceed the hybrid raw-log fallback in retrieval quality (precision, recall). This metric characterises the bootstrap period: a well-designed system should converge within approximately 8–12 sessions with organic usage, or fewer with seed import. Convergence is measured as the session at which the fraction of retrieval results originating from the graph path consistently exceeds the fraction from the raw-log fallback.

### 11.8 Graph Health Recovery Rate

The fraction of detected graph health anomalies (orphaned nodes, contradiction spikes, retrieval hit-rate drops) automatically resolved within one consolidation cycle. This measures the effectiveness of the integrated health monitoring and cleanup mechanisms (Section 9.6). A well-functioning system should achieve >90% automated resolution, with the remainder flagged for manual review or deferred re-extraction.

### 11.9 Convergence Profiling

A supplementary metric that tracks the empirical iteration count at convergence for spreading activation across different query types and graph states. This validates the spectral convergence analysis (Section 6.2) and identifies graph topologies or query patterns that approach the worst-case iteration bound.

### 11.10 Evaluation Requirements

Evaluation must be longitudinal with real users on real tasks over weeks and months. Established benchmarks — LoCoMo for multi-session conversational memory, LongMemEval for long-term interactive memory across five capability dimensions, and MemoryBench for continual learning from accumulated feedback — provide useful starting points for component-level validation. However, these benchmarks are insufficient for the claims this system makes — particularly regarding longitudinal degradation, the three-regime density model, and the emergent benefits of rich associative structure over extended use.

---

## 12. Discussion

### 12.1 Limitations

Several limitations of the current design warrant acknowledgement:

**Consolidation latency.** Because full memory extraction is deferred to offline processing, there is an inherent delay between an interaction and the availability of its memories in structured graph form. The session buffer (Section 3.2) mitigates this for corrections, preferences, and explicit facts within the current session, and the hybrid raw-log fallback (Section 9.4) ensures prior session content is retrievable even before consolidation. However, full graph integration — with proper edge construction, strength scoring, and hierarchical indexing — remains asynchronous.

**Session buffer volatility.** The session buffer is an in-memory data structure that is lost on crash. Periodic persistence to a write-ahead log mitigates this, but a crash between flush intervals will lose buffered items. The window of vulnerability equals the flush interval (configurable; default recommended: 30 seconds).

**Salience subjectivity and inter-run consistency.** Initial salience scoring depends on LLM judgment (Stage 2 of consolidation), introducing subjectivity and potential inconsistency. LLM-dependent operations (salience scoring, non-obvious edge construction, contradiction resolution) are inherently non-deterministic. We bound the impact of this non-determinism through three mechanisms: (a) salience is one of five scoring dimensions and is bounded to $[0, 1]$ — inconsistency in salience scoring produces bounded perturbation in the overall strength score; (b) edge construction errors are self-correcting through graph health monitoring (orphaned nodes are detected and re-linked); and (c) contradiction resolution escalates high-stakes conflicts to user confirmation rather than auto-resolving them. The system is designed to be *robust to* LLM judgment variability, not *dependent on* LLM judgment consistency.

**Extraction non-determinism.** Because structured memories are stochastic projections over raw logs rather than deterministic materialised views, graph states are not reproducible across independent extraction passes. Versioned snapshots (Section 9.5) preserve specific graph states but should be understood as capturing one valid interpretation, not the only possible interpretation. The convergence guarantee (probability of recovery approaching 1 over repeated passes) is an asymptotic property, not a per-pass guarantee.

**Symbolic pre-processing domain sensitivity.** Stage 1 of the consolidation pipeline relies on NER and entity-linking models whose quality varies across domains. Conversations heavy in domain-specific jargon, code, or non-English content may see lower Stage 1 extraction rates, routing more material to the more expensive Stage 2. The conversation content taxonomy (Section 7.1) makes this domain-dependence explicit: deployers can estimate their cost profile based on conversational domain. The system degrades gracefully — falling back to full LLM extraction rather than failing — but the cost savings of two-stage consolidation are domain-dependent.

**Graph maintenance complexity.** The progressive abstraction, defragmentation, and health monitoring operations, while necessary for long-term graph health, introduce implementation complexity and potential failure modes. Incorrect merges or splits could degrade memory quality. Versioned snapshots (Section 9.5) provide a rollback safety net, and graph health monitoring (Section 9.6) provides automated detection, but neither eliminates the possibility of subtle degradation that evades metric-based detection.

**Progressive abstraction is controlled-lossy.** While the three-tier architecture is strictly less lossy than systems like Mem0 — raw logs preserve the complete record — progressive abstraction is still a lossy operation at the active-graph level. When episodic nodes are absorbed into semantic summaries, the consolidation agent decides which details are salient, and this judgment can be wrong. The proactive re-extraction schedule (Section 7.2) mitigates this by systematically revisiting absorbed episodes, but recovery still depends on the re-extraction model identifying what was missed. We characterise this as *controlled lossiness* — bounded by raw-log preservation and systematically reduced over time — rather than the irrecoverable lossiness of systems that discard source material.

**Anchoring bias.** Despite the mitigation mechanisms described in Section 8.3 (relevance-threshold gating, epistemic framing, injection placement), proactive surfacing inherently risks anchoring the LLM's reasoning toward surfaced memories. The magnitude of this risk under real-world conditions is an empirical question requiring evaluation.

### 12.2 Open Questions

Several design decisions require empirical validation:

- **Optimal dimension weights** for the strength scoring model. The default weighting (utility and frequency higher) is principled but unvalidated. The adaptive heuristics (Section 5.5) address the most sensitive parameters, but the base weights still require empirical calibration.
- **Absorption threshold** ($\theta_{\text{abs}}$) for progressive abstraction. Too aggressive and valuable episodic detail is lost prematurely; too conservative and the graph grows unbounded.
- **Symmetric normalisation and convergence profile.** While the spectral argument (Section 6.2) guarantees geometric convergence, the empirical convergence profile across different query types and graph topologies requires validation.
- **Lateral inhibition coefficient** ($\gamma$) and overlap weighting ($\mu$). The optimal sparsity level and the balance between Jaccard overlap and branch co-membership depend on query type and graph density.
- **Similarity-gating exponent** ($\eta$). The square-root default is motivated but unvalidated; alternatives may perform better for specific use cases.
- **Recency floor** ($\phi$). The optimal floor depends on usage patterns — users who frequently reference recent work may benefit from higher $\phi$.
- **Consolidation frequency.** The optimal interval between consolidation passes may vary with usage patterns and may benefit from adaptive scheduling.
- **Hybrid fallback phase-out threshold.** The optimal point at which to disable the raw-log BM25 fallback in favour of pure graph retrieval depends on graph density and the specificity of the user's usage patterns.
- **Stage 1 confidence threshold** ($\theta_{\text{conf}}$). The boundary between "commit directly" and "route to Stage 2 LLM" in the two-stage consolidation pipeline affects both extraction quality and cost.
- **Snapshot retention policy.** The optimal balance between rollback depth and storage cost may vary across deployment scenarios.
- **Embedding migration threshold.** The recall-degradation threshold for aborting embedding model migration requires calibration against real-world embedding model transitions.
- **Density regime boundaries.** The approximate node counts and mean degrees defining the sparse, growth, and saturation regimes are design estimates requiring empirical calibration across diverse usage patterns.

### 12.3 Future Directions

Several extensions present themselves for future work:

- **Full Bayesian hyperparameter optimisation.** Replacing the simple adaptive heuristics (Section 5.5) with online Bayesian optimisation over the full parameter space, using a composite reward signal (retrieval precision, downstream task success, latency). This would enable zero-touch operation for deployments where manual parameter review is impractical.
- **Multi-level sharding for extreme-scale deployments.** For deployments approaching millions of nodes (multi-user, multi-year, enterprise-scale), partitioning the memory graph across shards with cross-shard bridge routing. This is unnecessary for typical personal-assistant deployments but may become relevant for shared enterprise memory systems.
- **Predictive prefetch via intent classification.** A lightweight intent classifier that predicts upcoming memory needs based on the ongoing reasoning trajectory, pre-loading relevant memories before they are explicitly needed. This is speculative and requires validation that the accuracy of intent prediction justifies the computational overhead.
- **Multi-user memory.** Extending the architecture to shared memory graphs where multiple users contribute to and retrieve from overlapping memory spaces, with appropriate access control and perspective handling.
- **Active learning for consolidation.** Using retrieval failure signals not just for re-extraction but for adaptive improvement of the extraction prompts themselves.
- **Federated memory.** Distributing the memory graph across multiple agents or systems while maintaining consistency and enabling cross-agent memory sharing.
- **Hardware-aware optimisation.** Adapting the graph storage and retrieval engine to exploit specific hardware characteristics (GPU-accelerated graph traversal, persistent memory tiers).

---

## 13. Conclusion

We have presented Activation-Weighted Hierarchical Memory, an external memory architecture for large language models that addresses the fundamental limitations of existing approaches. By separating memory formation (offline, via a two-stage hybrid pipeline with domain-dependent cost reduction) from memory retrieval (real-time, purely algorithmic), AWHM achieves high-fidelity, associative memory with sub-linear cost scaling. The three-tier storage design — append-only raw logs as permanent ground truth captured in real time, structured memories as stochastic projections re-derivable with probability approaching 1, and a session buffer for intra-session continuity with explicitly characterised extraction ceilings — ensures that consolidation quality improves monotonically in expectation while eliminating the intra-session amnesia gap that previous offline-consolidation designs suffered from. A cold-start bootstrapping protocol combining seed import with hybrid raw-log fallback ensures the system delivers value from the first session, transitioning measurably and automatically from simple to rich retrieval strategies as graph density permits.

The spreading activation retrieval engine, governed by symmetric sender-receiver normalisation, formally defined lateral inhibition via composite structural proximity, and similarity-gated strength scoring with power-law decay and adaptive parameter tuning, provides multi-hop, temporally aware, and importance-weighted retrieval that cosine-similarity baselines cannot match. Dual-register encoding bridges the gap between dense memory storage and colloquial query patterns, while temporal context tagging provides resilience against semantic drift over long deployment horizons. Proactive surfacing at natural reasoning breakpoints — with formal context-budget bounds and anchoring-bias mitigation — approximates associative human recall without requiring the model to manage its own memory or sacrificing context-window capacity. Versioned graph snapshots and integrated health monitoring provide operational resilience against consolidation failures and graph drift. A three-regime density model (sparse, growth, saturation) provides principled justification for the graph-bounding mechanisms and identifies the operating regime where the architecture provides maximal advantage. The modular deployment profile system (Lite, Standard, Full) demonstrates that the architecture's value is accessible incrementally, with each component layer independently testable.

We have proposed an eight-metric evaluation framework — anchored to established benchmarks (LoCoMo, LongMemEval, MemoryBench) — that foregrounds longitudinal performance, the regime most critical to real-world deployment and least served by existing benchmarks.

The core architectural insight is that memory is not a retrieval problem alone — it is a consolidation, decay, reinforcement, and association problem. By treating it as such, and by drawing on established principles from cognitive science and database systems, we arrive at a system whose memory improves with use rather than degrading with scale.

---

*Correspondence: [contact information]*
