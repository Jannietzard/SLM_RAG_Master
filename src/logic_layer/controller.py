"""
===============================================================================
AgenticController — static helpers for bridge-entity extraction & query rewriting
===============================================================================

Master's Thesis: "Enhancing Reasoning Fidelity in Quantized SLMs on Edge"
Artifact B: Agent-Based Query Processing — utility helpers used by the
production pipeline (src/pipeline/agent_pipeline.py).

For the thesis methodology section
----------------------------------
The production orchestrator of the three-agent pipeline (S_P → S_N → S_V)
is ``src.pipeline.agent_pipeline.AgentPipeline``. ``AgenticController`` in
this module is a namespace of stateless utility helpers (bridge-entity
extraction and hop-query rewriting) consumed by AgentPipeline's iterative
multi-hop path; it does not orchestrate any pipeline by itself.

B7 (audit 2026-05-15) — historical context:
    This module used to be the S_P → S_N → S_V orchestrator with two
    execution modes (LangGraph StateGraph + sequential fallback). Both modes
    were superseded by ``src.pipeline.agent_pipeline.AgentPipeline``, which
    is what every evaluation script calls. The orchestrator code, the
    LangGraph workflow, the AgentState TypedDict, the per-node methods,
    ``run()``, ``__call__()``, ``create_controller()``, and the ``__init__``
    were all removed in the 2026-05-15 cleanup audit.

    Only the static bridge-handling helpers are retained because
    ``AgentPipeline._iterative_navigate`` calls them by class-attribute
    reference (``AgenticController._extract_bridge_entities``,
    ``AgenticController._rewrite_hop_query_with_bridges``).

    For the full pipeline use::

        from src.pipeline import AgentPipeline, create_full_pipeline
        pipeline = create_full_pipeline()
        result = pipeline.process("Your query")

References:
- IRCoT (iterative retrieval-with-reasoning): Trivedi et al., 2023, ACL,
  arXiv:2212.10509.
- HippoRAG bridge-aware multi-hop: Gutiérrez et al., 2024, NeurIPS,
  arXiv:2405.14831.
- Self-Ask / DSP query rewriting: Press et al., 2022, EMNLP;
  Khattab et al., 2022, arXiv:2212.14024.

===============================================================================
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

from ._settings import _PROPER_NOUN_RE

logger = logging.getLogger(__name__)


# =============================================================================
# AGENTIC CONTROLLER — static helpers only (B7 cleanup)
# =============================================================================

class AgenticController:
    """
    Static-helper container for bridge-entity extraction and hop-query rewriting.

    This class is intentionally stateless after the B7 cleanup. It only
    exists as a namespace for class-level constants and ``@staticmethod`` /
    ``@classmethod`` helpers used by ``AgentPipeline._iterative_navigate``.

    Do NOT instantiate. Calling ``AgenticController()`` will succeed (no
    ``__init__`` is defined, so Python uses ``object.__init__``) but the
    resulting instance has no behavior beyond the static helpers.

    Surviving public surface (via class-attribute reference):
      - ``AgenticController._extract_bridge_entities(chunks, exclude, query)``
      - ``AgenticController._rewrite_hop_query_with_bridges(sub_query, bridges)``
      - ``AgenticController._score_bridge_candidate(...)`` — internal use
      - ``AgenticController._detect_expected_type(query)`` — internal use
    """

    # ── Class-level constants used by the static helpers ─────────────────────

    # Query-keyword stopwords for relevance ranking. Non-content words removed
    # before measuring proximity between a candidate entity and the query
    # keywords in the source chunk (§12.32 Fix 2).
    _QUERY_STOPWORDS = frozenset({
        "a", "an", "and", "are", "as", "at", "be", "by", "do", "does",
        "for", "from", "has", "have", "he", "her", "him", "his", "how",
        "in", "is", "it", "its", "of", "on", "or", "that", "the", "their",
        "they", "this", "to", "was", "were", "what", "when", "where",
        "which", "who", "whom", "whose", "why", "will", "with", "would",
        "you", "your", "i", "we", "us", "our", "she", "them",
    })

    # Expected entity-type hints from interrogative words. Used to prefer
    # candidates whose surface form matches the expected category.
    _QUERY_TYPE_HINTS = {
        # who/whose/whom → person
        "who": "PERSON", "whose": "PERSON", "whom": "PERSON",
        # where → location/place
        "where": "GPE",
        # when → date/year (matched by digit pattern, not name-style)
        "when": "DATE",
    }

    # ── Bridge-entity scoring (§12.32 Fix 2) ──────────────────────────────────

    @classmethod
    def _score_bridge_candidate(
        cls,
        candidate: str,
        chunk: str,
        query: str,
        expected_type: Optional[str] = None,
        chunk_rank: int = 0,
    ) -> float:
        """
        Relevance score for a bridge-entity candidate.

        Combines four signals:
            +α  Proximity to query keywords in the source chunk (the closer
                the candidate sits to terms the user actually asked about,
                the more likely it is the intended bridge).
            +β  Match to expected entity type (PERSON candidates score
                higher when the query says "who", GPE higher for "where").
            -γ  Position penalty for being far from the start of the chunk
                (top-of-chunk entities are usually the topic; later mentions
                are tangential — "Robert Durst... Galveston" appears after
                the subject's main definition).
            ×δ  C1 reciprocal chunk-rank prior 1/(1+rank): an entity from a
                top-ranked retrieval chunk is far more likely to be the bridge
                target than one from a low-ranked noise chunk. Reciprocal rank
                is the same primitive RRF uses (Cormack et al. 2009, SIGIR).
                The local score is clamped to be non-negative before the
                multiplicative prior so it cannot invert the ordering of
                negative-scoring candidates.

        Returns a non-negative float. Higher = more likely the bridge; 0 means
        "not found / no query proximity" (used by the C4 abstention floor).
        """
        chunk_lower = chunk.lower()
        candidate_lower = candidate.lower()
        cand_pos = chunk_lower.find(candidate_lower)
        if cand_pos < 0:
            return 0.0

        # ── Signal 1: proximity to query keywords ─────────────────────────
        query_keywords = [
            w.lower() for w in re.findall(r"\b\w{3,}\b", query)
            if w.lower() not in cls._QUERY_STOPWORDS
        ]
        min_dist = float("inf")
        for kw in query_keywords:
            kw_pos = chunk_lower.find(kw)
            if kw_pos >= 0:
                dist = abs(kw_pos - cand_pos)
                if dist < min_dist:
                    min_dist = dist
        if min_dist == float("inf"):
            proximity = 0.0
        else:
            proximity = 1.0 / (1.0 + min_dist / 200.0)

        # ── Signal 2: expected-type match ─────────────────────────────────
        type_bonus = 0.0
        # Tokens that, when present, mark the candidate as a role/title
        # rather than a person name.
        _ROLE_TOKENS = {
            "investigator", "director", "producer", "manager", "president",
            "chairman", "founder", "owner", "captain", "coach", "lawyer",
            "attorney", "officer", "secretary", "minister", "governor",
            "actor", "actress", "author", "writer", "composer",
            "pictures", "studios", "movie", "film", "company", "corporation",
            "investigations", "agency", "department", "division", "group",
            "industries", "limited", "incorporated", "associates",
            "private", "public",
        }
        if expected_type == "PERSON":
            tokens = candidate.split()
            tokens_lower = [t.lower() for t in tokens]
            has_role_token = any(t in _ROLE_TOKENS for t in tokens_lower)
            if (
                2 <= len(tokens) <= 3
                and all(t[0].isupper() for t in tokens if t)
                and not has_role_token
            ):
                type_bonus = 0.5
            elif has_role_token:
                type_bonus = -0.3
        elif expected_type == "GPE":
            if len(candidate.split()) <= 2:
                type_bonus = 0.3

        # ── Signal 3: position penalty ────────────────────────────────────
        position_penalty = min(0.5, cand_pos / 600.0)

        local_score = proximity

        # Type and length features MODULATE a candidate only when it has
        # lexical proximity to the question. Absent proximity, a candidate has
        # no positive evidence of being the answer (it merely "looks like" the
        # right type or length), so it must not clear the C4 abstention floor.
        if proximity > 0.0:
            local_score += type_bonus
            # Token-count preference (folded in from the former Pass-2 external
            # adjustment so there is a SINGLE scoring function — 2-token names
            # are typically the intended bridge; long spans are usually noise).
            n_tokens = len(candidate.split())
            if n_tokens == 2:
                local_score += 0.10
            elif n_tokens >= 4:
                local_score -= 0.20

        local_score -= position_penalty

        # ── Signal 4 (C1): reciprocal chunk-rank prior ────────────────────
        rank_prior = 1.0 / (1.0 + max(0, chunk_rank))
        return max(0.0, local_score) * rank_prior

    @classmethod
    def _detect_expected_type(cls, query: str) -> Optional[str]:
        """Infer the expected bridge-entity type from interrogative words.

        Returns the first matching hint type, or None if the query has no
        recognised question word. Conservative — only fires on canonical
        question starters."""
        q = query.lower().strip()
        for prefix, etype in cls._QUERY_TYPE_HINTS.items():
            if q.startswith(prefix + " ") or f" {prefix} " in q:
                return etype
        # Special case: "the actress who"/"the director who" → PERSON
        if re.search(r"\bthe\s+\w+\s+who\b", q):
            return "PERSON"
        return None

    @staticmethod
    def _extract_bridge_entities(
        chunks: List[str], exclude: List[str], query: str = "",
    ) -> List[str]:
        """
        Extract candidate bridge entity names from retrieved text chunks.

        Three-pass heuristic (§12.32):

        Pass 0 — Location-context extraction (highest priority, GPE-only):
          Scans the top chunk for place names introduced by location
          prepositions ("in the city of X", "capital of X", "in X").
          Gated to queries whose expected type is GPE.

        Pass 1 — Surname-anchor search (higher precision):
          For each known entity (e.g. "Kasper Schmeichel"), look for its
          surname-length token ("Schmeichel", ≥6 chars) in ALL provided
          chunks using a unicode-aware pattern. Recovers names like
          "Peter Schmeichel" from stored text containing "Peter Bolesław
          Schmeichel" (where ł breaks ASCII-only _PROPER_NOUN_RE).

        Pass 2 — General proper-noun fallback, query-relevance ranked:
          Falls back to _PROPER_NOUN_RE over all chunks if Pass 1 yields no
          results. Each candidate is scored via _score_bridge_candidate
          and the top 3 are returned.
        """
        _LOCATION_CTX_RE = re.compile(
            r"(?:in\s+the\s+(?:city|town|village|capital|region|province|district)\s+of"
            r"|capital\s+of"
            r"|(?:in|at|near|of)\s+)"
            r"([A-Z][a-z]{2,}(?:[- ][A-Z][a-z]+)*)",
            re.UNICODE,
        )

        exclude_lower = {e.lower() for e in exclude}
        seen: set = set()
        candidates: List[str] = []

        # ── Pass 0: location-context extraction (GPE queries only) ────────
        expected_type = AgenticController._detect_expected_type(query)
        if chunks and expected_type == "GPE":
            for m in _LOCATION_CTX_RE.finditer(chunks[0]):
                place = m.group(1).strip()
                place_lower = place.lower()
                if (
                    place_lower not in exclude_lower
                    and place_lower not in seen
                    and len(place) >= 4
                ):
                    seen.add(place_lower)
                    candidates.append(place)

        if candidates:
            return candidates[:3]

        # ── Passes 1 & 2 unified (C2): one confidence-scored candidate pool ──
        # The former design ran a surname-anchor pass that RETURNED EARLY on
        # first match, so a low-precision reconstruction (e.g. a spurious
        # "Salisbury Gardens" built from an exclude entity's surname) could
        # preempt a stronger general-proper-noun candidate (the real "Thomas
        # Mawson") that the second pass would have found. Scoring both
        # generators in a single pool removes that priority-on-specificity
        # short-circuit: candidates compete on the same scoring function and
        # the strongest wins regardless of which generator proposed it.
        expected_type = AgenticController._detect_expected_type(query)

        # Substring-aware exclusion: a compound exclude entity ("Salisbury
        # Woodland Gardens") also excludes its multi-token sub-phrases so a
        # partial variant cannot be proposed as a "new" bridge.
        excluded_subphrases: set = set(exclude_lower)
        for known in exclude:
            tokens = known.split()
            for i in range(len(tokens)):
                for j in range(i + 2, len(tokens) + 1):
                    if tokens[i].lower() in {"in", "on", "of", "at", "by", "the", "a", "an"}:
                        continue
                    if tokens[j - 1].lower() in {"in", "on", "of", "at", "by", "the", "a", "an"}:
                        continue
                    excluded_subphrases.add(" ".join(tokens[i:j]).lower())

        # Candidate generators → (candidate_text, source_chunk, chunk_rank).
        # Chunks arrive in RRF rank order, so the list index IS the rank.
        proposals: List[Tuple[str, str, int]] = []

        # Generator A — surname-anchor reconstruction (former Pass 1).
        for known in exclude:
            tokens = known.split()
            if len(tokens) not in (2, 3):
                continue
            if any(t.lower() in {"in", "on", "of", "at", "by", "the", "a", "an"}
                   for t in tokens):
                continue
            surname = tokens[-1]
            if len(surname) < 6:
                continue
            pat = re.compile(
                r"\b([A-Z][^\s,.()\[\]]{1,})\s+(?:[A-Z][^\s,.()\[\]]+\s+)?"
                + re.escape(surname)
                + r"\b",
                re.UNICODE,
            )
            for rank, chunk in enumerate(chunks):
                for m in pat.finditer(chunk):
                    first = m.group(1)
                    full = f"{first} {surname}"
                    if (
                        len(full) > 4
                        and first not in {"The", "A", "An", "This", "In", "Of"}
                        and ":" not in first
                    ):
                        proposals.append((full, chunk, rank))

        # Generator B — general proper-noun (former Pass 2).
        for rank, chunk in enumerate(chunks):
            for m in _PROPER_NOUN_RE.finditer(chunk):
                phrase = m.group(1)
                if len(phrase) > 4:
                    proposals.append((phrase, chunk, rank))

        # Score the merged pool with a single function; keep the best-scoring
        # (best-rank) instance of each distinct candidate.
        best: Dict[str, Tuple[float, str]] = {}
        for cand, chunk, rank in proposals:
            cl = cand.lower()
            if cl in excluded_subphrases or cl in seen:
                continue
            score = AgenticController._score_bridge_candidate(
                cand, chunk, query, expected_type, chunk_rank=rank,
            )
            if cl not in best or score > best[cl][0]:
                best[cl] = (score, cand)

        scored = sorted(best.values(), key=lambda x: -x[0])

        # ── C4: abstention floor ──────────────────────────────────────────
        # A candidate scoring 0 was either not found in its chunk or had no
        # query proximity. Returning a confidently-wrong bridge actively
        # misdirects hop-2 retrieval (and reranker hints), which is worse than
        # returning none — hop-2 then falls back to its un-rewritten sub-query.
        # So when the BEST candidate scores <= 0, abstain.
        if not scored or scored[0][0] <= 0.0:
            return []
        return [text for score, text in scored[:3] if score > 0.0]

    @staticmethod
    def _rewrite_hop_query_with_bridges(
        sub_query: str, bridges: List[str],
    ) -> str:
        """
        §12.37 Fix F: inject resolved bridge entities into a hop's sub-query.

        Iterative multi-hop retrieval relies on each hop being able to find
        its supporting paragraph in the index. Planner-generated sub-queries
        are written before retrieval runs, so Hop-2 sub-queries are usually
        under-specified (e.g. "What is the population?" with no entity).
        After Hop-1 resolves a bridge entity (e.g. "Strasbourg"), Hop-2 must
        be rewritten to "What is the population of Strasbourg?" before the
        retriever sees it.

        Heuristic (conservative — only fires when SAFE):
          - If sub_query already mentions any of the bridge entities, do
            nothing (no double-injection).
          - Otherwise append `" — about <bridge_1>, <bridge_2>"` to the
            sub-query.

        Refs: IRCoT (Trivedi et al., 2023, ACL, arXiv:2212.10509);
              Self-Ask (Press et al., 2022, EMNLP).
        """
        if not bridges or not sub_query:
            return sub_query
        sq_lower = sub_query.lower()
        new_bridges = [b for b in bridges if b.lower() not in sq_lower]
        if not new_bridges:
            return sub_query
        injection = ", ".join(new_bridges[:3])
        rewritten = f"{sub_query.rstrip(' ?.')} — about {injection}"
        logger.info(
            "[BridgeRewrite] %r → %r",
            sub_query[:60], rewritten[:80],
        )
        return rewritten
