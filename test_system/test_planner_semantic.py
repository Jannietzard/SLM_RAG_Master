"""
Semantic correctness tests for S_P (Planner).

Tests verify that planner.plan() produces structurally and semantically
correct RetrievalPlans — not just that it runs without errors.

Run without live LLM:
    python -X utf8 -m pytest test_system/test_planner_semantic.py -v
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.logic_layer.planner import (
    Planner, PlannerConfig, QueryType, RetrievalStrategy,
    EntityInfo, RetrievalPlan, create_planner, SPACY_AVAILABLE,
)


@pytest.fixture(scope="module")
def planner():
    return create_planner()  # auto-loads config/settings.yaml


# ── Classification accuracy ────────────────────────────────────────────────────

class TestQueryClassification:

    def test_single_hop_capital(self, planner):
        plan = planner.plan("What is the capital of France?")
        assert plan.query_type == QueryType.SINGLE_HOP
        assert plan.strategy == RetrievalStrategy.VECTOR_ONLY

    def test_comparison_nationality(self, planner):
        plan = planner.plan("Were Scott Derrickson and Ed Wood of the same nationality?")
        assert plan.query_type == QueryType.COMPARISON, \
            f"Expected COMPARISON, got {plan.query_type.value}"
        assert plan.strategy == RetrievalStrategy.HYBRID

    def test_multihop_director_film(self, planner):
        plan = planner.plan("Who is the director of the film that stars Tom Hanks?")
        assert plan.query_type == QueryType.MULTI_HOP

    def test_temporal_year(self, planner):
        plan = planner.plan("What happened after World War 2?")
        assert plan.query_type == QueryType.TEMPORAL

    def test_intersection_both(self, planner):
        plan = planner.plan("Which movies star both Brad Pitt and Leonardo DiCaprio?")
        assert plan.query_type == QueryType.INTERSECTION

    def test_multihop_capital_country_born(self, planner):
        plan = planner.plan("What is the capital of the country where Albert Einstein was born?")
        assert plan.query_type == QueryType.MULTI_HOP


# ── Sub-query quality ──────────────────────────────────────────────────────────

class TestSubQueryQuality:

    def test_comparison_attr_map_rewrite(self, planner):
        """ATTR_MAP must rewrite 'same nationality' into factual lookups."""
        plan = planner.plan("Were Scott Derrickson and Ed Wood of the same nationality?")
        for sq in plan.sub_queries:
            assert "nationality" in sq.lower(), \
                f"ATTR_MAP rewrite failed — sub-query missing 'nationality': {sq}"

    def test_comparison_no_original_query_appended(self, planner):
        """Original query must NOT be appended as a 3rd sub-query (causes RRF cross-query noise)."""
        plan = planner.plan("Were Scott Derrickson and Ed Wood of the same nationality?")
        assert len(plan.sub_queries) == 2, \
            f"Expected 2 sub-queries, got {len(plan.sub_queries)}: {plan.sub_queries}"

    def test_multihop_minimum_two_subqueries(self, planner):
        plan = planner.plan("What is the capital of the country where Albert Einstein was born?")
        assert len(plan.sub_queries) >= 2, \
            f"Multi-hop must produce ≥2 sub-queries, got {plan.sub_queries}"

    def test_sub_queries_are_questions(self, planner):
        plan = planner.plan("Who directed the film starring Tom Hanks?")
        for sq in plan.sub_queries:
            assert sq.endswith("?"), f"Sub-query not a question: {sq!r}"

    def test_single_hop_sub_query_equals_original(self, planner):
        query = "What is the capital of France?"
        plan = planner.plan(query)
        assert plan.sub_queries == [query]


# ── Entity extraction quality ──────────────────────────────────────────────────

class TestEntityExtraction:

    def test_full_name_not_partial(self, planner):
        """'Scott Derrickson' must be extracted as a unit, not as 'Scott' alone."""
        plan = planner.plan("Were Scott Derrickson and Ed Wood of the same nationality?")
        entity_texts = [e.text for e in plan.entities]
        assert any("Derrickson" in t for t in entity_texts), \
            f"Full name not extracted: {entity_texts}"

    def test_no_spurious_verb_entities(self, planner):
        """'Were' must not appear as an entity."""
        plan = planner.plan("Were Scott Derrickson and Ed Wood of the same nationality?")
        entity_texts_lower = [e.text.lower() for e in plan.entities]
        assert "were" not in entity_texts_lower, \
            f"Spurious verb 'Were' extracted as entity: {entity_texts_lower}"

    def test_entity_count_within_max(self, planner):
        plan = planner.plan("Were Scott Derrickson and Ed Wood of the same nationality?")
        assert len(plan.entities) <= planner.config.max_entities

    def test_bridge_entity_flagged_multihop(self, planner):
        """Bridge detection must not crash; bridge flag must be bool."""
        plan = planner.plan("Who is the director of the film that stars Tom Hanks?")
        for e in plan.entities:
            assert isinstance(e.is_bridge, bool)

    def test_entity_confidence_in_range(self, planner):
        plan = planner.plan("What did Albert Einstein discover?")
        for e in plan.entities:
            assert 0.0 <= e.confidence <= 1.0, \
                f"Entity confidence out of range: {e.text} = {e.confidence}"


# ── RetrievalPlan structural invariants ───────────────────────────────────────

class TestPlanStructure:

    def test_estimated_hops_matches_hop_sequence(self, planner):
        """estimated_hops must always equal len(hop_sequence)."""
        for query in [
            "What is the capital of France?",
            "Is Berlin older than Munich?",
            "Who directed the film that won the Oscar?",
            "What is the capital of the country where Einstein was born?",
        ]:
            plan = planner.plan(query)
            assert plan.estimated_hops == len(plan.hop_sequence), \
                f"estimated_hops={plan.estimated_hops} != len(hop_sequence)={len(plan.hop_sequence)} for {query!r}"

    def test_empty_query_does_not_crash(self, planner):
        plan = planner.plan("")
        assert isinstance(plan, RetrievalPlan)
        assert plan.query_type == QueryType.SINGLE_HOP
        assert plan.estimated_hops == 0
        assert plan.hop_sequence == []

    def test_none_query_does_not_crash(self, planner):
        """plan(None) must return a valid empty plan after the None-guard fix."""
        plan = planner.plan(None)
        assert isinstance(plan, RetrievalPlan)
        assert plan.query_type == QueryType.SINGLE_HOP
        assert plan.estimated_hops == 0

    def test_to_dict_serialisable(self, planner):
        plan = planner.plan("Who is the director of the film that stars Tom Hanks?")
        d = plan.to_dict()
        json_str = json.dumps(d)  # must not raise TypeError
        assert isinstance(json_str, str)

    def test_to_json_valid(self, planner):
        plan = planner.plan("What is the capital of France?")
        json_str = plan.to_json()
        parsed = json.loads(json_str)
        assert parsed["original_query"] == "What is the capital of France?"

    def test_temporal_constraint_extracted(self, planner):
        plan = planner.plan("Who was president in 1990?")
        assert "years" in plan.constraints, \
            f"Year not extracted into constraints: {plan.constraints}"
        assert "1990" in plan.constraints["years"]

    def test_confidence_in_range(self, planner):
        plan = planner.plan("Is Berlin older than Munich?")
        assert 0.0 <= plan.confidence <= 1.0

    def test_hop_step_ids_sequential(self, planner):
        plan = planner.plan("What is the capital of the country where Einstein was born?")
        for i, hop in enumerate(plan.hop_sequence):
            assert hop.step_id == i, \
                f"step_id={hop.step_id} not sequential at index {i}"


# ── Factory and settings compliance ──────────────────────────────────────────

class TestFactoryAndSettings:

    def test_create_planner_loads_settings(self):
        """Factory must auto-load settings.yaml — not use hardcoded defaults."""
        p = create_planner()
        assert p.config.min_entity_confidence == 0.7
        assert p.config.max_entities == 10
        assert p.config.classifier_spacy_weight == 1.5
        assert p.config.classifier_confidence_cap == 0.95

    def test_planner_config_from_yaml(self):
        cfg = {
            "planner": {
                "min_entity_confidence": 0.5,
                "max_entities": 5,
                "classifier_spacy_weight": 2.0,
            },
            "ingestion": {"spacy_model": "en_core_web_sm"},
        }
        config = PlannerConfig.from_yaml(cfg)
        assert config.min_entity_confidence == 0.5
        assert config.max_entities == 5
        assert config.classifier_spacy_weight == 2.0

    def test_planning_time_recorded(self):
        p = create_planner()
        plan = p.plan("What is the capital of France?")
        assert "planning_time_ms" in plan.metadata
        assert plan.metadata["planning_time_ms"] >= 0


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    p = create_planner()

    print("=" * 60)
    print("PLANNER SEMANTIC SMOKE CHECK")
    print(f"SpaCy: {'available' if SPACY_AVAILABLE else 'unavailable'}")
    print("=" * 60)

    cases = [
        ("What is the capital of France?", QueryType.SINGLE_HOP),
        ("Were Scott Derrickson and Ed Wood of the same nationality?", QueryType.COMPARISON),
        ("What is the capital of the country where Einstein was born?", QueryType.MULTI_HOP),
        ("Who was president in 1990?", QueryType.TEMPORAL),
    ]
    for q, expected in cases:
        plan = p.plan(q)
        ok = "OK  " if plan.query_type == expected else "FAIL"
        print(f"[{ok}] {q[:60]}")
        print(f"       type={plan.query_type.value}  subs={plan.sub_queries}")
    print("=" * 60)
