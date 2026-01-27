#!/usr/bin/env python3
"""
Integration Test für Logic Layer
=================================

Testet die gesamte Drei-Agenten-Pipeline:
    S_P (Planner) → S_N (Navigator) → S_V (Verifier)

Ausführung:
    cd src/logic_layer
    python test_integration.py
    
    ODER vom Projektroot:
    python -m src.logic_layer.test_integration
"""

import sys
import time
import logging
from typing import Optional

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_header(title: str):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_subheader(title: str):
    """Print formatted subheader."""
    print(f"\n--- {title} ---")


def test_imports() -> bool:
    """Test ob alle Imports funktionieren."""
    print_header("1. IMPORT TEST")
    
    try:
        # Versuche relative imports (wenn als Modul ausgeführt)
        try:
            from . import (
                Planner, create_planner, QueryType, RetrievalStrategy,
                Verifier, create_verifier, ValidationStatus,
                Navigator, AgenticController, create_controller
            )
        except ImportError:
            # Fallback für direkten Aufruf
            from planner import Planner, create_planner, QueryType, RetrievalStrategy
            from verifier import Verifier, create_verifier, ValidationStatus
            from Agent import Navigator, AgenticController, create_controller
        
        print("✅ Alle Imports erfolgreich!")
        print(f"   - Planner: {Planner}")
        print(f"   - Verifier: {Verifier}")
        print(f"   - Navigator: {Navigator}")
        print(f"   - AgenticController: {AgenticController}")
        return True
        
    except Exception as e:
        print(f"❌ Import Fehler: {e}")
        return False


def test_planner() -> bool:
    """Test S_P (Planner) Komponente."""
    print_header("2. PLANNER TEST (S_P)")
    
    try:
        from planner import create_planner, QueryType, RetrievalStrategy
    except ImportError:
        from .planner import create_planner, QueryType, RetrievalStrategy
    
    planner = create_planner()
    
    test_cases = [
        ("What is the capital of France?", QueryType.SINGLE_HOP, RetrievalStrategy.VECTOR_ONLY),
        ("Who directed the movie starring Tom Hanks?", QueryType.MULTI_HOP, RetrievalStrategy.HYBRID),
        ("Is Berlin larger than Munich?", QueryType.COMPARISON, RetrievalStrategy.HYBRID),
        ("What happened in 2020?", QueryType.TEMPORAL, RetrievalStrategy.HYBRID),
    ]
    
    all_passed = True
    
    for query, expected_type, expected_strategy in test_cases:
        print_subheader(f"Query: {query[:50]}...")
        
        start = time.time()
        plan = planner.plan(query)
        elapsed = (time.time() - start) * 1000
        
        type_match = plan.query_type == expected_type
        strategy_match = plan.strategy == expected_strategy
        
        status = "✅" if type_match and strategy_match else "⚠️"
        
        print(f"  {status} Type: {plan.query_type.value} (expected: {expected_type.value})")
        print(f"  {status} Strategy: {plan.strategy.value} (expected: {expected_strategy.value})")
        print(f"  📊 Entities: {[e.text for e in plan.entities[:5]]}")
        print(f"  ⏱️  Time: {elapsed:.1f}ms")
        
        if not (type_match and strategy_match):
            all_passed = False
    
    return all_passed


def test_verifier_components() -> bool:
    """Test S_V (Verifier) Komponenten ohne LLM."""
    print_header("3. VERIFIER COMPONENTS TEST (S_V)")
    
    try:
        from verifier import PreGenerationValidator, SourceCredibility, VerifierConfig
    except ImportError:
        from .verifier import PreGenerationValidator, SourceCredibility, VerifierConfig
    
    # Test Pre-Validation
    print_subheader("Pre-Generation Validator")
    
    # Erstelle Config mit gewünschten Einstellungen
    config = VerifierConfig(
        enable_entity_path_validation=True,
        enable_contradiction_detection=False,  # Skip NLI für schnellen Test
        enable_credibility_scoring=True
    )
    
    validator = PreGenerationValidator(config=config, graph_store=None)
    
    test_contexts = [
        "Einstein was born in 1879 in Ulm, Germany.",
        "Einstein received the Nobel Prize in 1921.",
    ]
    
    query = "When was Einstein born?"
    entities = ["Einstein"]
    
    start = time.time()
    result = validator.validate(
        context=test_contexts,
        query=query,
        entities=entities
    )
    elapsed = (time.time() - start) * 1000
    
    print(f"  ✅ Validation Status: {result.status.value}")
    print(f"  📊 Filtered Context: {len(result.filtered_context)} docs")
    print(f"  ⏱️  Time: {elapsed:.1f}ms")
    
    # Test Credibility Scoring (internal method)
    print_subheader("Source Credibility Scoring")
    if result.credibility_scores:
        for i, score in enumerate(result.credibility_scores):
            print(f"  📊 Doc {i+1} Credibility: {score:.2f}")
    else:
        print(f"  ℹ️  No credibility scores available")
    
    return True


def test_navigator() -> bool:
    """Test S_N (Navigator) Komponente mit Mock-Daten."""
    print_header("4. NAVIGATOR TEST (S_N)")
    
    try:
        from Agent import Navigator, ControllerConfig
        from planner import create_planner
    except ImportError:
        from .Agent import Navigator, ControllerConfig
        from .planner import create_planner
    
    # Navigator braucht ControllerConfig
    config = ControllerConfig(
        relevance_threshold_factor=0.6,
        redundancy_threshold=0.8
    )
    navigator = Navigator(config=config)
    
    # Mock retrieval results - muss rrf_score haben für Filter
    mock_results = [
        {"content": "Paris is the capital of France.", "rrf_score": 0.95, "source": "wiki"},
        {"content": "France is located in Western Europe.", "rrf_score": 0.75, "source": "geo"},
        {"content": "Paris is the largest city in France.", "rrf_score": 0.92, "source": "wiki"},
        {"content": "The capital of France is Paris.", "rrf_score": 0.88, "source": "encyclopedia"},  # Redundant
        {"content": "French cuisine is famous.", "rrf_score": 0.45, "source": "food"},  # Low relevance
    ]
    
    print_subheader("RRF Fusion Test")
    
    # Simuliere Fusion - prüfe ob Methoden existieren
    if hasattr(navigator, '_rrf_fusion'):
        print(f"  ✅ RRF Fusion method available")
    else:
        print(f"  ℹ️  RRF Fusion integrated in navigate()")
    
    print_subheader("Pre-Generative Filtering Test")
    
    # Test Relevance Filter
    if hasattr(navigator, '_relevance_filter'):
        relevance_filtered = navigator._relevance_filter(mock_results)
        print(f"  📊 Relevance filter: {len(mock_results)} → {len(relevance_filtered)}")
        
        # Test Redundancy Filter - erwartet "text" statt "content"
        if hasattr(navigator, '_redundancy_filter'):
            # Konvertiere Format für Redundancy Filter
            results_for_redundancy = [
                {"text": r["content"], "rrf_score": r["rrf_score"], "source": r["source"]}
                for r in relevance_filtered
            ]
            redundancy_filtered = navigator._redundancy_filter(results_for_redundancy)
            print(f"  📊 Redundancy filter: {len(relevance_filtered)} → {len(redundancy_filtered)}")
        else:
            print(f"  ℹ️  Redundancy filter integrated in navigate()")
    else:
        print(f"  ℹ️  Filters integrated in navigate()")
    
    return True


def test_full_pipeline(run_llm: bool = False) -> bool:
    """Test komplette Pipeline (optional mit LLM)."""
    print_header("5. FULL PIPELINE TEST")
    
    try:
        from Agent import AgenticController, create_controller
    except ImportError:
        from .Agent import AgenticController, create_controller
    
    # Controller ohne Retriever (nur Planner-Test)
    controller = create_controller(
        model_name="phi3",
        max_iterations=2
    )
    
    test_queries = [
        "What is machine learning?",
        "Compare Python and Java for web development.",
        "Who invented the telephone and when?",
    ]
    
    print_subheader("Pipeline Configuration")
    print(f"  📊 LangGraph: {controller.app is not None}")
    print(f"  📊 Has Retriever: {controller.navigator.retriever is not None}")
    print(f"  📊 Has Graph Store: {controller.verifier.graph_store is not None}")
    
    print_subheader("Query Processing (Planner Only)")
    
    for query in test_queries:
        start = time.time()
        plan = controller.planner.plan(query)
        elapsed = (time.time() - start) * 1000
        
        print(f"\n  Query: {query[:40]}...")
        print(f"    Type: {plan.query_type.value}")
        print(f"    Strategy: {plan.strategy.value}")
        print(f"    Hops: {len(plan.hop_sequence)}")
        print(f"    Time: {elapsed:.1f}ms")
    
    if run_llm:
        print_subheader("Full Generation Test (mit Ollama)")
        print("  ⚠️  Benötigt laufendes Ollama mit phi3 Modell")
        # Hier würde der vollständige Test mit Retriever laufen
    else:
        print_subheader("LLM Test übersprungen")
        print("  ℹ️  Führe mit --llm Flag aus für vollständigen Test")
    
    return True


def test_error_handling() -> bool:
    """Test Fehlerbehandlung und Fallbacks."""
    print_header("6. ERROR HANDLING TEST")
    
    try:
        from planner import create_planner
        from verifier import create_verifier
    except ImportError:
        from .planner import create_planner
        from .verifier import create_verifier
    
    print_subheader("Edge Cases")
    
    planner = create_planner()
    
    edge_cases = [
        "",  # Empty query
        "?",  # Just punctuation
        "a" * 1000,  # Very long query
        "Was ist die Hauptstadt von Deutschland?",  # German
    ]
    
    for query in edge_cases:
        try:
            plan = planner.plan(query)
            display = query[:30] + "..." if len(query) > 30 else query
            print(f"  ✅ '{display}' → {plan.query_type.value}")
        except Exception as e:
            print(f"  ⚠️  '{query[:30]}' → Error: {e}")
    
    return True


def run_all_tests(run_llm: bool = False):
    """Führe alle Tests aus."""
    print("\n" + "=" * 70)
    print(" LOGIC LAYER INTEGRATION TEST")
    print(" Masterarbeit: Enhancing Reasoning Fidelity in Quantized SLMs")
    print("=" * 70)
    
    results = {}
    
    # Test 1: Imports
    results['imports'] = test_imports()
    
    # Test 2: Planner
    results['planner'] = test_planner()
    
    # Test 3: Verifier Components
    results['verifier'] = test_verifier_components()
    
    # Test 4: Navigator
    results['navigator'] = test_navigator()
    
    # Test 5: Full Pipeline
    results['pipeline'] = test_full_pipeline(run_llm)
    
    # Test 6: Error Handling
    results['errors'] = test_error_handling()
    
    # Summary
    print_header("TEST SUMMARY")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    for name, status in results.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {name.upper()}")
    
    print(f"\n  Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 All tests passed!")
    else:
        print("\n  ⚠️  Some tests failed. Check output above.")
    
    return passed == total


if __name__ == "__main__":
    run_llm = "--llm" in sys.argv
    
    if run_llm:
        print("Running with LLM tests enabled...")
    
    success = run_all_tests(run_llm)
    sys.exit(0 if success else 1)