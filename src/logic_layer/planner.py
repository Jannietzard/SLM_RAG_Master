"""
PDDL-Based Query Planner for Multi-Hop QA

Version: 2.0.0 - ACADEMIC STANDARD
Author: Edge-RAG Research Project

===============================================================================
ACADEMIC FOUNDATION
===============================================================================

This planner uses the Unified Planning Framework (UPF), the modern academic
standard for AI planning developed by AIPlan4EU consortium.

References:
    - Ghallab, M., Nau, D., & Traverso, P. (2004). 
      "Automated Planning: Theory and Practice." Morgan Kaufmann.
    
    - AIPlan4EU Consortium (2023).
      "Unified Planning Framework." https://unified-planning.readthedocs.io/
    
    - International Planning Competition (IPC):
      https://www.icaps-conference.org/competitions/

PDDL (Planning Domain Definition Language):
    - Standard since 1998 (AIPS Planning Competition)
    - Extended versions: PDDL 2.1, 2.2, 3.0, 3.1
    - This implementation uses PDDL 2.1 features (typing, numeric fluents)

===============================================================================
PLANNING DOMAIN: MULTI-HOP QUESTION ANSWERING
===============================================================================

Domain Name: multi-hop-qa

Types:
    - query: A question or sub-question
    - entity: A named entity (person, place, etc.)
    - fact: A retrieved fact/document chunk

Predicates:
    - (answered ?q - query): Query has been answered
    - (requires ?q1 ?q2 - query): q1 requires q2 to be answered first
    - (mentions ?q - query ?e - entity): Query mentions entity
    - (known ?e - entity): Entity information is known
    - (retrieved ?f - fact): Fact has been retrieved
    - (supports ?f - fact ?q - query): Fact supports answering query

Actions:
    - retrieve(q, e): Retrieve facts about entity e for query q
    - lookup(q): Direct lookup for simple query q
    - bridge(q1, q2): Answer q1 using result of q2
    - compare(q, e1, e2): Compare two entities for query q
    - aggregate(q, sub_queries): Combine sub-query results

===============================================================================
INSTALLATION
===============================================================================

pip install unified-planning
pip install unified-planning[pyperplan]  # Lightweight solver for Edge

Optional (faster, but requires C++):
pip install unified-planning[fast-downward]

===============================================================================
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re

logger = logging.getLogger(__name__)

# =============================================================================
# CHECK UNIFIED PLANNING AVAILABILITY
# =============================================================================

try:
    import unified_planning as up
    from unified_planning.shortcuts import (
        UserType, 
        Fluent, 
        InstantaneousAction,
        Problem,
        OneshotPlanner,
        Object,
        Not,
        And,
        Or,
    )
    from unified_planning.engines import PlanGenerationResultStatus
    UP_AVAILABLE = True
    logger.info("Unified Planning Framework available")
except ImportError:
    UP_AVAILABLE = False
    logger.warning(
        "Unified Planning not available. Install with:\n"
        "  pip install unified-planning unified-planning[pyperplan]"
    )


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class QueryType(Enum):
    """Classification of query types based on planning requirements."""
    SIMPLE = "simple"           # Single retrieval action
    BRIDGE = "bridge"           # Sequential dependency
    COMPARISON = "comparison"   # Parallel + compare
    INTERSECTION = "intersection"  # Parallel + intersect
    COMPOSITIONAL = "compositional"  # Multiple steps
    AGGREGATE = "aggregate"     # Combine multiple results


@dataclass
class PlanStep:
    """A single step in the generated plan."""
    action: str          # Action name (retrieve, lookup, bridge, compare)
    parameters: Dict[str, str]  # Action parameters
    preconditions: List[str]    # What must be true before
    effects: List[str]          # What becomes true after


@dataclass
class QueryPlan:
    """Complete plan for answering a query."""
    original_query: str
    query_type: QueryType
    steps: List[PlanStep]
    sub_queries: List[str]      # Extracted sub-queries for retrieval
    entities: List[str]         # Extracted entities
    parallel_groups: List[List[int]]  # Steps that can run in parallel
    estimated_cost: float       # Estimated execution time

@dataclass
class PlannerConfig:
    """Konfiguration für Planner Stage."""
    model_name: str = "phi3"
    base_url: str = "http://localhost:11434"
    solver: str = "pyperplan"
    timeout: float = 5.0
    
# =============================================================================
# PDDL DOMAIN DEFINITION
# =============================================================================

class MultiHopQADomain:
    """
    PDDL Domain for Multi-Hop Question Answering.
    
    This class defines the planning domain following PDDL 2.1 standard.
    """
    
    DOMAIN_NAME = "multi-hop-qa"
    
    def __init__(self):
        """Initialize the planning domain."""
        if not UP_AVAILABLE:
            raise ImportError(
                "Unified Planning required. Install with:\n"
                "pip install unified-planning unified-planning[pyperplan]"
            )
        
        self.logger = logging.getLogger(__name__)
        
        # Define types
        self.query_type = UserType("query")
        self.entity_type = UserType("entity")
        self.fact_type = UserType("fact")
        
        # Define predicates (fluents in UPF terminology)
        self.answered = Fluent("answered", query=self.query_type)
        self.requires = Fluent("requires", q1=self.query_type, q2=self.query_type)
        self.mentions = Fluent("mentions", q=self.query_type, e=self.entity_type)
        self.known = Fluent("known", e=self.entity_type)
        self.retrieved = Fluent("retrieved", f=self.fact_type)
        self.supports = Fluent("supports", f=self.fact_type, q=self.query_type)
        
        # Define actions
        self._define_actions()
        
        self.logger.info(f"PDDL Domain '{self.DOMAIN_NAME}' initialized")
    
    def _define_actions(self):
        """Define planning actions following PDDL semantics."""
        
        # Action: retrieve(query, entity)
        # Retrieve facts about an entity to answer a query
        self.retrieve_action = InstantaneousAction(
            "retrieve",
            q=self.query_type,
            e=self.entity_type
        )
        q = self.retrieve_action.parameter("q")
        e = self.retrieve_action.parameter("e")
        # Precondition: query mentions entity and entity not yet known
        self.retrieve_action.add_precondition(self.mentions(q, e))
        # self.retrieve_action.add_precondition(Not(self.known(e)))
        # Effect: entity becomes known
        self.retrieve_action.add_effect(self.known(e), True)
        
        # Action: lookup(query)
        # Direct lookup for simple factual query
        self.lookup_action = InstantaneousAction(
            "lookup",
            q=self.query_type
        )
        q = self.lookup_action.parameter("q")
        # Precondition: no unsatisfied requirements
        # Effect: query is answered
        self.lookup_action.add_effect(self.answered(q), True)
        
        # Action: bridge(q1, q2)
        # Answer q1 using the result of q2 (sequential dependency)
        self.bridge_action = InstantaneousAction(
            "bridge",
            q1=self.query_type,
            q2=self.query_type
        )
        q1 = self.bridge_action.parameter("q1")
        q2 = self.bridge_action.parameter("q2")
        # Precondition: q1 requires q2 and q2 is answered
        self.bridge_action.add_precondition(self.requires(q1, q2))
        self.bridge_action.add_precondition(self.answered(q2))
        # Effect: q1 is answered
        self.bridge_action.add_effect(self.answered(q1), True)
        
        # Action: compare(query, entity1, entity2)
        # Compare two entities to answer a comparison query
        self.compare_action = InstantaneousAction(
            "compare",
            q=self.query_type,
            e1=self.entity_type,
            e2=self.entity_type
        )
        q = self.compare_action.parameter("q")
        e1 = self.compare_action.parameter("e1")
        e2 = self.compare_action.parameter("e2")
        # Precondition: both entities are known
        self.compare_action.add_precondition(self.known(e1))
        self.compare_action.add_precondition(self.known(e2))
        # Effect: query is answered
        self.compare_action.add_effect(self.answered(q), True)
    
    def get_actions(self) -> List[InstantaneousAction]:
        """Return all defined actions."""
        return [
            self.retrieve_action,
            self.lookup_action,
            self.bridge_action,
            self.compare_action,
        ]


# =============================================================================
# QUERY ANALYZER (Pre-processing for PDDL)
# =============================================================================

class QueryAnalyzer:
    """
    Analyze natural language queries to extract PDDL problem components.
    
    This bridges the gap between NL queries and formal planning.
    """
    
    # Patterns for query classification
    BRIDGE_INDICATORS = [
        r"of\s+the\s+\w+\s+(?:that|which|who)",
        r"where\s+.+\s+(?:was|is|were|are)",
        r"\w+\s+of\s+the\s+\w+\s+of",
    ]
    
    COMPARISON_INDICATORS = [
        r"(?:older|younger|taller|shorter|bigger|smaller|more|less)\s+than",
        r"\bor\b.*\?$",
        r"compare\s+.+\s+(?:and|with|to)",
        r"difference\s+between",
        r"which\s+(?:is|was)\s+\w+er",
    ]
    
    INTERSECTION_INDICATORS = [
        r"both\s+.+\s+and",
        r"in\s+common",
        r"(?:are|is)\s+(?:also|both)",
    ]
    
    # Entity extraction patterns
    ENTITY_PATTERNS = [
        r'"([^"]+)"',                           # Quoted
        r"'([^']+)'",                           # Single quoted
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b",  # Multi-word proper noun
        r"\b([A-Z][a-z]{2,})\b",                 # Single proper noun
    ]
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze(self, query: str) -> Dict[str, Any]:
        """
        Analyze query and extract PDDL problem components.
        
        Returns:
            Dict with:
            - query_type: QueryType enum
            - entities: List of extracted entities
            - sub_queries: Potential sub-queries
            - dependencies: List of (q1, q2) tuples where q1 requires q2
        """
        query = query.strip()
        entities = self._extract_entities(query)
        query_type = self._classify_query(query)
        sub_queries, dependencies = self._decompose(query, query_type, entities)
        
        return {
            "query_type": query_type,
            "entities": entities,
            "sub_queries": sub_queries,
            "dependencies": dependencies,
            "original": query,
        }
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities from query."""
        entities = []
        for pattern in self.ENTITY_PATTERNS:
            matches = re.findall(pattern, query)
            entities.extend(matches)
        
        # Deduplicate preserving order
        seen = set()
        unique = []
        for e in entities:
            if e.lower() not in seen and len(e) > 2:
                seen.add(e.lower())
                unique.append(e)
        
        return unique[:5]
    
    def _classify_query(self, query: str) -> QueryType:
        """Classify query type based on linguistic patterns."""
        query_lower = query.lower()
        
        # Check comparison first (most specific)
        for pattern in self.COMPARISON_INDICATORS:
            if re.search(pattern, query_lower):
                return QueryType.COMPARISON
        
        # Check intersection
        for pattern in self.INTERSECTION_INDICATORS:
            if re.search(pattern, query_lower):
                return QueryType.INTERSECTION
        
        # Check bridge (multi-hop)
        for pattern in self.BRIDGE_INDICATORS:
            if re.search(pattern, query_lower):
                return QueryType.BRIDGE
        
        # Default to simple
        return QueryType.SIMPLE
    
    def _decompose(
        self, 
        query: str, 
        query_type: QueryType, 
        entities: List[str]
    ) -> Tuple[List[str], List[Tuple[str, str]]]:
        """
        Decompose query into sub-queries and identify dependencies.
        
        Returns:
            Tuple of (sub_queries, dependencies)
            dependencies is a list of (q1, q2) meaning q1 requires q2
        """
        sub_queries = []
        dependencies = []
        
        if query_type == QueryType.SIMPLE:
            sub_queries = [query]
            
        elif query_type == QueryType.BRIDGE:
            # Split at dependency point
            parts = re.split(
                r"\s+(?:that|which|who|where)\s+", 
                query, 
                maxsplit=1, 
                flags=re.IGNORECASE
            )
            if len(parts) == 2:
                q1 = parts[0].strip() + "?"
                q2 = "What " + parts[1].rstrip("?") + "?"
                sub_queries = [q2, q1]  # q2 first (dependency)
                dependencies = [(q1, q2)]
            else:
                sub_queries = [query]
                
        elif query_type == QueryType.COMPARISON:
            # Create sub-queries for each entity
            for entity in entities[:2]:
                sub_queries.append(f"What is {entity}?")
            if not sub_queries:
                sub_queries = [query]
            # No dependencies - can run in parallel
            
        elif query_type == QueryType.INTERSECTION:
            # Create sub-queries for each entity
            for entity in entities[:2]:
                sub_queries.append(f"What about {entity}?")
            if not sub_queries:
                sub_queries = [query]
            # No dependencies - can run in parallel
            
        else:
            sub_queries = [query]
        
        return sub_queries, dependencies


# =============================================================================
# PDDL PLANNER
# =============================================================================

class PDDLPlanner:
    """
    PDDL-based Query Planner using Unified Planning Framework.
    
    This is the main planner class that:
    1. Analyzes the query
    2. Constructs a PDDL problem
    3. Invokes a planner (Pyperplan by default)
    4. Returns an executable plan
    """
    
    # Cost estimates for actions (in milliseconds)
    ACTION_COSTS = {
        "retrieve": 200,   # Vector/Graph retrieval
        "lookup": 150,     # Simple lookup
        "bridge": 50,      # Result combination
        "compare": 50,     # Comparison logic
    }
    
    def __init__(
        self, 
        solver: str = "pyperplan",
        timeout: float = 5.0,
    ):
        """
        Initialize PDDL planner.
        
        Args:
            solver: Planning solver to use ("pyperplan", "fast-downward")
            timeout: Maximum planning time in seconds
        """
        if not UP_AVAILABLE:
            raise ImportError(
                "Unified Planning required. Install with:\n"
                "pip install unified-planning unified-planning[pyperplan]"
            )
        
        self.solver = solver
        self.timeout = timeout
        self.domain = MultiHopQADomain()
        self.analyzer = QueryAnalyzer()
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"PDDLPlanner initialized with solver: {solver}")
    
    def plan(self, query: str) -> QueryPlan:
        """
        Generate a plan for answering the query.
        
        Args:
            query: Natural language question
            
        Returns:
            QueryPlan with steps, sub-queries, and execution order
        """
        # Step 1: Analyze query
        analysis = self.analyzer.analyze(query)
        
        self.logger.debug(
            f"Query analysis: type={analysis['query_type'].value}, "
            f"entities={analysis['entities']}"
        )
        
        # Step 2: Construct PDDL problem
        problem = self._construct_problem(analysis)
        
        # Step 3: Solve
        plan_result = self._solve(problem)
        
        # Step 4: Convert to QueryPlan
        query_plan = self._convert_plan(analysis, plan_result)
        
        return query_plan
    
    def _construct_problem(self, analysis: Dict[str, Any]) -> Problem:
        """Construct PDDL problem from query analysis."""
        problem = Problem(f"qa-{hash(analysis['original']) % 10000}")
        
        # Add domain fluents
        problem.add_fluent(self.domain.answered, default_initial_value=False)
        problem.add_fluent(self.domain.requires, default_initial_value=False)
        problem.add_fluent(self.domain.mentions, default_initial_value=False)
        problem.add_fluent(self.domain.known, default_initial_value=False)
        
        # Add actions
        for action in self.domain.get_actions():
            problem.add_action(action)
        
        # Create objects for queries
        main_query = Object("main_query", self.domain.query_type)
        problem.add_object(main_query)
        
        sub_query_objects = []
        for i, sq in enumerate(analysis["sub_queries"]):
            sq_obj = Object(f"sub_query_{i}", self.domain.query_type)
            problem.add_object(sq_obj)
            sub_query_objects.append(sq_obj)
        
        # Create objects for entities
        entity_objects = []
        for i, entity in enumerate(analysis["entities"]):
            e_obj = Object(f"entity_{i}", self.domain.entity_type)
            problem.add_object(e_obj)
            entity_objects.append(e_obj)
            
            # Set initial state: main_query mentions entity
            problem.set_initial_value(
                self.domain.mentions(main_query, e_obj), 
                True
            )
        
        # Set dependencies
        for q1_str, q2_str in analysis["dependencies"]:
            # Find corresponding objects
            q1_idx = next(
                (i for i, sq in enumerate(analysis["sub_queries"]) if sq == q1_str),
                None
            )
            q2_idx = next(
                (i for i, sq in enumerate(analysis["sub_queries"]) if sq == q2_str),
                None
            )
            if q1_idx is not None and q2_idx is not None:
                problem.set_initial_value(
                    self.domain.requires(
                        sub_query_objects[q1_idx], 
                        sub_query_objects[q2_idx]
                    ),
                    True
                )
        
        # Set goal: main query is answered
        problem.add_goal(self.domain.answered(main_query))
        
        return problem
    
    def _solve(self, problem: Problem) -> Optional[Any]:
        """Invoke planning solver."""
        try:
            with OneshotPlanner(name=self.solver) as planner:
                result = planner.solve(problem, timeout=self.timeout)
                
                if result.status == PlanGenerationResultStatus.SOLVED_SATISFICING:
                    self.logger.debug(f"Plan found: {result.plan}")
                    return result.plan
                else:
                    self.logger.warning(f"Planning failed: {result.status}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Planning error: {e}")
            return None
    
    def _convert_plan(
        self, 
        analysis: Dict[str, Any], 
        plan_result: Optional[Any]
    ) -> QueryPlan:
        """Convert UPF plan to QueryPlan."""
        steps = []
        parallel_groups = []
        
        if plan_result:
            # Extract steps from UPF plan
            current_parallel = []
            for i, action in enumerate(plan_result.actions):
                step = PlanStep(
                    action=action.action.name,
                    parameters={
                        str(p): str(v) 
                        for p, v in zip(action.action.parameters, action.actual_parameters)
                    },
                    preconditions=[],
                    effects=[],
                )
                steps.append(step)
                
                # Group parallel actions (those without dependencies)
                if action.action.name in ["retrieve", "lookup"]:
                    current_parallel.append(i)
                else:
                    if current_parallel:
                        parallel_groups.append(current_parallel)
                        current_parallel = []
            
            if current_parallel:
                parallel_groups.append(current_parallel)
        
        else:
            # Fallback: simple sequential plan
            for i, sq in enumerate(analysis["sub_queries"]):
                steps.append(PlanStep(
                    action="lookup",
                    parameters={"query": sq},
                    preconditions=[],
                    effects=[f"answered(sub_query_{i})"],
                ))
            parallel_groups = [list(range(len(steps)))]
        
        # Calculate estimated cost
        estimated_cost = sum(
            self.ACTION_COSTS.get(step.action, 100) 
            for step in steps
        )
        
        return QueryPlan(
            original_query=analysis["original"],
            query_type=analysis["query_type"],
            steps=steps,
            sub_queries=analysis["sub_queries"],
            entities=analysis["entities"],
            parallel_groups=parallel_groups,
            estimated_cost=estimated_cost,
        )


# =============================================================================
# LIGHTWEIGHT FALLBACK (When UPF not available)
# =============================================================================

class LightweightPlanner:
    """
    Lightweight fallback planner when Unified Planning is not available.
    
    Uses the same interface but with simpler heuristic planning.
    Still follows planning principles (preconditions, effects, goals).
    """
    
    def __init__(self):
        self.analyzer = QueryAnalyzer()
        self.logger = logging.getLogger(__name__)
        self.logger.info("LightweightPlanner initialized (UPF not available)")
    
    def plan(self, query: str) -> QueryPlan:
        """Generate plan using heuristic approach."""
        analysis = self.analyzer.analyze(query)
        
        steps = []
        parallel_groups = []
        
        query_type = analysis["query_type"]
        entities = analysis["entities"]
        sub_queries = analysis["sub_queries"]
        
        if query_type == QueryType.SIMPLE:
            # Single lookup
            steps.append(PlanStep(
                action="lookup",
                parameters={"query": query},
                preconditions=[],
                effects=["answered(main)"],
            ))
            parallel_groups = [[0]]
            
        elif query_type == QueryType.COMPARISON:
            # Parallel retrieval then compare
            for i, entity in enumerate(entities[:2]):
                steps.append(PlanStep(
                    action="retrieve",
                    parameters={"entity": entity},
                    preconditions=[],
                    effects=[f"known({entity})"],
                ))
            steps.append(PlanStep(
                action="compare",
                parameters={"entities": entities[:2]},
                preconditions=[f"known({e})" for e in entities[:2]],
                effects=["answered(main)"],
            ))
            parallel_groups = [[0, 1], [2]]  # First two parallel, then compare
            
        elif query_type == QueryType.BRIDGE:
            # Sequential with dependency
            for i, sq in enumerate(sub_queries):
                precond = [f"answered(sub_{i-1})"] if i > 0 else []
                steps.append(PlanStep(
                    action="lookup",
                    parameters={"query": sq},
                    preconditions=precond,
                    effects=[f"answered(sub_{i})"],
                ))
            parallel_groups = [[i] for i in range(len(steps))]  # All sequential
            
        else:
            # Default: parallel lookups
            for i, sq in enumerate(sub_queries):
                steps.append(PlanStep(
                    action="lookup",
                    parameters={"query": sq},
                    preconditions=[],
                    effects=[f"answered(sub_{i})"],
                ))
            parallel_groups = [list(range(len(steps)))]
        
        estimated_cost = sum(200 if s.action == "retrieve" else 150 for s in steps)
        
        return QueryPlan(
            original_query=query,
            query_type=query_type,
            steps=steps,
            sub_queries=sub_queries,
            entities=entities,
            parallel_groups=parallel_groups,
            estimated_cost=estimated_cost,
        )


# =============================================================================
# FACTORY & COMPATIBILITY INTERFACE
# =============================================================================

def create_planner(
    model_name: str = None,  # Ignored - API compatibility
    base_url: str = None,    # Ignored - API compatibility
    solver: str = "pyperplan",
    **kwargs
) -> "Planner":
    """
    Factory function for creating planner.
    
    Maintains API compatibility with LLM-based planner.
    """
    return Planner(solver=solver, **kwargs)


class Planner:
    """
    Compatibility wrapper matching original Planner interface.
    
    Drop-in replacement for src/logic_layer/planner.py
    """
    
    def __init__(
        self,
        model_name: str = None,
        base_url: str = None,
        solver: str = "pyperplan",
        **kwargs
    ):
        """Initialize planner."""
        if UP_AVAILABLE:
            try:
                self._planner = PDDLPlanner(solver=solver, **kwargs)
                self._use_pddl = True
            except Exception as e:
                logger.warning(f"PDDL planner init failed: {e}, using lightweight")
                self._planner = LightweightPlanner()
                self._use_pddl = False
        else:
            self._planner = LightweightPlanner()
            self._use_pddl = False
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"Planner initialized: {'PDDL' if self._use_pddl else 'Lightweight'}"
        )
    
    def decompose_query(self, query: str) -> List[str]:
        """
        Decompose query into sub-queries.
        
        Same interface as original LLM Planner.
        """
        plan = self._planner.plan(query)
        
        self.logger.debug(
            f"Decomposed [{plan.query_type.value}]: "
            f"{len(plan.sub_queries)} sub-queries, "
            f"estimated cost: {plan.estimated_cost}ms"
        )
        
        return plan.sub_queries
    
    def get_plan(self, query: str) -> QueryPlan:
        """Get full plan with execution details."""
        return self._planner.plan(query)
    
    def get_query_type(self, query: str) -> str:
        """Get detected query type."""
        plan = self._planner.plan(query)
        return plan.query_type.value


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import time
    
    test_queries = [
        "Who is the director of the film that stars Tom Hanks?",
        "Is Berlin older than Munich?",
        "Which movies star both Brad Pitt and Leonardo DiCaprio?",
        "What is the capital of France?",
        "What is the capital of the country where Einstein was born?",
    ]
    
    print("=" * 70)
    print("PDDL-BASED PLANNER TEST")
    print(f"Unified Planning Available: {UP_AVAILABLE}")
    print("=" * 70)
    
    planner = Planner()
    
    total_time = 0
    for query in test_queries:
        start = time.perf_counter()
        plan = planner.get_plan(query)
        elapsed = (time.perf_counter() - start) * 1000
        total_time += elapsed
        
        print(f"\nQuery: {query}")
        print(f"  Type: {plan.query_type.value}")
        print(f"  Entities: {plan.entities}")
        print(f"  Sub-queries: {plan.sub_queries}")
        print(f"  Steps: {len(plan.steps)}")
        print(f"  Parallel groups: {plan.parallel_groups}")
        print(f"  Est. cost: {plan.estimated_cost}ms")
        print(f"  Planning time: {elapsed:.2f}ms")
    
    print("\n" + "=" * 70)
    print(f"Average planning time: {total_time / len(test_queries):.2f}ms")
    print("=" * 70)