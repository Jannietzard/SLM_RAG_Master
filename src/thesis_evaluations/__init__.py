"""
Thesis evaluation suite for:
    "Enhancing Reasoning Fidelity in Quantized Small Language Models:
     Agentic Verification for Hybrid Retrieval-Augmented Generation on
     Resource-Constrained Devices"

This package contains everything needed to produce the empirical chapters
of the thesis. See README.md in this directory for the script-to-chapter
mapping and the suggested execution order.

Modules:
    benchmark_datasets         — core evaluation engine (S_P + S_N + S_V).
    quantization_sweep         — Tier-1A: quantization × model-size matrix.
    agentic_ablation           — Tier-1B: planner/verifier contribution table.
    latency_memory_profile     — Tier-1C: edge-device resource profile.
    thesis_results_aggregator  — produces LaTeX tables + plots from JSONL.
"""
