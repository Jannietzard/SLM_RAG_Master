"""
diagnose_verbose.py — Full pipeline trace (mega-file).

Runs the complete S_P -> S_N -> S_V pipeline for a single question and
prints the output of EVERY function it reaches. No production code is
modified; all hooks are installed via runtime monkey-patching.

Usage:
    python -X utf8 diagnose_verbose.py --idx 0
    python -X utf8 diagnose_verbose.py --idx 5 --skip-llm
    python -X utf8 diagnose_verbose.py --question "Who founded YG Entertainment?"
    python -X utf8 diagnose_verbose.py --idx 0 --trace-calls   # trace every src/ call
    python -X utf8 diagnose_verbose.py --idx 0 --no-color > trace.txt

Flags:
    --idx N           Question N from data/hotpotqa/questions.json (default: 0)
    --question TEXT   Custom free-text question instead of questions.json
    --gold TEXT       Gold answer (optional; auto-read from questions.json)
    --skip-llm        Skip the Verifier (retrieval debugging without LLM wait)
    --trace-calls     sys.settrace: print every function call in src/
    --no-color        Plain output (for pipe / file capture)

Gold-tracking:
    When --gold is provided (or auto-loaded from questions.json), the tool
    checks after EVERY filter step whether the gold answer is still present
    in any of the remaining chunks. A "GOLD LOST" marker indicates that the
    preceding step is the culprit.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from textwrap import wrap

# ─── Projekt-Root ins sys.path ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ─── Farben ─────────────────────────────────────────────────────────────────
USE_COLOR = True

# ─── Gold answer for stage tracking ──────────────────────────────────────────
# Wird in main() gesetzt und von allen Patch-Funktionen über Closure gelesen.
_GOLD_ANSWER: str = ""
_HOP_COUNTER: list = [0]   # mutable container für Hop-Zähler in Closures

def _c(code: str, text: str) -> str:
    return f"{code}{text}\033[0m" if USE_COLOR else text

def cyan(t):    return _c("\033[96m", t)
def green(t):   return _c("\033[92m", t)
def yellow(t):  return _c("\033[93m", t)
def red(t):     return _c("\033[91m", t)
def bold(t):    return _c("\033[1m", t)
def dim(t):     return _c("\033[2m", t)
def magenta(t): return _c("\033[95m", t)
def blue(t):    return _c("\033[94m", t)


# ─── Gold-answer tracking ────────────────────────────────────────────────────

def _gold_words(gold: str) -> list:
    """Important words from the gold answer (>=4 chars, no stopword)."""
    _stops = {"that", "this", "with", "from", "have", "been", "were", "will", "would"}
    return [w for w in gold.lower().split() if len(w) >= 4 and w not in _stops]


def _gold_check_texts(texts, stage: str, gold: str = "") -> None:
    """
    Print whether the gold answer is still contained in at least one of the
    provided texts.
    texts: list of str OR list of dicts with a "text" key.
    """
    if not gold:
        gold = _GOLD_ANSWER
    if not gold or gold in ("?", "(unknown)"):
        return
    words = _gold_words(gold)
    if not words:
        return
    hits = []
    for i, item in enumerate(texts):
        t = (item if isinstance(item, str) else item.get("text", "")).lower()
        if all(w in t for w in words):
            hits.append(i + 1)
    bar = "  " + "·" * 60
    if hits:
        extra = f" (+ {len(hits)-1} more)" if len(hits) > 1 else ""
        print(f"{bar}")
        print(f"  {green(bold('OK GOLD'))}  '{gold}'  ->  chunk #{hits[0]}{extra}  [{stage}]")
        print(f"{bar}")
    else:
        print(f"{bar}")
        print(f"  {red(bold('XX GOLD LOST'))}  '{gold}'  no longer in remaining chunks  [{stage}]")
        print(f"{bar}")


# ─── Output helpers ──────────────────────────────────────────────────────────

def section(title: str) -> None:
    bar = "═" * 72
    print(f"\n{bold(bar)}")
    print(f"  {bold(cyan(title))}")
    print(f"{bold(bar)}")

def subsection(title: str) -> None:
    print(f"\n  {bold(yellow('▶ ' + title))}")
    print(f"  {'─' * 68}")

def field(label: str, value, indent_lvl: int = 4) -> None:
    prefix = " " * indent_lvl
    label_str = bold(label + ":")
    value_str = str(value)
    if "\n" in value_str or len(value_str) > 100:
        print(f"{prefix}{label_str}")
        for line in value_str.splitlines():
            for wrapped in wrap(line, width=90) or [""]:
                print(f"{prefix}  {dim(wrapped)}")
    else:
        print(f"{prefix}{label_str} {value_str}")

def chunk_block(idx: int, text: str, score=None, extra: str = "", max_chars: int = 300) -> None:
    score_str = f"  {dim(f'score={score:.4f}')}" if score is not None else ""
    extra_str = f"  {dim(extra)}" if extra else ""
    print(f"  {bold(green(f'Chunk #{idx+1}'))}{score_str}{extra_str}")
    preview = text[:max_chars].replace("\n", " ")
    if len(text) > max_chars:
        preview += dim("…")
    for line in wrap(preview, width=90):
        print(f"    {dim(line)}")

def removed_block(idx: int, text: str, reason: str) -> None:
    print(f"  {red(f'✗ Chunk #{idx+1} ENTFERNT')}  {dim(f'[{reason}]')}")
    preview = text[:150].replace("\n", " ")
    for line in wrap(preview, width=88):
        print(f"    {dim(line)}")

def prompt_block(prompt: str) -> None:
    print(f"  {bold(magenta('PROMPT →'))}")
    border = "  " + "·" * 68
    print(border)
    for line in prompt.splitlines():
        for wrapped in wrap(line, width=86) or [""]:
            print(f"  {dim(wrapped)}")
    print(border)

def answer_block(answer: str, latency_ms: float = None) -> None:
    lat = f"  {dim(f'({latency_ms:.0f} ms)')}" if latency_ms else ""
    color = red if answer.startswith("[Error:") else green
    print(f"  {bold(color('ANTWORT →'))}{lat}")
    border = "  " + "·" * 68
    print(border)
    for line in answer.splitlines():
        for wrapped in wrap(line, width=86) or [""]:
            print(f"  {color(wrapped)}")
    print(border)


# ─── Monkey-Patch Utilities ──────────────────────────────────────────────────

def _patch(obj, method_name: str, wrapper_factory):
    original = getattr(obj, method_name)
    setattr(obj, method_name, wrapper_factory(original))


# =============================================================================
# FRAGE 1: sys.settrace — alle Funktionsaufrufe in src/ tracken
# =============================================================================

_call_log: list = []          # (datei, funktion)
_seen_files: set = set()      # unique src/ files

def _make_tracer(src_root: Path):
    """
    Gibt eine trace-Funktion zurück, die jeden Funktionsaufruf innerhalb
    von src/ mitschreibt. Über --trace-calls aktivierbar.
    """
    src_str = str(src_root / "src")

    def _tracer(frame, event, arg):
        if event != "call":
            return _tracer
        filename = frame.f_code.co_filename
        if src_str in filename:
            rel = Path(filename).relative_to(src_root)
            func = frame.f_code.co_name
            _call_log.append((str(rel), func))
            _seen_files.add(str(rel))
        return _tracer

    return _tracer

def print_call_trace() -> None:
    section("FUNCTION CALL TRACE (alle src/-Aufrufe)")
    subsection(f"Unique files ({len(_seen_files)})")
    for f in sorted(_seen_files):
        print(f"    {blue('📄')} {f}")

    subsection(f"Aufruf-Sequenz ({len(_call_log)} Aufrufe)")
    prev_file = None
    prev_func = None
    repeat_count = 0

    def _flush_repeat():
        if repeat_count > 1:
            print(f"    {dim('→')} {prev_func}()  {dim(f'×{repeat_count}')}")
        elif repeat_count == 1:
            print(f"    {dim('→')} {prev_func}()")

    for rel, func in _call_log:
        if rel != prev_file:
            _flush_repeat()
            repeat_count = 0
            print(f"\n  {bold(blue(rel))}")
            prev_file = rel
            prev_func = None
        if func == prev_func:
            repeat_count += 1
        else:
            _flush_repeat()
            prev_func = func
            repeat_count = 1
    _flush_repeat()


# =============================================================================
# PLANNER HOOK
# =============================================================================

def patch_planner(planner) -> None:
    def _wrap_plan(original):
        def _plan(query: str):
            section("S_P — PLANNER")
            field("Query", query)

            t0 = time.time()
            result = original(query)
            ms = (time.time() - t0) * 1000

            subsection("RetrievalPlan")
            field("query_type",  result.query_type.value)
            field("strategy",    result.strategy.value)
            field("confidence",  f"{result.confidence:.3f}")

            field("sub_queries", "")
            for i, sq in enumerate(result.sub_queries, 1):
                print(f"      {bold(str(i) + '.')} {sq}")

            field("entities", "")
            if result.entities:
                for e in result.entities:
                    # EntityInfo has: text, label, confidence (no entity_type!)
                    print(f"      {bold(e.text)}"
                          f"  {dim(e.label)}"
                          f"  {dim(f'conf={e.confidence:.2f}')}"
                          f"  {dim('bridge') if e.is_bridge else ''}")
            else:
                print(f"      {red('(no entities detected — entity-mention filter will be disabled!)')}")

            if result.hop_sequence:
                field("hop_sequence", "")
                for hop in result.hop_sequence:
                    print(f"      {dim(str(hop))}")

            field("Duration", f"{ms:.0f} ms")
            return result
        return _plan

    _patch(planner, "plan", _wrap_plan)


# =============================================================================
# HYBRID RETRIEVER HOOK — shows GLiNER entities + raw vector/graph results
# =============================================================================

def patch_retriever(retriever) -> None:
    """
    Patch HybridRetriever.retrieve() to expose GLiNER query entities,
    raw vector/graph results, and the fused top-K.
    This is the most important hook: it determines whether the target
    chunk is even pulled from the database.

    GLiNER entities are read from metrics.query_entities (no double call
    of the extractor).
    """
    orig_retrieve = retriever.retrieve

    def _retrieve(query: str, top_k=None, entity_hints=None):
        subsection(f"HybridRetriever.retrieve()  query={query!r}")

        # ── execute retrieve() ───────────────────────────────────────────────
        results, metrics = orig_retrieve(query, top_k, entity_hints=entity_hints)

        # ── read GLiNER entities from metrics (no double call) ───────────────
        print(f"    {bold('GLiNER query entities:')}")
        if metrics.query_entities:
            for e in metrics.query_entities:
                print(f"      {bold(e)}  {dim('(used for graph search)')}")
        else:
            print(f"      {red('(no entities detected -> graph search will be SKIPPED!)')}")

        # ── show metrics ─────────────────────────────────────────────────────
        print(f"\n    {bold('Retrieval metrics:')}")
        print(f"      Vector: {metrics.vector_results} hits  "
              f"{dim(f'({metrics.vector_time_ms:.0f} ms)')}")
        print(f"      Graph:  {metrics.graph_results} hits  "
              f"{dim(f'({metrics.graph_time_ms:.0f} ms)')}")
        print(f"      Fused:  {metrics.final_results} results  "
              f"{dim(f'({metrics.fusion_time_ms:.0f} ms)')}")

        # ── top-5 results with retrieval method and matched entities ─────────
        print(f"\n    {bold('Top-5 fused results:')}")
        for i, r in enumerate(results[:5]):
            src     = getattr(r, "source_doc", getattr(r, "source", "?"))
            score   = getattr(r, "rrf_score", getattr(r, "score", 0))
            method  = getattr(r, "retrieval_method", "?")
            matched = getattr(r, "matched_entities", [])
            v_score = getattr(r, "vector_score", None)
            g_score = getattr(r, "graph_score", None)
            txt     = getattr(r, "text", str(r))[:120].replace("\n", " ")

            method_color = blue if method == "graph" else (cyan if method == "hybrid" else dim)
            print(f"      {bold(f'#{i+1}')}"
                  f"  {green(f'rrf={score:.4f}')}"
                  f"  [{method_color(method)}]"
                  f"  {dim(f'src={src}')}")
            if v_score is not None:
                print(f"        {dim(f'vector_score={v_score:.4f}')}", end="")
            if g_score is not None:
                print(f"  {dim(f'graph_score={g_score:.4f}')}", end="")
            if v_score is not None or g_score is not None:
                print()
            if matched:
                print(f"        {dim(f'matched_entities: {matched}')}")
            for line in wrap(txt, 84):
                print(f"        {dim(line)}")

        return results, metrics

    retriever.retrieve = _retrieve


# =============================================================================
# NAVIGATOR HOOKS
# =============================================================================

def patch_navigator(navigator) -> None:

    # ── _rrf_fusion ──────────────────────────────────────────────────────────
    def _wrap_rrf(original):
        def _rrf(results, k=None):
            fused = original(results, k)
            k_val = navigator.config.rrf_k if k is None else k
            subsection(f"RRF Fusion  k={k_val}  "
                       f"({len(results)} Roh-Einträge → {len(fused)} einzigartige Chunks)")
            for i, r in enumerate(fused[:8]):
                score = r.get("rrf_score", 0)
                src   = r.get("source_count", "?")
                qc    = r.get("query_count", "?")
                txt   = r["text"][:120].replace("\n", " ")
                # Mark chunks that contain the gold answer
                gold_marker = ""
                if _GOLD_ANSWER and _gold_words(_GOLD_ANSWER):
                    words = _gold_words(_GOLD_ANSWER)
                    if all(w in txt.lower() for w in words):
                        gold_marker = f"  {green('← GOLD')}"
                print(f"    {bold(f'#{i+1}')}"
                      f"  {green(f'rrf={score:.4f}')}"
                      f"  {dim(f'src_count={src} query_count={qc}')}"
                      f"{gold_marker}")
                for line in wrap(txt, 86):
                    print(f"      {dim(line)}")
            if len(fused) > 8:
                print(f"    {dim(f'… {len(fused)-8} weitere Chunks')}")
            _gold_check_texts(fused, "nach RRF-Fusion")
            return fused
        return _rrf

    # ── _relevance_filter ────────────────────────────────────────────────────
    def _wrap_relevance(original):
        def _filt(results):
            before = len(results)
            threshold = 0.0
            if results:
                max_score = max(r["rrf_score"] for r in results)
                threshold = navigator.config.relevance_threshold_factor * max_score
            filtered = original(results)
            after = len(filtered)
            removed = before - after
            subsection(f"Filter 1 - Relevance  ({before} -> {after}"
                       + (f", {red(str(removed) + ' removed')})" if removed else ")"))
            if results:
                print(f"    {bold('Threshold:')} {threshold:.4f}  "
                      f"{dim(f'= {navigator.config.relevance_threshold_factor} x max({max_score:.4f})')}")
            if removed:
                kept_texts = {r["text"] for r in filtered}
                for i, r in enumerate(results):
                    if r["text"] not in kept_texts:
                        removed_block(i, r["text"],
                                      f"rrf={r['rrf_score']:.4f} < {threshold:.4f}")
            else:
                print(f"    {dim('(all chunks above threshold — no filtering)')}")
            _gold_check_texts(filtered, "after relevance filter")
            return filtered
        return _filt

    # ── _redundancy_filter ───────────────────────────────────────────────────
    def _wrap_redundancy(original):
        def _filt(results):
            before = len(results)
            filtered = original(results)
            after = len(filtered)
            removed = before - after
            subsection(f"Filter 2 - Redundancy  Jaccard threshold={navigator.config.redundancy_threshold}"
                       f"  ({before} -> {after}"
                       + (f", {red(str(removed) + ' removed')})" if removed else ")"))
            if removed:
                kept_texts = {r["text"] for r in filtered}
                for i, r in enumerate(results):
                    if r["text"] not in kept_texts:
                        removed_block(i, r["text"], "Jaccard duplicate")
            if not removed:
                print(f"    {dim('(no duplicates)')}")
            _gold_check_texts(filtered, "after redundancy filter")
            return filtered
        return _filt

    # ── _contradiction_filter ────────────────────────────────────────────────
    def _wrap_contradiction(original):
        def _filt(results):
            before = len(results)
            filtered = original(results)
            after = len(filtered)
            removed = before - after
            subsection(f"Filter 3 - Contradiction  overlap>={navigator.config.contradiction_overlap_threshold}"
                       f"  ratio>={navigator.config.contradiction_ratio_threshold}"
                       f"  min_value>={navigator.config.contradiction_min_value}"
                       f"  ({before} -> {after}"
                       + (f", {red(str(removed) + ' removed')})" if removed else ")"))
            if removed:
                kept_texts = {r["text"] for r in filtered}
                for i, r in enumerate(results):
                    if r["text"] not in kept_texts:
                        removed_block(i, r["text"], "Numeric contradiction")
            if not removed:
                print(f"    {dim('(no contradictions)')}")
            _gold_check_texts(filtered, "after contradiction filter")
            return filtered
        return _filt

    # ── _entity_overlap_pruning ──────────────────────────────────────────────
    def _wrap_entity_overlap(original):
        def _filt(results):
            before = len(results)
            filtered = original(results)
            after = len(filtered)
            removed = before - after
            subsection(f"Filter 4 - Entity overlap  ({before} -> {after}"
                       + (f", {red(str(removed) + ' removed')})" if removed else ")"))
            if removed:
                kept_texts = {r["text"] for r in filtered}
                for i, r in enumerate(results):
                    if r["text"] not in kept_texts:
                        removed_block(i, r["text"], "Entity set is a subset")
            if not removed:
                print(f"    {dim('(no subsets)')}")
            _gold_check_texts(filtered, "after entity-overlap filter")
            return filtered
        return _filt

    # ── _entity_mention_filter ───────────────────────────────────────────────
    def _wrap_entity_mention(original):
        def _filt(results, entity_names):
            before = len(results)
            filtered = original(results, entity_names)
            after = len(filtered)
            removed = before - after

            # Detect safety-fallback: all chunks would have been filtered but
            # the filter returned the full list unchanged as a last-resort.
            # Heuristic: removed==0, entity_names non-empty, and no chunk
            # actually contains any entity → fallback must have fired.
            safety_fallback = False
            if removed == 0 and entity_names and results:
                import re as _re
                def _mentions(text, names):
                    t = text.lower()
                    for name in names:
                        tokens = name.lower().split()
                        if len(tokens) > 1:
                            if name.lower() in t:
                                return True
                        else:
                            if len(name) >= 5 and _re.search(r'\b' + _re.escape(name.lower()) + r'\b', t):
                                return True
                    return False
                safety_fallback = not any(_mentions(r["text"], entity_names) for r in results)

            label = f"{before} -> {after}"
            if removed:
                label += f", {red(str(removed) + ' removed')}"
            elif safety_fallback:
                label += f", {yellow('safety fallback!')}"
            subsection(f"Filter 5 - Entity mention  ({label})")

            print(f"    {bold('Searched entities:')} "
                  + (", ".join(bold(e) for e in entity_names) if entity_names
                     else red("(none!) -> filter disabled, all chunks kept")))

            if not entity_names:
                print(f"    {yellow('* CAUSE: planner entities empty')}")
                print(f"    {yellow('* CONSEQUENCE: irrelevant chunks pass through — Verifier sees poor context')}")
            elif safety_fallback:
                print(f"    {yellow('* SAFETY FALLBACK: no chunk contains any of the searched entities.')}")
                print(f"    {yellow('* CAUSE: article likely not in the database (missing from ingestion)')}")
                print(f"    {yellow('* CONSEQUENCE: all 10 irrelevant chunks stay — Verifier has useless context')}")
            elif removed:
                kept_texts = {r["text"] for r in filtered}
                for i, r in enumerate(results):
                    if r["text"] not in kept_texts:
                        removed_block(i, r["text"], "No entity mention in the text")

            if after < before or not entity_names or safety_fallback:
                print(f"    {bold('Remaining chunks:')}")
                for i, r in enumerate(filtered):
                    chunk_block(i, r["text"], r.get("rrf_score"))
            _gold_check_texts(filtered, "after entity-mention filter")
            return filtered
        return _filt

    # ── _context_shrinkage ───────────────────────────────────────────────────
    def _wrap_shrinkage(original):
        def _filt(results, max_chars_per_chunk=None):
            shrunk = original(results, max_chars_per_chunk)
            limit = max_chars_per_chunk or navigator.config.max_chars_per_doc
            total_before = sum(len(r["text"]) for r in results)
            total_after  = sum(len(r["text"]) for r in shrunk)
            reduction = 100 * (1 - total_after / max(total_before, 1))
            subsection(f"Filter 6 - Context shrinkage  "
                       f"limit={limit} chars/chunk  "
                       f"({total_before} -> {total_after} chars, {reduction:.0f}% reduction)")
            for i, r in enumerate(shrunk):
                orig_len = len(results[i]["text"]) if i < len(results) else "?"
                new_len  = len(r["text"])
                trunc = f"truncated {orig_len}->{new_len}" if orig_len != new_len else "unchanged"
                chunk_block(i, r["text"], r.get("rrf_score"), extra=trunc)
            _gold_check_texts(shrunk, "after context shrinkage")
            return shrunk
        return _filt

    _patch(navigator, "_rrf_fusion",             _wrap_rrf)
    _patch(navigator, "_relevance_filter",       _wrap_relevance)
    _patch(navigator, "_redundancy_filter",      _wrap_redundancy)
    _patch(navigator, "_contradiction_filter",   _wrap_contradiction)
    _patch(navigator, "_entity_overlap_pruning", _wrap_entity_overlap)
    _patch(navigator, "_entity_mention_filter",  _wrap_entity_mention)
    _patch(navigator, "_context_shrinkage",      _wrap_shrinkage)

    # ── navigate() ───────────────────────────────────────────────────────────
    orig_navigate = navigator.navigate

    def _nav_wrapper(retrieval_plan, sub_queries, entity_names=None):
        _HOP_COUNTER[0] += 1
        hop_label = f"Hop {_HOP_COUNTER[0]}" if _HOP_COUNTER[0] > 1 else "Single-Pass"
        section(f"S_N — NAVIGATOR  [{hop_label}]")
        field("Sub-Queries", len(sub_queries))
        for i, sq in enumerate(sub_queries, 1):
            print(f"    {bold(str(i) + '.')} {sq}")
        field("Entity names (for entity-mention filter)", entity_names or [])
        print(f"    {bold('max_context_chunks:')} {navigator.config.max_context_chunks}")
        print(f"    {bold('top_k_per_subquery:')} {navigator.config.top_k_per_subquery}")

        result = orig_navigate(retrieval_plan, sub_queries, entity_names)

        subsection("NAVIGATOR RESULT")
        field("Raw chunks (before all filters)",    len(result.raw_context))
        field("Filtered chunks (after all filters)", len(result.filtered_context))
        print(f"\n    {bold('Filter counters:')}")
        filter_keys = [
            "pre_filter_count",
            "after_relevance_filter",
            "after_redundancy_filter",
            "after_contradiction_filter",
            "after_entity_overlap_pruning",
            "after_entity_mention_filter",
        ]
        prev = result.metadata.get("pre_filter_count", "?")
        for k in filter_keys:
            v = result.metadata.get(k, "?")
            diff = ""
            if k != "pre_filter_count" and isinstance(v, int) and isinstance(prev, int):
                delta = v - prev
                if delta < 0:
                    diff = f"  {red(f'-{abs(delta)} removed')}"
                elif delta == 0:
                    diff = f"  {dim('unchanged')}"
            print(f"      {bold(k)}: {v}{diff}")
            prev = v

        if result.metadata.get("retrieval_errors"):
            print(f"    {red('Retrieval errors:')} {result.metadata['retrieval_errors']}")

        # Gold check on final Navigator result
        _gold_check_texts(result.filtered_context, f"Navigator result [{hop_label}]")
        return result

    navigator.navigate = _nav_wrapper


# =============================================================================
# VERIFIER HOOKS
# =============================================================================

def patch_verifier(verifier) -> None:

    # ── _reorder_by_question_relevance ────────────────────────────────────────
    orig_reorder = verifier._reorder_by_question_relevance

    def _reorder_wrapper(query, context):
        reordered = orig_reorder(query, context)
        subsection(f"_reorder_by_question_relevance()  {len(context)} chunks")
        if reordered != context:
            print(f"    {bold('Order changed')} — LLM sees chunks in this order:")
        else:
            print(f"    {dim('(order unchanged)')}")
        for i, c in enumerate(reordered):
            orig_pos = context.index(c) + 1 if c in context else "?"
            gold_marker = ""
            if _GOLD_ANSWER:
                words = _gold_words(_GOLD_ANSWER)
                if words and all(w in c.lower() for w in words):
                    gold_marker = f"  {green('<- GOLD ok')}"
            pos_change = f"  {dim(f'(was #{orig_pos})')}" if orig_pos != i + 1 else ""
            print(f"    {bold(f'#{i+1}')}{pos_change}{gold_marker}")
            preview = c[:120].replace("\n", " ")
            for line in wrap(preview, 86):
                print(f"      {dim(line)}")
        _gold_check_texts(reordered, "after reorder (LLM input)")
        return reordered

    verifier._reorder_by_question_relevance = _reorder_wrapper

    # ── _format_context ───────────────────────────────────────────────────────
    orig_format = verifier._format_context

    def _fmt_wrapper(context):
        formatted = orig_format(context)
        subsection(f"_format_context()  {len(context)} chunks -> {len(formatted)} chars")
        max_docs = verifier.config.max_docs
        max_chars = verifier.config.max_chars_per_doc
        print(f"    {bold('Limits:')} max_docs={max_docs}, max_chars_per_doc={max_chars}, "
              f"max_context_chars={verifier.config.max_context_chars}")
        if len(context) > max_docs:
            print(f"    {yellow(f'* {len(context)} chunks -> only the first {max_docs} will be used!')}")
        return formatted

    verifier._format_context = _fmt_wrapper

    # ── _extract_claims ───────────────────────────────────────────────────────
    orig_extract = verifier._extract_claims

    def _claims_wrapper(answer):
        claims = orig_extract(answer)
        print(f"\n    {bold('Extracted claims')} ({len(claims)}):")
        for c in claims:
            print(f"      {dim('·')} {c}")
        return claims

    verifier._extract_claims = _claims_wrapper

    # ── _call_llm ─────────────────────────────────────────────────────────────
    orig_call_llm = verifier._call_llm

    def _wrap_call_llm(prompt: str):
        prompt_block(prompt)
        answer, latency_ms = orig_call_llm(prompt)
        answer_block(answer, latency_ms)
        return answer, latency_ms

    verifier._call_llm = _wrap_call_llm

    # ── generate_and_verify ───────────────────────────────────────────────────
    orig_gen = verifier.generate_and_verify

    def _wrap_generate(query, context, entities=None, hop_sequence=None):
        section("S_V — VERIFIER")
        field("Query",       query)
        field("Chunks in",   len(context))
        field("Entities",    entities or [])

        # Run pre-validation and display its result
        subsection("Pre-Generation Validation")
        try:
            pre_val = verifier.pre_validator.validate(
                context=context, query=query,
                entities=entities, hop_sequence=hop_sequence,
            )
            print(f"    {bold('Status:')}          {pre_val.status.value}")
            print(f"    {bold('entity_path_valid:')} {pre_val.entity_path_valid}")
            print(f"    {bold('Contradictions:')}")
            if pre_val.contradictions:
                for c in pre_val.contradictions[:3]:
                    print(f"      {red('XX')} {dim(str(c)[:120])}")
            else:
                print(f"      {dim('(none)')}")
            print(f"    {bold('Filtered context:')} "
                  f"{len(pre_val.filtered_context)}/{len(context)} chunks kept")
            if len(pre_val.filtered_context) < len(context):
                kept = set(id(c) for c in pre_val.filtered_context)
                for i, c in enumerate(context):
                    if id(c) not in kept:
                        print(f"      {red(f'XX chunk #{i+1} removed by pre-validator:')}")
                        print(f"        {dim(c[:100])}")
            print(f"    {bold('Credibility scores:')}")
            if hasattr(pre_val, "credibility_scores") and pre_val.credibility_scores:
                for i, sc in enumerate(pre_val.credibility_scores[:5]):
                    score_val = sc.score if hasattr(sc, "score") else sc
                    print(f"      Chunk #{i+1}: {score_val:.3f}")
            else:
                print(f"      {dim('(none)')}")
        except Exception as ex:
            print(f"    {red(f'Pre-validation error: {ex}')}")

        print(f"\n  {bold(yellow('Context chunks sent to Verifier:'))}")
        for i, c in enumerate(context):
            chunk_block(i, c)

        result = orig_gen(query, context, entities, hop_sequence)

        subsection("VERIFIER RESULT")
        field("Answer",       result.answer)
        conf = result.confidence
        conf_str = conf.value if hasattr(conf, "value") else str(conf)
        field("Confidence",   conf_str)
        field("Iterations",   result.iterations)
        field("All verified", result.all_verified)
        if result.verified_claims:
            print(f"    {bold('Verified claims')} ({len(result.verified_claims)}):")
            for c in result.verified_claims:
                print(f"      {green('✓')} {dim(c)}")
        if result.violated_claims:
            print(f"    {bold('Violated claims')} ({len(result.violated_claims)}):")
            for c in result.violated_claims:
                print(f"      {red('✗')} {dim(c)}")
        return result

    verifier.generate_and_verify = _wrap_generate


# =============================================================================
# PIPELINE AUFBAUEN
# =============================================================================

def load_question(idx: int) -> dict:
    questions_path = PROJECT_ROOT / "data" / "hotpotqa" / "questions.json"
    if not questions_path.exists():
        print(red(f"Error: {questions_path} not found"))
        sys.exit(1)
    with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)
    if idx >= len(questions):
        print(red(f"Index {idx} too large (max: {len(questions)-1})"))
        sys.exit(1)
    return questions[idx]


# =============================================================================
# CONTROLLER HOOK — make bridge-entity extraction visible
# =============================================================================

def patch_controller_bridge(controller) -> None:
    """
    Patch AgenticController._extract_bridge_entities so that every detection
    of bridge entities appears in the output. This shows whether the
    iterative multi-hop is actually running and which entities it discovers.
    """
    from src.logic_layer.controller import AgenticController as _AC
    orig_extract = _AC._extract_bridge_entities  # raw function via class

    def _wrapped_extract(chunks, exclude):
        result = orig_extract(chunks, exclude)
        bar = "  " + "·" * 60
        print(f"\n{bar}")
        if result:
            print(f"  {green(bold('** BRIDGE ENTITIES DETECTED:'))}  "
                  + "  ".join(bold(e) for e in result))
            print(f"  {dim(f'(from {len(chunks)} chunk(s), exclude={exclude})')}")
        else:
            print(f"  {yellow(bold('* NO BRIDGE ENTITIES FOUND'))}  "
                  f"{dim(f'(exclude={exclude}, chunks: {len(chunks)})')}")
            if chunks:
                preview = chunks[0][:120].replace("\n", " ")
                print(f"  {dim(f'Chunk preview: {preview}')}")
        print(f"{bar}\n")
        return result

    # Instance-dict patch: Python looks in self.__dict__ first -> overrides class staticmethod
    controller._extract_bridge_entities = _wrapped_extract


def build_pipeline(cfg: dict):
    import yaml
    from src.data_layer.embeddings import BatchedOllamaEmbeddings
    from src.data_layer.hybrid_retriever import HybridRetriever, RetrievalConfig, RetrievalMode
    from src.data_layer.storage import HybridStore, StorageConfig
    from src.logic_layer import AgenticController, ControllerConfig, create_planner, create_verifier

    vector_path = PROJECT_ROOT / "data" / "hotpotqa" / "vector"
    graph_path  = PROJECT_ROOT / "data" / "hotpotqa" / "graph"

    emb_cfg = cfg.get("embeddings", {})
    embeddings = BatchedOllamaEmbeddings(
        model_name=emb_cfg.get("model_name", "nomic-embed-text"),
        base_url=emb_cfg.get("base_url", "http://localhost:11434"),
    )
    storage_cfg = StorageConfig(vector_db_path=vector_path, graph_db_path=graph_path)
    store = HybridStore(config=storage_cfg, embeddings=embeddings)

    rag_cfg   = cfg.get("rag", {})
    ee_gliner = cfg.get("entity_extraction", {}).get("gliner", {})
    ret_cfg = RetrievalConfig(
        mode=RetrievalMode.HYBRID,
        vector_top_k=rag_cfg.get("top_k_vectors", 10),
        final_top_k=rag_cfg.get("top_k_vectors", 10),
        rrf_k=rag_cfg.get("rrf_k", 60),
        cross_source_boost=rag_cfg.get("cross_source_boost", 1.2),
        query_ner_confidence=ee_gliner.get("confidence_threshold", 0.15),
        query_entity_types=ee_gliner.get("entity_types") or None,
    )
    retriever = HybridRetriever(
        hybrid_store=store,
        embeddings=embeddings,
        config=ret_cfg,
    )

    ctrl_cfg = ControllerConfig.from_yaml(cfg)
    planner  = create_planner(cfg)
    verifier = create_verifier(cfg=cfg)
    verifier.set_graph_store(store.graph_store)

    controller = AgenticController(
        config=ctrl_cfg,
        planner=planner,
        verifier=verifier,
        full_cfg=cfg,
    )
    controller.set_retriever(retriever)

    return controller, planner, verifier, store, retriever


# =============================================================================
# MAIN
# =============================================================================

def main():
    global USE_COLOR, _GOLD_ANSWER

    parser = argparse.ArgumentParser(
        description="Full pipeline trace with output from every function"
    )
    parser.add_argument("--idx",         type=int,  default=0)
    parser.add_argument("--question",    type=str,  default=None)
    parser.add_argument("--gold",        type=str,  default=None,
                        help="Gold answer for stage tracking (otherwise read from questions.json)")
    parser.add_argument("--skip-llm",    action="store_true")
    parser.add_argument("--trace-calls", action="store_true",
                        help="sys.settrace: show every src/ function call")
    parser.add_argument("--no-color",    action="store_true")
    args = parser.parse_args()

    if args.no_color:
        USE_COLOR = False

    # ── Load the question ────────────────────────────────────────────────────
    if args.question:
        q_text = args.question
        gold   = args.gold or "(unknown)"
        q_type = "custom"
    else:
        q = load_question(args.idx)
        q_text = q["question"]
        gold   = args.gold or q.get("answer", "?")
        q_type = q.get("question_type", "?")

    # Set the gold answer globally (all patch hooks read it from here)
    _GOLD_ANSWER = gold

    # ── Load config ──────────────────────────────────────────────────────────
    import yaml
    with open(PROJECT_ROOT / "config" / "settings.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ── Header ───────────────────────────────────────────────────────────────
    print("\n" + bold("═" * 72))
    print(f"  {bold(cyan('PIPELINE VERBOSE TRACE'))}")
    print(bold("═" * 72))
    field("Question",     q_text)
    field("Gold answer",  gold)
    field("Type",         q_type)
    if args.skip_llm:
        print(f"\n  {yellow('* --skip-llm: Verifier will NOT be executed')}")
    if args.trace_calls:
        print(f"\n  {blue('* --trace-calls: every src/ function call will be logged')}")

    # ── sys.settrace aktivieren ───────────────────────────────────────────────
    if args.trace_calls:
        sys.settrace(_make_tracer(PROJECT_ROOT))

    # ── Pipeline bauen ────────────────────────────────────────────────────────
    print(f"\n{dim('Lade Pipeline...')} ", end="", flush=True)
    t0 = time.time()
    controller, planner, verifier, store, retriever = build_pipeline(cfg)
    print(f"{green('OK')}  {dim(f'({(time.time()-t0)*1000:.0f} ms)')}")

    # ── Hooks einsetzen ───────────────────────────────────────────────────────
    patch_planner(planner)
    patch_retriever(retriever)
    patch_navigator(controller.navigator)
    patch_controller_bridge(controller)

    if not args.skip_llm:
        patch_verifier(verifier)
    else:
        def _skip_verifier(query, context, entities=None, hop_sequence=None):
            from src.logic_layer import VerificationResult
            section("S_V — VERIFIER  (übersprungen via --skip-llm)")
            field("Kontext-Chunks", len(context))
            if context:
                for i, c in enumerate(context):
                    gold_marker = ""
                    if _GOLD_ANSWER:
                        words = _gold_words(_GOLD_ANSWER)
                        if words and all(w in c.lower() for w in words):
                            gold_marker = f"  {green('← GOLD ✓')}"
                    chunk_block(i, c)
                    if gold_marker:
                        print("    " + green("  <- GOLD ANSWER '" + _GOLD_ANSWER + "' IN THIS CHUNK"))
                _gold_check_texts(context, "Verifier input (skip-llm)")
            else:
                print(f"    {red('* NO CONTEXT — Navigator returned 0 chunks!')}")
                print(f"    {yellow('  -> Possible causes: retrieval error, entity-mention filter too aggressive')}")
            result = VerificationResult(
                answer="[skipped]",
                iterations=0,
                verified_claims=[],
                violated_claims=[],
                all_verified=False,
                pre_validation=None,
                iteration_history=[],
                timing_ms=0.0,
            )
            subsection("VERIFIER RESULT  (simulated)")
            field("Answer",     result.answer)
            field("Confidence", result.confidence.value)
            field("Iterations", result.iterations)
            return result
        verifier.generate_and_verify = _skip_verifier

    # ── Run pipeline ─────────────────────────────────────────────────────────
    t_start = time.time()
    final_state = controller.run(q_text)
    total_ms = (time.time() - t_start) * 1000

    # ── Disable sys.settrace ─────────────────────────────────────────────────
    if args.trace_calls:
        sys.settrace(None)
        print_call_trace()

    # ── Summary ──────────────────────────────────────────────────────────────
    section("SUMMARY")
    pred = final_state.get("answer", "")
    field("Question",   q_text)
    field("Gold",       gold)
    field("Prediction", pred)
    field("Total time", f"{total_ms:.0f} ms")

    timings = final_state.get("stage_timings", {})
    if timings:
        field("S_P", f"{timings.get('planner_ms', 0):.0f} ms")
        field("S_N", f"{timings.get('navigator_ms', 0):.0f} ms")
        field("S_V", f"{timings.get('verifier_ms', 0):.0f} ms")

    # EM-Check
    import re
    def _norm(t: str) -> str:
        t = t.lower()
        t = re.sub(r'\b(a|an|the)\b', ' ', t)
        t = re.sub(r'[^\w\s]', '', t)
        return ' '.join(t.split())

    pred_n, gold_n = _norm(pred), _norm(gold)
    em = pred_n == gold_n or (
        gold_n and bool(re.search(r'\b' + re.escape(gold_n) + r'\b', pred_n))
    )
    print(f"\n  Result: {bold(green('OK CORRECT') if em else red('XX WRONG'))}")

    errors = final_state.get("errors", [])
    if errors:
        print(f"\n  {red('Pipeline errors:')}")
        for e in errors:
            print(f"    {red('!')} {dim(e)}")

    print(f"\n{bold('═' * 72)}\n")


if __name__ == "__main__":
    main()
