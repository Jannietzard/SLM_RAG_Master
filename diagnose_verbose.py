"""
diagnose_verbose.py — Vollständiger Pipeline-Trace (Megafile)

Läuft die komplette S_P → S_N → S_V Pipeline für eine Frage durch und zeigt
den Output JEDER aufgerufenen Funktion an. Kein einziger Produktionscode wird
geändert — alle Hooks werden via Monkey-Patching zur Laufzeit eingehängt.

Usage:
    python -X utf8 diagnose_verbose.py --idx 0
    python -X utf8 diagnose_verbose.py --idx 5 --skip-llm
    python -X utf8 diagnose_verbose.py --question "Wer gründete YG Entertainment?"
    python -X utf8 diagnose_verbose.py --idx 0 --trace-calls   # alle src/-Funktionen tracken
    python -X utf8 diagnose_verbose.py --idx 0 --no-color > trace.txt

Flags:
    --idx N           Frage N aus data/hotpotqa/questions.json (default: 0)
    --question TEXT   Eigene Freitextfrage statt questions.json
    --skip-llm        Verifier überspringen (Retrieval-Debugging ohne LLM-Wartezeit)
    --trace-calls     sys.settrace: zeigt JEDEN Funktionsaufruf in src/ mit Dateiname
    --no-color        Farbloses Output (für Pipe/Dateiausgabe)
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


# ─── Ausgabe-Hilfsfunktionen ─────────────────────────────────────────────────

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
_seen_files: set = set()      # einzigartige src/-Dateien

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
    subsection(f"Einzigartige Dateien ({len(_seen_files)})")
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
                    # EntityInfo hat: text, label, confidence (kein entity_type!)
                    print(f"      {bold(e.text)}"
                          f"  {dim(e.label)}"
                          f"  {dim(f'conf={e.confidence:.2f}')}"
                          f"  {dim('bridge') if e.is_bridge else ''}")
            else:
                print(f"      {red('(keine Entitäten erkannt — Entity-Mention Filter wird deaktiviert!)')}")

            if result.hop_sequence:
                field("hop_sequence", "")
                for hop in result.hop_sequence:
                    print(f"      {dim(str(hop))}")

            field("Dauer", f"{ms:.0f} ms")
            return result
        return _plan

    _patch(planner, "plan", _wrap_plan)


# =============================================================================
# HYBRID RETRIEVER HOOK — zeigt GLiNER-Entities + Vector/Graph-Roh-Ergebnisse
# =============================================================================

def patch_retriever(retriever) -> None:
    """
    Patcht HybridRetriever.retrieve() um GLiNER-Query-Entities,
    rohe Vector-/Graph-Ergebnisse und fusionierten Top-K sichtbar zu machen.
    Das ist der wichtigste Hook: hier entscheidet sich ob Ed Wood überhaupt
    aus der DB geholt wird.

    GLiNER-Entities werden aus metrics.query_entities gelesen (kein
    Doppelaufruf des Extraktors).
    """
    orig_retrieve = retriever.retrieve

    def _retrieve(query: str, top_k=None, entity_hints=None):
        subsection(f"HybridRetriever.retrieve()  query={query!r}")

        # ── retrieve() ausführen ─────────────────────────────────────────────
        results, metrics = orig_retrieve(query, top_k, entity_hints=entity_hints)

        # ── GLiNER-Entities aus Metrics lesen (kein Doppelaufruf) ────────────
        print(f"    {bold('GLiNER Query-Entities:')}")
        if metrics.query_entities:
            for e in metrics.query_entities:
                print(f"      {bold(e)}  {dim('(wird für Graph-Search verwendet)')}")
        else:
            print(f"      {red('(keine Entities erkannt → Graph-Search wird ÜBERSPRUNGEN!)')}")

        # ── Metriken anzeigen ─────────────────────────────────────────────────
        print(f"\n    {bold('Retrieval-Metriken:')}")
        print(f"      Vector: {metrics.vector_results} Treffer  "
              f"{dim(f'({metrics.vector_time_ms:.0f} ms)')}")
        print(f"      Graph:  {metrics.graph_results} Treffer  "
              f"{dim(f'({metrics.graph_time_ms:.0f} ms)')}")
        print(f"      Fused:  {metrics.final_results} Ergebnisse  "
              f"{dim(f'({metrics.fusion_time_ms:.0f} ms)')}")

        # ── Top-5 Ergebnisse mit Retrieval-Methode und matched Entities ───────
        print(f"\n    {bold('Top-5 fusionierte Ergebnisse:')}")
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
                print(f"    {bold(f'#{i+1}')}"
                      f"  {green(f'rrf={score:.4f}')}"
                      f"  {dim(f'src_count={src} query_count={qc}')}")
                for line in wrap(txt, 86):
                    print(f"      {dim(line)}")
            if len(fused) > 8:
                print(f"    {dim(f'… {len(fused)-8} weitere Chunks')}")
            return fused
        return _rrf

    # ── _relevance_filter ────────────────────────────────────────────────────
    def _wrap_relevance(original):
        def _filt(results):
            before = len(results)
            # Compute threshold before calling original so we can reference it
            # in the removed-chunk display below.
            threshold = 0.0
            if results:
                max_score = max(r["rrf_score"] for r in results)
                threshold = navigator.config.relevance_threshold_factor * max_score
            filtered = original(results)
            after = len(filtered)
            removed = before - after
            # Section header first, threshold info second.
            subsection(f"Filter 1 — Relevanz  ({before} → {after}"
                       + (f", {red(str(removed) + ' entfernt')})" if removed else ")"))
            if results:
                print(f"    {bold('Schwelle:')} {threshold:.4f}  "
                      f"{dim(f'= {navigator.config.relevance_threshold_factor} × max({max_score:.4f})')}")
            if removed:
                kept_texts = {r["text"] for r in filtered}
                for i, r in enumerate(results):
                    if r["text"] not in kept_texts:
                        removed_block(i, r["text"],
                                      f"rrf={r['rrf_score']:.4f} < {threshold:.4f}")
            else:
                print(f"    {dim('(alle Chunks über Schwelle — kein Filtering)')}")
            return filtered
        return _filt

    # ── _redundancy_filter ───────────────────────────────────────────────────
    def _wrap_redundancy(original):
        def _filt(results):
            before = len(results)
            filtered = original(results)
            after = len(filtered)
            removed = before - after
            subsection(f"Filter 2 — Redundanz  Jaccard-Threshold={navigator.config.redundancy_threshold}"
                       f"  ({before} → {after}"
                       + (f", {red(str(removed) + ' entfernt')})" if removed else ")"))
            if removed:
                kept_texts = {r["text"] for r in filtered}
                for i, r in enumerate(results):
                    if r["text"] not in kept_texts:
                        removed_block(i, r["text"], "Jaccard-Duplikat")
            if not removed:
                print(f"    {dim('(keine Duplikate)')}")
            return filtered
        return _filt

    # ── _contradiction_filter ────────────────────────────────────────────────
    def _wrap_contradiction(original):
        def _filt(results):
            before = len(results)
            filtered = original(results)
            after = len(filtered)
            removed = before - after
            subsection(f"Filter 3 — Widerspruch  overlap>={navigator.config.contradiction_overlap_threshold}"
                       f"  ratio>={navigator.config.contradiction_ratio_threshold}"
                       f"  ({before} → {after}"
                       + (f", {red(str(removed) + ' entfernt')})" if removed else ")"))
            if removed:
                kept_texts = {r["text"] for r in filtered}
                for i, r in enumerate(results):
                    if r["text"] not in kept_texts:
                        removed_block(i, r["text"], "Numerischer Widerspruch")
            if not removed:
                print(f"    {dim('(keine Widersprüche)')}")
            return filtered
        return _filt

    # ── _entity_overlap_pruning ──────────────────────────────────────────────
    def _wrap_entity_overlap(original):
        def _filt(results):
            before = len(results)
            filtered = original(results)
            after = len(filtered)
            removed = before - after
            subsection(f"Filter 4 — Entity-Overlap  ({before} → {after}"
                       + (f", {red(str(removed) + ' entfernt')})" if removed else ")"))
            if removed:
                kept_texts = {r["text"] for r in filtered}
                for i, r in enumerate(results):
                    if r["text"] not in kept_texts:
                        removed_block(i, r["text"], "Entity-Set ist Teilmenge")
            if not removed:
                print(f"    {dim('(keine Teilmengen)')}")
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

            label = f"{before} → {after}"
            if removed:
                label += f", {red(str(removed) + ' entfernt')}"
            elif safety_fallback:
                label += f", {yellow('Safety-Fallback!')}"
            subsection(f"Filter 5 — Entity-Mention  ({label})")

            print(f"    {bold('Gesuchte Entities:')} "
                  + (", ".join(bold(e) for e in entity_names) if entity_names
                     else red("(keine!) → Filter deaktiviert, alle Chunks behalten")))

            if not entity_names:
                print(f"    {yellow('⚠ URSACHE: Planner-Entities leer')}")
                print(f"    {yellow('⚠ FOLGE: Irrelevante Chunks kommen durch — Verifier hat schlechten Kontext')}")
            elif safety_fallback:
                print(f"    {yellow('⚠ SAFETY-FALLBACK: Kein einziger Chunk enthält eine der gesuchten Entities.')}")
                print(f"    {yellow('⚠ URSACHE: Artikel wahrscheinlich nicht in der Datenbank (fehlt in Ingestion)')}")
                print(f"    {yellow('⚠ FOLGE: Alle 10 irrelevanten Chunks bleiben — Verifier hat nutzlosen Kontext')}")
            elif removed:
                kept_texts = {r["text"] for r in filtered}
                for i, r in enumerate(results):
                    if r["text"] not in kept_texts:
                        removed_block(i, r["text"], "Keine Entity-Erwähnung im Text")

            if after < before or not entity_names or safety_fallback:
                print(f"    {bold('Verbleibende Chunks:')}")
                for i, r in enumerate(filtered):
                    chunk_block(i, r["text"], r.get("rrf_score"))
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
            subsection(f"Filter 6 — Context Shrinkage  "
                       f"limit={limit} Zeichen/Chunk  "
                       f"({total_before} → {total_after} Zeichen, {reduction:.0f}% Reduktion)")
            for i, r in enumerate(shrunk):
                orig_len = len(results[i]["text"]) if i < len(results) else "?"
                new_len  = len(r["text"])
                trunc = f"gekürzt {orig_len}→{new_len}" if orig_len != new_len else "unverändert"
                chunk_block(i, r["text"], r.get("rrf_score"), extra=trunc)
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
        section("S_N — NAVIGATOR")
        field("Sub-Queries", len(sub_queries))
        for i, sq in enumerate(sub_queries, 1):
            print(f"    {bold(str(i) + '.')} {sq}")
        field("Entity Names (für Entity-Mention Filter)", entity_names or [])
        print(f"    {bold('max_context_chunks:')} {navigator.config.max_context_chunks}")
        print(f"    {bold('top_k_per_subquery:')} {navigator.config.top_k_per_subquery}")

        result = orig_navigate(retrieval_plan, sub_queries, entity_names)

        subsection("NAVIGATOR ERGEBNIS")
        field("Raw Chunks (vor allen Filtern)",    len(result.raw_context))
        field("Filtered Chunks (nach allen Filtern)", len(result.filtered_context))
        print(f"\n    {bold('Filter-Zähler:')}")
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
                    diff = f"  {red(f'−{abs(delta)} entfernt')}"
                elif delta == 0:
                    diff = f"  {dim('unverändert')}"
            print(f"      {bold(k)}: {v}{diff}")
            prev = v

        if result.metadata.get("retrieval_errors"):
            print(f"    {red('Retrieval-Fehler:')} {result.metadata['retrieval_errors']}")
        return result

    navigator.navigate = _nav_wrapper


# =============================================================================
# VERIFIER HOOKS
# =============================================================================

def patch_verifier(verifier) -> None:

    # ── _format_context ───────────────────────────────────────────────────────
    orig_format = verifier._format_context

    def _fmt_wrapper(context):
        formatted = orig_format(context)
        subsection(f"_format_context()  {len(context)} Chunks → {len(formatted)} Zeichen")
        max_docs = verifier.config.max_docs
        max_chars = verifier.config.max_chars_per_doc
        print(f"    {bold('Limits:')} max_docs={max_docs}, max_chars_per_doc={max_chars}, "
              f"max_context_chars={verifier.config.max_context_chars}")
        if len(context) > max_docs:
            print(f"    {yellow(f'⚠ {len(context)} Chunks → nur die ersten {max_docs} werden verwendet!')}")
        return formatted

    verifier._format_context = _fmt_wrapper

    # ── _extract_claims ───────────────────────────────────────────────────────
    orig_extract = verifier._extract_claims

    def _claims_wrapper(answer):
        claims = orig_extract(answer)
        print(f"\n    {bold('Extrahierte Claims')} ({len(claims)}):")
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
        field("Query",    query)
        field("Chunks eingehend", len(context))
        field("Entities", entities or [])

        # Pre-Validation direkt aufrufen und anzeigen
        subsection("Pre-Generation Validation")
        try:
            pre_val = verifier.pre_validator.validate(
                context=context, query=query,
                entities=entities, hop_sequence=hop_sequence,
            )
            print(f"    {bold('Status:')}          {pre_val.status.value}")
            print(f"    {bold('entity_path_valid:')} {pre_val.entity_path_valid}")
            print(f"    {bold('Widersprüche:')}")
            if pre_val.contradictions:
                for c in pre_val.contradictions[:3]:
                    print(f"      {red('✗')} {dim(str(c)[:120])}")
            else:
                print(f"      {dim('(keine)')}")
            print(f"    {bold('Filtered Context:')} "
                  f"{len(pre_val.filtered_context)}/{len(context)} Chunks behalten")
            if len(pre_val.filtered_context) < len(context):
                kept = set(id(c) for c in pre_val.filtered_context)
                for i, c in enumerate(context):
                    if id(c) not in kept:
                        print(f"      {red(f'✗ Chunk #{i+1} von Pre-Validator entfernt:')}")
                        print(f"        {dim(c[:100])}")
            print(f"    {bold('Credibility Scores:')}")
            if hasattr(pre_val, "credibility_scores") and pre_val.credibility_scores:
                for i, sc in enumerate(pre_val.credibility_scores[:5]):
                    score_val = sc.score if hasattr(sc, "score") else sc
                    print(f"      Chunk #{i+1}: {score_val:.3f}")
            else:
                print(f"      {dim('(keine)')}")
        except Exception as ex:
            print(f"    {red(f'Pre-Validation Fehler: {ex}')}")

        print(f"\n  {bold(yellow('Kontext-Chunks an Verifier:'))}")
        for i, c in enumerate(context):
            chunk_block(i, c)

        result = orig_gen(query, context, entities, hop_sequence)

        subsection("VERIFIER ERGEBNIS")
        field("Antwort",      result.answer)
        conf = result.confidence
        conf_str = conf.value if hasattr(conf, "value") else str(conf)
        field("Konfidenz",    conf_str)
        field("Iterationen",  result.iterations)
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
        print(red(f"Fehler: {questions_path} nicht gefunden"))
        sys.exit(1)
    with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)
    if idx >= len(questions):
        print(red(f"Index {idx} zu groß (max: {len(questions)-1})"))
        sys.exit(1)
    return questions[idx]


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
    global USE_COLOR

    parser = argparse.ArgumentParser(
        description="Vollständiger Pipeline-Trace mit Output jeder Funktion"
    )
    parser.add_argument("--idx",         type=int,  default=0)
    parser.add_argument("--question",    type=str,  default=None)
    parser.add_argument("--skip-llm",    action="store_true")
    parser.add_argument("--trace-calls", action="store_true",
                        help="sys.settrace: zeigt jeden src/-Funktionsaufruf")
    parser.add_argument("--no-color",    action="store_true")
    args = parser.parse_args()

    if args.no_color:
        USE_COLOR = False

    # ── Frage laden ──────────────────────────────────────────────────────────
    if args.question:
        q_text, gold, q_type = args.question, "(unbekannt)", "custom"
    else:
        q = load_question(args.idx)
        q_text = q["question"]
        gold   = q.get("answer", "?")
        q_type = q.get("question_type", "?")

    # ── Config laden ─────────────────────────────────────────────────────────
    import yaml
    with open(PROJECT_ROOT / "config" / "settings.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ── Header ───────────────────────────────────────────────────────────────
    print("\n" + bold("═" * 72))
    print(f"  {bold(cyan('PIPELINE VERBOSE TRACE'))}")
    print(bold("═" * 72))
    field("Frage",        q_text)
    field("Gold-Antwort", gold)
    field("Typ",          q_type)
    if args.skip_llm:
        print(f"\n  {yellow('⚠ --skip-llm: Verifier wird NICHT ausgeführt')}")
    if args.trace_calls:
        print(f"\n  {blue('ℹ --trace-calls: alle src/-Funktionsaufrufe werden geloggt')}")

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

    if not args.skip_llm:
        patch_verifier(verifier)
    else:
        def _skip_verifier(query, context, entities=None, hop_sequence=None):
            from src.logic_layer import VerificationResult
            section("S_V — VERIFIER  (übersprungen via --skip-llm)")
            field("Kontext-Chunks", len(context))
            if context:
                for i, c in enumerate(context):
                    chunk_block(i, c)
            else:
                print(f"    {red('⚠ KEIN KONTEXT — Navigator hat 0 Chunks geliefert!')}")
                print(f"    {yellow('  → Mögliche Ursachen: Retrieval-Fehler, Entity-Mention-Filter zu aggressiv')}")
            result = VerificationResult(
                answer="[übersprungen]",
                iterations=0,
                verified_claims=[],
                violated_claims=[],
                all_verified=False,
                pre_validation=None,
                iteration_history=[],
                timing_ms=0.0,
            )
            subsection("VERIFIER ERGEBNIS  (simuliert)")
            field("Antwort",     result.answer)
            field("Konfidenz",   result.confidence.value)
            field("Iterationen", result.iterations)
            return result
        verifier.generate_and_verify = _skip_verifier

    # ── Pipeline ausführen ────────────────────────────────────────────────────
    t_start = time.time()
    final_state = controller.run(q_text)
    total_ms = (time.time() - t_start) * 1000

    # ── sys.settrace deaktivieren ─────────────────────────────────────────────
    if args.trace_calls:
        sys.settrace(None)
        print_call_trace()

    # ── Zusammenfassung ───────────────────────────────────────────────────────
    section("ZUSAMMENFASSUNG")
    pred = final_state.get("answer", "")
    field("Frage",       q_text)
    field("Gold",        gold)
    field("Vorhersage",  pred)
    field("Gesamtdauer", f"{total_ms:.0f} ms")

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
    print(f"\n  Ergebnis: {bold(green('✓ RICHTIG') if em else red('✗ FALSCH'))}")

    errors = final_state.get("errors", [])
    if errors:
        print(f"\n  {red('Pipeline-Fehler:')}")
        for e in errors:
            print(f"    {red('!')} {dim(e)}")

    print(f"\n{bold('═' * 72)}\n")


if __name__ == "__main__":
    main()
