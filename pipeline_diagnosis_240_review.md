# Edge-RAG Pipeline Diagnosis — Senior-Engineer Review
## Categories A–F | Model: qwen2:1.5b | Dataset: HotpotQA | Traces T1–T9

> Companion analysis to `pipeline_diagnosis_findings.md`. Each category is reviewed
> in four steps: (1) data review & clarification check, (2) systemic root-cause
> analysis, (3) proposed architectural/code changes that generalize to the 500+
> set (not narrow per-trace patches), (4) per-change SWOT with cross-component
> impact. A final cross-category synthesis collapses the 7 categories into 3 root
> defects and gives a recommended build order.

**Context note:** the 500-question traces analyzed here were generated **after**
Fixes A/C/D/E from the prior engineering session (Pattern-F interrogative guard,
bridge-injection removal from the verifier scaffold, Pattern-F bare-pronoun guard,
sqrt-length-normalized reordering). This matters most for Category C — see below.

---

## Category A — Planner / Query Decomposition Failures

### Step 1 — Data Review & Clarification Check
The evidence is dense and self-consistent. Three open clarifications that fork specific recommendations:

1. **T1, T2** ("classified multi_hop but emitted 1 sub-query identical to the full query") is exactly the failure mode the §12.31 classification-decomposition consistency fallback was built to prevent, and which Fix D's connector-split path further addressed. **Were these traces generated before or after Fixes A/C/D/E?** If after, the consistency fallback is silently not firing for entity-less questions — a live bug, not the one the evidence implies.
2. **T4, T5** both use the `"Who or what is {X}?"` hop-1 template. Need the `matched_pattern` each carried: Pattern E relational-noun bridge, generic 2-hop fallback, or a separate "lookup" path? The fix location differs.
3. **T6** classified `single_hop` conf=0.500. Is 0.500 the **fallback floor confidence** (no-signal branch)? That distinguishes a scoring-threshold problem from a true classification miss.

### Step 2 — Systemic Root Cause Analysis
Eight traces, **three systemic root causes**, all from one decision: **the Planner's decomposition is entity-anchored, with no fallback when the anchor is absent, generic, or malformed.**

- **RC1 — Entity-absence collapses multi-hop to single-hop (T1, T2, T6).** When NER returns no usable PERSON/WORK/ORG anchor (T1: only a DATE; T2: partial song title + year; T6: misspelled name), the planner silently degrades to single-hop and emits the unsplit query — succeeds only by luck. Deep issue: the planner anchors on *entities* rather than the question's *syntactic structure*. "What arcade game is named after the player who…" is structurally a bridge regardless of whether NER found an entity — the relative clause is the bridge signal.
- **RC2 — `"Who or what is {X}?"` is a lookup, not a constraint (T4, T5).** T4: the "entity" is actually a constraint ("1993 HoF inductee") so the lookup pulls generic HoF articles. T5: the template was fed a *truncated* entity ("Woodland Gardens" not "Salisbury Woodland Gardens"), discarding the disambiguator. One template serves two semantically different hop types (resolution vs. constraint), and its input is sometimes corrupted upstream.
- **RC3 — Malformed/fragmented NER anchors trusted blindly (T7, T9, contributing to T4).** T7: "The Hook-Handed Man" split into 3 fragments. T9: "Are" absorbed into "Are  Chrysalis"; "Look" never extracted → both comparison hops share one anchor. No validation layer between NER output and sub-query construction.

Through-line: **single-pass, entity-trusting, template-driven decomposer with no structural fallback and no NER sanity layer.**

### Step 3 — Proposed Changes
- **A1 — Structure-first decomposition with relative-clause/bridge-syntax detector (RC1).** Fire on syntactic bridge signals independent of NER (SpaCy dep-parse, reusing Pattern E/G/H infra). When no named entity exists, hop-1's target is the relative clause *as a descriptive retrieval query*, not an entity lookup. Hard rule: if `query_type == multi_hop`/bridge, never emit a hop_sequence whose only sub-query equals the original — enforce with an assertion + structural fallback that always produces ≥2 distinct sub-queries.
- **A2 — Two-template hop generation: resolve vs. constrain (RC2).** Clean proper noun → `"Who or what is {X}?"` (resolution). Constraint signal (year, ordinal, superlative, role+temporal qualifier) → constraint-style sub-query ("Who was inducted into the Baseball Hall of Fame in 1993?"). Detection structural (4-digit year token, ordinal/superlative POS, measure-phrase), not a keyword list. Never truncate the entity (T5).
- **A3 — NER sanity/normalization layer between extraction and sub-query construction (RC3).** De-fragment overlapping hyphenated spans (keep longest); strip leading interrogatives ("Are "); backfill comparison anchors from sub-query text when NER yields fewer anchors than conjuncts ("Look"). *(This is the same layer as B1 — build once, serves A/B/C/E.)*
- **A4 — Confidence-gated structural override (T6).** When classifier confidence is at/near the fallback floor AND the question contains a latent bridge ("class of X", "[attribute] of the [thing] that…"), run A1's detector as a tie-breaker before committing to single-hop. Low confidence should *trigger* structural re-analysis, not lock in the default.

### Step 4 — SWOT
- **A1** — *S:* highest leverage; entity-independent; reuses SpaCy. *W:* descriptive hop-1 is a weaker retrieval signal; risk of over-decomposing genuine single-hops. *O:* feeds the iterative navigator (§12.37) which can resolve the bridge from partial results. *T:* cascading precision loss if it forces 2 hops on a misclassified single-hop — **must ship with A4's gate**; +1 retrieval pass latency.
- **A2** — *S:* fixes generic-article noise (T4); ~0 latency. *W:* constraint detection is heuristic. *O:* better alignment with how gold paragraphs are written → lifts temporal/ordinal SF-Recall. *T:* low; the T5 full-span fix could interact with the entity-mention filter (verify vs Category E).
- **A3** — *S:* removes noise at the source; benefits A/B/C/E. *W:* comparison-anchor backfill is the trickiest piece. *O:* highest cross-category multiplier. *T:* over-aggressive de-fragmentation could drop a legitimately distinct short entity — only de-fragment on offset *overlap*, not mere substring.
- **A4** — *S:* targets *uncertain* (not wrong) classifications; low blast radius. *W:* helps only a narrow confidence band. *O:* logged "low-confidence → structural recovery" rate for the thesis. *T:* amplifies A1 false positives precisely where the system is shaky — tie activation to a hard structural signal.

**Sequencing:** A3 first (deterministic, de-risks A1/A2). A1+A4 together (A4 is the safety valve against over-decomposition). A2 last (verify T5 full-span against Category E first).

---

## Category B — NER / Entity Extraction Failures

### Step 1 — Data Review & Clarification Check
Tightly coupled to Category A (T1/T4/T7/T9 are the same failures from the NER side). Four clarifications:

1. **GLiNER threshold** is 0.15 (recall-optimized) — won't filter the conf=0.75–1.00 wrong entities. **Is "Look" (T9) dropped at threshold, or never proposed by GLiNER at all?** Threshold problem vs. label-coverage problem — different fix.
2. **Entity-type whitelist** has no "date" type, yet T1/T2/T4 show DATE entities. **Is there a separate SpaCy pass in the Planner producing these, distinct from GLiNER?** The fix location differs.
3. **Graph node-matching:** does the store match exact strings only, or is the Hop-0 alias cascade (exact → CONTAINS → substring) firing for T4's "National 1993 Baseball Hall of Fame"?
4. **Misspelling (T6):** is "Apatim"→"Apratim" a recurring class in the 500-set, or a one-off? Fuzzy matching isn't justified for a single trace.

### Step 2 — Systemic Root Cause Analysis
Six traces → **three root causes; none is "GLiNER is bad."** The failures are in the **absence of a post-extraction contract** between NER and its two consumers (graph anchors, filter tokens).

- **RC1 — No span-boundary normalization (T7, T9, T4).** Interrogative absorption ("Are  Chrysalis"); hyphen fragmentation ("The Hook-Handed Man" → 3 spans); numeric absorption ("National 1993 Baseball Hall of Fame"). All **boundary** errors, not classification errors — content is right, extent is wrong.
- **RC2 — Filter tokens and graph anchors have no quality gate (T1, T2).** T1: only entity is a DATE → graph anchor matches no node, and after DATE-filtering the entity-mention filter has nothing → silently disables → passes all 10 chunks. T2: "Something" (common word) → retains irrelevant chunks. No notion of "is this entity *useful* as an anchor?"
- **RC3 — No surface-form reconciliation between question and corpus (T6, partially T4).** Typo (T6) or compositional mismatch (T4) → exact graph matching fails → survives only on partial/surname matches by luck.

Through-line: **NER output is consumed raw — no boundary repair, no discriminativeness gate, no surface-form reconciliation.**

### Step 3 — Proposed Changes
- **B1 — Span-boundary normalization pass (RC1).** (1) Trim leading stopwords/interrogatives (`is/are/was/were/do/does/did/who/what/which/the/a/an`) — "Are  Chrysalis" → "Chrysalis"; collapse double-space. (2) Consolidate hyphenated expressions to the maximal hyphen-bounded span; drop overlapping fragments. (3) Strip embedded 4-digit years for anchor purposes — emit year-stripped entity as the graph anchor AND the year as a separate temporal constraint (consumed by A2). Must run where raw GLiNER char-offsets are still available.
- **B2 — Discriminativeness gate on filter tokens and graph anchors (RC2).** Reject as filter token if the entity is a single common English word (kills "Something"). Reject as graph anchor if it's a pure temporal/measure phrase with no proper-noun token (kills "7 consecutive seasons"). **Critical: when all entities are rejected, do NOT silently pass all chunks** — fall back to a defined behavior (skip filtering + log a structural warning + rely on RRF/reranker, or trigger A1 descriptive anchor). Generalizes the existing `_is_junk_entity`/DEFAULT_STOPLIST work to also gate graph anchors.
- **B3 — Corpus-aware surface-form reconciliation (RC3) — *conditional on clarification #4*.** Fuzzy fallback (Levenshtein ≤2, same first char) when exact+substring graph match is below a floor (T6: 0.025). Compositional fallback (year-stripped form from B1.3) for T4. Gate B3 on whether typos recur.

### Step 4 — SWOT
- **B1** — *S:* deterministic, ~0 latency, four downstream beneficiaries (graph, filter, reranker, bridge). *W:* year-split is inert without A2; hyphen-merge needs char offsets. *O:* precondition for trustworthy graph anchoring across the 500-set. *T:* over-trimming "The Who"/"The Hook-Handed Man" — only trim leading interrogatives/auxiliaries unconditionally; trim "The/A/An" only if the remaining span keeps ≥1 capitalized token.
- **B2** — *S:* fixes the most damaging *silent* failure (T1 disable-all); reuses stoplist infra. *W:* "discriminative" is a heuristic boundary. *O:* the "never silently no-op on empty input" rule generalizes pipeline-wide. *T:* **The "Look" vs "Something" paradox** — a flat stoplist cannot suppress "Something" (T2) without also killing "Look" (T9). Resolution: gate on **syntactic role** (comparison conjunct, relcl head), not dictionary frequency. This couples B2 to A's structural parser; they cannot be solved independently.
- **B3** — *S:* recovers luck-only gold (T6). *W:* edit-distance scan needs an index (compounds the existing O(N) pandas scan). *O:* a reportable robustness result. *T:* false merges ("Allen"→"Allan") — bound tightly (same first char, length ±1, last-resort only).

**Key finding:** B1 (span normalization) **is the same layer as A3** — build once, pays off in A/B/C/E. The "Look" vs "Something" paradox is the subtlest issue: a structure-aware gate, not a flat stoplist.

---

## Category C — Bridge Entity Extractor Failures

### Step 1 — Data Review & Clarification Check (resolved: traces are POST-Fix-C)
1. **Relationship to Fix C — confirmed post-Fix-C.** Fix C removed bridge-entity injection from the **verifier prompt scaffold only**; it did NOT touch `_extract_bridge_entities` or the iterative navigator's `_rewrite_hop_query_with_bridges`/reranker hints (§12.37). So wrong bridges ("Nora Ephron", "National League", "Salisbury Gardens") **still poison hop-2 retrieval** (query rewrite + `[ENTITIES: …]` reranker hints), but no longer poison the verifier scaffold. The damage path narrowed from two channels (prompt + retrieval) to one (retrieval) — and the surviving channel is the consequential one (determines whether hop-2 gold is retrieved).
2. **§12.32 query-aware scoring (`_score_bridge_candidate`) — is it live in these traces?** T4 says "scored all 8 chunks equally," but §12.32 should apply a position penalty + type bonus. The T4 fix differs entirely depending on whether that scorer is running.
3. **Exclude-list mechanics:** in T5 a spurious bridge was built from a surname *of an exclude entity*; in T8 the needed target ("10,000 metres") was *in* the exclude list (as part of "1967 Pan American Games"?) and suppressed. Confirm how the exclude list is built (`plan.entities`?).

### Step 2 — Systemic Root Cause Analysis
Seven traces (5 real + 2 N/A). **Meta-observation: the extractor is asked to do an impossible job in 3 of 5 "failures."**
- **T1, T6:** no bridge step ran (upstream Category A).
- **T8:** the correct target ("10,000 metres" article) **was never in hop-1 chunks** (Category D). The extractor cannot extract what isn't there.

**Only T4, T5, T7 are genuine extractor failures.** Cascade-masking overstates the extractor's defect rate.

- **RC1 — No chunk-rank prior in candidate scoring (T4, T7).** Pass 2 scores by *local* signals (proximity, type, position) but ignores the **rank of the source chunk**. Reggie Jackson (chunk #1, top-RRF) loses to "National League"/"Goose Goslin" from lower-ranked noise chunks. The RRF rank *is* the system's best estimate of which chunk holds the answer — discarding it throws away the strongest prior.
- **RC2 — Pass ordering lets weak heuristics short-circuit strong ones (T5).** Pass 1 (surname-anchor) runs before Pass 2 and *returns early*. In T5, Pass 1 fired on "Salisbury" (from an exclude entity), built spurious "Salisbury Gardens", returned — so Pass 2 never reached "Thomas Mawson". Cascade priority-ordered by heuristic *specificity*, not *confidence*.
- **RC3 — Span fragmentation propagates into bridge candidates (T7).** "Lemony Snicket's A Series of Unfortunate Events" fragmented by Pass 2's proper-noun regex — same RC1 from Category B resurfacing inside the extractor.

### Step 3 — Proposed Changes
- **C1 — Chunk-rank prior in `_score_bridge_candidate`.** `final = local_score × rank_prior(source_rank)` with soft decay (`1/(1+rank)`, or a top-2 boost mirroring §12.33). Recovers Reggie Jackson (chunk #1) over National League (noise chunk); generalizes because answer entities cluster in top-ranked chunks.
- **C2 — Confidence-scored single-pass selection, not priority-ordered early-return.** Passes 0/1/2 become *candidate generators*, each attaching calibrated confidence (Pass 1 surname-reconstruction is inherently lower-confidence than a clean Pass 2 match); select top-3 from the merged pool. Eliminates the T5 short-circuit. Additionally: reject Pass-1 reconstructions that are substring-variants of an exclude entity.
- **C3 — Inherit normalized spans from B1** instead of re-detecting proper nouns with an independent regex. Keeps the full film title whole (T7). Consolidation, not new logic.
- **C4 — Empty/absent-target confidence floor (T8 manifestation).** Below a confidence floor, return empty (no bridge) so hop-2 falls back to the original query — converts a *harmful* confidently-wrong bridge into a *neutral* one.

### Step 4 — SWOT
- **C1** — *S:* aligns extractor with the retriever's strongest prior; recovers T4/T7; cheap. *W:* amplifies a wrong rank-1 chunk. *O:* pairs with the iterative navigator. *T:* **retrieval-error amplification** — conditional on hop-1 quality; use soft decay, not top-1-only.
- **C2** — *S:* eliminates the short-circuit class; debuggable confidences. *W:* per-pass weight calibration burden. *O:* unifies with C1/C4 into one scorer. *T:* could regress questions that worked by lucky pass-ordering — validate net delta on the 500-set.
- **C3** — *S:* removes fragmentation with zero new logic. *W:* hard dependency on B1. *O:* single source of truth for span handling. *T:* low (very long titles → let the graph alias cascade handle, don't re-fragment).
- **C4** — *S:* converts the most harmful failure into neutral; cheap. *W:* floor is a recall/precision dial. *O:* C1's rank-prior gives a principled confidence to threshold on. *T:* too-high floor falls back to single-hop-equivalent behavior.

**Post-Fix-C tightening of C4:** the floor should gate *both* the hop-2 query rewrite *and* whether hop-2's chunks are merged into the final verifier context (§12.37 accumulates raw+filtered context across hops) — a confidently-wrong bridge shouldn't inject its noise chunks into answer generation.

**Key findings:** the extractor's defect rate is overstated by cascade-masking (only T4/T5/T7 real). C1/C2/C4 should be designed as **one scoring function**, not three patches. The dominant threat (C1/C4) is retrieval-error amplification — couples C's safety to Category D's quality.

---

## Category D — Retrieval / Missing Gold Paragraph

### Step 1 — Data Review & Clarification Check
1. **T4 — does "Greatest Sports Legends" exist in the corpus at all?** Grep `_TEXT_TO_TITLE`/corpus. If gold isn't in the corpus, the question is **unanswerable and should be excluded from the denominator, not counted as a pipeline failure.** Single most important fact in the category.
2. **T8/T9 — is the `max_docs=5` the same cap Fix E reorders into, and was Fix E live?** If yes, length-normalization alone didn't rescue these → confirms D2 (provenance-awareness) is the missing piece.
3. **`max_context_chunks=8` vs `max_docs=5` discrepancy.** Two caps appear: navigator keeps 8, `_format_context` truncates to 5. Which is binding (the verifier sees 5 or 8)?

### Step 2 — Systemic Root Cause Analysis
Seven traces → **four classes; only one is true retrieval.**
- **Class 1 — Cascade from single-hop collapse (T1, T6) — NOT Category D.** hop-2 never ran; the retriever was never *asked*. Belongs to Category A's accounting.
- **Class 2 — Genuine unreachable gold (T4) — the only true retrieval failure.** Sub-cases: absent from corpus (unanswerable) vs. present-but-graph-isolated (the fixable defect — connects to isolated-entity/§3f and Ed-Wood disambiguation).
- **Class 3 — `max_docs` cap severs retrieved gold (T8, T9) — the dominant *fixable* failure.** Gold retrieved, survived every filter, then cut by `_format_context`'s top-5 truncation. T8: gold at #6, answer sentence at #8. T9: gold at #9. The reorder that decides which 5 survive (Fix E) scores on **keyword overlap only** — no signal for "graph-anchored bridge target" (T8) or "second comparison entity's article" (T9).
- **Class 4 — Reorder scoring ignores entity-anchor provenance (T9).** The "Look" article has zero keyword overlap (the question is *about* it but doesn't describe it), so Fix E scored it near-zero. **Reorder relevance ≠ structural necessity.**

Through-line: **the system retrieves better than it preserves. Gold reaches the final stage and is discarded by a cap + a reorder optimizing for keyword relevance over structural necessity.**

### Step 3 — Proposed Changes
- **D1 — Structural-necessity protection in the context window (T8, T9).** Before `_format_context` truncates, reserve guaranteed slots: comparison → ≥1 chunk per comparison entity (T9); bridge/multi-hop → the hop-2 graph-anchored chunk (T8). Tag chunks with provenance (`_matched_entity`, `_hop`, `_graph_anchored`); RRF fusion already tags `_best_sub_query` (§12.30) so the infra exists. Cap fills reserved slots first, then ranks the remainder by Fix-E score.
- **D2 — Make reorder scoring provenance-aware, not keyword-only (T9 root).** Augment Fix E's `_score` with a structural term: a chunk graph-anchored on a query/bridge entity, or the article for a comparison conjunct, gets a score floor keyword-overlap cannot drop it below. Natural extension of Fix E (E fixed length bias; D2 fixes necessity blindness).
- **D3 — Raise/soften `max_docs` with confidence-aware sizing (Class 3 mechanism).** (a) Decouple caps: if navigator keeps 8 but `_format_context` truncates to 5, raise to match (recovers T8's #6). (b) Token-budget instead of doc-count (short gold rides along cheaply). **Ship behind D1/D2, never alone.**
- **D4 — Corpus-presence audit + denominator correction (T4 measurement).** Verify gold exists in the corpus before counting "missing gold" as a pipeline failure. Exclude un-ingested-gold questions from the EM denominator (report separately as corpus-coverage). T4-class failures may be inflating the 55% pipeline-failure rate.

### Step 4 — SWOT
- **D1** — *S:* recovers gold already paid for; reuses §12.30 tags; highest ROI. *W:* depends on correct provenance tagging (coupled to Category C bridge correctness). *O:* "reserved slot" generalizes; likely lifts comparison EM beyond the flat 50%. *T:* reserving slots evicts general-relevance chunks — with cap=5, reserving 2 leaves only 3 free; pairs best with D3.
- **D2** — *S:* fixes the *root* of T9; small extension to Fix E. *W:* three scoring terms to balance. *O:* unifies with D1 (D2's structural score *is* D1's slot signal). *T:* a wrong bridge entity (Category C) gets the wrongly-anchored chunk promoted.
- **D3** — *S:* T8's #6 recovered free by 5→8. *W:* **directly fights the latency/timeout fix** — bigger context = slower SLM, the 60s cliff returns. *O:* token-budget sizing. *T:* **most dangerous change** — lost-in-the-middle for a 1.5B model; gold at position #7 may be retrieved-but-ignored. **Never ship alone.**
- **D4** — *S:* measurement integrity; true failure rate likely <55%. *W:* improves no answer. *O:* a defensible thesis statistic (answerable-subset EM). *T:* reviewer scrutiny — report both raw and answerable-subset numbers with a principled exclusion criterion.

**Key findings:** "Missing gold" is four bugs; only T4 is retrieval (T1/T6→A, T8/T9→cap, T5→none). **The largest fixable sub-class (T8/T9) is a preservation bug, not a retrieval bug.** D1+D2 are one provenance-aware preservation layer, shipped before D3. D4 may be the highest-value item for the thesis.

---

## Category E — Entity-Mention Filter Over-Aggressiveness

### Step 1 — Data Review & Clarification Check
1. **§12.33 Top-K RRF immunity — live in these traces?** Top-2 chunks immune to the filter. T7 (10→2) and T5 hop-2 (10→3) — survivors may be the immune top-2 plus matches. If active, T7's "one bad retrieval from dropping gold" risk is partly mitigated.
2. **Filter binary vs. score-boost.** T9: the Look chunk "had no positive survival signal" → implies a **hard binary gate** (mention→keep, else→drop), not a soft signal feeding the reorder. Does an entity match contribute a *score* that propagates to Fix-E reordering, or pure pass/drop?
3. **Empty-entity behavior (T1) — same `silent disable` as Category B?** Confirm it's one code path (the empty-entity branch) so it's fixed once.

### Step 2 — Systemic Root Cause Analysis
Seven traces → **two derivative classes (fix upstream) + two genuine filter-design defects.**
- **Derivative (T1, T2, T6, T9-input):** the filter faithfully executed on garbage NER input. T1 (empty list → silent disable), T2 ("Something" common word), T6 (misspelled token), T9 (malformed "Are Chrysalis" + missing "Look"). **All four are Category B; fixing the filter for them would be the narrow over-fitting the protocol forbids.** B1+B2 fix the input; filter behavior corrects automatically.
- **Genuine defect 1 — Hard binary gate destroys the survival signal (T9).** The filter is boolean pass/drop, so a kept chunk gains **no priority signal**. The "Look" chunk survived the filter but the downstream reorder demoted it to #9 and the cap killed it. A binary gate cannot express "keep AND prioritize."
- **Genuine defect 2 — No floor on survivor count → fragility (T5 hop-2, T7).** The filter can reduce 10→2 (T7) or 10→3 (T5) with no lower bound — "one bad retrieval from dropping gold." Even with perfect entities, an over-specific match can strand the verifier. §12.33's top-2 immunity is a partial version, but T7's 10→2 is itself the fragile minimum, not a floor.

Through-line: **the filter conflates "remove non-matching noise" with "make a hard keep/drop decision," when it should (a) emit a graded signal informing the reorder, and (b) respect a survivor floor.**

### Step 3 — Proposed Changes
- **E1 — Convert the filter from a hard gate to a graded re-ranking signal (defect 1: T9).** Compute an entity-match score per chunk (count/quality × B2 discriminativeness) and feed it into Fix-E's reorder as an additive term. Hard-drop retained only as a *tail* operation: drop chunks that have zero entity match AND rank below the survivor floor AND aren't structurally required. The "Look" chunk (matches a comparison entity) gets a strong positive score and rises, instead of merely not being dropped.
- **E2 — Survivor floor + structural-requirement protection (defect 2: T5, T7).** Never reduce below `N_floor` (≈ the verifier window); if matching would, keep the top-`N_floor` by RRF. Converts T7's 10→2 into a safe 10→5. Combined with D1, guarantees comparison/bridge gold chunks are never filtered out.

### Step 4 — SWOT
- **E1** — *S:* fixes information-destruction; recovers T9; unifies with Fix E + D2; a wrong entity now deprioritizes rather than deletes. *W:* more marginal chunks survive into ranking — benefit needs D1/D2 downstream; shipped alone could increase verifier noise. *O:* precision+recall signal; helps every comparison/bridge question. *T:* precision regression where the hard gate was correctly removing noise (T5/T7 hop-1) — keep the hard-drop tail; measure net on the 500-set.
- **E2** — *S:* eliminates the "one bad retrieval" fragility with a hard guarantee; extends §12.33 (2→N_floor). *W:* forces possibly-noisy chunks back in when the filter was correctly aggressive (T5 hop-2). *O:* lets E1 be aggressive safely (floor is the net). *T:* **interaction with the cap** — floor=5 + cap=5 leaves zero room for D1 reserved slots; floor, cap, and reserved slots must be one coherent budget.

**Key findings:** Category E is ~70% derivative — fix B1/B2 and four traces resolve. Only two genuine filter defects (hard-gate info loss, missing floor). **The filter, reorder, and cap are one system pretending to be three** — each discards information the next needs; T8/T9 die in the seams. Correct architecture: a single provenance-aware ranking-and-budgeting stage.

---

## Category F — max_docs Cap Cutting Gold Context

### Step 1 — Data Review & Clarification Check
1. **Cap interaction order.** `max_docs=5`, `max_chars_per_doc=800`, `max_context_chars=3500` (5×800=4000 > 3500, so the total can bind before doc-count). T1 shows 943→601 — **601 < 800, so a fourth implicit limit may exist.** Confirm the application order in `_format_context` and what produced 601.
2. **Is Fix E live in the F-traces?** T8/T9 show keyword-proximity reorder demoting gold; if Fix E was active, length-normalization alone didn't rescue them → confirms D2 (provenance) is the missing half.

### Step 2 — Systemic Root Cause Analysis
Seven traces → three ways. **The cap is the primary failure in only 2 of 7 (T8, T9), plus a char-truncation issue (T1).**
- **Class 1 — Cap innocent (T4, T5, T6, T7).** Gold within cap; failure upstream (missing article / single-hop) or answered correctly (reorder *helped* in T5).
- **Class 2 — Reorder demotes structurally-required gold below the cap (T8, T9) — the real defect (= Category D Class 4 at its lethal endpoint).** T8: "25 laps" chunk (low keyword overlap) ranked #6 → cut → SLM hallucinates "2" from a visible relay "second place." T9: generic "women's magazines" chunks score high; specific Chrysalis/Look articles land #7–9 → cut → SLM answers "No" from absence. **Keyword-overlap reordering systematically promotes generic chunks over specific gold** — the gold states the fact in *different words* than the abstract question.
- **Class 3 — Per-doc char truncation cuts answer-bearing tails (T1) — new, distinct defect.** Even within the doc cap, `max_chars_per_doc` head-truncation is position-blind: keeps first N chars regardless of where the answer sentence sits.

Through-line: **three position-blind, content-blind cuts (doc-count, per-doc chars, total chars) on top of a reorder that ranks by lexical echo rather than answer specificity. Gold dies between "retrieved" and "shown."**

### Step 3 — Proposed Changes
- **F1 — Answer-specificity term in reorder scoring (Class 2: T8, T9).** (a) **IDF weighting** over the candidate set: a term in many candidate chunks ("women's magazines" across 8 articles in T9) is down-weighted; a rare term ("Chrysalis"/"Look") becomes decisive. (b) **Entity-anchor/provenance bonus** (= D2 + E1): a graph-anchored chunk gets a score floor (T8's "25 laps" hop-2 target protected even with low overlap). This is the **missing half of Fix E** — Fix E removed *length* bias; F1 removes *genericness* bias (IDF) and *necessity* blindness (provenance).
- **F2 — Sentence-level, answer-aware truncation (Class 3: T1).** Replace position-blind `max_chars_per_doc` head-truncation with sentence-aware selection: keep the highest query/entity-relevance sentences (incl. bridge-entity mentions) so a tail answer survives. Reuses F1's scoring at sentence granularity.
- **F3 — Unify the cap into the budgeting module (consolidation of D1/D2/E1/E2/F1).** Replace `_format_context`'s three independent cuts with one budget allocation: score chunks on {RRF rank (C1), entity-match (E1), IDF-keyword (F1), provenance/necessity (D2)}; reserve required slots (D1); enforce a survivor floor (E2); fill remaining **token** budget by score with sentence-level selection (F2). The cap becomes a budget-aware fill that *cannot* evict structurally-required gold.

### Step 4 — SWOT
- **F1** — *S:* fixes the most damaging confabulation path (T8→"2", T9→"No"); IDF is cheap over ~8–10 in-hand chunks; compounds with Fix E. *W:* IDF over a tiny set is noisy; provenance bonus depends on Category C correctness; three-term calibration. *O:* helps every abstract-category-term comparison; could lift the flat 50% comparison EM. *T:* over-correcting toward rare terms (a typo could get huge IDF weight) — bound the term, add-one smoothing.
- **F2** — *S:* recovers answer tails (T1); content-aware; more answer per token (helps latency vs. raising the cap). *W:* segmentation cost (SpaCy already in stack); could drop a sentence a pronoun needs. *O:* same sentence-relevance primitive D2/F1 need. *T:* **coherence loss** for a 1.5B model — disjointed selected sentences may confuse more than a truncated-but-coherent passage. Test against the SLM specifically.
- **F3** — *S:* closes every seam T8/T9/T1 fell through; one tunable surface instead of 3 magic numbers; ablatable for the thesis. *W:* largest refactor; highest regression surface. *O:* maintainable, measurable methodological contribution. *T:* **integration over-subscription** (floor + reserved + cap must be one budget with explicit precedence: required > floor > score-fill); preserve currently-passing T5/T7 behavior — validate net delta on the 500-set.

---

## Final Cross-Category Synthesis (all 7 categories)

### The 7 categories are 3 root architectural defects, not 7 bugs
- **Defect I — Entity-anchored decomposition with no structural fallback** (Category A; upstream half of C/D). *Fix:* structure-first decomposition (A1), confidence-gated override (A4), never-degrade contract.
- **Defect II — Raw NER consumption with no normalization or quality gate** (Category B, E; input half of C). *Fix:* span-normalization layer (B1) + discriminativeness gate (B2). Resolves ~40% of all traces.
- **Defect III — Filter / reorder / cap are three blind stages that discard each other's information** (Categories D, E, F). *Fix:* one provenance-aware context-budgeting module (D1+D2+E1+E2+F1+F2+F3).

### Cascade-masking is severe and must shape measurement
Each named category overstated its own defect rate:
- **Category C:** 5 "failures," only 3 real (T1/T6 are A, T8 is D).
- **Category D:** 7 "failures," only 1 true retrieval (T4); 2 are A, 2 are F.
- **Category E:** 7 "failures," only 2 genuine filter defects; 4 are B.
- **Category F:** 7 "failures," only 2 cap-primary (T8/T9) + 1 char (T1); 4 innocent.

**Fix Defects I and II first, and the apparent failure counts in C/D/E/F drop without touching those components.** Measure each fix's impact only *after* upstream fixes land, or the gains will be mis-attributed.

### The two strongest single claims
- **Highest-ROI change: Defect II's NER normalization layer (B1+B2)** — deterministic, cheap, low-risk, resolves traces across B, C, E, and the inputs to D/F. Build first.
- **Most dangerous change: enlarging the cap (D3) — don't, alone.** T8/T9 prove gold dies from *mis-ranking*, not from too small a window. Enlarging without fixing the ranking (F1) just moves gold from "cut at #6" to "ignored at position #6" by a lost-in-the-middle 1.5B model.

### Recommended build order
1. **B1 + B2** — NER normalization + discriminativeness gate. Deterministic; unblocks everything; resolves the most traces.
2. **A1 + A4** — structure-first decomposition + confidence gate. Recovers the single-hop-collapse class (T1, T6, and bridge questions currently passing by luck).
3. **Unified context-budgeting module** (D1+D2+E1+E2+F1+F2) — closes the filter/reorder/cap seams. Ship F1's IDF + provenance *before* touching cap size.
4. **C1 + C2 + C4** — bridge scorer. Now safe: upstream retrieval/decomposition is better, so the rank-prior won't amplify garbage.
5. **D4** — corpus-coverage audit; report answerable-subset EM (measurement integrity).
6. **D3** — cap size, last, as a tuning knob within the budget, never alone.

### Open clarifications carried across the diagnosis
- A: fix-ordering of these traces; `matched_pattern` for T4/T5; is 0.500 the floor?
- B: is "Look" suppressed by threshold or never proposed; DATE provenance (GLiNER vs SpaCy); graph alias-cascade firing; do typos recur?
- C: is §12.32 `_score_bridge_candidate` live; how is the exclude list built?
- D: does "Greatest Sports Legends" exist in corpus; is Fix E live; binding cap 5 or 8?
- E: is §12.33 top-2 immunity live; filter binary or scored?
- F: cap application order; what produced T1's 601-char truncation; is Fix E live?

**Headline: you don't have 7 problems, you have 3 — and the cheapest one (NER normalization) unlocks the most.**
