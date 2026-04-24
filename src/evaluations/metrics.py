"""
Canonical evaluation metrics for the Edge-RAG thesis evaluation.

All three evaluation entry-points (evaluate_hotpotqa, ablation_study,
agent_pipeline.BatchProcessor) import from here to guarantee identical
metric computation across every reported number.

Metrics follow the HotpotQA official evaluation script:
    Yang et al. (2018). HotpotQA: A Dataset for Diverse, Explainable
    Multi-hop Question Answering. EMNLP 2018.
    https://hotpotqa.github.io/

Normalisation pipeline (normalize_answer):
    1. Lowercase
    2. Remove punctuation
    3. Remove articles (a, an, the)
    4. Collapse whitespace

Exact Match (compute_exact_match):
    Returns True when the normalised strings are identical OR when the
    normalised gold string appears as a whole-word span inside the
    normalised prediction (word-boundary anchored to avoid substring false
    positives such as "no" matching "cannot").

F1 (compute_f1):
    Token-level F1 using occurrence counts (not set intersection), matching
    the official HotpotQA evaluator. This correctly handles repeated tokens
    like "the the" vs "the".
"""

import re
import string


def normalize_answer(text: str) -> str:
    """
    Apply the standard HotpotQA normalisation pipeline to a string.

    Steps: lowercase → remove punctuation → remove articles → collapse whitespace.
    """
    text = text.lower()
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text.strip()


def compute_exact_match(prediction: str, gold: str) -> bool:
    """
    Compute Exact Match (EM) between a model prediction and a gold answer.

    Returns True when:
    1. The normalised strings are identical, OR
    2. The normalised gold appears as a whole-word span inside the normalised
       prediction (word-boundary anchored substring check).

    The substring fallback handles cases where the model produces a correct
    answer embedded in a longer sentence (e.g. "The answer is Paris" vs "Paris").

    Args:
        prediction: Raw model output string.
        gold:       Ground-truth answer string.

    Returns:
        True if the prediction is considered an exact match.
    """
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)

    if pred_norm == gold_norm:
        return True

    if gold_norm and re.search(r"\b" + re.escape(gold_norm) + r"\b", pred_norm):
        return True

    return False


def compute_f1(prediction: str, gold: str) -> float:
    """
    Compute token-level F1 between a model prediction and a gold answer.

    Uses occurrence counts (not set intersection) so that repeated tokens
    are handled correctly.  This matches the official HotpotQA evaluator.

    F1 = 2 * precision * recall / (precision + recall)
    where precision = matched_tokens / prediction_tokens
    and   recall    = matched_tokens / gold_tokens

    Args:
        prediction: Raw model output string.
        gold:       Ground-truth answer string.

    Returns:
        F1 score in [0.0, 1.0].
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common_words = set(pred_tokens) & set(gold_tokens)
    if not common_words:
        return 0.0

    num_common = sum(
        min(pred_tokens.count(w), gold_tokens.count(w)) for w in common_words
    )

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)

    if precision + recall == 0.0:
        return 0.0

    return 2.0 * precision * recall / (precision + recall)
