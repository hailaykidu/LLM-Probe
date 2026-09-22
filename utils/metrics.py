"""Scoring metrics for LLM Probe.

Translation quality is reported with chrF and character-level BLEU rather
than word-level BLEU-4. Tigrinya references in this lexicon average ~1.37
whitespace tokens, so word-level 4-grams are largely undefined: a dedicated
MT system scoring 16.24% exact match on this data still yields
BLEU-4(word) = 0.24. Character-level metrics are the meaningful signal at
lexicon scale. `compute_bleu` is retained for the word-level figure but
should not be used as the headline translation metric.

Gold references may hold comma-separated synonyms ("ንእሽቶይ, ቁሩብ,ውሕድ"); each
is a valid translation, so the corpus-level helpers accept multiple
references per item.
"""
import nltk
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
from nltk.translate.chrf_score import corpus_chrf

nltk.download("punkt", quiet=True)

SMOOTH = SmoothingFunction().method4


def compute_accuracy(results):
    correct = sum(1 for r in results if r.get("Match"))
    return correct / len(results) if results else 0.0


def split_references(raw):
    """Comma-separated gold synonyms -> list of individual references."""
    return [r.strip() for r in str(raw).split(",") if r.strip()]


def compute_bleu(reference, candidate):
    """Word-level sentence BLEU-4. See module docstring: near-zero by
    construction on single-token references, retained for comparability."""
    ref_tokens = [reference.split()]
    cand_tokens = candidate.split()
    return sentence_bleu(ref_tokens, cand_tokens, smoothing_function=SMOOTH)


def compute_corpus_bleu(references, hypotheses, char_level=True):
    """Corpus BLEU-4 over multi-reference items.

    references: list of lists of reference strings (one list per item)
    hypotheses: list of hypothesis strings
    """
    if not hypotheses:
        return 0.0
    if char_level:
        refs = [[list(r.replace(" ", "")) for r in item] for item in references]
        hyps = [list(h.replace(" ", "")) for h in hypotheses]
    else:
        refs = [[r.split() for r in item] for item in references]
        hyps = [h.split() for h in hypotheses]
    return corpus_bleu(refs, hyps, smoothing_function=SMOOTH) * 100


def compute_chrf(references, hypotheses):
    """Corpus chrF. Uses the first reference per item (nltk's corpus_chrf is
    single-reference); scored against the primary gold translation."""
    if not hypotheses:
        return 0.0
    first = [item[0] for item in references]
    return corpus_chrf(first, list(hypotheses)) * 100


def score_translations(references, hypotheses):
    """All translation metrics for a model's outputs, as reported in the
    corrected results table."""
    pairs = [(r, h) for r, h in zip(references, hypotheses) if r]
    if not pairs:
        return {}
    refs, hyps = [p[0] for p in pairs], [p[1] for p in pairs]
    return {
        "n": len(hyps),
        "chrf": compute_chrf(refs, hyps),
        "bleu4_char": compute_corpus_bleu(refs, hyps, char_level=True),
        "bleu4_word": compute_corpus_bleu(refs, hyps, char_level=False),
        "exact": sum(h in r for r, h in zip(refs, hyps)) / len(hyps) * 100,
    }
