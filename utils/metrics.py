import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Download required NLTK resources (only runs once)
nltk.download('punkt', quiet=True)

def compute_accuracy(results):
    correct = sum(1 for r in results if r.get("Match"))
    return correct / len(results) if results else 0.0

def compute_bleu(reference, candidate):
    ref_tokens = [reference.split()]
    cand_tokens = candidate.split()
    smoothie = SmoothingFunction().method4
    return sentence_bleu(ref_tokens, cand_tokens, smoothing_function=smoothie)
