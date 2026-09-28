"""Re-score saved model outputs under defensible matching rules.

No inference: every number here is computed from the ModelOutput fields
already stored in results/evaluation_reports/<task>/<task>_<model>.json by
the September 2026 runs. The experimental records are unchanged; only the
scoring rule differs.

The task scripts score POS and morphosyntax with
`bool(expected_tags & output_tags)` -- a set intersection over whole answers
that may run to 128 tokens. An answer that names several parts of speech is
counted correct as soon as one of them matches, so verbose models are
rewarded for hedging. This reports three rules side by side:

  set_intersection  what the task scripts computed (for comparison)
  first_token       the model's first word, the answer it actually commits to
  exact             the whole cleaned answer equals the gold label

and two reference points a model must beat to demonstrate anything:

  majority          always answer the most frequent gold label
  english_only      an English POS tagger that never sees the Tigrinya

Usage:  python scripts/rescore_tasks.py [--task pos_tagging] [--json OUT]
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WORD_RE = re.compile(r"[a-z]+")
# Prompt example the morphosyntax template shows the model; answers that
# merely echo it are not evidence of analysis.
MORPHO_EXAMPLE = "preposition, noun, singular"

TASKS = {
    "pos_tagging": "ExpectedTags",
    "morphosyntax_probe": "ExpectedTags",
}


def tags(text):
    return set(WORD_RE.findall(str(text).lower()))


def first_token(text):
    words = WORD_RE.findall(str(text).lower())
    return words[0] if words else ""


def score_rows(rows, gold_key):
    n = len(rows)
    if not n:
        return None
    si = ft = ex = echo = 0
    for r in rows:
        gold = tags(r[gold_key])
        out = str(r.get("ModelOutput", ""))
        if gold & tags(out):
            si += 1
        if first_token(out) in gold:
            ft += 1
        if str(out).strip().lower() == str(r[gold_key]).strip().lower():
            ex += 1
        if MORPHO_EXAMPLE in out.lower():
            echo += 1
    return {
        "n": n,
        "set_intersection": si / n * 100,
        "first_token": ft / n * 100,
        "exact": ex / n * 100,
        "echoes_prompt_example": echo / n * 100,
    }


def majority_baseline(rows, gold_key):
    counts = Counter(str(r[gold_key]).strip().lower() for r in rows)
    label, hits = counts.most_common(1)[0]
    return label, hits / len(rows) * 100


def english_only_baseline(rows, gold_key):
    """NLTK POS tagger given only the English gloss, never the Tigrinya."""
    try:
        import nltk
        for res, pkg in [("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
                         ("tokenizers/punkt_tab", "punkt_tab")]:
            try:
                nltk.data.find(res)
            except LookupError:
                nltk.download(pkg, quiet=True)
    except Exception:
        return None
    mapping = {
        "NN": "noun", "NNS": "noun", "NNP": "noun", "NNPS": "noun",
        "VB": "verb", "VBD": "verb", "VBG": "verb", "VBN": "verb",
        "VBP": "verb", "VBZ": "verb",
        "JJ": "adjective", "JJR": "adjective", "JJS": "adjective", "DT": "adjective",
        "RB": "adverb", "RBR": "adverb", "RBS": "adverb",
        "IN": "preposition", "PRP": "pronoun", "PRP$": "pronoun",
        "CC": "conjunction", "UH": "interjection",
    }
    ok = 0
    for r in rows:
        eng = str(r.get("English", "")).split(",")[0].strip()
        if not eng:
            continue
        toks = nltk.word_tokenize(eng)
        if not toks:
            continue
        tag = nltk.pos_tag(toks)[-1][1]
        if mapping.get(tag, "") in tags(r[gold_key]):
            ok += 1
    return ok / len(rows) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(TASKS), action="append")
    ap.add_argument("--json", help="write results to this path")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = {}
    for task in (args.task or sorted(TASKS)):
        gold_key = TASKS[task]
        pattern = os.path.join(root, "results/evaluation_reports", task, f"{task}_*.json")
        # "probe" appears in the morphosyntax_probe_* filenames themselves,
        # so only exclude the auxiliary probe outputs by their own prefixes.
        paths = sorted(p for p in glob.glob(pattern)
                       if not any(os.path.basename(p).startswith(s)
                                  for s in ("nllb_probe", "fewshot_probe"))
                       and not any(s in os.path.basename(p) for s in ("accuracy", "_bleu")))
        if not paths:
            continue

        print(f"\n=== {task} ===")
        print(f"{'model':<13}{'n':>7}{'set-int':>10}{'first-tok':>11}{'exact':>9}{'echo%':>9}")
        print("-" * 59)
        task_out = {}
        ref_rows = None
        for p in paths:
            model = os.path.basename(p)[len(task) + 1:-5]
            rows = json.load(open(p, encoding="utf-8"))
            rows = [r for r in rows if r.get(gold_key) is not None]
            s = score_rows(rows, gold_key)
            if not s:
                continue
            ref_rows = ref_rows or rows
            task_out[model] = s
            print(f"{model:<13}{s['n']:>7}{s['set_intersection']:>10.2f}"
                  f"{s['first_token']:>11.2f}{s['exact']:>9.2f}{s['echoes_prompt_example']:>9.1f}")

        if ref_rows:
            label, maj = majority_baseline(ref_rows, gold_key)
            eng = english_only_baseline(ref_rows, gold_key)
            print("-" * 59)
            print(f"{'BASELINE majority (always ' + label + ')':<41}{maj:>18.2f}")
            if eng is not None:
                print(f"{'BASELINE English-only tagger (no Tigrinya)':<41}{eng:>18.2f}")
            task_out["_baselines"] = {"majority_label": label, "majority": maj, "english_only": eng}
        out[task] = task_out

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\nwritten to {args.json}")


if __name__ == "__main__":
    main()
