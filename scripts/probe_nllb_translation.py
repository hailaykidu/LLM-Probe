"""Evaluate NLLB-200 on English->Tigrinya against the same gold translations.

NLLB has Tigrinya (tir_Ethi) as an explicitly supported target language,
unlike the general-purpose LLMs benchmarked in tasks/translation_fidelity.py.
This establishes whether the near-zero BLEU those models produce is a ceiling
of the task/metric or of the models.

Scored identically to scripts/compute_bleu.py so numbers are comparable.
Gold data is untouched; results are written to a separate probe file, not to
the canonical translation_fidelity_<model>.json set.
"""
import argparse
import json
import os
import re
import sys

import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.chrf_score import corpus_chrf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.nllb_loader import load_model  # noqa: E402

SMOOTH = SmoothingFunction().method4
MARKUP_RE = re.compile(r"[*_`]+")
# Ethiopic wordspace/full-stop/comma plus ASCII equivalents, at either end.
EDGE_PUNCT_RE = re.compile(r"^[\s፠-፨.,;:]+|[\s፠-፨.,;:]+$")
SEP_RE = re.compile(r"[፣,]")


def clean(text):
    """Normalize output form only -- never substitutes or reorders content.

    NLLB emits a single lexicon term but sometimes decorates it: a trailing
    Ethiopic comma ("ስክሩብ፣"), or the same term repeated as a comma list
    ("ጎደና፣ ጎደና"). Both are surface artifacts of decoding a one-word target,
    so the punctuation is trimmed and an all-identical repeat list collapses
    to its single distinct term. A list with genuinely different terms is
    left alone -- collapsing that would be choosing an answer for the model.
    """
    text = " ".join(MARKUP_RE.sub("", str(text)).split()).strip()
    parts = [EDGE_PUNCT_RE.sub("", p) for p in SEP_RE.split(text)]
    parts = [p for p in parts if p]
    if parts and len(set(parts)) == 1:
        return parts[0]
    return EDGE_PUNCT_RE.sub("", text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="facebook/nllb-200-distilled-600M")
    ap.add_argument("--n", type=int, default=0, help="0 = all rows")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gold = pd.read_json(os.path.join(root, "data/gold_labels/translations.json"))
    gold = gold.rename(columns={"Reference": "ExpectedTigrigna"})
    gold = gold.drop_duplicates(subset=["English", "ExpectedTigrigna"]).dropna(
        subset=["English", "ExpectedTigrigna"]
    )
    if args.n:
        gold = gold.sample(n=min(args.n, len(gold)), random_state=args.seed)

    print(f"checkpoint={args.checkpoint}  rows={len(gold)}")
    pipe = load_model(args.checkpoint)

    inputs = [str(e) for e in gold["English"]]
    hyps = [clean(r["generated_text"]) for r in pipe(inputs)]
    refs = [[x.strip() for x in str(v).split(",") if x.strip()] for v in gold["ExpectedTigrigna"]]

    pairs = [(r, h) for r, h in zip(refs, hyps) if r]
    refs, hyps = [p[0] for p in pairs], [p[1] for p in pairs]

    char_refs = [[list(x.replace(" ", "")) for x in r] for r in refs]
    char_hyps = [list(h.replace(" ", "")) for h in hyps]
    word_refs = [[x.split() for x in r] for r in refs]
    word_hyps = [h.split() for h in hyps]

    res = {
        "checkpoint": args.checkpoint,
        "n": len(hyps),
        "chrf": corpus_chrf([r[0] for r in refs], hyps) * 100,
        "bleu4_char": corpus_bleu(char_refs, char_hyps, smoothing_function=SMOOTH) * 100,
        "bleu4_word": corpus_bleu(word_refs, word_hyps, smoothing_function=SMOOTH) * 100,
        "exact": sum(h in r for r, h in zip(refs, hyps)) / len(hyps) * 100,
    }

    print(f"\n{'n':>7}{'chrF':>9}{'BLEU-4c':>10}{'BLEU-4w':>10}{'exact%':>9}")
    print("-" * 45)
    print(f"{res['n']:>7}{res['chrf']:>9.2f}{res['bleu4_char']:>10.2f}"
          f"{res['bleu4_word']:>10.2f}{res['exact']:>9.2f}")

    print("\nfirst 15 outputs:")
    for i in range(min(15, len(hyps))):
        mark = "OK " if hyps[i] in refs[i] else "   "
        print(f"  {mark}{str(gold['English'].iloc[i]):<22} ref={refs[i][0]:<16} got={hyps[i]}")

    tag = args.checkpoint.split("/")[-1]
    dest = os.path.join(root, "results/evaluation_reports/translation_fidelity",
                        f"nllb_probe_{tag}.json")
    res["samples"] = [
        {"English": str(gold["English"].iloc[i]), "Expected": refs[i], "Output": hyps[i]}
        for i in range(min(60, len(hyps)))
    ]
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\nwritten to {dest}")


if __name__ == "__main__":
    main()
