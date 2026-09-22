"""A/B probe: does few-shot prompting raise translation BLEU?

Runs the existing one-shot prompt and the new few-shot prompt over the SAME
sample of gold rows for one model, and reports chrF / BLEU / exact-match for
each. Gold data is untouched -- only the prompt differs between arms.

Rows used as few-shot exemplars are excluded from the evaluation sample so a
model is never scored on an item whose answer is in its own prompt.

Writes results/evaluation_reports/translation_fidelity/fewshot_probe_<model>.json
and prints a comparison. This is a diagnostic, not a benchmark run: it does
not touch the canonical translation_fidelity_<model>.json files.
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

from models.span_infilling import to_span_infilling_prompts  # noqa: E402

SMOOTH = SmoothingFunction().method4
EXTRA_ID_RE = re.compile(r"<extra_id_\d+>")
OUTPUT_PREFIX_RE = re.compile(r"(?i)^\s*[\*_]*\s*(output|answer)\s*[\*_]*\s*:\s*")
MARKUP_RE = re.compile(r"[*_`]+")
GEEZ_RUN = re.compile(r"[ሀ-፿]+(?:\s+[ሀ-፿]+)*")

LOADERS = {
    "gemma-2b": "models.gemma_loader",
    "gemma-7b": "models.gemma7b_loader",
    "mistral-7b": "models.mistral_loader",
    "mt5-small": "models.mt5_loader",
    "mt5-large": "models.mt5_large_loader",
    "byt5": "models.byt5_loader",
    "qwen-7b": "models.qwen_loader",
    "falcon-7b": "models.falcon_loader",
}


def clean(text):
    text = EXTRA_ID_RE.sub("", str(text))
    text = OUTPUT_PREFIX_RE.sub("", text)
    text = MARKUP_RE.sub("", text)
    return " ".join(text.split()).strip()


def first_geez_span(text):
    """Longest-prefix Ge'ez span: what the model most likely intends as its answer.

    Applied identically to both arms, so it cannot bias the comparison.
    """
    m = GEEZ_RUN.search(str(text))
    return m.group(0).strip() if m else ""


def extract_text(response):
    if response is None:
        return ""
    if isinstance(response, list):
        if response and isinstance(response[0], dict):
            return response[0].get("generated_text") or response[0].get("text") or ""
        return str(response[0]) if response else ""
    if isinstance(response, dict):
        return response.get("generated_text") or response.get("text") or ""
    return str(response)


def score(refs_list, hyps):
    pairs = [(r, h) for r, h in zip(refs_list, hyps) if r]
    if not pairs:
        return {}
    refs_list, hyps = zip(*pairs)
    first_refs = [r[0] for r in refs_list]
    char_refs = [[list(x.replace(" ", "")) for x in r] for r in refs_list]
    char_hyps = [list(h.replace(" ", "")) for h in hyps]
    exact = sum(h in r for r, h in zip(refs_list, hyps)) / len(hyps) * 100
    return {
        "n": len(hyps),
        "chrf": corpus_chrf(first_refs, list(hyps)) * 100,
        "bleu4_char": corpus_bleu(char_refs, char_hyps, smoothing_function=SMOOTH) * 100,
        "exact": exact,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(LOADERS))
    ap.add_argument("--n", type=int, default=300, help="rows to sample")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gold = pd.read_json(os.path.join(root, "data", "gold_labels", "translations.json"))
    gold = gold.rename(columns={"Reference": "ExpectedTigrigna"})
    gold = gold.drop_duplicates(subset=["English", "ExpectedTigrigna"]).dropna(
        subset=["English", "ExpectedTigrigna"]
    )

    one_shot = open(os.path.join(root, "data/prompts/translation_fidelity.txt"), encoding="utf-8").read()
    few_shot = open(os.path.join(root, "data/prompts/translation_fidelity_fewshot.txt"), encoding="utf-8").read()

    # Exclude any row whose English appears as a few-shot exemplar.
    exemplars = set(re.findall(r"^English:\s*(.+)$", few_shot, re.M))
    exemplars |= set(re.findall(r"^English:\s*(.+)$", one_shot, re.M))
    exemplars.discard("{english_sentence}")
    sample = gold[~gold["English"].astype(str).isin(exemplars)]
    sample = sample.sample(n=min(args.n, len(sample)), random_state=args.seed)
    print(f"model={args.model}  sample={len(sample)} rows  (excluded {len(exemplars)} exemplar phrases)")

    import importlib

    pipe = importlib.import_module(LOADERS[args.model]).load_model()

    refs = [[x.strip() for x in str(v).split(",") if x.strip()] for v in sample["ExpectedTigrigna"]]
    out = {"model": args.model, "n": len(sample), "arms": {}}
    records = []

    for arm, template in (("one_shot", one_shot), ("few_shot", few_shot)):
        prompts = [template.replace("{english_sentence}", str(e)) for e in sample["English"]]
        prompts = to_span_infilling_prompts(prompts, args.model)
        raw = [extract_text(r) for r in pipe(prompts)]
        cleaned = [clean(t) for t in raw]
        spans = [first_geez_span(t) for t in cleaned]

        out["arms"][arm] = {
            "as_written": score(refs, cleaned),
            "geez_span": score(refs, spans),
        }
        records.append((arm, raw, cleaned, spans))

    print(f"\n{'arm':<10}{'variant':<12}{'n':>6}{'chrF':>9}{'BLEU-4c':>10}{'exact%':>9}")
    print("-" * 56)
    for arm in ("one_shot", "few_shot"):
        for variant in ("as_written", "geez_span"):
            s = out["arms"][arm][variant]
            if s:
                print(f"{arm:<10}{variant:<12}{s['n']:>6}{s['chrf']:>9.2f}{s['bleu4_char']:>10.2f}{s['exact']:>9.2f}")

    dest = os.path.join(root, "results/evaluation_reports/translation_fidelity",
                        f"fewshot_probe_{args.model}.json")
    out["samples"] = [
        {
            "English": e, "Expected": r,
            **{f"{arm}_raw": rec[1][i] for arm, rec in zip(("one_shot", "few_shot"), records)},
            **{f"{arm}_span": rec[3][i] for arm, rec in zip(("one_shot", "few_shot"), records)},
        }
        for i, (e, r) in enumerate(zip(sample["English"], sample["ExpectedTigrigna"]))
    ][:40]
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nwritten to {dest}")


if __name__ == "__main__":
    main()
