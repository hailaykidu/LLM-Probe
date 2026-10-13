"""Recompute the translation-fidelity Match flag from saved outputs.

The task scorer split comma-separated gold synonyms without dropping empty
strings, so a reference ending in a trailing comma ("ምኽፋል ዕዳ,") admitted ""
as a valid answer and an empty generation counted as an exact match. This
rescores the saved per-model result files under the corrected rule, with no
model inference.

The matching rule is otherwise unchanged and mirrors
tasks/translation_fidelity.py: the normalized whole output, or any single
punctuation-stripped word within it, equal to any gold synonym.

Usage:
  python scripts/rescore_translation_match.py            # report only
  python scripts/rescore_translation_match.py --write     # rewrite the files
"""
import argparse
import glob
import json
import os
import re

RESULTS_DIR = os.path.join("results", "evaluation_reports", "translation_fidelity")

# Identical to tasks/translation_fidelity.py, so this scores what matching saw.
WORD_RE = re.compile(r"[^\s.,:;!?()\[\]{}*_\"'`]+")


def normalize(text):
    return str(text).strip().replace(" ", "").replace("​", "")


def output_words(text):
    return [normalize(w) for w in WORD_RE.findall(str(text))]


def candidates(expected, drop_empty):
    out = [normalize(c) for c in str(expected).split(",")]
    return [c for c in out if c] if drop_empty else out


def is_match(row, drop_empty):
    cands = candidates(row["ExpectedTigrigna"], drop_empty)
    out = row.get("ModelOutput", "")
    return normalize(out) in cands or any(w in cands for w in output_words(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    ap.add_argument("--write", action="store_true",
                    help="rewrite the result and accuracy files in place")
    args = ap.parse_args()

    skip = ("accuracy", "_bleu", "nllb_probe", "fewshot_probe")
    paths = sorted(
        p for p in glob.glob(os.path.join(args.results_dir, "translation_fidelity_*.json"))
        if not any(s in os.path.basename(p) for s in skip)
    )
    if not paths:
        raise SystemExit(f"no result files in {args.results_dir}")

    print(f"{'model':<13}{'rows':>7}{'old':>8}{'new':>8}{'empty-hits':>12}")
    print("-" * 48)
    for path in paths:
        model = os.path.basename(path)[len("translation_fidelity_"):-len(".json")]
        rows = json.load(open(path, encoding="utf-8"))

        old_hits = new_hits = empty_hits = 0
        for r in rows:
            old = is_match(r, drop_empty=False)
            new = is_match(r, drop_empty=True)
            old_hits += old
            new_hits += new
            if old and not new:
                empty_hits += 1
            r["Match"] = new

        n = len(rows)
        print(f"{model:<13}{n:>7}{100 * old_hits / n:>7.2f}%"
              f"{100 * new_hits / n:>7.2f}%{empty_hits:>12}")

        if args.write:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)
            acc_path = os.path.join(
                args.results_dir, f"translation_fidelity_accuracy_{model}.txt")
            with open(acc_path, "w", encoding="utf-8") as f:
                f.write(f"{new_hits / n:.4f}")

    if not args.write:
        print("\nreport only; pass --write to rewrite the files")


if __name__ == "__main__":
    main()
