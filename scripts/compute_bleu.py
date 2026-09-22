"""Compute BLEU for translation fidelity from already-saved model outputs.

Reads the per-model result JSONs written by tasks/translation_fidelity.py and
scores them offline -- no model inference, no GPU.

Reference fields hold comma-separated synonyms ("ንእሽቶይ, ቁሩብ,ውሕድ"); each is a
valid translation, so they are passed to sentence_bleu as multiple references
rather than one string.

Three variants are reported because the choice is not neutral for this data:
Tigrinya references here average ~1.4 whitespace tokens, so word-level BLEU-4
is structurally near-zero regardless of translation quality. chrF and
character-level BLEU are the meaningful signals; word BLEU-4 is included only
because it is what the unused utils.metrics.compute_bleu would have produced.
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import score_translations, split_references  # noqa: E402

# Same cleanup the task scorer applies, so BLEU sees what matching saw.
MARKUP_RE = re.compile(r"[*_`]+")


def clean(text):
    return " ".join(MARKUP_RE.sub("", str(text)).split()).strip()


def score_file(path):
    rows = json.load(open(path, encoding="utf-8"))
    rows = [r for r in rows if r.get("ExpectedTigrigna") and r.get("ModelOutput") is not None]
    if not rows:
        return None

    refs, hyps = [], []
    for r in rows:
        item = [x for x in (clean(v) for v in split_references(r["ExpectedTigrigna"])) if x]
        if not item:
            continue
        refs.append(item)
        hyps.append(clean(r["ModelOutput"]))

    scores = score_translations(refs, hyps)
    if not scores:
        return None
    # Exact-match is taken from the task scorer's own Match flag rather than
    # recomputed, so this column stays identical to the reported accuracy.
    scores["rows"] = scores.pop("n")
    scores["exact"] = sum(bool(r.get("Match")) for r in rows) / len(rows) * 100
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-dir",
        default="results/evaluation_reports/translation_fidelity",
        help="directory of translation_fidelity_<model>.json files",
    )
    args = ap.parse_args()

    pattern = os.path.join(args.results_dir, "translation_fidelity_*.json")
    # Skip this script's own output and any accuracy summaries -- only
    # per-model result files hold the row records BLEU is computed from.
    skip = ("accuracy", "_bleu")
    paths = sorted(
        p for p in glob.glob(pattern)
        if not any(s in os.path.basename(p) for s in skip)
    )
    if not paths:
        raise SystemExit(f"no result files matched {pattern}")

    print(f"{'model':<14}{'rows':>7}{'BLEU-4(w)':>11}{'BLEU-4(c)':>11}{'chrF':>9}{'exact%':>9}")
    print("-" * 61)
    out = {}
    for p in paths:
        model = os.path.basename(p)[len("translation_fidelity_"):-len(".json")]
        s = score_file(p)
        if not s:
            print(f"{model:<14}{'(no scorable rows)':>49}")
            continue
        out[model] = s
        print(
            f"{model:<14}{s['rows']:>7}{s['bleu4_word']:>11.2f}"
            f"{s['bleu4_char']:>11.2f}{s['chrf']:>9.2f}{s['exact']:>9.2f}"
        )

    dest = os.path.join(args.results_dir, "translation_fidelity_bleu.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nwritten to {dest}")


if __name__ == "__main__":
    main()
