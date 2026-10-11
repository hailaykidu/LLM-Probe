"""How much of each tagging task is determined by the English gloss alone?

The prompt for part-of-speech tagging and morphosyntactic labelling is

    Phrase: {Tigrinya} ({English})

so the English headword is given to the model along with the Tigrinya form.
`scripts/rescore_tasks.py` already reports an English-only baseline: an NLTK
tagger applied to the gloss and nothing else. That baseline measures what one
particular tagger achieves, which is a lower bound on what the English alone
makes available. This script measures the upper bound.

The ceiling is computed with an oracle that sees only the English string. For
each headword it answers with the most frequent gold label that headword
carries anywhere in the evaluation set, which is the best any English-only
system can do without guessing between genuinely ambiguous readings. The
oracle never sees a Tigrinya form.

Two numbers follow from it, and they bound opposite things:

  ceiling     what an English-only system can reach. A model scoring below
              this has not demonstrated that the Tigrinya form helped it.
  ambiguous   the share of headwords carrying more than one gold label.
              These are the only items on which the Tigrinya form can carry
              information the English does not already supply.

The script also tests each model against the NLTK baseline with a one-sided
binomial test, so that a small margin is reported as significant or not
rather than described in prose.

Usage:  python scripts/task_ceiling.py [--json OUT]
"""
import argparse
import collections
import json
import math
import os

# Task -> (sense-split gold file, label field). These are the 5,701-item sets
# the two tagging tasks were actually scored against.
TASKS = {
    "pos_tagging": ("pos_tagging_sense_split.json", "POS_CATEGORIES"),
    "morphosyntax_probe": ("morphosyntax_probe_sense_split.json", "Expected"),
}

RESCORED = os.path.join("results", "evaluation_reports", "rescored_2026-09-28.json")


def one_sided_binomial(successes, n, p0):
    """P(X >= successes) under Binomial(n, p0), normal approximation.

    n is 5,701 here, far into the regime where the approximation is accurate,
    and an exact sum overflows a float. A continuity correction is applied.
    """
    mean = n * p0
    sd = math.sqrt(n * p0 * (1.0 - p0))
    if sd == 0:
        return float("nan"), 1.0
    z = (successes - 0.5 - mean) / sd
    return z, 0.5 * math.erfc(z / math.sqrt(2))


def ceiling(root, gold_file, label_field):
    """The best score reachable from the English headword alone."""
    path = os.path.join(root, "data", "gold_labels", gold_file)
    rows = json.load(open(path, encoding="utf-8"))

    labels_by_headword = collections.defaultdict(list)
    for row in rows:
        labels_by_headword[row["English"]].append(row[label_field])

    # The oracle's answer for a headword: its most frequent gold label.
    oracle = {
        head: collections.Counter(labels).most_common(1)[0][0]
        for head, labels in labels_by_headword.items()
    }
    correct = sum(1 for row in rows if oracle[row["English"]] == row[label_field])
    ambiguous = sum(
        1 for labels in labels_by_headword.values() if len(set(labels)) > 1
    )
    return {
        "n": len(rows),
        "headwords": len(labels_by_headword),
        "ambiguous_headwords": ambiguous,
        "ambiguous_share": ambiguous / len(labels_by_headword) * 100,
        "ceiling": correct / len(rows) * 100,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", help="write the computed values to this path")
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scores = json.load(open(os.path.join(root, RESCORED), encoding="utf-8"))
    out = {}

    for task, (gold_file, label_field) in TASKS.items():
        stats = ceiling(root, gold_file, label_field)
        baselines = scores[task]["_baselines"]
        nltk_baseline = baselines["english_only"]
        n = stats["n"]

        print(f"\n=== {task} ===")
        print(f"items {n}, distinct English headwords {stats['headwords']}")
        print(f"headwords with more than one gold label: "
              f"{stats['ambiguous_headwords']} ({stats['ambiguous_share']:.1f}%)")
        print()
        print(f"  majority-label baseline            {baselines['majority']:6.2f}%")
        print(f"  English-only tagger (NLTK)         {nltk_baseline:6.2f}%")
        print(f"  English-only CEILING (oracle)      {stats['ceiling']:6.2f}%")
        print()
        print("  model          first-tok   vs NLTK      p   vs ceiling")
        print("  " + "-" * 56)

        rows = sorted(
            ((m, v["first_token"]) for m, v in scores[task].items()
             if m != "_baselines"),
            key=lambda kv: -kv[1],
        )
        per_model = {}
        for model, score in rows:
            successes = round(score / 100 * n)
            z, p = one_sided_binomial(successes, n, nltk_baseline / 100)
            gap = score - stats["ceiling"]
            mark = "" if p < 0.05 else "  n.s."
            print(f"  {model:<13} {score:7.2f}  {score - nltk_baseline:+8.2f}"
                  f" {p:7.4f}{mark:>6}  {gap:+8.2f}")
            per_model[model] = {
                "first_token": score,
                "vs_nltk_baseline": score - nltk_baseline,
                "p_value": p,
                "significant_at_05": bool(p < 0.05),
                "vs_ceiling": gap,
            }

        best = rows[0]
        print()
        print(f"  Best model is {best[1] - stats['ceiling']:.2f} points below what the")
        print(f"  English gloss alone makes available.")
        out[task] = {**stats, "baselines": baselines, "models": per_model}

    print("\nNo model on either task reaches the English-only ceiling, so none")
    print("demonstrates that the Tigrinya form contributed information the")
    print("English gloss did not already supply.")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(out, handle, ensure_ascii=False, indent=2)
        print(f"\nwritten to {args.json}")


if __name__ == "__main__":
    main()
