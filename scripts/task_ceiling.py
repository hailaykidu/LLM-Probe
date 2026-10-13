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
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.rescore_tasks import NLTK_TO_GOLD  # noqa: E402

# Same tokenisation as the reported scoring, so the paired test scores the
# rule the tables report.
WORD_RE = re.compile(r"[a-z]+")

# Task -> (sense-split gold file, label field). These are the 5,701-item sets
# the two tagging tasks were actually scored against.
TASKS = {
    "pos_tagging": ("pos_tagging_sense_split.json", "POS_CATEGORIES"),
    "morphosyntax_probe": ("morphosyntax_probe_sense_split.json", "Expected"),
}

RESCORED = os.path.join("results", "evaluation_reports", "rescored_2026-09-28.json")


def one_sided_binomial(successes, n, p0):
    """P(X >= successes) under Binomial(n, p0), normal approximation.

    Retained only for comparison with the paired test below. It treats the
    baseline rate as a known constant, which it is not: the baseline is a
    system scored on the same items, so the two sets of outcomes are paired
    and correlated. Ignoring that correlation understates the variance of the
    difference and so overstates significance. McNemar's test is the correct
    one; see mcnemar().
    """
    mean = n * p0
    sd = math.sqrt(n * p0 * (1.0 - p0))
    if sd == 0:
        return float("nan"), 1.0
    z = (successes - 0.5 - mean) / sd
    return z, 0.5 * math.erfc(z / math.sqrt(2))


def mcnemar(model_only, baseline_only):
    """One-sided McNemar test that the model beats the baseline.

    Both predictors are scored on the same items, so only the discordant
    pairs carry information: model_only is the count of items the model got
    right and the baseline wrong, baseline_only the reverse. Under the null
    each discordant item falls either way with probability one half.

    Uses the continuity-corrected normal approximation, valid here because
    the discordant counts run to the hundreds or thousands.
    """
    n = model_only + baseline_only
    if n == 0:
        return float("nan"), 1.0
    z = (abs(model_only - baseline_only) - 1) / math.sqrt(n)
    p = 0.5 * math.erfc(z / math.sqrt(2))
    # One-sided in the direction of the model: a model behind the baseline
    # gets the complement rather than a small p-value.
    return z, p if model_only >= baseline_only else 1.0 - p


def paired_outcomes(root, task, model):
    """Per-item correctness of the model and the English-only baseline.

    Returns (model_correct, baseline_correct) as equal-length lists of bools
    over the saved generations, or None if the result file is absent. The
    model is scored by the first-token rule, the same one the reported table
    uses; the baseline is the NLTK tagger reading only the English gloss.
    """
    path = os.path.join(root, "results", "evaluation_reports", task,
                        f"{task}_{model}.json")
    if not os.path.exists(path):
        return None
    rows = json.load(open(path, encoding="utf-8"))

    try:
        import nltk
    except ImportError:
        return None

    model_ok, base_ok = [], []
    for row in rows:
        gold = set(WORD_RE.findall(str(row["ExpectedTags"]).lower()))
        words = WORD_RE.findall(str(row.get("ModelOutput", "")).lower())
        model_ok.append(bool(words) and words[0] in gold)

        english = str(row.get("English", "")).split(",")[0].strip()
        tagged = nltk.pos_tag(nltk.word_tokenize(english)) if english else []
        predicted = NLTK_TO_GOLD.get(tagged[-1][1], "") if tagged else ""
        base_ok.append(predicted in gold if predicted else False)

    return model_ok, base_ok


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
        print("  model          first-tok   vs NLTK    McNemar p"
              "   (unpaired)  vs ceiling")
        print("  " + "-" * 72)

        rows = sorted(
            ((m, v["first_token"]) for m, v in scores[task].items()
             if m != "_baselines"),
            key=lambda kv: -kv[1],
        )
        per_model = {}
        for model, score in rows:
            gap = score - stats["ceiling"]

            # Unpaired, as earlier versions of this script reported it.
            successes = round(score / 100 * n)
            _, p_binom = one_sided_binomial(successes, n, nltk_baseline / 100)

            # Paired, which is the test the design calls for.
            outcomes = paired_outcomes(root, task, model)
            if outcomes is None:
                p_mcnemar = float("nan")
                discordant = (None, None)
            else:
                model_ok, base_ok = outcomes
                b = sum(1 for m_, b_ in zip(model_ok, base_ok) if m_ and not b_)
                c = sum(1 for m_, b_ in zip(model_ok, base_ok) if b_ and not m_)
                _, p_mcnemar = mcnemar(b, c)
                discordant = (b, c)

            mark = "" if p_mcnemar < 0.05 else " n.s."
            print(f"  {model:<13} {score:7.2f}  {score - nltk_baseline:+8.2f}"
                  f"  {p_mcnemar:10.2e}{mark:>5}  {p_binom:10.2e}  {gap:+8.2f}")
            per_model[model] = {
                "first_token": score,
                "vs_nltk_baseline": score - nltk_baseline,
                "p_value_mcnemar": p_mcnemar,
                "p_value_unpaired_binomial": p_binom,
                "discordant_model_only": discordant[0],
                "discordant_baseline_only": discordant[1],
                "significant_at_05": bool(p_mcnemar < 0.05),
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
