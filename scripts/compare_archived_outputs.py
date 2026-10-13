"""Check that the reported results are new runs, not a re-scoring.

Section 7 claims the generations behind this paper are new, against the
archived pre-fix run. That is checkable: if the outputs had merely been
re-scored, every shared item would carry an identical generation. This
compares the two sets item by item and reports how many match.

It also reports the archived accuracy files, which read 0.0000 for every
model because the archived gold stored the source dictionary's parenthetical
notation ("(adv) adverb") and the scorer of the time compared whole strings.

Usage:  python scripts/compare_archived_outputs.py [--task pos_tagging]
"""
import argparse
import glob
import json
import os

CURRENT = os.path.join("results", "evaluation_reports")
ARCHIVED = os.path.join("results", "evaluation_reports_pre_fix_backup")


def outputs_by_headword(path):
    """English headword -> cleaned generation, for one result file."""
    rows = json.load(open(path, encoding="utf-8"))
    return {
        str(r.get("English", "")).strip(): str(r.get("ModelOutput", "")).strip()
        for r in rows
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="pos_tagging")
    args = ap.parse_args()

    pattern = os.path.join(CURRENT, args.task, f"{args.task}_*.json")
    total_shared = total_same = 0

    print(f"{'model':<13}{'shared':>8}{'identical':>11}{'archived acc':>14}")
    print("-" * 46)
    for new_path in sorted(glob.glob(pattern)):
        model = os.path.basename(new_path)[len(args.task) + 1:-len(".json")]
        if "accuracy" in model:
            continue
        old_path = os.path.join(ARCHIVED, args.task, f"{args.task}_{model}.json")
        if not os.path.exists(old_path):
            continue

        old, new = outputs_by_headword(old_path), outputs_by_headword(new_path)
        shared = set(old) & set(new)
        same = sum(1 for k in shared if old[k] == new[k])
        total_shared += len(shared)
        total_same += same

        acc_path = os.path.join(
            ARCHIVED, args.task, f"{args.task}_accuracy_{model}.txt")
        acc = open(acc_path).read().strip() if os.path.exists(acc_path) else "-"
        print(f"{model:<13}{len(shared):>8}{same:>11}{acc:>14}")

    if not total_shared:
        raise SystemExit(f"no model appears in both {CURRENT} and {ARCHIVED}")

    print("-" * 46)
    print(f"{'TOTAL':<13}{total_shared:>8}{total_same:>11}")
    print(f"\n{total_same} of {total_shared} shared items carry an identical "
          f"generation ({100 * total_same / total_shared:.2f}%).")
    print("A re-scoring of the same outputs would give 100%.")


if __name__ == "__main__":
    main()
