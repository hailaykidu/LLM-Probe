"""Reconstruct the sense-split evaluation set used by POS and morphosyntax.

The POS tagging and morphosyntactic labelling results in
`results/evaluation_reports/{pos_tagging,morphosyntax_probe}/` cover 5,701
items, while the committed gold files hold 5,783 rows which the task scripts
reduce to 5,775 by `drop_duplicates(["English","Tigrigna"]).dropna(...)`.
The two sets are not the same: 182 gold rows are absent from the results and
108 result rows are absent from the gold.

The difference is systematic, not a sampling accident. Those runs were scored
against a gold file in which a multi-sense entry had been split to its first
sense -- `("a little", "ንእሽቶይ, ቁሩብ,ውሕድ")` was evaluated as
`("a little", "ንእሽቶይ")`. All 108 result-only rows are first senses of a
multi-sense gold headword. That intermediate file was never committed, so
re-running the task scripts today evaluates a different item set than
sections 4.1 and 4.2 of the paper report.

This script regenerates that set from the committed gold, so the evaluated
items are reproducible from tracked artifacts rather than resting on an
uncommitted file. It is a reconstruction, not a recovered original: the
output is verified row-for-row against the saved model outputs by
`--verify`, which reports any divergence instead of hiding it.

Reconstruction rule, applied to the already-corrected gold:

  1. `drop_duplicates(["English","Tigrigna"], keep="first")` then `dropna`,
     exactly as the task scripts do -- 5,783 -> 5,775 rows.
  2. Keep the first comma-separated Tigrinya sense of each entry.
  3. Deduplicate again on the resulting (English, Tigrinya) pair, keeping the
     first occurrence -- 5,775 -> 5,701 rows.

Step 3 matters for one row. `effect` appears twice in the gold, as
`("effect", "ሳዕቤን, ውጽኢት", noun)` and `("effect", "ሳዕቤን", verb)`. Both reduce
to `("effect", "ሳዕቤን")`, and the saved outputs carry `noun`, so the earlier
row wins. Deduplicating before the split would select `verb` and diverge from
the record.

Usage:
    python scripts/build_sense_split_eval_set.py --verify
    python scripts/build_sense_split_eval_set.py --write
"""
import argparse
import json
import os

import pandas as pd

# Gold file and label column per task, matching what each task script loads.
TASKS = {
    "pos_tagging": ("pos_tags_fixed.json", "POS_CATEGORIES"),
    "morphosyntax_probe": ("morpho_features_fixed.json", "Expected"),
}

OUT_SUFFIX = "_sense_split.json"


def first_sense(text):
    """The first comma-separated sense of a Tigrinya entry."""
    for part in str(text).split(","):
        part = part.strip()
        if part:
            return part
    return ""


def build(root, gold_file, label_column):
    """Reconstruct the evaluated item set from the committed gold."""
    path = os.path.join(root, "data", "gold_labels", gold_file)
    gold = pd.read_json(path)
    # Step 1: exactly what the task scripts apply before evaluating.
    gold = gold.drop_duplicates(
        subset=["English", "Tigrigna"], keep="first"
    ).dropna(subset=["English", "Tigrigna", label_column])

    # Steps 2 and 3: split to the first sense, then dedupe on the result.
    # The order matters -- see the module docstring on "effect".
    rows = []
    for _, row in gold.iterrows():
        sense = first_sense(row["Tigrigna"])
        if not sense:
            continue
        rows.append({
            "English": row["English"],
            "Tigrigna": sense,
            label_column: row[label_column],
        })
    frame = pd.DataFrame(rows).drop_duplicates(
        subset=["English", "Tigrigna"], keep="first"
    )
    return frame


def saved_outputs(root, task):
    """(English, Tigrinya) -> gold label, from any saved result file."""
    directory = os.path.join(root, "results", "evaluation_reports", task)
    if not os.path.isdir(directory):
        return None, None
    for name in sorted(os.listdir(directory)):
        if not name.startswith(f"{task}_") or not name.endswith(".json"):
            continue
        if "accuracy" in name or "_bleu" in name or name.endswith(OUT_SUFFIX):
            continue
        rows = json.load(open(os.path.join(directory, name), encoding="utf-8"))
        if rows and "ExpectedTags" in rows[0]:
            return name, {
                (r["English"], r["Tigrigna"]): r["ExpectedTags"] for r in rows
            }
    return None, None


def verify(root, task, frame, label_column):
    """Check the reconstruction against what the models were actually scored on."""
    name, saved = saved_outputs(root, task)
    if not saved:
        print(f"  {task}: no saved outputs to verify against")
        return True

    built = {
        (r["English"], r["Tigrigna"]): r[label_column]
        for _, r in frame.iterrows()
    }
    missing = set(saved) - set(built)
    extra = set(built) - set(saved)
    differing = [k for k in set(saved) & set(built) if saved[k] != built[k]]

    print(f"  {task}: reconstructed {len(built)}, saved {len(saved)} "
          f"(against {name})")
    print(f"    rows in saved but not reconstructed: {len(missing)}")
    print(f"    rows reconstructed but not saved:    {len(extra)}")
    print(f"    label mismatches:                    {len(differing)}")
    for key in sorted(missing)[:5]:
        print(f"      missing: {key}")
    for key in sorted(extra)[:5]:
        print(f"      extra:   {key}")
    for key in sorted(differing)[:5]:
        print(f"      differs: {key} saved={saved[key]!r} built={built[key]!r}")
    return not (missing or extra or differing)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true",
                        help="compare the reconstruction with the saved outputs")
    parser.add_argument("--write", action="store_true",
                        help="write data/gold_labels/<task>_sense_split.json")
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    all_ok = True
    for task, (gold_file, label_column) in TASKS.items():
        frame = build(root, gold_file, label_column)
        print(f"{task}: {len(frame)} items from {gold_file}")
        if args.verify:
            all_ok &= verify(root, task, frame, label_column)
        if args.write:
            out = os.path.join(
                root, "data", "gold_labels", f"{task}{OUT_SUFFIX}")
            frame.to_json(out, orient="records", force_ascii=False, indent=2)
            print(f"  written to {os.path.relpath(out, root)}")

    if args.verify:
        print("\nreconstruction matches the saved outputs exactly"
              if all_ok else "\nreconstruction DIVERGES from the saved outputs")
    if not (args.verify or args.write):
        print("\nnothing written; pass --verify and/or --write")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
