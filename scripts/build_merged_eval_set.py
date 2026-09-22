"""Build the merged evaluation set from the bidirectional source lexicon.

`results/evaluation_reports/Combined_POS_Lexicon.csv` holds two datasets
concatenated under one pair of headers:

    rows 0..3586    English -> Tigrinya   (English column holds English)
    rows 3587..end  Tigrinya -> English   (English column holds TIGRINYA)

The reverse-direction block is stored with the columns in source order, so
its "English" field actually contains Ge'ez script and its "Tigrigna" field
contains English glosses. Evaluating that block as-is presents Tigrinya input
in a field labelled English -- the bug that affected the originally published
results. This script reorients that block, merges both directions, and writes
the gold files the task scripts read.

The 7,234 pairs in the source count each direction separately; merging the
two directions yields 5,783 unique evaluation items.

Usage:
    python scripts/build_merged_eval_set.py [--dry-run]
"""
import argparse
import json
import os
import re
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GEEZ_RE = re.compile(r"[ሀ-፿]")
# Gold POS labels were published as "(adv) adverb"; no model emits the
# parenthetical abbreviation, so exact-match scoring failed on every row.
# The bare tag is what gets written out.
POS_PREFIX_RE = re.compile(r"^\s*\([^)]*\)\s*")

SOURCE_CSV = os.path.join("results", "evaluation_reports", "Combined_POS_Lexicon.csv")
GOLD_DIR = os.path.join("data", "gold_labels")


def is_geez(value):
    return bool(GEEZ_RE.search(str(value)))


def reorient(df):
    """Swap English/Tigrigna for rows whose 'English' field holds Ge'ez.

    Detection is by script rather than row index so the boundary is derived
    from the data instead of hardcoded -- the source file's split point is a
    property of how it was concatenated, not a guarantee.
    """
    flipped = df["English"].map(is_geez)
    out = df.copy()
    out.loc[flipped, ["English", "Tigrigna"]] = df.loc[flipped, ["Tigrigna", "English"]].values
    return out, int(flipped.sum())


def strip_pos_prefix(value):
    return POS_PREFIX_RE.sub("", str(value)).strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=SOURCE_CSV)
    ap.add_argument("--out-dir", default=GOLD_DIR)
    ap.add_argument("--dry-run", action="store_true", help="report counts, write nothing")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(root, args.source)
    df = pd.read_csv(src)
    print(f"source: {args.source}  rows={len(df)}")

    df, n_flipped = reorient(df)
    print(f"  reoriented reverse-direction rows: {n_flipped}")
    print(f"  forward-direction rows:            {len(df) - n_flipped}")

    df = df.dropna(subset=["English", "Tigrigna"])
    df["English"] = df["English"].astype(str).str.strip()
    df["Tigrigna"] = df["Tigrigna"].astype(str).str.strip()
    # Distinct senses of one headword are separate evaluation items, so the
    # key is the pair -- deduplicating on English alone would silently drop
    # them.
    merged = df.drop_duplicates(subset=["English", "Tigrigna"], keep="first")
    print(f"  merged unique (English, Tigrigna): {len(merged)}")

    pos = merged.assign(POS_CATEGORIES=merged["POS_CATEGORIES"].map(strip_pos_prefix))
    n_prefixed = int(merged["POS_CATEGORIES"].astype(str).str.startswith("(").sum())
    print(f"  POS labels with '(abbr)' prefix stripped: {n_prefixed}")

    outputs = {
        "pos_tags_fixed.json": pos[["English", "Tigrigna", "POS_CATEGORIES"]],
        "morpho_features_fixed.json": pos[["English", "Tigrigna", "POS_CATEGORIES"]].rename(
            columns={"POS_CATEGORIES": "Expected"}
        ),
        "translations.json": merged[["English", "Tigrigna"]].rename(
            columns={"Tigrigna": "Reference"}
        ),
    }

    if args.dry_run:
        print("\n--dry-run: nothing written")
        for name, frame in outputs.items():
            print(f"  would write {name}: {len(frame)} rows, cols={list(frame.columns)}")
        return

    out_dir = os.path.join(root, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    for name, frame in outputs.items():
        dest = os.path.join(out_dir, name)
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(frame.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
        print(f"  wrote {dest}  ({len(frame)} rows)")


if __name__ == "__main__":
    main()
