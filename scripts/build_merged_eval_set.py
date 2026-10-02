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
two directions yields the 5,783 rows of the committed gold files.

Three details of that merge are load-bearing, and an earlier revision of this
script got them wrong, so that it produced 5,775 rows rather than the 5,783
the evaluations actually read. `--verify` now checks the output against the
committed gold and reports any divergence.

1. **Rows with no Tigrinya content are dropped before reorienting**, not
   after. Eight source rows carry an English headword and an empty Tigrinya
   field; reorienting first would move the empty value into `English` and
   leave an item with no English side.

2. **The deduplication key is the (English, Tigrinya, POS) triple**, not the
   pair. The source lists `estimate -> ግምት` as both noun and verb, and
   `accidentally -> ብሓደጋ` as both adjective and adverb. These are distinct
   senses of one form and are kept as separate items, consistent with the
   project's rule that evaluation identities are not collapsed on English
   alone.

3. **A label is reduced to its leading category word**, and a label naming
   only gender becomes `uncategorized`. `(nm) noun masculine` and
   `(npl) noun plural` become `noun`; `(m) masculine` and `(f) feminine`
   carry no category and become `uncategorized`. Those two come from the
   one-row Masculine and Feminine sheets of the source workbook, whose
   columns are themselves swapped.

The gold additionally holds three rows that duplicate another row exactly
(`grocer -> በዓል ድኳን`, `lion -> ኣንበሳ`, `paste -> ባስታ`, each twice). They are
preserved rather than silently removed: the task scripts drop them at load
time, so they affect no reported number, and removing them here would change
the committed artifact.

Usage:
    python scripts/build_merged_eval_set.py [--dry-run] [--verify]
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


# The category vocabulary the gold files use. A source label that reduces to
# something outside this set names only gender and carries no category.
POS_CATEGORIES = {
    "noun", "verb", "adjective", "adverb", "preposition", "pronoun",
    "conjunction", "interjection", "uncategorized",
}


def strip_pos_prefix(value):
    """Reduce a source label to the bare category word used in the POS gold.

    "(nm) noun masculine" -> "noun";  "(m) masculine" -> "uncategorized".
    """
    text = POS_PREFIX_RE.sub("", str(value)).strip().lower()
    head = text.split()[0] if text else ""
    return head if head in POS_CATEGORIES else "uncategorized"


def morph_features(value):
    """Reduce a source label to the feature string used in the morphosyntax gold.

    The morphosyntax gold keeps the gender and number a source label carries,
    where the POS gold discards them -- this is the only difference between
    the two files, and what makes the morphosyntax task nominally distinct:

        "(nm) noun masculine" -> "noun, masculine"
        "(npl) noun plural"   -> "noun, plural"
        "(m) masculine"       -> "uncategorized, masculine"
        "(an) adjective noun" -> "adjective, noun-derived"
        "(n) noun"            -> "noun"
    """
    text = POS_PREFIX_RE.sub("", str(value)).strip().lower()
    words = text.split()
    if not words:
        return "uncategorized"
    head = words[0] if words[0] in POS_CATEGORIES else "uncategorized"
    rest = words if head == "uncategorized" else words[1:]
    features = []
    for word in rest:
        if word in ("masculine", "feminine"):
            features.append(word)
        elif word in ("plural", "pl"):
            features.append("plural")
        elif word == "noun" and head == "adjective":
            features.append("noun-derived")
    return ", ".join([head] + features) if features else head


def drop_empty_tigrinya(df):
    """Drop source rows that carry no Tigrinya side.

    Applied before reorienting: these rows hold an English headword with an
    empty Tigrigna field, and swapping them first would leave the English
    side empty instead.
    """
    missing = df["Tigrigna"].isna() & ~df["English"].map(is_geez)
    return df[~missing], int(missing.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=SOURCE_CSV)
    ap.add_argument("--out-dir", default=GOLD_DIR)
    ap.add_argument("--dry-run", action="store_true", help="report counts, write nothing")
    ap.add_argument("--verify", action="store_true",
                    help="compare the rebuild with the committed gold files")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(root, args.source)
    df = pd.read_csv(src)
    print(f"source: {args.source}  rows={len(df)}")

    df, n_empty = drop_empty_tigrinya(df)
    print(f"  dropped rows with no Tigrinya side: {n_empty}")

    df, n_flipped = reorient(df)
    print(f"  reoriented reverse-direction rows: {n_flipped}")
    print(f"  forward-direction rows:            {len(df) - n_flipped}")

    df = df.copy()
    df["English"] = df["English"].astype(str).str.strip()
    df["Tigrigna"] = df["Tigrigna"].astype(str).str.strip()
    df = df[(df["English"] != "") & (df["Tigrigna"] != "") & (df["Tigrigna"] != "nan")]
    n_prefixed = int(df["POS_CATEGORIES"].astype(str).str.startswith("(").sum())
    # The morphosyntax gold keeps gender/number; the POS gold does not. Derive
    # the feature string before the POS column is reduced to its head word.
    df["Expected"] = df["POS_CATEGORIES"].map(morph_features)
    df["POS_CATEGORIES"] = df["POS_CATEGORIES"].map(strip_pos_prefix)
    # Distinct senses of one form are separate evaluation items, so the key is
    # the (pair, features) triple: "estimate -> ግምት" is kept as both noun and
    # verb. Deduplicating on the pair alone would discard one of them.
    #
    # The key uses the morphosyntactic feature string rather than the reduced
    # POS category, because the two differ for three rows. "grocer -> በዓል ድኳን"
    # appears as both "(n) noun" and "(nm) noun masculine": identical once
    # reduced to "noun", but distinct as "noun" and "noun, masculine". Keying
    # on the POS column would drop one and leave the morphosyntax gold three
    # rows short.
    merged = df.drop_duplicates(
        subset=["English", "Tigrigna", "Expected"], keep="first")
    print(f"  merged unique (English, Tigrigna, features): {len(merged)}")

    pos = merged
    print(f"  POS labels with '(abbr)' prefix stripped: {n_prefixed}")

    outputs = {
        "pos_tags_fixed.json": pos[["English", "Tigrigna", "POS_CATEGORIES"]],
        "morpho_features_fixed.json": pos[["English", "Tigrigna", "Expected"]],
        "translations.json": merged[["English", "Tigrigna"]].rename(
            columns={"Tigrigna": "Reference"}
        ),
    }

    if args.verify:
        print()
        ok = True
        for name, frame in outputs.items():
            dest = os.path.join(root, args.out_dir, name)
            if not os.path.exists(dest):
                print(f"  {name}: not present, nothing to verify against")
                continue
            committed = pd.read_json(dest)
            columns = list(frame.columns)
            rebuilt_rows = sorted(
                tuple(str(v) for v in row) for row in frame[columns].itertuples(index=False)
            )
            committed_rows = sorted(
                tuple(str(v) for v in row)
                for row in committed[columns].itertuples(index=False)
            )
            # The committed gold keeps three rows that duplicate another row
            # exactly; the rebuild emits each distinct row once. Compare the
            # distinct content, and account for the duplicates separately.
            extra = len(committed_rows) - len(set(committed_rows))
            same = sorted(set(rebuilt_rows)) == sorted(set(committed_rows))
            ok &= same
            print(f"  {name}: rebuilt {len(frame)}, committed {len(committed)} "
                  f"({extra} exact duplicate rows) -- content "
                  f"{'identical' if same else 'DIFFERS'}")
            if not same:
                r, c = set(rebuilt_rows), set(committed_rows)
                for row in sorted(r - c)[:5]:
                    print(f"    rebuilt only:   {row}")
                for row in sorted(c - r)[:5]:
                    print(f"    committed only: {row}")
        print("\nrebuild reproduces the committed gold"
              if ok else "\nrebuild DIVERGES from the committed gold")
        if not args.dry_run:
            return 0 if ok else 1

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
