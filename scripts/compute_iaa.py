"""Compute inter-annotator agreement from the annotation workbook.

No inference and no new annotation: every number here is read out of
`data/POS_english_to_tigrigna_Annotated.xlsx`, which holds the 500-item
sample, both annotators' sheets and the adjudicated version.

Why this script exists
----------------------
Section 2.3 of the paper reports no inter-annotator agreement coefficient for
this dataset. That is a claim about what the records support, so this script
derives it from them rather than asserting it. It reports three things:

1. **Cohen's kappa per dimension**, pairwise-complete (rows where both
   annotators left a value), and again with a blank treated as its own
   label. The two conventions disagree by up to 0.39 on the same records,
   so no single coefficient is well defined.

2. **Each annotator against the adjudicated sheet.** Where the annotators
   differ on POS, the adjudication reproduces Annotator 2 on every row and
   Annotator 1 on none; on the handful of Gender/Agreement/Alignment
   disagreements it reproduces Annotator 1. The adjudication is assembled
   from the two columns rather than decided separately, so agreement with it
   is not evidence about either. Gender and Number show zero disagreements
   across 420+ jointly-labelled items, which independent annotation does not
   produce.

3. **How the 500 items were drawn.** Each sampled row is traced back to the
   POS sheet it came from. The sample is alphabetical from the top of each
   sheet, not random: it is 61% adjectives and contains no verbs at all,
   against a corpus that is 54% nouns and 25% verbs. A kappa computed on it
   would not describe the corpus even if the columns were independent.

The workbook is read straight from its zip/XML so no spreadsheet library is
required (openpyxl is not installed in this project's environment).

Usage:  python scripts/compute_iaa.py [--json OUT]
"""
import argparse
import collections
import os
import re
import zipfile
from xml.etree import ElementTree as ET

MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
RELS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
NS = {"m": MAIN[1:-1]}

WORKBOOK = "data/POS_english_to_tigrigna_Annotated.xlsx"

# Column letter -> dimension, shared by both annotator sheets and Sheet1.
DIMENSIONS = {
    "POS": "D",
    "Gender": "E",
    "Number": "F",
    "Agreement": "G",
    "Alignment": "H",
}

# The eleven part-of-speech sheets that make up the source corpus, in the
# order the workbook lists them.
POS_SHEETS = [
    "Masculine", "Feminine", "Adjective", "Noun", "Verb", "Adverb",
    "Preposition", "Pronoun", "Conjunction", "Interjection", "Uncategorized",
]

ANNOTATOR_A = "Annotator 1"
ANNOTATOR_B = "Annotator 2"
ADJUDICATED = "Sheet1"

GEEZ_RE = re.compile(r"[ሀ-፿]")


class Workbook:
    """Minimal read-only xlsx reader: sheet name -> list of {column: value}."""

    def __init__(self, path):
        self.zip = zipfile.ZipFile(path)
        rels = ET.fromstring(self.zip.read("xl/_rels/workbook.xml.rels"))
        targets = {r.get("Id"): r.get("Target") for r in rels}
        book = ET.fromstring(self.zip.read("xl/workbook.xml"))
        self.sheets = {
            s.get("name"): targets[s.get(RELS + "id")]
            for s in book.findall("m:sheets/m:sheet", NS)
        }
        self.strings = []
        if "xl/sharedStrings.xml" in self.zip.namelist():
            shared = ET.fromstring(self.zip.read("xl/sharedStrings.xml"))
            for si in shared.findall("m:si", NS):
                self.strings.append(
                    "".join(t.text or "" for t in si.iter(MAIN + "t"))
                )

    def rows(self, name):
        target = self.sheets[name]
        path = target if target.startswith("xl/") else "xl/" + target.lstrip("/")
        sheet = ET.fromstring(self.zip.read(path))
        out = []
        for row in sheet.findall("m:sheetData/m:row", NS):
            cells = {}
            for c in row.findall("m:c", NS):
                column = "".join(ch for ch in c.get("r") if ch.isalpha())
                value = c.find("m:v", NS)
                if c.get("t") == "s" and value is not None:
                    cells[column] = self.strings[int(value.text)]
                else:
                    cells[column] = value.text if value is not None else None
            out.append(cells)
        return out

    def close(self):
        self.zip.close()


def cell(row, column):
    return (row.get(column) or "").strip()


def annotations(workbook, sheet):
    """Item_ID -> {dimension: label} for one annotation sheet."""
    table = {}
    for row in workbook.rows(sheet)[1:]:
        item_id = cell(row, "A")
        if not item_id:
            # Trailing rows carry no Item_ID and no labels; both annotator
            # sheets report 585 rows for 500 items because of them.
            continue
        table[item_id] = {d: cell(row, col) for d, col in DIMENSIONS.items()}
    return table


def cohens_kappa(pairs):
    """Cohen's kappa for a list of (label_a, label_b).

    Returns (kappa, n, observed_agreement). kappa is None when expected
    agreement is 1.0, which makes the statistic undefined.
    """
    n = len(pairs)
    if n == 0:
        return None, 0, 0.0
    observed = sum(1 for a, b in pairs if a == b) / n
    count_a = collections.Counter(a for a, _ in pairs)
    count_b = collections.Counter(b for _, b in pairs)
    expected = sum(
        (count_a[k] / n) * (count_b[k] / n) for k in set(count_a) | set(count_b)
    )
    if expected >= 1.0:
        return None, n, observed
    return (observed - expected) / (1 - expected), n, observed


def report_kappa(a, b):
    print("## 1. Cohen's kappa, Annotator 1 vs Annotator 2")
    print()
    print("Two conventions for the cells an annotator left empty:")
    print("  pairwise  -- drop rows where either annotator left a blank")
    print("  blank=label -- treat a blank as its own category, over all 500")
    print()
    print("%-12s %7s %9s %9s %9s %9s" % (
        "dimension", "n", "P(obs)", "pairwise", "n(all)", "blank=lbl"))
    print("-" * 60)
    shared = sorted(set(a) & set(b), key=lambda x: int(x) if x.isdigit() else 0)
    results = {}
    for dimension in DIMENSIONS:
        pairs = [
            (a[i][dimension], b[i][dimension])
            for i in shared
            if a[i][dimension] and b[i][dimension]
        ]
        kappa, n, observed = cohens_kappa(pairs)
        with_blanks = [
            (a[i][dimension] or "BLANK", b[i][dimension] or "BLANK")
            for i in shared
        ]
        kappa_blank, n_blank, _ = cohens_kappa(with_blanks)
        results[dimension] = {
            "n": n,
            "observed_agreement": observed,
            "kappa_pairwise": kappa,
            "n_all": n_blank,
            "kappa_blank_as_label": kappa_blank,
        }
        print("%-12s %7d %9.4f %9s %9d %9s" % (
            dimension, n, observed,
            "undef" if kappa is None else "%.4f" % kappa,
            n_blank,
            "undef" if kappa_blank is None else "%.4f" % kappa_blank))
    print()
    print("Shared Item_IDs: %d" % len(shared))
    spread = max(
        abs(r["kappa_pairwise"] - r["kappa_blank_as_label"])
        for r in results.values()
        if r["kappa_pairwise"] is not None
        and r["kappa_blank_as_label"] is not None
    )
    print("Largest gap between the two conventions: %.4f" % spread)
    return results


def report_adjudication(a, b, adjudicated):
    print()
    print("## 2. Each annotator against the adjudicated sheet (%s)" % ADJUDICATED)
    print()
    print("Where the annotators disagree, a separately-decided adjudication")
    print("would side with each of them sometimes. One assembled from their")
    print("columns reproduces one or the other wholesale.")
    print()
    print("%-12s %7s %10s %10s %9s" % (
        "dimension", "disagr", "adj = A1", "adj = A2", "neither"))
    print("-" * 52)
    results = {}
    shared = sorted(set(a) & set(b), key=lambda x: int(x) if x.isdigit() else 0)
    for dimension in DIMENSIONS:
        both = [
            i for i in shared
            if a[i][dimension] and b[i][dimension] and i in adjudicated
        ]
        disagreed = [i for i in both if a[i][dimension] != b[i][dimension]]
        with_a = sum(
            1 for i in disagreed if adjudicated[i][dimension] == a[i][dimension]
        )
        with_b = sum(
            1 for i in disagreed if adjudicated[i][dimension] == b[i][dimension]
        )
        results[dimension] = {
            "disagreements": len(disagreed),
            "adjudication_follows_a1": with_a,
            "adjudication_follows_a2": with_b,
            "neither": len(disagreed) - with_a - with_b,
        }
        print("%-12s %7d %10d %10d %9d" % (
            dimension, len(disagreed), with_a, with_b,
            len(disagreed) - with_a - with_b))
    print()
    print("Dimensions with zero disagreements are ones the two sheets never")
    print("differ on at all, across every jointly-labelled item.")
    print()
    blanks_a = {d: sum(1 for i in a if not a[i][d]) for d in DIMENSIONS}
    blanks_b = {d: sum(1 for i in b if not b[i][d]) for d in DIMENSIONS}
    print("Blank cells, Annotator 1: %s" % blanks_a)
    print("Blank cells, Annotator 2: %s" % blanks_b)
    print()
    print("Blanks are dropped pairwise above, so the kappa rows are computed")
    print("on exactly the subset where both annotators committed to a label.")
    return {"differences": results, "blanks_a": blanks_a, "blanks_b": blanks_b}


def report_sample(workbook, adjudicated):
    print()
    print("## 3. How the 500 sampled items were drawn")
    print()
    source = {}
    corpus = collections.Counter()
    swapped = 0
    total = 0
    for sheet in POS_SHEETS:
        for row in workbook.rows(sheet)[1:]:
            english, tigrinya = cell(row, "A"), cell(row, "B")
            source.setdefault((english, tigrinya), sheet)
            corpus[sheet] += 1
            total += 1
            if GEEZ_RE.search(english):
                swapped += 1

    traced = collections.Counter()
    untraced = 0
    for item_id, _ in sorted(
        adjudicated.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0
    ):
        traced[item_id] = 0
    for row in workbook.rows(ADJUDICATED)[1:]:
        if not cell(row, "A"):
            continue
        key = (cell(row, "B"), cell(row, "C"))
        if key in source:
            traced[source[key]] = traced.get(source[key], 0) + 1
        else:
            untraced += 1
    sample = {k: v for k, v in traced.items() if k in POS_SHEETS and v}

    print("%-16s %10s %10s %9s" % (
        "source sheet", "corpus", "sampled", "% of sheet"))
    print("-" * 48)
    for sheet in POS_SHEETS:
        got = sample.get(sheet, 0)
        share = (got / corpus[sheet] * 100) if corpus[sheet] else 0.0
        print("%-16s %10d %10d %8.1f%%" % (sheet, corpus[sheet], got, share))
    print("-" * 48)
    print("%-16s %10d %10d" % ("TOTAL", total, sum(sample.values())))
    if untraced:
        print("untraced sample rows: %d" % untraced)
    print()
    print("The sample takes whole small sheets and the alphabetical head of")
    print("the large ones. Verb is the second-largest category in the corpus")
    print("and contributes %d items." % sample.get("Verb", 0))
    print()
    print("Column swap in the source sheets: %d of %d rows (%.1f%%) carry"
          % (swapped, total, swapped / total * 100))
    print("Ge'ez script in the English column -- the same defect recorded for")
    print("Combined_POS_Lexicon.csv in CORRECTIONS.md section 1.2.")
    return {
        "corpus": dict(corpus),
        "sampled": sample,
        "untraced": untraced,
        "swapped_rows": swapped,
        "total_rows": total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", help="write the computed values to this path")
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, WORKBOOK)
    if not os.path.exists(path):
        raise SystemExit("workbook not found: %s" % path)

    workbook = Workbook(path)
    try:
        a = annotations(workbook, ANNOTATOR_A)
        b = annotations(workbook, ANNOTATOR_B)
        adjudicated = annotations(workbook, ADJUDICATED)

        print("Inter-annotator agreement, computed from %s" % WORKBOOK)
        print("=" * 62)
        print()
        print("Items with an Item_ID: A1=%d  A2=%d  adjudicated=%d"
              % (len(a), len(b), len(adjudicated)))
        print()

        kappa = report_kappa(a, b)
        adjudication = report_adjudication(a, b, adjudicated)
        sample = report_sample(workbook, adjudicated)

        print()
        print("## Conclusion")
        print()
        print("Section 2.3 reports the blank=label column, counting a cell")
        print("one annotator left empty as a disagreement. Three properties")
        print("bound what those figures support: the columns are not")
        print("independent of the adjudication, agreement on filled labels")
        print("is near-total so the divergence is in coverage rather than")
        print("label choice, and the subset is alphabetical and verb-free,")
        print("so it is not an estimate for the corpus.")

        if args.json:
            import json
            payload = {
                "workbook": WORKBOOK,
                "kappa": kappa,
                "adjudication": adjudication,
                "sample": sample,
            }
            with open(args.json, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            print()
            print("written to %s" % args.json)
    finally:
        workbook.close()


if __name__ == "__main__":
    main()
