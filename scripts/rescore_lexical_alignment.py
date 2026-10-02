"""Re-score saved lexical_alignment outputs under a strict matching rule.

No inference: every number here is computed from the ModelOutput fields
already stored in
results/evaluation_reports/lexical_alignment/lexical_alignment_<model>.json.
The experimental records are unchanged; only the scoring rule differs.

Why: tasks/lexical_alignment.py scores SemanticMatch with

    all(tig.replace(" ", "") in output.replace(" ", "") for tig in gold_rhs)

i.e. the gold Tigrinya only has to appear as a SUBSTRING anywhere in the
output. Outputs run to hundreds of characters, so a model that rambles hits
the gold string incidentally while its actual answer is wrong. Observed in
the September 2026 records (mistral-7b):

    gold: nose→ኣፍንጫ
    out : nose→ኣፍ, snout→ኣፍንጫ I'd like to see a program that can do this
          for longer sentences and possibly even whole documents ...

The asserted pair is nose→ኣፍ, which is wrong; the row scored correct
because ኣፍንጫ occurs later in unrelated text. Mean output length among
mistral-7b's "correct" rows is ~150 characters for a single-word task.

Rules reported side by side:

  substring     what the task script computed (for comparison)
  strict_pair   the RHS of the FIRST arrow pair the model emits must equal
                the gold RHS -- the answer the model actually commits to
  contains_pair the full "English→Tigrinya" gold pair appears as a unit,
                a middle ground that still tolerates surrounding text

Usage:  python scripts/rescore_lexical_alignment.py [--json OUT]
"""
import argparse
import glob
import json
import os
import re

TASK = "lexical_alignment"

# The task script normalizes by deleting spaces before comparing; the same
# normalization is applied here so the "substring" column reproduces the
# stored SemanticMatch exactly.
def _norm(text):
    return str(text).replace(" ", "").replace("​", "")


def gold_pairs(expected):
    """[(english, tigrinya)] from a gold "eng→tig, eng2→tig2" string."""
    pairs = []
    for part in str(expected).split(","):
        if "→" in part:
            lhs, rhs = part.split("→", 1)
            pairs.append((lhs.strip(), rhs.strip()))
    return pairs


def score_substring(expected, output):
    """The task script's rule: every gold RHS appears anywhere in output."""
    rhs = [t for _, t in gold_pairs(expected)]
    if not rhs:
        return False
    out = _norm(output)
    return all(_norm(t) in out for t in rhs)


FIRST_PAIR_RE = re.compile(r"[^→,\n]*→([^,\n]*)")


def score_strict_pair(expected, output):
    """The RHS of the first arrow pair emitted must equal the gold RHS.

    This is the answer the model commits to for the headword it was asked
    about, ignoring anything it appends afterwards.
    """
    rhs = [t for _, t in gold_pairs(expected)]
    if not rhs:
        return False
    first_line = str(output).splitlines()[0] if str(output).strip() else ""
    m = FIRST_PAIR_RE.match(first_line)
    if not m:
        return False
    asserted = _norm(m.group(1))
    return all(_norm(t) == asserted for t in rhs)


def score_contains_pair(expected, output):
    """The complete gold "english→tigrinya" pair appears as a unit."""
    pairs = gold_pairs(expected)
    if not pairs:
        return False
    out = _norm(output)
    return all(_norm(f"{e}→{t}") in out for e, t in pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="write results to this path")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pattern = os.path.join(root, "results/evaluation_reports", TASK, f"{TASK}_*.json")
    paths = sorted(
        p for p in glob.glob(pattern)
        if not any(s in os.path.basename(p) for s in ("accuracy", "_bleu"))
    )
    if not paths:
        raise SystemExit(f"no result files matched {pattern}")

    out = {}
    print("%-24s %7s %11s %13s %13s" % (
        "model", "n", "substring%", "contains%", "strict%"))
    print("-" * 72)
    for path in paths:
        model = os.path.basename(path)[len(TASK) + 1:-len(".json")]
        rows = json.load(open(path, encoding="utf-8"))
        if not rows:
            continue
        n = len(rows)
        sub = con = strict = 0
        for r in rows:
            exp, o = r.get("ExpectedAlignment", ""), r.get("ModelOutput", "")
            sub += score_substring(exp, o)
            con += score_contains_pair(exp, o)
            strict += score_strict_pair(exp, o)
        out[model] = {
            "n": n,
            "substring": sub / n * 100,
            "contains_pair": con / n * 100,
            "strict_pair": strict / n * 100,
        }
        print("%-24s %7d %10.2f%% %12.2f%% %12.2f%%" % (
            model, n, out[model]["substring"],
            out[model]["contains_pair"], out[model]["strict_pair"]))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\nwritten to {args.json}")


if __name__ == "__main__":
    main()
