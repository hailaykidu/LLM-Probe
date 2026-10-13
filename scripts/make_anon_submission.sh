#!/usr/bin/env bash
# Build the anonymous submission bundle: LaTeX source with every comment
# stripped, and the supplementary material without the .git directory.
#
# Comments in the working source record what to change at camera-ready --
# which author goes last, that not every author annotated, that the byline
# order differs from the earlier paper's. Together with the citation to that
# paper, those notes identify the authors, so none of them may reach a
# reviewer. Rather than keep the working source free of notes, this strips
# them on the way out.
#
# Usage:  ./scripts/make_anon_submission.sh [OUTDIR]
set -euo pipefail

cd "$(dirname "$0")/.."
out="${1:-dist/anon-submission}"
rm -rf "$out"
mkdir -p "$out/paper"

# --- LaTeX source, comments removed ---------------------------------------
# Strips whole-line comments and trailing comments, while preserving a
# literal \% and the comment-as-line-join idiom (a line ending in a bare %
# with no space before it suppresses the newline; removing it would insert
# an unwanted space). Lines inside verbatim-like environments are left alone
# because this document has none.
for f in paper/main.tex paper/references.bib; do
    python3 - "$f" "$out/$f" <<'PY'
import re, sys

src, dest = sys.argv[1], sys.argv[2]
out = []
for line in open(src, encoding="utf-8"):
    stripped = line.lstrip()
    # A whole-line comment disappears entirely.
    if stripped.startswith("%"):
        continue
    # A trailing comment is cut at the first unescaped %, unless it is the
    # line-join idiom (no whitespace before the % and nothing after it).
    m = re.search(r"(?<!\\)%", line)
    if m:
        before, after = line[:m.start()], line[m.end():].strip()
        if after or before.endswith((" ", "\t")):
            line = before.rstrip() + "\n"
    out.append(line)

# Collapse the blank runs the removals leave behind.
text = re.sub(r"\n{3,}", "\n\n", "".join(out))
open(dest, "w", encoding="utf-8").write(text)
PY
done

cp paper/acl.sty paper/acl_natbib.bst "$out/paper/" 2>/dev/null || true
cp paper/main.pdf "$out/paper/" 2>/dev/null || true

# --- Supplementary material, without the git history ----------------------
# Every commit carries the author's name, so the history cannot ship with an
# anonymous submission. git archive writes the tracked tree at HEAD and no
# .git directory.
git archive --format=tar HEAD | (mkdir -p "$out/supplementary" && tar -x -C "$out/supplementary")
rm -rf "$out/supplementary/paper"

echo "wrote $out"
echo
echo "checks:"
# Only unescaped % is a comment; \% is a literal percent sign in the text.
printf '  LaTeX comments left:            %s\n' \
    "$(grep -cP '(?<!\\)%' "$out/paper/main.tex" || true)"
printf '  .git in supplementary:          %s\n' \
    "$([ -d "$out/supplementary/.git" ] && echo PRESENT || echo absent)"

# The bibliography must cite the earlier paper by its real authors, and that
# paper shares an author with this one. Those names are therefore expected in
# references.bib and in the \citet that reads it, and nowhere else. Anything
# outside those two files is a leak.
#
# The strings to search for are taken from the bibliography entry rather than
# written here, so this script does not itself carry the names into the
# anonymous bundle it produces.
# Every name token in that entry's author field, "and" dropped.
patterns=$(
    sed -n '/@inproceedings{llmprobe-2026/,/^}/p' paper/references.bib \
    | sed -n 's/.*author *= *{\(.*\)}.*/\1/p' \
    | tr -d '{},' | tr ' ' '\n' \
    | grep -E '^[A-Za-z]{3,}$' | grep -vx 'and' | sort -u
)
# The serving institution's prefix, if the local environment still sets it,
# so a bundle built on a configured machine is checked for it too.
[ -n "${LLM_ENDPOINT_LEGACY_PREFIX:-}" ] &&
    patterns="$patterns ${LLM_ENDPOINT_LEGACY_PREFIX%%_*}"

for pattern in $patterns; do
    hits=$(grep -rli "$pattern" "$out" 2>/dev/null \
        | grep -v 'paper/references.bib' | grep -v 'paper/main.tex' || true)
    n=$([ -z "$hits" ] && echo 0 || echo "$hits" | wc -l)
    printf '  %-14s outside paper/:  %s\n' "$pattern" "$n"
    [ -n "$hits" ] && echo "$hits" | sed 's/^/      /'
done
echo
echo "Note: git archive reads HEAD, so commit before building the bundle or"
echo "the supplementary will lag the working tree."
