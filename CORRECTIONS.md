# Corrections and reproduction audit

Post-publication audit of *LLM Probe: Evaluating LLMs for Low-Resource
Languages* (LLMs4SSH @ LREC 2026, pp. 224–234).

**Status: under review with co-authors. Not yet filed as a correction with
the venue.**

This file records what was found. It does not overwrite the original numbers
in `README.md` or in `results/evaluation_reports_pre_fix_backup/` — the
original record is kept intact so the correction can be checked against it.

---

## 1. What was wrong

### 1.1 Gold POS labels carried an abbreviation prefix

Gold labels were stored as `(adv) adverb`, `(n) noun`, `(v) verb` — the
notation used in Table 1 of the paper. Models emit the bare tag (`adverb`).
Exact-match scoring therefore failed on **every row**, regardless of whether
the model was correct.

This zeroed out POS tagging and morphosyntactic probing completely: all eight
models scored `0.0000` across all 3,899 evaluated rows.

Fixed: gold labels are written without the prefix
(`scripts/build_merged_eval_set.py`), and `normalize_tags` in the task
scripts now strips any parenthetical abbreviation and splits on whitespace as
well as commas.

**Reconciliation with the archived zeros.** The scoring predicate itself is
unchanged across the fix — `bool(expected_tags & output_tags)` in both the
initial commit and now. Only label normalisation changed. The old
`normalize_tags` deleted spaces *before* splitting on commas, so `(n) noun`
became the single token `(n)noun`, which no model output can equal:

| Gold | Old tokens | New tokens | Model says `noun` |
|---|---|---|---|
| `(n) noun` | `{(n)noun}` | `{noun}` | old ✗ / new ✓ |
| `(adv) adverb` | `{(adv)adverb}` | `{adverb}` | old ✗ / new ✗ |

Re-scoring the archived outputs confirms the mechanism exactly. Taking
`results/evaluation_reports_pre_fix_backup/pos_tagging/pos_tagging_gemma-7b.json`
(3,899 rows, gold stored as `(adv) adverb`) and applying each rule to the
same stored model outputs:

- old `normalize_tags`: 0/3,899 = `0.0000`, reproducing the archived file;
- new `normalize_tags`: 3,121/3,899 = `0.8005`.

The archived `0.0000` and the corrected `0.8188` are therefore the same model
outputs under two label-handling rules, not two different experiments. The
row count differs (3,899 archived vs 5,701 current) because the lexicon
correction in §1.2 changed the evaluation set as well; that is a separate
change from the normalisation fix.

### 1.2 The reverse-direction lexicon block was never reoriented

`results/evaluation_reports/Combined_POS_Lexicon.csv` holds two datasets
concatenated under one pair of headers:

| rows | direction | `English` column contains | `Tigrigna` column contains |
|---|---|---|---|
| 0–3586 | English→Tigrinya | English | Tigrinya |
| 3587–7233 | Tigrinya→English | **Tigrinya** | **English** |

The reverse block was evaluated as-is, so 3,647 items presented Ge'ez-script
input in a field labelled English — models were asked to tag or translate the
wrong language under a mislabelled prompt.

Fixed: `scripts/build_merged_eval_set.py` detects the reverse block by script
(not by hardcoded index) and swaps the columns before merging.

### 1.3 Two model identifiers do not match the checkpoints loaded

| Published as | Actually loaded |
|---|---|
| Falcon-10B | `tiiuae/falcon-7b-instruct` (7B) |
| mT5-base | `google/mt5-small` |

Falcon-7B is the correct model; the published "Falcon-10B" label is
incorrect. All reported Falcon results, including the full evaluation run,
were produced by the 7B instruction-tuned checkpoint. Result filenames and
JSON keys now use `falcon-7b` and `mt5-small`.

### 1.4 BLEU was never wired into scoring

`utils/metrics.py` defined `compute_bleu` but nothing called it; translation
fidelity reported exact-match accuracy only. The BLEU column in the published
Table 5 (21.8–26.5) has no generating code in this repository.

Fixed: `utils/metrics.py` now provides corpus-level chrF and character-level
BLEU, used by `scripts/compute_bleu.py`.

### 1.5 Dataset size (clarification, not an error)

7,234 is correct as a count of **annotated phrase pairs**, counted
bidirectionally (3,587 forward + 3,647 reverse). Merging the two directions
yields **5,775 unique evaluation items**, which is what accuracy figures are
computed over. Both numbers are right; the paper does not distinguish them.

---

### 1.6 Provenance: the source lexicon was not new and was not changed

`results/evaluation_reports/Combined_POS_Lexicon.csv` is the source lexicon
underlying the published benchmark, digitized from the Swansea
Tigrinya–English dictionary with additional entries contributed by
native-speaker linguists (paper §3.1). It was not created or modified during
this audit.

This is verifiable from files already in the repository. Compared
case-insensitively against `data/gold_labels/backup_pre_split/pos_tags_fixed.json`
— the pre-audit state — the two are **identical**: 7,217 distinct
`(English, POS_CATEGORIES)` pairs on both sides, with no pair present in one
and absent from the other. The only textual difference is letter case in 64
rows, where the CSV reads `(uncategorized) Uncategorized` and the backup
reads `(uncategorized) uncategorized`.

The file first appears in version control in the commit of 22 September 2026
because that commit is the first time it was tracked, not because it was
altered; its modification time on disk predates that commit.

**No annotation was added, removed or revised.** Every correction in §1.1–1.5
concerns how this file's two direction-blocks were split, and how gold labels
were matched against model output — not the annotated content itself. The
scores changed because the split and the matching changed.

Verify with:

```bash
python - <<'EOF'
import pandas as pd, json
c = pd.read_csv('results/evaluation_reports/Combined_POS_Lexicon.csv')
b = json.load(open('data/gold_labels/backup_pre_split/pos_tags_fixed.json', encoding='utf-8'))
cs = {(str(a).strip(), str(p).strip().lower()) for a, p in zip(c['English'], c['POS_CATEGORIES'])}
bs = {(str(r['English']).strip(), str(r['POS_CATEGORIES']).strip().lower()) for r in b}
print('identical:', cs == bs, '| pairs:', len(cs), len(bs))
EOF
```

### 1.7 Job logs for the September 2026 runs

SLURM stdout/stderr for every run behind the corrected numbers is in `logs/`:

| Job | Log | Produces |
|---|---|---|
| 75167 | `eval_75167.out` | corrected POS / morphosyntax / lexical alignment |
| 76683 | `eval_tf_rerun_76683.out` | translation fidelity rerun |
| 77601 | `fewshot_probe_77601.out` | one-shot vs few-shot prompting probe |
| 77608 | `nllb_probe_77608.out` | NLLB-200-600M probe |
| 77617 | `nllb_sweep_77617.out` | NLLB-200 600M/1.3B/3.3B sweep |
| 75279 | `eval_xlmr_75279.out` | xlm-roberta follow-up |
| 77595 | `eval_falcon3_77595.out` | falcon3-10b attempt (failed to load, no results) |

These were added in a commit after the initial audit commit; they were
present on disk throughout but had not been tracked.

**Provenance of job 75167.** The file was written 2026-09-21 06:08 and first
committed 2026-09-25 (`f27e5de`). It therefore postdates the published paper
and records the *corrected* code, not the code the paper was written from:
the label-prefix fix in §1.1 landed 2026-09-22 (`ec3ee0d`), one day after the
run. The log is evidence for the corrected numbers in §2 and for nothing
about the published ones. The same holds for every log in the table above.

### 1.8 Table 2 (inter-annotator agreement) does not reproduce as published

The paper reports Cohen's κ of 0.89 (POS), 0.86 (Gender), 0.88 (Number),
0.84 (Agreement) and 0.91 (Lexical alignment) over 500 pairs "independently
annotated by two trained linguists".

The annotation records are released as
`data/POS_english_to_tigrigna_Annotated.xlsx`: a 500-item sample, two
annotator sheets, an adjudicated sheet and an annotation guide.
`scripts/compute_iaa.py` recomputes the agreement figures from them.

Both annotators sometimes left a cell empty where the other assigned a label.
Those are not matching judgements, so they are counted as disagreements and κ
is computed over all 500 items with a blank treated as its own category:

| Dimension | Published κ | Recomputed κ | Observed agreement |
|---|---|---|---|
| POS | 0.89 | 0.9076 | 0.9500 |
| Gender | 0.86 | 0.8079 | 0.9240 |
| Number | 0.88 | 0.6762 | 0.9280 |
| Agreement | 0.84 | 0.5912 | 0.9000 |
| Lexical alignment | 0.91 | 0.8487 | 0.9480 |

Restricting instead to items both annotators labelled gives 0.9076, 1.0000,
1.0000, 0.9815 and 0.9850 on 500, 420, 428, 445 and 474 items. Neither
convention reproduces the published row, and the two differ by up to 0.39,
so the published values cannot be confirmed from these records either way.

Three properties of the subset bound what any of these figures support.

**The columns are not independent of the adjudication.** On the 25 POS items
where the annotators differ, the adjudicated sheet reproduces Annotator 2 on
all 25 and Annotator 1 on none; on the three Gender/Agreement/Alignment
disagreements it reproduces Annotator 1. The adjudication is assembled from
the two columns rather than decided separately.

**Agreement on filled labels is near-total.** Where both annotators assigned
a gender or number value they never differ — 0 disagreements across 420 and
428 items. The divergence between them is in coverage, not in label choice.

**The subset is not a random sample.** The 500 items are whole small sheets
(all 9 prepositions, all 10 pronouns, all 6 interjections) plus the
alphabetical head of the large ones (305 adjectives, 150 nouns). The corpus
is 54% noun and 25% verb; the subset is 61% adjective and contains **no
verbs**.

These are statements about what the released records support. They are not a
claim about how the published values were produced.

A separate observation from the same file: the annotators' free-text notes on
201 of the 500 items document mislabelled parts of speech in the source
dictionary, inflected forms given as citation forms, and mistranslations
(`at least` → ብብዚሒ, which means *at most*). These support §1.6's statement
that the POS labels are the source dictionary's and were not independently
verified.

### 1.9 POS and morphosyntax were scored on an uncommitted item set

The POS tagging and morphosyntactic labelling results cover 5,701 items,
while the committed gold files hold 5,783 rows which the task scripts reduce
to 5,775. The two sets are not the same: 182 gold rows are absent from the
results and 108 result rows are absent from the gold.

The difference is systematic. Those runs were scored against a gold file in
which each multi-sense entry had been split to its first sense — `("a
little", "ንእሽቶይ, ቁሩብ,ውሕድ")` was evaluated as `("a little", "ንእሽቶይ")`. All 108
result-only rows are first senses of a multi-sense gold headword. That
intermediate file was never committed, so re-running the task scripts today
evaluates a different item set than the POS and morphosyntax tables report.

`scripts/build_sense_split_eval_set.py` reconstructs the set from the
committed gold and verifies it row-for-row against the saved model outputs:
5,701 items, no row missing, none extra, no label mismatch. The result is
released as `data/gold_labels/{pos_tagging,morphosyntax_probe}_sense_split.json`.

This is a reconstruction, not a recovered original. It is reproducible and
verified against the experimental record, but the file the September runs
actually read no longer exists.

## 2. What the corrected runs show

**The label-prefix fix changes every POS and morphosyntax score from zero to
a non-zero value.** The figures below are the task scripts' own
set-intersection accuracies, which is what the paper reported.

| Model | POS before | POS after | Morph before | Morph after |
|---|---|---|---|---|
| Gemma-7B | 0.00 | **81.88** | 0.00 | **88.16** |
| Mistral-7B | 0.00 | 78.76 | 0.00 | 79.56 |
| Gemma-2B | 0.00 | 73.29 | 0.00 | 75.60 |
| Falcon-7B | 0.00 | 43.96 | 0.00 | 74.27 |
| Qwen-7B | 0.00 | 17.07 | 0.00 | 53.96 |
| ByT5 | 0.00 | 0.05 | 0.00 | 37.41 |
| mT5-large | 0.00 | 0.09 | 0.00 | 0.72 |
| mT5-small | 0.00 | 0.00 | 0.00 | 2.44 |

"before" = `results/evaluation_reports_pre_fix_backup/`, "after" =
`results/evaluation_reports/`.

**These numbers should not be read as a comparison between architectures,
and an earlier revision of this file did so in error.** Three findings
documented since make that reading unsupportable:

- The scorer counts a row correct if *any* word of the answer matches a gold
  label word (§3). Gemma-7B's median answer is 14 words and scores 0.00%
  under whole-answer matching, 79.39% on its first word alone.
- A constant `noun` answer scores 53.73% on the same items, and an English
  POS tagger that never sees the Tigrinya scores 60.38%. Falcon-7B and
  Qwen-7B fall below both.
- The five causal models are instruction-tuned checkpoints; mT5 and ByT5 are
  raw span-denoising models that were never trained to follow instructions,
  as `models/span_infilling.py` records. The contrast is between models that
  can follow an instruction and models that cannot, not between
  architectures.

The seq2seq models produce degenerate repetition (`ኣብ ኣብ ኣብ …`) rather than
valid labels, reproduced independently in both runs. That is a statement
about those checkpoints under this prompt, not about encoder-decoder models
in general.

`scripts/rescore_tasks.py` reports these outputs under three matching rules
with both baselines; `PAPER_DRAFT.md` §4.1–4.2 present that analysis, and
supersede the architecture claim made here.

### Translation

All eight models score effectively zero (chrF ≤ 0.81, exact ≤ 0.28%). This is
a property of the models, not the benchmark or the metric: NLLB-200, which
supports Tigrinya (`tir_Ethi`), reaches **chrF 16.05 / 16.24% exact** on the
same 5,775 rows with the same scorer.

| Model | chrF | BLEU-4(char) | BLEU-4(word) | exact% |
|---|---|---|---|---|
| NLLB-200-3.3B | 16.05 | 14.23 | 0.24 | 16.24 |
| NLLB-200-1.3B | 15.01 | 13.29 | 0.23 | 15.00 |
| NLLB-200-600M | 14.11 | 12.59 | 0.25 | 14.42 |
| ByT5 | 0.81 | 0.01 | 0.00 | 0.05 |
| Gemma-2B | 0.39 | 0.01 | 0.01 | 0.07 |
| Mistral-7B | 0.26 | 0.00 | 0.00 | 0.03 |
| mT5-small | 0.20 | 0.04 | 0.02 | 0.05 |
| Gemma-7B | 0.16 | 0.00 | 0.00 | 0.03 |
| Falcon-7B | 0.13 | 0.00 | 0.00 | 0.03 |
| mT5-large | 0.13 | 0.04 | 0.01 | 0.02 |
| Qwen-7B | 0.04 | 0.00 | 0.00 | 0.28 |

**Word-level BLEU-4 is not a usable metric on this data.** References average
~1.37 whitespace tokens, so 4-grams are largely undefined: NLLB-3.3B scores
BLEU-4(word) = 0.24 while getting 16.24% of items exactly right. chrF and
character-level BLEU are the appropriate metrics at lexicon scale.

Attempts to raise translation scores by other means were tried and did not
work: few-shot prompting with 8 exemplars and an explicit output-format
constraint moved gemma-7b from chrF 0.16 to 0.66 with exact match unchanged
at 0.00; extracting the Ge'ez span from verbose output recovered 2 of 5,775
rows. Median character-similarity between model output and reference is 0.000.

---

## 3. What is still unresolved

**The values in the published Table 5 do not match any run producible from
this codebase, before or after the fixes. The source of those specific
numbers has not been located.**

Specifically:

- The pre-fix run scored `0.0000` on POS, morphosyntax and translation for
  all eight models — not the 70–80% reported.
- The post-fix run reproduces the causal-model POS figures approximately
  (Gemma-2B 73.29 vs 74.5 published) but not the seq2seq ones (ByT5 0.05 vs
  78.0; mT5-base/small 0.00 vs 78.9).
- Reconstructing the pre-fix data with the prefix stripped gives ByT5 and
  mT5-base 0.0% — their raw outputs contain no POS tag under any scoring
  rule.
- No run produces the lexical alignment value `1.0000` reported for four
  models.
- The BLEU column has no generating code, and 21.8–26.5 is not reachable on
  this data by any model tested, including a purpose-built MT system.

This is a statement about what could not be reproduced here. It is not a
claim about how the published values were produced; that has not been
determined.

---

## 4. Reproducing this audit

```bash
source env/bin/activate
python scripts/build_merged_eval_set.py --dry-run --verify  # rebuilds the 5,783 gold rows
python scripts/build_sense_split_eval_set.py --verify  # 5,701 POS/morph items (§1.9)
python scripts/compute_iaa.py                       # agreement figures (§1.8)
python scripts/rescore_tasks.py --models gemma-2b,gemma-7b,mistral-7b,\
falcon-7b,qwen-7b,mt5-small,mt5-large,byt5          # three rules + baselines (§2)
./scripts/run_all_evaluations.sh                    # full sweep (cluster)
python scripts/compute_bleu.py                      # chrF / BLEU from saved outputs
python scripts/probe_nllb_translation.py \
    --checkpoint facebook/nllb-200-3.3B --n 0       # MT baseline
```

Original pre-fix results are preserved under
`results/evaluation_reports_pre_fix_backup/` and
`results/evaluation_reports/translation_fidelity_pre_wordmatch_fix_backup_2026-09-21/`.
