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
(`scripts/build_merged_eval_set.py`).

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

## 2. What is now confirmed

**Causal models substantially outperform sequence-to-sequence models on POS
tagging and morphosyntactic probing** once the label-prefix bug is fixed.
This reverses the direction reported in the paper.

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

Averaged by architecture on POS: causal 59.0, seq2seq 0.0. The paper reports
causal 74.1, seq2seq 79.0.

The seq2seq models produce degenerate repetition (`ኣብ ኣብ ኣብ …`) rather than
valid labels. This is reproduced independently in both runs.

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
python scripts/build_merged_eval_set.py --dry-run   # verify 5,775 merged items
./scripts/run_all_evaluations.sh                    # full sweep (cluster)
python scripts/compute_bleu.py                      # chrF / BLEU from saved outputs
python scripts/probe_nllb_translation.py \
    --checkpoint facebook/nllb-200-3.3B --n 0       # MT baseline
```

Original pre-fix results are preserved under
`results/evaluation_reports_pre_fix_backup/` and
`results/evaluation_reports/translation_fidelity_pre_wordmatch_fix_backup_2026-09-21/`.
