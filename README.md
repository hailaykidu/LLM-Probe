# LLM-Probe

Code, data and evaluation records for a lexicon-grounded evaluation of eight
general-purpose LLMs on Tigrinya, across part-of-speech tagging,
morphosyntactic labelling, English→Tigrinya translation and lexical
alignment. All models are evaluated zero-shot, by prompting only; nothing is
fine-tuned here.

The paper is in [`paper/`](paper/). Every number it reports is recomputed
from the artifacts in this repository by the commands below, none of which
requires re-running a model.

## Layout

```
data/
  POS_english_to_tigrigna_Annotated.xlsx   annotation workbook (§3.3)
  lexicon_combined_fixed.csv               merged, reoriented source lexicon
  lexicon.json                             the same lexicon as records
  gold_labels/
    pos_tags_fixed.json                    POS gold, 5,783 rows
    morpho_features_fixed.json             morphosyntax gold
    translations.json                      translation references
    lexical_alignment.json                 word alignments
    *_sense_split.json                     the 5,701-item tagging sets
    statistics.json                        corpus statistics
  prompts/                                 one template per task
models/                 one loader per model, plus shared loading code
tasks/                  one evaluation script per task
scripts/                dataset construction, scoring and probes
utils/                  metrics and logging helpers
results/
  evaluation_reports/
    Combined_POS_Lexicon.csv               digitised source, as received
    rescored_2026-09-28.json               three matching rules + baselines
    <task>/                                per-model outputs and accuracies
paper/                  LaTeX source, bibliography and compiled PDF
```

## Reproducing the reported numbers

```bash
python -m venv env && source env/bin/activate
pip install -r requirements.txt
python setup_nltk.py

ROSTER=gemma-2b,gemma-7b,mistral-7b,falcon-7b,qwen-7b,mt5-small,mt5-large,byt5

python scripts/build_merged_eval_set.py --dry-run --verify   # gold files, §3.1
python scripts/build_sense_split_eval_set.py --verify        # 5,701-item sets
python scripts/compute_iaa.py                                # agreement, §3.3
python scripts/task_ceiling.py                               # ceiling, §4.2
python scripts/rescore_tasks.py --models "$ROSTER"           # §5.1, §5.2
python scripts/compute_bleu.py                               # §5.3
```

`--verify` rebuilds an artifact from its source and compares it against what
is committed, reporting any divergence rather than overwriting.

`--models` pins a table to a fixed set. Without it, every result file present
is scored, so a later run over additional models changes the output.

## Evaluation sets

Three sizes appear in the paper and are not interchangeable:

| Count | What it is |
|---|---|
| 5,783 | gold rows, deduplicated on (English, Tigrinya, features) |
| 5,775 | items for translation and lexical alignment, after each task drops duplicate pairs and rows with an empty side |
| 5,701 | items for the two tagging tasks, each multi-sense entry reduced to its first sense |

Appendix A of the paper gives the row-level derivation.

## Scoring

Model answers are frequently verbose, so the matching rule materially changes
the reported number. `scripts/rescore_tasks.py` reports three rules side by
side — exact, first-token and set-intersection — together with two baselines
that use no Tigrinya at all (a constant majority answer, and an English POS
tagger given only the English gloss), and the rate at which an answer merely
echoes the prompt's own example.

The per-task `*_accuracy_<model>.txt` files hold the set-intersection rule,
which is the most permissive of the three.

## Known limitations

- 2,208 of 5,783 lexical alignment labels are positional fallbacks rather
  than verified annotation, so that task is not well posed as scored.
- The morphosyntax gold is 96.8% identical to the POS gold; the two tasks
  measure nearly the same thing.
- The prompt supplies the English gloss alongside the Tigrinya form, and an
  English-only tagger scores 60.4% on POS, so the task is substantially
  solvable without Tigrinya.
- The evaluated roster mixes instruction-tuned causal models with raw
  span-denoising seq2seq checkpoints, so it cannot separate architecture from
  instruction-tuning.
- Entries are dictionary headwords, not running text.

## The annotation workbook

`data/POS_english_to_tigrigna_Annotated.xlsx` holds the annotation records
behind §3.3 of the paper. It is not an input to any model run; it documents
how a 500-item subset of the lexicon was annotated.

A 500-item subset was annotated independently by two of the paper's authors
for part of speech, gender, number, category agreement and lexical alignment.
Each assigned labels separately using the `Guide` sheet, which defines every
permitted value. The two sheets were then compared to locate disagreements,
and the final sheet was compiled from them. **No third annotator took part,
and nobody outside the author group adjudicated**, so the sheet named
`Sheet1` is a compilation of the two annotation columns rather than an
independent third judgement.

`python scripts/compute_iaa.py` recomputes every agreement figure from the
workbook and reports three things: Cohen's κ per dimension under both blank
conventions, which annotator the compiled sheet follows wherever the two
disagree, and how the 500 items were drawn. The paper reports no single
agreement coefficient for the dataset, and §3.3 states the three reasons.

## Provenance

The lexicon is digitised from a Tigrinya–English dictionary, with additional
entries contributed by native-speaker linguists. The source file
`results/evaluation_reports/Combined_POS_Lexicon.csv` is included as
received, before correction, so that the construction of the gold labels can
be checked end to end: `scripts/build_merged_eval_set.py --verify` rebuilds
the gold from it and reports any divergence from what is committed.

Two defects in that source are corrected rather than silently cleaned, and
both are documented in the paper. Its second block stores the two translation
directions under one pair of column headers, so the column named `English`
holds Ge'ez script for roughly half the rows; and gender and number markers
are embedded inline in the Tigrinya field rather than in the label column.

## Licence

Code is released under the MIT License; data and evaluation records under
CC BY 4.0. See [`LICENSE`](LICENSE), which also notes the terms attaching to
the source dictionary, to the third-party model checkpoints whose outputs are
recorded here, and to the ACL style files in `paper/`.

## Running a full evaluation

```bash
./scripts/run_all_evaluations.sh
```

Each task script is resumable: it loads any existing partial result file and
skips completed (English, Tigrinya) pairs. That resume key is also a hazard —
after a change to the gold data or a prompt, existing files look complete and
are skipped, silently mixing old and new scoring. Move the affected files
aside before resubmitting.

Models served over HTTP rather than loaded locally read their endpoint
address and key from `KBS_INFERENCE_BASE_URL` and `KBS_INFERENCE_API_KEY`;
neither is stored in this repository.
