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
  gold_labels/          gold answers per task, plus corpus statistics
  prompts/              one prompt template per task
  lexicon*.{csv,json}   merged source lexicon
  POS_english_to_tigrigna_Annotated.xlsx   annotation workbook (§3.3)
models/                 one loader per model
tasks/                  one evaluation script per task
scripts/                dataset construction, scoring and probes
utils/                  metrics and logging helpers
results/
  evaluation_reports/   per-model, per-task outputs and accuracy files
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
