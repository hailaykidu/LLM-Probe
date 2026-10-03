# LLM Probe

A multi-model, multi-task probing harness that measures how well general-purpose
LLMs (none fine-tuned for Tigrinya) handle Tigrinya at four different linguistic
levels: word-level translation, lexical alignment, part-of-speech tagging, and
morphosyntactic feature identification. Every model is evaluated **zero-shot**,
via prompting only -- no fine-tuning happens in this project (that's what
[`LLAMA3_M`](../../../../LLAMA3_M) and [`EnTiMT`](../../../EnTiMT) do instead).

## Status

Current results are in `results/evaluation_reports/<task>/`, and the analysis
built on them is in [`PAPER_DRAFT.md`](PAPER_DRAFT.md). Every number in that
draft is regenerable from this repository; §7 of it lists the command and the
artifact behind each table.

Earlier runs are archived under `results/evaluation_reports_pre_fix_backup/`
and `results/evaluation_reports/translation_fidelity_pre_wordmatch_fix_backup_2026-09-21/`.
They are kept so the current scoring can be checked against what preceded it,
and their numbers should not be cited or mixed with current results.

## Data

- **Source lexicon**: `results/evaluation_reports/Combined_POS_Lexicon.csv`,
  7,234 rows, digitised from a bilingual dictionary. It holds two
  concatenated blocks, English→Tigrinya followed by Tigrinya→English, under
  one pair of headers, so in the second block the `English` column carries
  Ge'ez script. `scripts/build_merged_eval_set.py` detects that block by
  script, reorients it and merges both directions; `--verify` checks the
  output against the committed gold files.
- **Gold labels** (`data/gold_labels/`), **5,783 rows**, all built by that
  script. Deduplication keys on (English, Tigrinya, features), so distinct
  senses of one form survive as separate items: `estimate → ግምት` is kept as
  both noun and verb.
  - `pos_tags_fixed.json` — part of speech only (`noun`, `verb`, `adjective`, …).
  - `morpho_features_fixed.json` — part of speech plus the gender and number
    the source encodes (`noun, masculine`, `noun, plural`). Only 185 of 5,783
    rows carry a feature beyond the POS tag, which bounds what this task can
    measure.
  - `translations.json` — each row's Tigrinya form.
  - `lexical_alignment.json` — word-to-word alignments. 3,575 rows carry a
    manually verified alignment; the remaining 2,208 are a positional
    fallback flagged `"AlignmentSource": "auto"` and are not verified gold.
  - `statistics.json` — corpus statistics.
  - `{pos_tagging,morphosyntax_probe}_sense_split.json` — the 5,701-item sets
    the POS and morphosyntax runs used, in which a multi-sense entry is
    represented by its first sense. Built and verified against the saved
    model outputs by `scripts/build_sense_split_eval_set.py`.
- Each task loads its own gold file directly and applies its own
  deduplication at load time, evaluating **5,701–5,775 items per model**
  depending on the task. The three counts are explained in §2.1 of
  [`PAPER_DRAFT.md`](PAPER_DRAFT.md).
- `data/gold_labels/backup_pre_split/` holds earlier versions of these files,
  kept for reference.

## Models evaluated

Loaded via per-model wrapper scripts in `models/` (`*_loader.py`), each
returning a callable matching the `transformers` pipeline call contract:
`gemma-2b`, `gemma-7b`, `mistral-7b`, `mt5-small`, `mt5-large`, `byt5`,
`qwen-7b`, `falcon-7b`.

Two models are relabeled from earlier naming to match the checkpoint each
loader actually loads: `falcon-7b` (`models/falcon_loader.py` loads
`tiiuae/falcon-7b-instruct`, 7B parameters) and `mt5-small`
(`models/mt5_loader.py` loads `google/mt5-small`). Files under
`results/evaluation_reports_pre_fix_backup/` carry the earlier
`falcon-10b`/`mt5-base` keys, since they record runs made under those labels.

Two models were configured but did **not** produce results:
- `xlm-roberta`: errored on every task ("No mask_token (`<mask>`) found on the
  input") -- it was probed with an open-ended generation prompt, which doesn't
  suit an encoder-only MLM without a fill-mask-style prompt.
- `apertus-8b`: skipped -- its download stalled for hours on the HF CDN (see
  `models/apertus_loader.py`; excluded at the shell-script level in
  `scripts/run_all_evaluations.sh` rather than removed from the code).

## Tasks & prompting

Each task has a fixed prompt template in `data/prompts/*.txt`, filled in per
item and passed to every model identically:

| Task | Prompt (abridged) | What's scored |
|---|---|---|
| `translation_fidelity` | "Translate the following English phrase into Tigrinya. Answer only with the Tigrinya translation." | exact match against any acceptable gold translation |
| `lexical_alignment` | "Align the following English sentence with its Tigrinya translation... EnglishWord→TigrinyaWord" | exact format match **and** a separate semantic-match flag |
| `pos_tagging` | "Identify the part of speech of the following Tigrigna item. Answer only with one word (e.g., noun, verb, adjective)." | exact match against gold POS tag |
| `morphosyntax_probe` | "Identify the morphosyntactic features of the following Tigrigna phrase. Answer only with a comma-separated list of lowercase terms." | exact match against gold feature list |

Task scripts (`tasks/*.py`) strip common formatting noise before scoring
(`<extra_id_N>` placeholder tokens from T5-family models, echoed "Output:"/
"Answer:" prefixes, echoed "output format" instructions) via
`strip_special_tokens()`, then apply `clean_and_enforce_format()` (lexical
alignment only) before comparing to gold.

## Scoring

Model answers are frequently verbose, so the matching rule changes the
reported number substantially. `scripts/rescore_tasks.py` reports three rules
side by side for POS tagging and morphosyntactic labelling:

- **exact** — the cleaned answer equals the gold label.
- **first-token** — the answer's first word is a gold label word. This is the
  answer the model commits to, and the rule the analysis leads with.
- **set-intersection** — any word of the answer is a gold label word. This is
  permissive: a long answer naming several parts of speech is correct as soon
  as one fits. The per-task `*_accuracy_<model>.txt` files hold this rule.

It also reports two baselines a model must beat to demonstrate Tigrinya
knowledge — always answering the majority label, and an English POS tagger
given only the English gloss — and how often an answer merely echoes the
prompt's own example.

Translation is scored with chrF and character-level BLEU
(`scripts/compute_bleu.py`). Tigrinya references average 1.37 whitespace
tokens, so word-level BLEU-4 is near-undefined at this scale.

## Known limitations

- 2,208 of 5,783 lexical alignment labels are positional fallbacks, not
  verified annotation, so that task is not well-posed as scored.
- The morphosyntax gold is 96.8% identical to the POS gold, so the two tasks
  measure nearly the same thing.
- The prompt supplies the English gloss alongside the Tigrinya form, and an
  English-only tagger scores 60.4% on POS, so the task is substantially
  solvable without the Tigrinya.
- The evaluated roster mixes instruction-tuned causal models with raw
  span-denoising seq2seq checkpoints, so it cannot separate architecture from
  instruction-tuning.
- Entries are dictionary headwords, not running text.

## Reproducing

```bash
sbatch submit_evaluation_job.sh
```

Runs `scripts/run_all_evaluations.sh` on the `ampere` partition (4 GPUs, 32
CPUs, 128GB RAM). Each task script is resumable: it loads any existing
partial results file and skips already-completed `(English, Tigrigna)` pairs,
so an interrupted run restarts without redoing finished work.

That resume key is also a hazard. After a change to the gold data or a
prompt, existing result files hold keys that look complete and will be
skipped, so the run silently mixes old and new scoring. Move the affected
files aside before resubmitting.

To regenerate the analysis from saved outputs, without any inference:

```bash
python scripts/build_merged_eval_set.py --dry-run --verify   # gold files
python scripts/build_sense_split_eval_set.py --verify        # 5,701-item sets
python scripts/compute_iaa.py                                # annotation agreement
python scripts/rescore_tasks.py --models <comma-separated>   # three rules + baselines
python scripts/compute_bleu.py                               # chrF / char-BLEU
```

`--models` pins a table to a fixed roster; without it, every result file
present is scored, so a later run changes the output.

## Environment

`env/` is a Python venv with the packages in `requirements.txt`: `transformers`,
`torch`, `sentencepiece`, `accelerate`, `protobuf`, `nltk` (for BLEU, defined
in `utils/metrics.py` but not currently wired into any task's scoring), `pandas`,
`tqdm`.

## Files

```
LLM_Probe/
├── data/
│   ├── lexicon.json                      # corrected English-Tigrinya lexicon (5,783 entries)
│   ├── lexicon_combined_fixed.csv        # merged, reoriented source lexicon
│   ├── POS_english_to_tigrigna_Annotated.xlsx  # annotation workbook (§2.3)
│   ├── prompts/*.txt                     # one prompt template per task
│   └── gold_labels/
│       ├── *.json                        # gold answers + corpus stats
│       └── backup_pre_split/             # earlier versions, for reference
├── models/*_loader.py                    # one HF pipeline loader per model
├── tasks/*.py                            # one evaluation script per task
├── utils/{logger,metrics}.py             # result logging + accuracy/BLEU helpers
├── scripts/
│   ├── build_merged_eval_set.py          # source CSV -> gold files (--verify)
│   ├── build_sense_split_eval_set.py     # the 5,701-item POS/morph sets
│   ├── rescore_tasks.py                  # three matching rules + baselines
│   ├── compute_bleu.py                   # chrF / character BLEU
│   ├── compute_iaa.py                    # annotation agreement
│   └── run_all_evaluations.sh            # runs all 4 tasks across all models
├── submit_evaluation_job.sh              # SLURM entry point
├── results/
│   ├── evaluation_reports/               # per-model/per-task JSON + accuracy .txt
│   │   └── Combined_POS_Lexicon.csv      # source lexicon, as digitised
│   └── evaluation_reports_pre_fix_backup/ # earlier run -- do not cite
└── logs/                                 # SLURM stdout/stderr per job id
```
