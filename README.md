# LLM Probe

A multi-model, multi-task probing harness that measures how well general-purpose
LLMs (none fine-tuned for Tigrinya) handle Tigrinya at four different linguistic
levels: word-level translation, lexical alignment, part-of-speech tagging, and
morphosyntactic feature identification. Every model is evaluated **zero-shot**,
via prompting only -- no fine-tuning happens in this project (that's what
[`LLAMA3_M`](../../../../LLAMA3_M) and [`EnTiMT`](../../../EnTiMT) do instead).

## Status

**Re-running (as of 2026-09-18)** after fixing a series of data-format,
prompt, and pipeline bugs that made every prior run (including the one
behind the published paper's Table 5 / thesis Table 5.9) unreliable. See
"Fixes applied" below. The previous "complete" run (finished 2026-07-17,
8,234-row dataset) is archived under `results/evaluation_reports_pre_fix_backup/`
for reference and is superseded -- do not cite its numbers.

## Data

- **Source of truth**: `results/evaluation_reports/Combined_POS_Lexicon.csv`
  (7,234 raw rows). This file turned out to have its `English`/`Tigrigna`
  columns swapped for exactly its second half (rows 3587-7233, an artifact of
  the bidirectional English->Tigrinya / Tigrinya->English construction never
  being normalized back to one column convention), plus 8 rows with no
  Tigrigna translation and 1,443 rows that were exact duplicates once the
  swap was corrected. `data/lexicon_combined_fixed.csv` is the corrected
  version: **5,783 rows**, 5,025 unique English headwords. Distinct senses of
  the same headword (e.g. "abuse" as noun vs. verb, "administrator"
  masculine vs. feminine noun) are preserved as separate rows rather than
  deduplicated away.
- **Source lexicon**: `data/lexicon.json` -- regenerated from the corrected
  CSV (5,783 entries, superseding the old 3,561-entry file, which is archived
  at `data/gold_labels/backup_pre_split/lexicon_original.json`).
- **Gold labels** (`data/gold_labels/`), all regenerated from the corrected
  CSV:
  - `pos_tags_fixed.json` -- part-of-speech only (`noun`, `verb`, `adjective`, ...).
  - `morpho_features_fixed.json` -- now genuinely encodes gender/number
    (`feminine`, `masculine`, `plural`) where the source data provides it, no
    longer a duplicate of the POS tag under a different name (see Fixes
    below). Only ~3% of rows carry a feature beyond "none" -- the source
    lexicon mostly doesn't encode gender/number, which limits what this task
    can measure regardless of pipeline correctness.
  - `translations.json` -- each row's Tigrigna form, losslessly regenerated.
  - `lexical_alignment.json` -- word-to-word alignments. 3,575 of 5,783 rows
    carry the original manually-verified alignment; the remaining 2,208 use a
    naive positional fallback (flagged via `"AlignmentSource": "auto"`) and
    should be prioritized for human review before being treated as gold.
  - `statistics.json` -- regenerated corpus stats for the corrected dataset.
  - Pre-correction versions of all four files are archived under
    `data/gold_labels/backup_pre_split/`.
- Each task loads its own already-corrected gold file directly (no more
  fragile `English`-only merge against `lexicon.json`) and evaluates
  **5,700-5,775 items per model**, depending on the task's own dedup key.

## Models evaluated

Loaded via per-model wrapper scripts in `models/` (`*_loader.py`), each
returning a callable matching the `transformers` pipeline call contract:
`gemma-2b`, `gemma-7b`, `mistral-7b`, `mt5-small`, `mt5-large`, `byt5`,
`qwen-7b`, `falcon-7b`.

Two models are relabeled from earlier naming to match the checkpoint each
loader actually loads (fixes #5 and #12 below): `falcon-10b` ->
`falcon-7b` (`models/falcon_loader.py` loads `tiiuae/falcon-7b-instruct`,
7B parameters) and `mt5-base` -> `mt5-small` (`models/mt5_loader.py` loads
`google/mt5-small`). Historical result files under
`results/evaluation_reports_pre_fix_backup/` still carry the old
`falcon-10b`/`mt5-base` filenames/keys, since they document runs made under
those mistaken labels.

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

## Fixes applied (2026-09-18)

A discrepancy between this project's results and the published paper
("LLM Probe: Evaluating LLMs for Low-Resource Languages", LLMs4SSH @ LREC
2026 / thesis Table 5.9) led to a full pipeline audit. Every prior run,
including the one behind the paper's Table 5, turned out to rest on a
combination of the following bugs. All are fixed as of this run:

1. **Empty-output generation bug.** `models/base_loader.py`'s
   text-generation pipeline had no `max_new_tokens`/`min_new_tokens`, so an
   instruct model whose first generated token was EOS returned an empty
   string once `return_full_text=False` stripped the prompt back out. This
   silently zeroed out every output for several causal models in an earlier
   run. Fixed: explicit `max_new_tokens=128`, `min_new_tokens=1`.
2. **Tokenizer forced to the slow path.** `use_fast=False` forced every
   model onto the slow, pure-Python tokenizer. Switched to `use_fast=True`
   (the well-tested Rust-backed path) on general principle, though testing
   afterward showed this was *not* the source of the mixed-script output
   (`ገSitz`, `ገΑνα`) seen in earlier runs -- re-encoding/decoding `ገዛ`
   through the fast tokenizer round-trips correctly. That output is real
   greedy-decoded model generation (e.g. Gemma-2B-it inserting German
   tokens like `Sitz`/`Besitzer` mid-response on a Tigrinya prompt), not a
   tokenizer artifact, and is left as-is since it reflects actual model
   behavior on out-of-distribution input rather than a pipeline bug --
   changing decoding strategy to suppress it would also deviate from the
   paper's stated greedy/temperature-0.0 methodology.
3. **English/Tigrigna columns swapped for half the source data.**
   `Combined_POS_Lexicon.csv`'s second half (rows 3587-7233) had English and
   Tigrigna reversed -- a bidirectional-lexicon artifact never normalized
   back to one column convention. Fixed at the data level (see "Data"
   above); this was also present in every downstream gold-label file.
4. **Many-to-many merges cross-multiplying/corrupting rows.** All four task
   scripts merged the lexicon and gold tables `on="English"` without
   deduplicating a non-unique key, silently cross-multiplying rows for any
   headword with more than one sense. Fixed by regenerating each gold file
   directly from the corrected source (no more merge against `lexicon.json`)
   and deduplicating on `(English, Tigrigna)` -- not `English` alone -- so
   distinct senses are preserved rather than discarded.
5. **Falcon mislabeled as 10B.** `models/falcon_loader.py` loads
   `tiiuae/falcon-7b-instruct` (7B). Relabeled `falcon-10b` -> `falcon-7b`
   throughout.
6. **Gold-label parenthetical format bug (POS tagging & morphosyntax
   probing).** Gold values like `"(v) verb"` were normalized by stripping
   spaces *before* splitting on commas, gluing the abbreviation onto the
   word into one unmatchable token (`"(v)verb"`). This alone accounts for
   the flat 0% these two tasks showed in every prior run, independent of
   whether the model's answer was actually correct. Fixed: strip the
   parenthetical, tokenize on words.
7. **Unstripped "Output:" self-echo.** Every prompt's few-shot example ends
   with a literal `Output: ...` line, and models routinely echo that label
   back (`"Output: noun"`). None of the four tasks stripped it before
   comparing to gold. Added a shared regex across all four task scripts.
8. **Multi-word/multi-sense "item" in POS & morphosyntax prompts.** ~6-19%
   of rows filled the prompt's single-word "Phrase:" slot with several
   comma-separated Tigrinya forms at once (e.g. `"ንእሽቶይ, ቁሩብ, ውሕድ"`), asking
   the model for one POS/feature answer covering all of them -- an ill-posed
   question regardless of model quality. Fixed: use only the first form.
9. **`morpho_features_fixed.json`'s "Expected" field duplicated the POS
   tag** rather than encoding actual gender/number/agreement features, so no
   model could ever score correctly on real morphosyntactic content. Fixed
   by splitting the source CSV's compound tags (e.g. `"(nf) noun feminine"`)
   into POS + feature components; `Expected` now reads e.g. `"noun,
   feminine"` where the source data supports it.
10. **SLURM script pointed at a venv that no longer exists.**
    `submit_evaluation_job.sh` activated a venv under an old project path
    that was never migrated when the project moved to `Project/LLM-Probe`.
    Fixed the path and a `--time=272:00:00` typo (should be `72:00:00`).
11. **`"text2text-generation"` pipeline task removed in transformers 5.17.**
    The freshly-installed `transformers` version dropped the
    `text2text-generation` pipeline task entirely (`pipeline()` now only
    supports `text-generation`, hardcoded to `AutoModelForCausalLM`), so
    `models/mt5_loader.py` and `models/byt5_loader.py` crashed at import
    time -- and because every task script builds its `models` dict eagerly
    at module load, this killed the *entire* task (all 8 models, not just
    the two seq2seq ones) before any evaluation ran. This is what killed job
    75151's first attempt. Fixed by adding `Seq2SeqPipeline` in
    `models/base_loader.py`, a minimal wrapper around `model.generate()`
    that reproduces the same call contract every task script expects,
    bypassing the `pipeline()` factory for `mt5`/`byt5` entirely.
12. **`mt5-base` was actually `google/mt5-small`.** Same class of bug as
    Falcon (#5): `models/mt5_loader.py` loads `google/mt5-small`, but every
    task script and result file labels it `"mt5-base"`. Relabeled to
    `mt5-small` throughout.
13. **Eager, unguarded model loading killed entire task scripts on any
    single model's load failure.** Every task script built its `models`
    dict as one literal with every `load_*()` call evaluated inline and
    unguarded. `xlm-roberta`'s load failure (a version-dependent
    `IndexError` in `accelerate`'s device-map inference, different from its
    previously-documented prompting-time failure) raised an exception that
    killed the whole module before the evaluation loop ran, silently
    discarding every other model's results for that task -- this is what
    caused job 75159 to print "All evaluations completed" while writing
    zero result rows for all four tasks. Fixed: each model now loads in its
    own `try/except`, with a failure logged and skipped rather than fatal.
14. **`lexical_alignment.txt`'s prompt showed multiple untrimmed candidate
    translations.** For 182 of 5,775 rows (~3.2%), the prompt's "Tigrigna:"
    line listed every comma-separated candidate (e.g. "abandon" ->
    "ገደፈ, ረጥረጠ"), but `ExpectedAlignment` is always annotated against only
    the first one shown -- the model had no way to know which candidate the
    gold answer expected. Fixed: the prompt now shows only the first
    candidate; the stored `Tigrigna` field on each result row is unchanged
    (full multi-value string), since `clean_and_enforce_format`'s fallback
    already keys off the first candidate specifically. Applied while job
    75167 was mid-run (after gemma-2b's lexical alignment had already
    completed under the unfixed prompt) -- see the caveat on that run's
    results below.

**Not fixed, flagged instead:** `lexical_alignment.json`'s word-level
alignments are only manually verified for 3,575 of 5,783 rows; the remaining
2,208 use a naive positional fallback (`"AlignmentSource": "auto"`) pending
human review. `morpho_features_fixed.json` only carries a real feature
beyond "none" for ~3% of rows, since the source lexicon mostly doesn't
encode gender/number -- this limits what the morphosyntax task can measure
regardless of pipeline correctness.

The previous run's results are archived at
`results/evaluation_reports_pre_fix_backup/` and should not be cited; they
reflect the bugs above, not model capability.

**Caveat on job 75167 specifically:** its lexical alignment result for
`gemma-2b` was scored before fix #14 above landed, so ~3.2% of its rows were
evaluated against the untrimmed, multi-candidate prompt. Every other
model/task combination in that run used the fixed prompt. Re-run
`gemma-2b`'s lexical alignment alone (delete
`results/evaluation_reports/lexical_alignment/lexical_alignment_gemma-2b.json`
and its accuracy `.txt`, then resubmit) before treating its lexical
alignment number as directly comparable to the other seven models.

## Reproducing

```bash
sbatch submit_evaluation_job.sh
```

Runs `scripts/run_all_evaluations.sh` on the `ampere` partition (4 GPUs, 32
CPUs, 128GB RAM). Each task script is resumable -- it loads any existing
partial results file and skips already-completed `(English, Tigrigna)` pairs
before continuing, so an interrupted run can restart without redoing
completed work. Before resuming into `results/evaluation_reports/`, make
sure any files there weren't produced by the pre-fix pipeline -- stale
`(English, Tigrigna)` keys from an old run will be wrongly treated as
"already done" and skipped. Move old results out of the way (as was done
for this run, into `results/evaluation_reports_pre_fix_backup/`) before
resubmitting after a data/prompt change.

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
│   ├── lexicon_combined_fixed.csv        # corrected source (swap fixed, POS/features split)
│   ├── prompts/*.txt                     # one prompt template per task
│   └── gold_labels/
│       ├── *.json                        # corrected gold answers + corpus stats
│       └── backup_pre_split/             # pre-correction versions, for reference
├── models/*_loader.py                    # one HF pipeline loader per model
├── tasks/*.py                            # one evaluation script per task
├── utils/{logger,metrics}.py             # result logging + accuracy/BLEU helpers
├── scripts/run_all_evaluations.sh        # runs all 4 tasks across all models
├── submit_evaluation_job.sh              # SLURM entry point
├── results/
│   ├── evaluation_reports/               # per-model/per-task JSON + accuracy .txt, plus all_results.json
│   │   └── Combined_POS_Lexicon.csv      # raw source lexicon, pre-correction (see Fixes)
│   └── evaluation_reports_pre_fix_backup/ # archived pre-fix run -- do not cite
└── logs/                                 # SLURM stdout/stderr per job id
```
