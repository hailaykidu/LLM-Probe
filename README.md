# LLM Probe

A multi-model, multi-task probing harness that measures how well general-purpose
LLMs (none fine-tuned for Tigrinya) handle Tigrinya at four different linguistic
levels: word-level translation, lexical alignment, part-of-speech tagging, and
morphosyntactic feature identification. Every model is evaluated **zero-shot**,
via prompting only -- no fine-tuning happens in this project (that's what
[`LLAMA3_M`](../../../../LLAMA3_M) and [`EnTiMT`](../../../EnTiMT) do instead).

## Status

**Complete.** Last run finished 2026-07-17. All four tasks ran to completion
across 8 of the 9 configured models, consolidated into
`results/evaluation_reports/all_results.json`.

## Data

- **Source lexicon**: `data/lexicon.json` -- 3,561 English-Tigrinya word/phrase
  entries with part-of-speech tags, each Tigrinya entry given as a list of
  acceptable translations (e.g. `"a little"` (adv) -> `["ንእሽቶይ", "ቁሩብ", "ውሕድ"]`).
- **Gold labels** (`data/gold_labels/`): derived from the lexicon --
  `translations.json`, `pos_tags.json` / `pos_tags_fixed.json`,
  `morpho_features_fixed.json`, plus `lexical_alignment.json` (word-to-word
  alignments) and `statistics.json` (corpus stats: 7,234 total entries, 7,068
  unique English phrases, 6,073 unique Tigrinya phrases, 967 multi-word
  English phrases / 2,000 multi-word Tigrinya phrases).
- Each task actually evaluates **3,899 items per model** (a subset of the full
  lexicon after dedup/filtering per task) -- 3,899 x 8 models = 31,192 rows
  per task in the consolidated results.

## Models evaluated

Loaded via per-model wrapper scripts in `models/` (`*_loader.py`), each
returning a `transformers` pipeline: `gemma-2b`, `gemma-7b`, `mistral-7b`,
`mt5-base`, `mt5-large`, `byt5`, `qwen-7b`, `falcon-10b`.

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
(`<extra_id_N>` placeholder tokens from T5-family models, echoed
"output format" instructions) via `strip_special_tokens()`, then apply
`clean_and_enforce_format()` before comparing to gold.

## Results

### Exact-match accuracy (strict -- output string must match gold exactly)

| Model | Translation fidelity | POS tagging | Morphosyntax probe | Lexical alignment (format) |
|---|---|---|---|---|
| gemma-2b | 0.44% | 0.00% | 0.00% | 0.31% |
| gemma-7b | 0.44% | 0.00% | 0.00% | 2.80% |
| mistral-7b | 0.44% | 0.00% | 0.00% | 1.15% |
| mt5-base | 0.00% | 0.00% | 0.00% | 0.95% |
| mt5-large | 0.00% | 0.00% | 0.00% | 0.03% |
| byt5 | 0.00% | 0.00% | 0.00% | 1.08% |
| qwen-7b | 0.44% | 0.00% | 0.00% | 0.62% |
| falcon-10b | 0.44% | 0.00% | 0.00% | 0.15% |

Exact match is close to 0% almost everywhere. This is largely a **format**
problem, not necessarily a total comprehension failure: models routinely wrap
answers in explanatory text ("The phrase ... translated into Tigrigna is
..."), continue generating past the requested single answer, or (T5-family
models) emit raw `<extra_id_0>` sentinel tokens instead of a clean
completion -- any of which fails an exact-match check regardless of whether
the core answer was right.

### Lexical alignment -- semantic match (softer metric: does the aligned
Tigrinya word count as a correct translation, ignoring exact formatting?)

| Model | Semantic match |
|---|---|
| mistral-7b | 88.0% |
| gemma-2b | 69.0% |
| falcon-10b | 42.8% |
| gemma-7b | 38.6% |
| mt5-large | 29.5% |
| qwen-7b | 24.3% |
| byt5 | 8.3% |
| mt5-base | 0.0% |

This is the one metric in this project where a real signal comes through:
**mistral-7b and gemma-2b actually align English-Tigrinya word pairs
correctly most of the time**, even though their raw output rarely matches the
expected string format exactly. `mt5-base` shows no semantic understanding at
all here.

## Interpreting these results together

- **Format-strictness dominates every task except lexical alignment's
  semantic check.** A model can "know" the right Tigrinya word and still
  score 0% if the harness expects a bare word and gets a full sentence.
- **Where format noise is factored out (semantic match), model quality
  differences become visible and match rough intuition**: mistral-7b and
  gemma-2b (both instruction-tuned, reasonably strong general models) lead;
  mt5-base (the smallest, least instruction-tuned model here) trails at 0%.
- **POS tagging and morphosyntax probing got no signal at all**, even
  semantically -- these are harder, more structured linguistic judgments than
  word-level translation, and none of these general-purpose models appear to
  have picked up Tigrinya morphology from pretraining.
- This is the empirical basis for treating **general-purpose LLM prompting as
  unreliable for Tigrinya generation tasks** elsewhere in this line of work
  (e.g. the decision to fine-tune dedicated models in `LLAMA3_M` and
  `EnTiMT` rather than rely on zero-shot prompting).

## Reproducing

```bash
sbatch submit_evaluation_job.sh
```

Runs `scripts/run_all_evaluations.sh` on the `ampere` partition (4 GPUs, 32
CPUs, 128GB RAM). Each task script is resumable -- it loads any existing
partial results file and skips already-completed `(English, Tigrigna)` pairs
before continuing, so an interrupted run can restart without redoing
completed work.

Note: `submit_evaluation_job.sh`'s `#SBATCH --time=272:00:00` looks like a
typo for `72:00:00` (the accompanying comment says "72 hours") -- left as-is
here since it's the actual file the job used to produce these results, but
worth fixing before the next run.

## Environment

`env/` is a Python venv with the packages in `requirements.txt`: `transformers`,
`torch`, `sentencepiece`, `accelerate`, `protobuf`, `nltk` (for BLEU, defined
in `utils/metrics.py` but not currently wired into any task's scoring), `pandas`,
`tqdm`.

## Files

```
LLM_Probe/
├── data/
│   ├── lexicon.json / lexicon.csv       # source English-Tigrinya lexicon
│   ├── prompts/*.txt                    # one prompt template per task
│   └── gold_labels/*.json               # derived gold answers + corpus stats
├── models/*_loader.py                   # one HF pipeline loader per model
├── tasks/*.py                           # one evaluation script per task
├── utils/{logger,metrics}.py            # result logging + accuracy/BLEU helpers
├── scripts/run_all_evaluations.sh       # runs all 4 tasks across all models
├── submit_evaluation_job.sh             # SLURM entry point
├── results/evaluation_reports/          # per-model/per-task JSON + accuracy .txt, plus all_results.json
└── logs/                                # SLURM stdout/stderr per job id
```
