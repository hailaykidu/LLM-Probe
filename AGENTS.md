# Agent Guide

## Project shape

This is a zero-shot Hugging Face evaluation harness for Tigrinya across four tasks:
translation fidelity, lexical alignment, POS tagging, and morphosyntax. Model wrappers live in `models/`; task-specific loading, normalization, scoring, and resumable evaluation live in `tasks/`; prompts and corrected gold labels live under `data/`.

Read [README.md](README.md) first for the authoritative dataset status, correction history, result caveats, and reproduction notes. Do not cite or mix files from `results/evaluation_reports_pre_fix_backup/` with current results.

## Running the project

Run commands from the repository root because task scripts use relative paths:

```bash
source env/bin/activate
python setup_nltk.py
./scripts/run_all_evaluations.sh
```

Use `python tasks/<task>.py` for one task. Use `EVAL_ONLY_MODELS=model-name` for a focused rerun, for example `EVAL_ONLY_MODELS=xlm-roberta`. The canonical cluster entry point is `sbatch submit_evaluation_job.sh`; see [submit_evaluation_job.sh](submit_evaluation_job.sh) for resources and environment setup. Dependencies are listed in [requirements.txt](requirements.txt); versions are intentionally not pinned.

There is no repository-owned test suite or build step. For Python-only changes, use `python -m compileall models tasks utils` as a cheap syntax check, then run the narrowest affected task when model access and GPU resources are available.

## Data and result invariants

- Use the corrected source and gold files under [data/](data/), not the archived/pre-fix data.
- Preserve distinct lexical senses. Evaluation identities generally use `(English, Tigrigna)`; do not deduplicate by `English` alone unless the task explicitly requires it.
- Each task reads its own corrected gold JSON directly. Avoid reintroducing an English-only merge against `data/lexicon.json`.
- Prompt templates in [data/prompts/](data/prompts/) are part of the evaluation contract. A prompt change requires a clean or deliberately scoped rerun because existing partial result files are resumable and may otherwise skip stale rows.
- Current outputs belong under `results/evaluation_reports/<task>/`; the orchestration script merges them into `all_results.json`.
- Lexical alignment labels marked `AlignmentSource: auto` are positional fallbacks, not manually verified gold.

## Model and task conventions

- Shared loading and generation behavior is in [models/base_loader.py](models/base_loader.py). Keep model-specific checkpoint choices in the corresponding `models/*_loader.py` wrapper.
- `mt5` and `byt5` use the local `Seq2SeqPipeline` because the installed Transformers version removed the `text2text-generation` pipeline task.
- Causal and seq2seq loaders generate with explicit `max_new_tokens=128` and `min_new_tokens=1`; preserve this unless the evaluation methodology intentionally changes.
- XLM-R is a masked model and is loaded onto one device to avoid the tied-weight `device_map="auto"` failure described in the README. Its open-ended prompts are not equivalent to fill-mask evaluation.
- Task scripts load models independently and skip a model whose loader fails. Do not make one optional model failure abort the whole task.
- Keep output cleanup and scoring rules local to the task unless the same normalization is clearly shared by all tasks.

## Change and validation checklist

1. Check whether the change affects prompts, gold labels, model labels/checkpoints, or resume keys; these can invalidate comparability with existing results.
2. Preserve UTF-8 and Tigrinya text exactly; avoid lossy CSV/JSON rewrites.
3. Run the narrowest available syntax or task check from the repository root.
4. For evaluation changes, record whether old result files must be moved aside or a model-specific rerun is required.

Useful implementation references: [models/base_loader.py](models/base_loader.py), [tasks/](tasks/), [scripts/run_all_evaluations.sh](scripts/run_all_evaluations.sh), [data/gold_labels/](data/gold_labels/), and [utils/metrics.py](utils/metrics.py). BLEU support exists in `utils/metrics.py` but is not currently part of task scoring.
