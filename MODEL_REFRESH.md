# 2026 model roster refresh

Replaces the evaluated checkpoints with current open-weight successors.
**Nothing here has been re-run**: no result file, results table or reported
number has been touched. The numbers in `PAPER_DRAFT.md`, `CORRECTIONS.md`
and `README.md` belong to the old roster and remain attached to it.

## Replacement map

| Old checkpoint | New checkpoint | Verified |
|---|---|---|
| `google/gemma-2b-it` | `google/gemma-4-E2B` | yes |
| `google/gemma-7b-it` | `google/gemma-4-12B` | yes |
| `mistralai/Mistral-7B-Instruct-v0.2` | `mistralai/Ministral-3-8B-Instruct-2512` | yes |
| `tiiuae/falcon-7b-instruct` | `tiiuae/Falcon-H1R-7B` | yes |
| `Qwen/Qwen1.5-7B-Chat` | `Qwen/Qwen3.6-27B` | yes |
| `google/mt5-small` | `google/t5gemma-2-270m-270m` | yes |
| `google/mt5-large` | `google/t5gemma-2-1b-1b` | yes |
| `google/byt5-small` | unchanged | — |

All nine IDs resolved through `huggingface_hub.model_info`. The old code used
instruction-tuned checkpoints throughout, so the Instruct/R variants were
chosen for Ministral and Falcon rather than the Base variants.

Aliases in the task scripts were renamed to match
(`gemma-2b` → `gemma-4-e2b`, `mt5-small` → `t5gemma-2-270m`, and so on), so
new result files cannot silently overwrite or be confused with old ones.

## Loading changes required

**Auto classes.** Only Falcon-H1 registers as `FalconH1ForCausalLM`. The rest
are `*ForConditionalGeneration`:

| Model | `model_type` | Resolves through |
|---|---|---|
| gemma-4-E2B / 12B | `gemma4`, `gemma4_unified` | `AutoModelForCausalLM` (works despite being multimodal) |
| Falcon-H1R-7B | `falcon_h1` | `AutoModelForCausalLM` |
| Qwen3.6-27B | `qwen3_5` | `AutoModelForCausalLM` |
| t5gemma-2 | `t5gemma2` | `AutoModelForSeq2SeqLM` |
| Ministral-3-8B | `mistral3` | **neither** — see below |

**Ministral-3 needs a new branch.** It registers only in
`MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES`; both `AutoModelForCausalLM` and
`AutoModelForSeq2SeqLM` fail. `base_loader.py` gained an `image_text` type
that loads it via `AutoModelForImageTextToText`. The vision tower is loaded
but never fed; generation runs on the text decoder.

**dtype must not be forced.** `base_loader.py` previously hardcoded
`torch.float16` for seq2seq and causal models. t5gemma-2 ships bfloat16
weights and fails immediately under float16:

```
RuntimeError: expected m1 and m2 to have the same dtype,
but got: c10::BFloat16 != c10::Half
```

Both branches now use `dtype="auto"`, which honours each checkpoint's stored
precision. This also affects the old models, which were published in float16
and are unchanged by `"auto"`.

**`T5Gemma2Config` has no `decoder_start_token_id`.** Accessing it raises
`AttributeError`. The local `Seq2SeqPipeline` calls `model.generate()`
without referencing it, so no change was needed — but any code that reads it
directly will break.

**Span-infilling no longer applies to the seq2seq models.**
`models/span_infilling.py` appended `<extra_id_0>` for raw mT5/ByT5
checkpoints, which are span-denoising only. t5gemma-2 is instruction-capable,
so a sentinel would suppress the answer rather than elicit it.
`SPAN_INFILLING_MODELS` is now `{"byt5"}`.

**Falcon-H1 kernels.** `mamba_ssm` and `causal_conv1d` are **not installed**.
The config declares `mamba_d_state=256`, `mamba_n_heads=24`, `mamba_expand=2`.
transformers falls back to a slower pure-PyTorch path when the fused kernels
are absent; installing them would speed up Falcon-H1 but is not required for
correctness. Flagged rather than installed, since it changes the environment.

**Minimum versions.** `requirements.txt` now pins `transformers>=5.17.0` and
`torch>=2.6`. The `gemma4`, `gemma4_unified`, `mistral3`, `falcon_h1` and
`t5gemma2` model types are not registered in earlier releases. Installed
here: transformers 5.17.0, torch 2.14.0+cu130.

## Tokenizer changes (step 4)

Tokens per whitespace word, measured on one Tigrinya and one Amharic
sentence:

| Model | vocab | Tigrinya | Amharic | padding |
|---|---|---|---|---|
| gemma-4-E2B / 12B | 262,144 | 3.17 | 1.50 | left |
| t5gemma-2 (both) | 262,144 | 3.17 | 1.50 | right |
| Falcon-H1R-7B | 130,048 | 4.67 | 4.50 | left |
| Qwen3.6-27B | 248,077 | 5.17 | 5.00 | right |
| Ministral-3-8B | 131,072 | 9.00 | 8.50 | right |
| byt5-small | 384 | 10.00 | 9.50 | right |

The Gemma-4 and T5Gemma-2 tokenizers are markedly more efficient on Ge'ez
script than the others — roughly half Qwen's token count and a third of
Ministral's. This is a substantive change from the old roster and will
affect how much context each model needs for the same prompt.

**Ministral tokenizer warning.** Loading it emits:

> The tokenizer you are loading from `mistralai/Ministral-3-8B-Instruct-2512`
> with an incorrect regex pattern ... You should set the
> `fix_mistral_regex=True` flag.

Checked on the Tigrinya sentence: output is **identical** with and without
the flag (54 tokens either way), so Ge'ez tokenization is unaffected. The
flag may still matter for other scripts and should be set if Latin-script
text is added.

No hard-coded token IDs or embedding resizing exist in the codebase. The only
token-id reference is a `pad_token_id or eos_token_id` fallback in
`base_loader.py`, which is tokenizer-agnostic.

## Compute (step 5) — suggested, not applied

The new roster is substantially larger. `submit_evaluation_job.sh` currently
requests **4 GPUs, 32 CPUs, 128 GB, 72 h** and has not been changed.

| Model | Old params | New params | Change |
|---|---|---|---|
| qwen | 7 B | 27 B | ~3.9× |
| gemma-7b | 7 B | 12 B | ~1.7× |
| mistral | 7 B | 8 B | ~1.1× |
| gemma-2b | 2 B | ~2 B (E2B) | ~1× |
| mt5-large | 1.2 B | 1 B + 1 B | ~1.7× |
| mt5-small | 0.3 B | 0.27 B + 0.27 B | ~1.8× |

Suggested changes, for review:

- **`--mem` 128G → 256G.** Qwen3.6-27B alone needs roughly 54 GB in bf16;
  the current 128 GB is tight once host-side loading buffers are counted.
- **`--gres=gpu:4` retained**, but Qwen3.6-27B will shard across several
  GPUs under `device_map="auto"`. If tasks run sequentially this is fine;
  running two large models concurrently is not.
- **`--time=72:00:00` retained.** The last full sweep took 2 d 18 h on the
  old roster; the new one will be slower, and 72 h may become tight. Worth
  splitting per-task if it exceeds the limit.
- **Precision:** `dtype="auto"` gives bf16 for most of the new roster, which
  is both faster and more numerically stable than the previous forced fp16.
- **Generation:** `max_new_tokens=128` is unchanged. The evaluation
  methodology depends on it, so it was not touched.
- **Batching:** generation is still one prompt at a time. This dominated
  runtime on the last sweep (GPU utilisation 13–16%). Batching would help far
  more than any hardware change, but it alters the evaluation path and was
  left alone.

No hyperparameter was changed silently.

## Tables mentioning old model names (step 6)

Left untouched, for your decision:

- `PAPER_DRAFT.md` — §3 model table, §4.1–4.4 result tables, §6 comparison
- `CORRECTIONS.md` — §2 before/after tables, translation table
- `README.md` — model list and correction notes

All report measurements of the **old** checkpoints and remain valid for them.
Re-running the new roster would produce a separate set of numbers that must
not be merged into these tables.

## Verification status

Done on CPU: ID resolution, Auto-class mapping, config inspection, tokenizer
loading and token counts, span-infilling routing, end-to-end generation
through the project loader for `t5gemma-2-270m-270m`.

Pending: full-roster load-and-generate smoke test on GPU (SLURM job 80651,
`logs/smoke_*.out`). The large models (Qwen3.6-27B, Gemma-4-12B) have not yet
been loaded on this machine.
