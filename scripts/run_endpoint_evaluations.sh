#!/usr/bin/env bash
# Run the four evaluation tasks for the KBS endpoint-served models only.
#
# These models generate over HTTP rather than on a local GPU, so this script
# does not need a GPU allocation and does not go through SLURM -- unlike
# scripts/run_all_evaluations.sh, which loads local checkpoints.
#
# Requires KBS_INFERENCE_API_KEY to be exported (the endpoint rejects
# unauthenticated requests). The key is deliberately not stored here.
#
# Run from the repository root; the task scripts use relative paths.
set -euo pipefail

if [[ -z "${KBS_INFERENCE_API_KEY:-}" ]]; then
    echo "KBS_INFERENCE_API_KEY is not set; export it before running." >&2
    exit 1
fi

# Only the endpoint models, so a run here never pulls a local checkpoint.
# deepseek-v4-flash-284b is excluded by default: it measured ~88s per
# single-word request, which is impractical across 5775 rows.
export EVAL_ONLY_MODELS="${EVAL_ONLY_MODELS:-qwen3.8-27b,gpt-oss-120b}"

echo "Models: ${EVAL_ONLY_MODELS}"

for task in translation_fidelity lexical_alignment pos_tagging morphosyntax_probe; do
    echo "=== ${task} ==="
    python "tasks/${task}.py"
done

echo "=== rescoring ==="
python scripts/compute_bleu.py
