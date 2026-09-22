#!/bin/bash
#SBATCH --job-name=llm_eval_xlmr
#SBATCH --output=logs/eval_xlmr_%j.out
#SBATCH --error=logs/eval_xlmr_%j.err
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1                   # xlm-roberta-base (~125M params) fits on a single GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --hint=nomultithread

# Follow-up run for xlm-roberta-base only, across all four evaluation tasks.
#
# Why this exists: the main run (submit_evaluation_job.sh, job 75167) skipped
# xlm-roberta-base in every task with
#   "Failed to load model 'xlm-roberta-base': list index out of range"
# -- a transformers 5.17.0 bug in infer_auto_device_map() that crashes on
# tied embedding/decoder weights when device_map="auto" splits the model
# across multiple GPUs. Fixed in models/base_loader.py by loading masked-LM
# models onto a single GPU directly instead of "auto"-sharding them (see the
# comment there). Since the main run already completed/is completing every
# other model, this follow-up uses EVAL_ONLY_MODELS to evaluate xlm-roberta
# alone -- each task script skips any (English, Tigrigna) row already present
# in its per-model results file, but since xlm-roberta has no prior results,
# this simply runs it fresh across the full dataset for all four tasks.
export EVAL_ONLY_MODELS=xlm-roberta

export PYTORCH_ALLOC_CONF=expandable_segments:True

source /homes/neumann/teklehaymanot/Project/LLM-Probe/env/bin/activate

cd /homes/neumann/teklehaymanot/Project/LLM-Probe

if [ ! -f "scripts/run_all_evaluations.sh" ]; then
  echo "❌ Missing run_all_evaluations.sh script. Aborting."
  exit 1
fi

./scripts/run_all_evaluations.sh
