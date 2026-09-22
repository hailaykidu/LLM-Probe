#!/bin/bash
#SBATCH --job-name=nllb_sweep
#SBATCH --output=logs/nllb_sweep_%j.out
#SBATCH --error=logs/nllb_sweep_%j.err
#SBATCH --partition=ampere
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --hint=nomultithread

# Full-set NLLB-200 sweep (600M / 1.3B / 3.3B) on all gold translation rows,
# to establish the maximum achievable BLEU on this benchmark with a
# Tigrinya-capable MT model. Same scorer as the LLM results.

export PYTORCH_ALLOC_CONF=expandable_segments:True
source /homes/neumann/teklehaymanot/Project/LLM-Probe/env/bin/activate
cd /homes/neumann/teklehaymanot/Project/LLM-Probe

for ckpt in facebook/nllb-200-distilled-600M facebook/nllb-200-1.3B facebook/nllb-200-3.3B; do
  echo "############ $ckpt ############"
  python scripts/probe_nllb_translation.py --checkpoint "$ckpt" --n 0 || echo "FAILED: $ckpt"
done
