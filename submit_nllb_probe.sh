#!/bin/bash
#SBATCH --job-name=nllb_probe
#SBATCH --output=logs/nllb_probe_%j.out
#SBATCH --error=logs/nllb_probe_%j.err
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --hint=nomultithread

# Baseline probe: NLLB-200 (Tigrinya = tir_Ethi is explicitly supported)
# on the same gold translations the general-purpose LLMs were scored on.
# Establishes whether near-zero BLEU is a model ceiling or a task ceiling.

export PYTORCH_ALLOC_CONF=expandable_segments:True
source /homes/neumann/teklehaymanot/Project/LLM-Probe/env/bin/activate
cd /homes/neumann/teklehaymanot/Project/LLM-Probe

python scripts/probe_nllb_translation.py \
  --checkpoint "${NLLB_CKPT:-facebook/nllb-200-distilled-600M}" \
  --n "${NLLB_N:-1000}"
