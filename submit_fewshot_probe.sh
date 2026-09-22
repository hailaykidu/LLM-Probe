#!/bin/bash
#SBATCH --job-name=fewshot_probe
#SBATCH --output=logs/fewshot_probe_%j.out
#SBATCH --error=logs/fewshot_probe_%j.err
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --hint=nomultithread

# A/B probe: one-shot vs few-shot translation prompting on a 300-row sample.
# Diagnostic only -- does not write canonical task results.

export PYTORCH_ALLOC_CONF=expandable_segments:True
source /homes/neumann/teklehaymanot/Project/LLM-Probe/env/bin/activate
cd /homes/neumann/teklehaymanot/Project/LLM-Probe

python scripts/probe_fewshot_translation.py --model "${PROBE_MODEL:-gemma-7b}" --n "${PROBE_N:-300}"
