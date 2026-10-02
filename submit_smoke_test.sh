#!/bin/bash
#SBATCH --job-name=smoke_models
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err
#SBATCH --partition=ampere
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=03:00:00
#SBATCH --hint=nomultithread

export PYTORCH_ALLOC_CONF=expandable_segments:True
source /homes/neumann/teklehaymanot/Project/LLM-Probe/env/bin/activate
cd /homes/neumann/teklehaymanot/Project/LLM-Probe
python scripts/smoke_test_models.py
