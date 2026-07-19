#!/bin/bash
#SBATCH --job-name=llm_eval_tigrinya
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --partition=ampere             # ✅ Use the ampere GPU partition
#SBATCH --gres=gpu:4                   # Request 4 GPUs for parallel model loading
#SBATCH --cpus-per-task=32             # Use 32 CPU cores for faster preprocessing
#SBATCH --mem=128G                     # Allocate 128 GB RAM for large model support
#SBATCH --time=272:00:00                # Set time limit to 72 hours
#SBATCH --hint=nomultithread           # Optional: disable hyperthreading

# ✅ Prevent CUDA memory fragmentation
export PYTORCH_ALLOC_CONF=expandable_segments:True  # Updated from deprecated PYTORCH_CUDA_ALLOC_CONF

# Activate your virtual environment
source /homes/neumann/teklehaymanot/TigrinyaTokenizer/MPETokenization/Paralleldata/MoVoC/LLM_Probe/env/bin/activate

# Navigate to project root
cd /homes/neumann/teklehaymanot/TigrinyaTokenizer/MPETokenization/Paralleldata/MoVoC/LLM_Probe

# Optional: check if required files exist before running
if [ ! -f "scripts/run_all_evaluations.sh" ]; then
  echo "❌ Missing run_all_evaluations.sh script. Aborting."
  exit 1
fi

# Run all evaluation scripts
./scripts/run_all_evaluations.sh
