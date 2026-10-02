#!/bin/bash
#SBATCH --job-name=roster2026_eval
#SBATCH --output=logs/roster2026_%j.out
#SBATCH --error=logs/roster2026_%j.err
#SBATCH --partition=ampere
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --hint=nomultithread

# Full evaluation on the 2026 model roster (gemma-4, Ministral-3, Falcon-H1R,
# Qwen3.6-27B, t5gemma-2, byt5). Results land in results/evaluation_reports/
# under the NEW aliases, so they cannot collide with the old-roster files that
# the paper's tables report.
#
# --mem raised 128G -> 256G: Qwen3.6-27B alone needs ~54GB in bf16.

export PYTORCH_ALLOC_CONF=expandable_segments:True
source /homes/neumann/teklehaymanot/Project/LLM-Probe/env/bin/activate
cd /homes/neumann/teklehaymanot/Project/LLM-Probe

echo "### smoke test first ###"
python scripts/smoke_test_models.py || echo "smoke test reported failures; continuing"

for t in pos_tagging morphosyntax_probe translation_fidelity lexical_alignment; do
  echo "############ $t ############"
  python tasks/$t.py || echo "FAILED: $t"
done
