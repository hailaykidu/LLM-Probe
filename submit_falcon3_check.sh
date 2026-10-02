#!/bin/bash
#SBATCH --job-name=falcon3_check
#SBATCH --output=logs/falcon3_check_%j.out
#SBATCH --error=logs/falcon3_check_%j.err
#SBATCH --partition=ampere
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --hint=nomultithread

# Verifies tiiuae/Falcon3-10B-Instruct loads and generates under dtype="auto"
# (bf16). Job 77595 failed here with a CUDA device-side assert under forced
# float16; this confirms the fix before committing GPU time to a full sweep.

export PYTORCH_ALLOC_CONF=expandable_segments:True
source /homes/neumann/teklehaymanot/Project/LLM-Probe/env/bin/activate
cd /homes/neumann/teklehaymanot/Project/LLM-Probe

python - <<'PYEOF'
import sys; sys.path.insert(0,'.')
import torch
from models.falcon3_loader import load_model
from scripts.smoke_test_models import extract_text
TI="ኣብ ገዛ ዝነብር ቆልዓ ንእሽቶይ እዩ"
pipe=load_model()
print("dtype:", next(pipe.model.parameters()).dtype)
for prompt in [f"Identify the part of speech.\nPhrase: ገዛ (house)\nOutput:",
               f"Translate the following English phrase into Tigrigna.\nEnglish: house\nOutput:"]:
    out=pipe([prompt])
    # The HF text-generation pipeline batches a list of prompts and returns
    # list[list[{...}]]; out[0]["generated_text"] indexed the inner list with
    # a string and raised TypeError (job 81327). extract_text handles both
    # that shape and the flat one Seq2SeqPipeline/endpoint loaders return.
    print("OK ->", repr(extract_text(out).strip()[:70]))
tk=pipe.tokenizer
print(f"tokens/word Tigrinya: {len(tk(TI)['input_ids'])/len(TI.split()):.2f}")
PYEOF
