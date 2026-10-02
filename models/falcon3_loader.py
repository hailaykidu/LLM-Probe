from .base_loader import load_model as base_load_model

# tiiuae/Falcon3-10B-Instruct publishes bfloat16 weights. An earlier attempt
# (SLURM job 77595) failed on all four tasks with "CUDA error: device-side
# assert triggered / probability tensor contains either inf, nan or element
# < 0" because base_loader forced float16 on every causal model, overflowing
# the bf16 logits. base_loader now passes dtype="auto", which honours the
# checkpoint's own precision.

def load_model():
    return base_load_model("tiiuae/Falcon3-10B-Instruct", task="text-generation")
