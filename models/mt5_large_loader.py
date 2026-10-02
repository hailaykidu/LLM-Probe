from .base_loader import load_model as base_load_model

def load_model():
    return base_load_model("google/t5gemma-2-1b-1b")
