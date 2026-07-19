from .base_loader import load_model as base_load_model

def load_model():
    return base_load_model("google/gemma-2b-it", task="text-generation")
