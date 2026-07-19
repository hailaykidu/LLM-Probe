from .base_loader import load_model as base_load_model

def load_model():
    return base_load_model("swiss-ai/Apertus-8B-2509", task="text-generation")
