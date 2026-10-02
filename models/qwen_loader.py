from .base_loader import load_model as base_load_model

def load_model():
    model_id = "Qwen/Qwen3.6-27B"
    return base_load_model(model_id, task="text-generation")
