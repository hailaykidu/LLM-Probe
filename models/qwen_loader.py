from .base_loader import load_model as base_load_model

def load_model():
    model_id = "Qwen/Qwen1.5-7B-Chat"
    return base_load_model(model_id, task="text-generation")
