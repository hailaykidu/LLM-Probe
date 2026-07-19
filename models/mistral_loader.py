from .base_loader import load_model as base_load_model

def load_model():
    return base_load_model("mistralai/Mistral-7B-Instruct-v0.2", task="text-generation")
