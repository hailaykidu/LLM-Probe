from .base_loader import load_model as base_load_model

def load_model():
    return base_load_model("mistralai/Ministral-3-8B-Instruct-2512", task="text-generation")
