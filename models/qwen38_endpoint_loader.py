from .endpoint_loader import load_model as base_load_model
from .endpoint_models import ENDPOINT_MODELS


def load_model():
    return base_load_model(ENDPOINT_MODELS["qwen3.8-27b"])
