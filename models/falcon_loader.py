from .base_loader import load_model as base_load_model

def load_model():
   return base_load_model("tiiuae/falcon-7b-instruct", task="text-generation")
