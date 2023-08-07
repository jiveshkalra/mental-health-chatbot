
# Use a pipeline as a high-level helper

from transformers import pipeline

pipe = pipeline("text-generation", model="meta-llama/Llama-2-7b-hf",use_auth_token=True)
print(pipe("Hey"))
