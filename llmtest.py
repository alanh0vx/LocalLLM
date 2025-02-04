import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "distilgpt2"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Test generation
prompt = "Hello, how are you today?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate text
outputs = model.generate(
    **inputs,
    max_length=50,
    do_sample=True,
    temperature=0.7,
)
print("Generated Text:\n", tokenizer.decode(outputs[0], skip_special_tokens=True))
