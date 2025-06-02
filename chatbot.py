import json
from llama_cpp import Llama

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load configuration
with open("config.json", "r") as f:
    config = json.load(f)

models = config["models"]
default_model_id = config.get("default_model")

# Let user choose model (or use default)
print("Available models:")
for i, model_id in enumerate(models.keys(), 1):
    print(f"  [{i}] {model_id}")

choice = input(f"Select a model [default={default_model_id}]: ").strip()
if choice.isdigit() and 1 <= int(choice) <= len(models):
    selected_model_id = list(models.keys())[int(choice) - 1]
else:
    selected_model_id = default_model_id

model_path = models[selected_model_id]
print(f"Loading model: {selected_model_id}")

# Load GGUF model
llm = Llama(
    model_path=model_path,
    n_ctx=4096,
    n_threads=6,
    n_batch=128,
    n_gpu_layers=32,
    verbose=False
)

# System prompt and history
system_prompt = "You are a helpful and concise AI assistant."
history = [{"role": "system", "content": system_prompt}]

print("Chatbot ready. Type 'exit' to quit.")
while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "exit":
        break

    history.append({"role": "user", "content": user_input})
    messages = [history[0]] + history[-8:]

    prompt = ""
    for m in messages:
        if m["role"] == "system":
            prompt += f"[System]: {m['content']}\n"
        elif m["role"] == "user":
            prompt += f"User: {m['content']}\n"
        elif m["role"] == "assistant":
            prompt += f"Assistant: {m['content']}\n"
    prompt += "Assistant:"

    response = llm(prompt, max_tokens=200, stop=["User:", "Assistant:", "[System]:"], temperature=0.7, top_p=0.95)
    reply = response["choices"][0]["text"].strip()

    print("Assistant:", reply)
    history.append({"role": "assistant", "content": reply})
