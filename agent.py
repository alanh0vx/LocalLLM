import os
import requests
from smolagents import OpenAIServerModel, CodeAgent, DuckDuckGoSearchTool, tool

LM_STUDIO_API_BASE = "http://localhost:1234"

# --- Step 1: Fetch loaded models ---
def get_loaded_models():
    try:
        res = requests.get(f"{LM_STUDIO_API_BASE}/api/v0/models")
        res.raise_for_status()
        models = res.json()["data"]
        loaded_llms = [m for m in models if m["state"] == "loaded" and m["type"] == "llm"]
        return loaded_llms
    except Exception as e:
        print(f"[Error] Failed to fetch models from LM Studio: {e}")
        return []

loaded_models = get_loaded_models()
if not loaded_models:
    print("No LLM models are currently loaded in LM Studio.")
    exit(1)

# --- Step 2: Ask user to pick a model ---
print("Available loaded models:")
for i, m in enumerate(loaded_models, 1):
    print(f"  [{i}] {m['id']}")

try:
    choice = int(input("Select a model [default=1]: ").strip() or "1")
    selected_model_id = loaded_models[choice - 1]["id"]
except (ValueError, IndexError):
    print("Invalid selection.")
    exit(1)

print(f"Using model: {selected_model_id}")

# --- Step 3: Create agent with chosen model ---
model = OpenAIServerModel(
    model_id=selected_model_id,
    api_base=f"{LM_STUDIO_API_BASE}/v1",
    api_key="n/a",
)

system_prompt = (
    "You are a helpful AI assistant with access to tools like web search. "
    "Always try to complete the user's task step by step. If unsure, use available tools. "
    "Keep the final answer precise, format it inside a answer box.\n"
)

agent = CodeAgent(
    model=model,
    tools=[DuckDuckGoSearchTool()]
)

# --- Step 4: Interactive loop ---
print("\n[Local Agent] Type your question. Type 'exit' to quit.")
while True:
    try:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response = agent.run(system_prompt + '\n' + user_input)
       
        print(f"Agent: {response}")
    except KeyboardInterrupt:
        print("\nSession ended.")
        break
    except Exception as e:
        print(f"[Error] {e}")
