import os
import json
import random
import warnings
from flask import Flask, render_template, request, jsonify
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer, util

warnings.filterwarnings("ignore")
app = Flask(__name__)

# === Load Configuration from config.json ===
CONFIG_PATH = "config.json"
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError("Missing config.json file.")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

models = config["models"]
default_model_id = config.get("default_model", next(iter(models)))

# Initial model load
current_model_id = default_model_id
llm = Llama(
    model_path=models[current_model_id],
    n_ctx=2048,
    n_threads=6,
    n_batch=128,
    verbose=False
)

# --- Select bot name ---
bot_name = random.choice(["Alan (AI)", "Alice (AI)", "Alex (AI)"])

# --- Load bank content ---
BANK_CONTENT_PATH = "bank_content.json"
if os.path.exists(BANK_CONTENT_PATH):
    with open(BANK_CONTENT_PATH, "r", encoding="utf-8") as f:
        bank_sections = json.load(f)
else:
    bank_sections = {"General": "No bank content provided."}

# --- Load SentenceTransformer ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- Semantic Section Search ---
def get_best_section(user_query: str, sections: dict, threshold: float = 0.3) -> (str, str):
    # Direct match with a known topic (exact or partial)
    for header in sections:
        if user_query.strip().lower() == header.lower():
            return header, sections[header]

    # Otherwise, fall back to semantic similarity
    query_embedding = embedder.encode(user_query, convert_to_tensor=True)
    best_score = -1
    best_header = None
    best_content = None

    for header, content in sections.items():
        section_text = f"{header}. {content}"
        section_embedding = embedder.encode(section_text, convert_to_tensor=True)
        score = util.pytorch_cos_sim(query_embedding, section_embedding).item()
        if score > best_score:
            best_score = score
            best_header = header
            best_content = content

    return (None, user_query) if best_score < threshold else (best_header, best_content)


# --- Prompt Generation ---
# --- Prompt Generation ---
def generate_final_answer(best_header: str, best_content: str, user_name: str, user_query: str) -> str:
    system_prompt = (
        "[System]: You are a helpful bank helpdesk assistant. "
        "You are only allowed to answer questions that are strictly related to banking topics "
        "such as those found in official customer service guides. "
        "If a question is unrelated to banking or outside your domain, politely respond that you can only help with bank-related topics.\n\n"
    )

    if best_header and best_header.lower() in user_query.lower():
        return (
            f"Thank you, {user_name}. Here's what you need to know about {best_header}:\n\n"
            f"{best_content}\n\nLet me know if you'd like help with another topic."
        )

    if best_header is None:
        prompt = (
            system_prompt +
            f"User: Thank you, {user_name}, for your question: {user_query}\n"
            "Assistant:"
        )
    else:
        prompt = (
            system_prompt +
            f"User: Thank you, {user_name}, for your question.\n"
            f"Section: {best_header}\n\n{best_content}\n\n"
            "Assistant:"
        )

    response = llm(prompt, max_tokens=200, temperature=0.5, top_p=0.95, stop=["User:", "Assistant:", "Section:", "[System]:"])
    return response["choices"][0]["text"].strip()



# --- Flask Routes ---
@app.route("/")
def index():
    bank_topics = sorted(bank_sections.keys())
    return render_template("index.html", bot_name=bot_name, bank_topics=bank_topics,
                           models=models.keys(), default_model=default_model_id)

@app.route("/chat", methods=["POST"])
def chat():
    global llm, current_model_id

    data = request.get_json()
    user_name = data.get("user_name", "User")
    user_input = data.get("user_input", "").strip()
    model_id = data.get("model", default_model_id)

    # Change model if needed
    if model_id != current_model_id and model_id in models:
        print(f"Switching to model: {model_id}")
        llm = Llama(
            model_path=models[model_id],
            n_ctx=2048,
            n_threads=6,
            n_batch=128,
            verbose=False
        )
        current_model_id = model_id

    GREETING_PHRASES = {"hi", "hello", "hey", "help", "can you help me", "i need help"}

    if user_input.lower() in GREETING_PHRASES:
        response_text = f"Hello {user_name}, how can I assist you today?"
    else:
        # Check for exact bank topic pattern: "I want to know more about {topic}"
        prefix = "i want to know more about "
        lowered_input = user_input.lower()
        if lowered_input.startswith(prefix):
            topic_key = user_input[len(prefix):].strip().title()
            matched = [k for k in bank_sections.keys() if k.lower() == topic_key.lower()]
            if matched:
                response_text = f"Thank you, {user_name}. Here's some helpful info about {matched[0]}:\n\n{bank_sections[matched[0]]}"
                return jsonify({"response": response_text})

        # Fall back to semantic search + LLM
        best_header, best_content = get_best_section(user_input, bank_sections)
        response_text = generate_final_answer(best_header, best_content, user_name, user_input)

    return jsonify({"response": response_text})

# --- Run the app ---
if __name__ == "__main__":
    app.run(debug=True)
