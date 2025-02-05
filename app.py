import os
import json
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
import warnings

from flask import Flask, render_template, request, jsonify

warnings.filterwarnings("ignore")

app = Flask(__name__)

# === Load Configuration from config.json ===
CONFIG_PATH = "config.json"
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
else:
    raise FileNotFoundError("Missing config.json file. Please create one based on config.example.json.")

MODEL_ID = config["MODEL_ID"]
HF_TOKEN = config["HF_TOKEN"]
CACHE_DIR = config["CACHE_DIR"]

# --- Select a bot name randomly ---
bot_name = random.choice(["Alan (AI)", "Alice (AI)", "Alex (AI)"])

# --- Load meta-llama model and tokenizer ---
print("Loading meta-llama model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    cache_dir=CACHE_DIR
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    cache_dir=CACHE_DIR,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("Meta-llama loaded.")

# --- Load Bank Content from JSON ---
BANK_CONTENT_PATH = "bank_content.json"
if os.path.exists(BANK_CONTENT_PATH):
    with open(BANK_CONTENT_PATH, "r", encoding="utf-8") as f:
        bank_sections = json.load(f)
else:
    bank_sections = {
        "General": "No bank content provided. (Please add your bank-related guidelines here.)"
    }
print("Bank content loaded from JSON.")

# --- Load SentenceTransformer for embeddings ---
print("Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("SentenceTransformer loaded.")


def get_best_section(user_query: str, sections: dict, threshold: float = 0.3) -> (str, str):
    """
    Computes embeddings for the user query and each bank section,
    then returns the section header and content with the highest cosine similarity.
    If the best similarity score is below the threshold, it returns None as the header
    and the original user_query as the content.
    """
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

    if best_score < threshold:
        return None, user_query
    return best_header, best_content


def generate_final_answer(best_header: str, best_content: str, user_name: str, user_query: str) -> str:
    """
    Uses the meta-llama model to generate a final, friendly answer.
    If a relevant bank section was found (best_header is not None), the prompt thanks the user,
    outputs the selected section in a fixed format, and invites further questions.
    Otherwise, the prompt uses the user's original query directly.
    """
    if best_header is None:
        prompt = (
            f"Thank you, {user_name}, for your question: {user_query}\n"
            "Please let me know if you have any further questions."
        )
    else:
        prompt = (
            f"Thank you, {user_name}, for your question.\n"
            f"Section: {best_header}\n\n{best_content}\n\n"
            "Please let me know if you have any further questions."
        )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    extra_tokens = 100
    input_length = inputs.input_ids.shape[1]
    max_length = input_length + extra_tokens

    outputs = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,
        temperature=0.5,
        no_repeat_ngram_size=3,
        repetition_penalty=1.3,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text.strip()


@app.route("/")
def index():
    # Pass the sorted bank topics to the template for selection
    bank_topics = sorted([
        "credit", "card", "mortgage", "saving", "loan", "bank", "deposit", "withdraw",
        "interest", "finance", "investment", "branch", "location", "address", "atm",
        "online banking", "account opening", "loan application"
    ])
    return render_template("index.html", bot_name=bot_name, bank_topics=bank_topics)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_name = data.get("user_name", "User")
    user_input = data.get("user_input", "")

    GREETING_PHRASES = {"hi", "hello", "hey", "help", "can you help me", "i need help"}
    BANK_KEYWORDS = {
        "credit", "card", "mortgage", "saving", "loan", "bank", "deposit", "withdraw",
        "interest", "finance", "investment", "branch", "location", "address", "atm",
        "online banking", "account opening", "loan application"
    }

    # If the input is a greeting
    if user_input.lower() in GREETING_PHRASES:
        response_text = f"Hello {user_name}! How can I assist you today?"
    # If the query does not contain any bank-related keywords
    elif not any(keyword in user_input.lower() for keyword in BANK_KEYWORDS):
        response_text = "I'm sorry, I can only answer bank-related questions."
    else:
        best_header, best_content = get_best_section(user_input, bank_sections)
        response_text = generate_final_answer(best_header, best_content, user_name, user_input)

    return jsonify({"response": response_text})


if __name__ == "__main__":
    app.run(debug=True)
