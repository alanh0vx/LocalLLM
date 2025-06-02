# Bank Chatbot with Local LLMs (GGUF + llama.cpp + Flask)

This project is a **privacy-preserving AI chatbot** that runs **entirely on your local machine**, powered by **GGUF-format LLMs** via `llama-cpp-python`. It includes:

- `chatbot.py` — Console LLM chat with multi-model support  
- `app.py` — Web-based chatbot interface (Flask) for LLM injection and banking support  
- `index.html` — Frontend styled with Bootstrap

---

## Setup Instructions

### 1. Clone This Repository

```bash
git clone https://github.com/alanh0vx/LocalLLM
cd LocalLLM
```

### 2. Install Python Dependencies

```bash
pip install llama-cpp-python flask sentence-transformers
```

For GPU support with `llama-cpp-python`, see [llama-cpp-python build instructions](https://github.com/abetlen/llama-cpp-python#installation).

---

## Models (GGUF Format)

This project uses **quantized local models** in `.gguf` format.

### Example Models:

| Model ID                 | Path                                                                 |
|--------------------------|----------------------------------------------------------------------|
| `gemma-3-4b-it`          | `D:/development/llm-models/.../gemma-3-4b-it-Q4_K_M.gguf`             |
| `deepseek-r1-qwen3-8b`   | `D:/development/llm-models/.../DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf` |

Define these in `config.json` like this:

```json
{
  "models": {
    "gemma-3-4b-it": "D:/path/to/gemma-3-4b-it-Q4_K_M.gguf",
    "deepseek-r1-qwen3-8b": "D:/path/to/DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf"
  },
  "default_model": "gemma-3-4b-it"
}
```

---

## How to Run

### 1. Basic Console Chat (`chatbot.py`)

```bash
python chatbot.py
```

- Prompts you to select a model from `config.json`
- Multi-turn memory
- Lightweight local chatbot for testing

---

### 2. Web Chat Interface (`app.py`)

```bash
python app.py
```

- Opens a web UI at [http://localhost:5000](http://localhost:5000)
- Includes topic dropdown, name input, and model selection
- Retrieves banking-related info from `bank_content.json`
- Supports switching between LLMs dynamically
- Useful for testing LLM injection and prompt control

---

## Security Testing: LLM Injection

This project can be used to simulate and test:

- Prompt injection attacks
- Hallucination and output control
- OWASP Top 10 for LLM Applications (LLM01–LLM10)

![image](https://github.com/user-attachments/assets/aeea56a6-6419-42eb-8231-17eac490c053)


---

## Files Overview

| File                | Purpose                                      |
|---------------------|----------------------------------------------|
| `chatbot.py`        | Basic terminal LLM chatbot                   |
| `app.py`            | Flask web app with banking support and LLM testing |
| `index.html`        | Chat frontend (Bootstrap UI)                 |
| `config.json`       | LLM configuration (GGUF paths and default)   |
| `bank_content.json` | Banking domain knowledge base                |

---
