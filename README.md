# Chatbot and AI Agent with Local LLMs 

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
pip install llama-cpp-python flask
```

For GPU support with `llama-cpp-python`, see [llama-cpp-python build instructions](https://github.com/abetlen/llama-cpp-python#installation).

---

## Models (GGUF Format)

This project uses **quantized local models** in `.gguf` format.

The models can be downloaded within  [lm studio](https://lmstudio.ai/)

![image](https://github.com/user-attachments/assets/b128be06-d35d-4d00-961b-98d24e4b4788)

![image](https://github.com/user-attachments/assets/84381a28-abac-43fa-ae47-d541fffaae90)



### Example Models:

| Model ID                 | Path                                                                   |
|--------------------------|------------------------------------------------------------------------|
| `gemma-3-4b-it`          | `D:/development/llm-models/.../gemma-3-4b-it-Q4_K_M.gguf`              |
| `gemma-3-27b-it`          | `D:/development/llm-models/.../gemma-3-27B-it-QAT-Q4_0.gguf`              |
| `deepseek-r1-qwen3-8b`   | `D:/development/llm-models/.../DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf`  |
| `Meta-Llama-3.1-8B`      | `D:/development/llm-models/.../Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` |

Define these in `config.json` like this: (rename `config.json.sample` to `config.json`)

```json
{
  "models": {
    "gemma-3-4b-it": "D:/development/llm-models/lmstudio-community/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_K_M.gguf",
    "gemma-3-27b-it": "D:/development/llm-models/lmstudio-community/gemma-3-27B-it-qat-GGUF/gemma-3-27B-it-QAT-Q4_0.gguf",
    "deepseek-r1-qwen3-8b": "D:/development/llm-models/lmstudio-community/DeepSeek-R1-0528-Qwen3-8B-GGUF/DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf",
    "Meta-Llama-3.1-8B": "D:/development/llm-models/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
  },
  "default_model": "gemma-3-27b-it"
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
| `agent.py`          | AI agent to use tools                        |

---

## Local AI Agent with LM Studio API
This project supports running fully local AI agents using LM Studio, which serves quantized GGUF models through a local OpenAI-compatible API (http://127.0.0.1:1234/v1). 


In `agent.py`, we use the SmolAgents framework with OpenAIServerModel to connect to LM Studio, allowing users to interact with tools like DuckDuckGoSearchTool and custom logic via a conversational interface. On startup, the script fetches all currently loaded models from the LM Studio API and lets the user select one for inference. The selected model is then used by a CodeAgent to reason and act using both LLM reasoning and real-time tool usage.

![image](https://github.com/user-attachments/assets/4204d871-031e-4a0c-be9b-5eb648fe0f3b)

![image](https://github.com/user-attachments/assets/bd62627b-2e84-4aed-8929-dfc94310be2c)


