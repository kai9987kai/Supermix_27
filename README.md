## About the Model (Supermix v27)

`Supermix_27` includes a **trained PyTorch chat model checkpoint** used by the local Python runtime:

- **Checkpoint path:** `runtime_python/champion_model_chat_supermix_v27_500k_ft.pth`
- **Runtime type:** PyTorch (`.pth`)
- **Primary execution path:** local Python runtime (`runtime_python/`)
- **Python web runtime:** `runtime_python/chat_web_app.py` (uses the real checkpoint)
- **Static web version:** `web_static/` is a browser UI that uses metadata retrieval and **does not run the PyTorch model directly**

### What the model is for
This checkpoint is packaged as the **runtime chat model** for the project’s local Python interfaces (terminal/web).  
The repository also includes the supporting training/build pipeline (`source/`) and retained datasets/databases used by the overall system.

### Naming breakdown (checkpoint filename)
`champion_model_chat_supermix_v27_500k_ft.pth`

Likely meaning (based on naming convention):
- `champion_model` → selected/best exported checkpoint
- `chat` → chat-oriented runtime/inference behavior
- `supermix` → project/model family name
- `v27` → version 27
- `500k_ft` → likely a fine-tuned checkpoint associated with a `500k` training step stage (inference from filename; confirm with metadata/training logs)

### Runtime architecture status (important)
The repository page confirms the checkpoint exists and is used by the Python runtime, but the public repo landing page does **not** expose enough detail to verify:

- exact architecture class (e.g. Transformer variant / custom module graph)
- parameter count
- hidden size / number of layers / heads
- tokenizer type / vocab size
- context window
- training objective(s)
- precision (fp32/fp16/bf16/int8)
- device requirements (CPU-only vs CUDA support)
- benchmark scores
- safety filtering / moderation behavior

If these details exist in model metadata files inside `runtime_python/`, they should be documented here as a formal model card.

---

## Model Card (Recommended fields to document)

### 1) Architecture
Document:
- model class name
- layer count
- hidden size
- attention heads
- FFN size
- positional encoding / rope / alibi (if used)
- parameter count (total + trainable)

### 2) Tokenization
Document:
- tokenizer library/type
- vocab size
- special tokens
- max sequence length / context window

### 3) Training / Fine-Tuning
Document:
- base model (if any)
- training datasets used (high-level summary)
- fine-tuning datasets (`datasets/*.jsonl` lineage)
- training step counts / epochs
- optimizer / LR schedule (if tracked)
- checkpoint selection criteria (“champion” meaning)

### 4) Inference Runtime
Document:
- required Python version
- PyTorch version
- CPU/GPU support
- RAM/VRAM needs
- expected latency (CPU/GPU)
- batching support (if any)

### 5) Capabilities
Document:
- intended tasks (chat, Q&A, retrieval-assisted chat, metadata-backed UI)
- strengths
- known failure modes
- unsupported use cases

### 6) Safety / Limitations
Document:
- hallucination risks
- prompt injection risks (if retrieval is used)
- dataset bias limitations
- privacy notes for SQLite/chat databases

---

## Static Web vs Real Model (for users)
To avoid confusion:

- **`runtime_python/` = real neural inference** using `champion_model_chat_supermix_v27_500k_ft.pth`
- **`web_static/` = static UI** (GitHub Pages compatible), using metadata retrieval in JavaScript
- **GitHub Pages cannot run the `.pth` checkpoint** without model conversion + browser runtime porting

---

## Suggested next improvement (high value)
Add a `runtime_python/MODEL_CARD.md` (or extend this README) with:
- architecture summary
- parameter count
- tokenizer details
- training/fine-tune provenance
- inference requirements
- example prompts + outputs
