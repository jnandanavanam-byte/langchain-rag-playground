# 🤖 LangChain RAG Playground

A Dual-Pipeline Retrieval-Augmented Generation (RAG) Benchmarking Platform built with:

- LangChain
- Streamlit
- Ollama
- FAISS
- Chroma
- HuggingFace Embeddings

Compare different chunking strategies, embedding models, vector stores, and LLMs using the same document set and the same question.

---

# 🎯 Why This Project?

Most RAG applications provide a single retrieval pipeline:

```text
Documents
    ↓
Chunking
    ↓
Embedding
    ↓
Vector Store
    ↓
LLM
```

This project allows you to compare two independent RAG pipelines side-by-side.

Example:

Pipeline 1

- Recursive Chunking
- MiniLM
- FAISS
- Llama 3.2

Pipeline 2

- Token Chunking
- BGE Small
- Chroma
- Mistral

The same question is executed against both pipelines to compare:

- Retrieval quality
- Source grounding
- Chunking effectiveness
- Embedding effectiveness
- Response performance

---

# 🚀 Features

## 📄 Upload Once, Compare Twice

Upload documents once and reuse them across both pipelines.

Supported file formats:

- PDF
- DOCX
- XLSX
- CSV

---

## 🔀 Dual Pipeline Benchmarking

Configure two completely independent RAG pipelines.

### Pipeline 1

Select:

- Chunking Strategy
- Embedding Model
- Vector Store
- LLM

### Pipeline 2

Select:

- Chunking Strategy
- Embedding Model
- Vector Store
- LLM

---

## ✂️ Chunking Strategies

Supported chunkers:

- Recursive Character
- Character Splitter
- Token Splitter
- Markdown Splitter

---

## 🧠 Embedding Models

Supported embeddings:

- MiniLM
- BGE Small
- E5 Small

---

## 🗃 Vector Databases

Supported vector stores:

### FAISS

High-performance vector similarity search.

### Chroma

Persistent vector database.

---

## 🤖 LLM Support

Using local Ollama models:

- Llama 3.2
- Mistral
- Phi-3

---

## ⚡ CUDA Auto Detection

The application automatically detects GPU availability.

```text
GPU Available
    ↓
CUDA Enabled

GPU Not Available
    ↓
CPU Mode
```

Example:

```text
🚀 CUDA Enabled
NVIDIA RTX A5000 Laptop GPU
```

---

## 📊 Benchmark Dashboard

Compare:

- Chunks Generated
- Vectors Created
- Average Chunk Size
- Retrieved Chunks
- Source Count
- Answer Length
- Response Time

---

## 🔎 Side-by-Side Answer Comparison

Ask one question:

```text
What is the difference between
a Purchase Order and
a Purchase Requisition?
```

Compare both responses:

### Pipeline 1

Answer

Sources

### Pipeline 2

Answer

Sources

---

# 🏗 Architecture

```text
Documents
    ↓
Loaders
    ↓
Chunkers
    ↓
Embeddings
    ↓
Vector Store
    ↓
Retriever
    ↓
Prompt
    ↓
LLM
    ↓
Response
```

---

# 📂 Project Structure

```text
langchain-rag-playground/

├── app/
│   ├── config.py
│   ├── chunkers.py
│   ├── embeddings.py
│   ├── loaders.py
│   ├── llms.py
│   ├── pipeline.py
│   ├── rag.py
│   ├── vectordb.py
│   ├── cleanup.py
│   │
│   └── vectorstores/
│       ├── faiss_store.py
│       ├── chroma_store.py
│       └── __init__.py
│
├── ui/
│   └── streamlit_app.py
│
├── run.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Installation

## Clone the Repository

```bash
git clone https://github.com/<your-github-account>/langchain-rag-playground.git

cd langchain-rag-playground
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv .venv

source .venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🦙 Ollama Setup

Install Ollama:

https://ollama.com

Pull required models:

```bash
ollama pull llama3.2

ollama pull mistral

ollama pull phi3
```

Start Ollama:

```bash
ollama serve
```

---

# ▶️ Run the Application

```bash
streamlit run run.py
```

---

# 🧪 Example Benchmark

## Pipeline 1

```text
Chunker      : Recursive Character
Embedding    : MiniLM
Vector Store : FAISS
LLM          : Llama 3.2
```

## Pipeline 2

```text
Chunker      : Token Splitter
Embedding    : BGE Small
Vector Store : Chroma
LLM          : Mistral
```

Question:

```text
What is the difference between
Purchase Requisition and
Purchase Order?
```

Compare:

- Retrieval Quality
- Answer Completeness
- Source Documents
- Response Time

---

# 📸 Screenshots



# 📸 Screenshots

Visual walkthroughs and benchmark examples will be added in a future release.

# 🛣 Roadmap

## Version 1.1

- 🎤 Voice Prompts (Whisper)
- 🔊 Voice Responses (Edge-TTS)
- 📑 Response History
- ⏹ Stop Build Processing
- 🔄 Reset Environment

---

## Version 1.2

- 📈 Radar Charts
- 🏆 Pipeline Ranking
- 📊 Advanced Benchmark Analytics
- 📤 Export Results to Excel

---

## Version 1.3

- Multi-Pipeline Benchmarking (3+ Pipelines)
- Streaming Responses
- Remote LLM Support
- RAG Evaluation Metrics

---

# 🤝 Contributing

Contributions, suggestions, and benchmarking ideas are welcome.

Feel free to submit:

- Issues
- Feature Requests
- Pull Requests

## Docker Support

Docker deployment is planned for a future release.

Current deployment:

streamlit run run.py

---

# 📄 License

MIT License

---

# ⭐ Support

If you found this project useful:

- Star the repository
- Share your benchmark results
- Contribute new chunkers, embeddings, vector stores, or evaluation metrics

Happy Benchmarking! 🚀