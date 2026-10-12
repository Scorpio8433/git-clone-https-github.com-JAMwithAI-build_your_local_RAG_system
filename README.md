# JAMwithAI Local RAG System

A local retrieval-augmented generation (RAG) project for asking questions about your own documents using a local language model. This repo is built as a simple Python starter for creating a private, local AI knowledge assistant.

## What this project does

This app:
- reads documents from a local folder
- splits files into manageable text chunks
- converts chunks into embeddings using sentence-transformers
- stores those embeddings in a FAISS vector database
- retrieves the most relevant document chunks for a user question
- sends the retrieved context to a local Ollama model
- returns a grounded answer based on your local files

This makes it useful for private knowledge search, project documentation Q&A, and local experimentation without sending data to a cloud AI provider.

## Why this repo exists

The goal of this project is to make a simple, understandable local RAG system that can run entirely on a developer machine. It is intended as a learning project and a solid starting point for building a larger document assistant or private knowledge chatbot.

## Features

- Local-first architecture
- Document indexing from local `data/` files
- Chunking and embedding pipeline
- FAISS-based semantic retrieval
- Local LLM generation with Ollama
- Simple Streamlit chat interface
- Easy to extend with more file types or data sources

## Tech stack

- Python 3.10+
- Streamlit
- sentence-transformers
- FAISS (CPU)
- Ollama
- NumPy
- Requests

## Project structure

```text
.
├── app.py                  # Streamlit UI
├── rag_system.py           # Core indexing, retrieval, and LLM logic
├── requirements.txt        # Python dependencies
├── .gitignore
├── data/
│   └── knowledge.md        # Sample local knowledge base
├── README.md
└── .venv/                  # Local virtual environment (not committed)
```

## Quick start

### 1. Prerequisites

Install:
- Python 3.10 or newer
- Ollama
- Git

### 2. Clone the repo

```bash
git clone https://github.com/Scorpio8433/git-clone-https-github.com-JAMwithAI-build_your_local_RAG_system.git
cd git-clone-https-github.com-JAMwithAI-build_your_local_RAG_system
```

### 3. Create a virtual environment

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Install a local model with Ollama

```bash
ollama pull llama3.2
```

If Ollama is not already running, start it before launching the app.

### 6. Run the app

```bash
streamlit run app.py
```

Then open the URL shown in the terminal, typically:

```text
http://localhost:8501
```

## Using the app

1. Add your `.txt`, `.md`, or `.csv` files in the `data/` folder.
2. Open the app in the browser.
3. Type a question in the text box.
4. The app will:
   - search the local indexed content
   - retrieve relevant chunks
   - send them to the local Ollama model
   - display a grounded answer based on your files

## How it works

The app follows a classic local RAG flow:

1. Read files from `data/`
2. Split the text into chunks
3. Generate embeddings for each chunk
4. Store vectors in a FAISS index
5. Embed the user question
6. Retrieve the most relevant chunks
7. Feed those chunks into a local LLM
8. Return the answer

## Example knowledge file

The repo includes a sample file at `data/knowledge.md`. If no documents are found, the app creates this sample content automatically so the project can run immediately.

## Notes

- This is a starter project designed for learning and experimentation.
- The app currently supports local text-based sources like `.txt`, `.md`, and `.csv`.
- It expects Ollama to be installed and accessible on `http://localhost:11434`.
- This is not a production-ready multi-user app yet, but it is a strong base for a local document assistant.

## Recommended next improvements

- Add PDF support
- Add file upload via the UI
- Persist the FAISS index between runs
- Improve conversation memory
- Add better model selection and error handling
- Add a FastAPI backend for a more scalable architecture

## License

This project is licensed under the MIT License.

## Repository purpose

This repository is a local AI starter project intended to help users build a private, document-aware chatbot that runs on the machine without requiring cloud-hosted APIs.
