# Local RAG Assistant

A simple local retrieval-augmented generation (RAG) project that lets you chat with your own documents using a local LLM.

Features
- Upload or place documents in `data/`
- Embed documents locally with `sentence-transformers`
- Search relevant chunks using FAISS
- Ask questions through a Streamlit chat interface
- Use a local Ollama model for generation

Quick start

1. Install Python 3.10+
2. Create a virtual environment
3. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

4. Install a local model in Ollama

```bash
ollama pull llama3.2
```

5. Start the app

```bash
streamlit run app.py
```

6. Open the local URL shown in the terminal.

Project structure

```
.
├── app.py
├── rag_system.py
├── requirements.txt
├── .gitignore
├── data/
│   └── knowledge.md
├── README.md
└── .venv/
```

How it works

- Files in `data/` are read and split into chunks
- Each chunk is embedded using a sentence-transformers model
- FAISS finds the most relevant chunks for a user question
- The chunks are sent to Ollama together with your question
- The model answers using the retrieved context

Notes

- This is a starter app designed for learning and local experimentation.
- The app expects Ollama to be installed and running locally on port 11434.
- If no docs are present, the app uses the sample knowledge file in `data/knowledge.md`.

Want to customize?

Update:
- `data/knowledge.md` to add your own knowledge base
- `rag_system.py` to change chunk size, indexing behavior, or search logic
- `app.py` to change the interface or add extra controls
