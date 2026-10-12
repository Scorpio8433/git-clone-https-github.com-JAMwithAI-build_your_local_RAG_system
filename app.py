from pathlib import Path

import streamlit as st

from rag_system import RAGSystem


st.set_page_config(page_title="Local RAG Assistant", page_icon="🧠", layout="wide")
st.title("Local RAG Assistant")
st.caption("Ask questions about your local documents using a local LLM.")

with st.sidebar:
    st.header("Settings")
    docs_dir = st.text_input("Documents folder", value="data")
    ollama_model = st.text_input("Ollama model", value="llama3.2")
    chunk_size = st.number_input("Chunk size", min_value=100, max_value=1000, value=400, step=50)
    overlap = st.number_input("Chunk overlap", min_value=20, max_value=200, value=80, step=10)
    refresh = st.button("Refresh index")

if "rag" not in st.session_state:
    st.session_state.rag = RAGSystem(
        docs_dir=docs_dir,
        ollama_model=ollama_model,
        chunk_size=int(chunk_size),
        overlap=int(overlap),
    )

if refresh:
    st.session_state.rag = RAGSystem(
        docs_dir=docs_dir,
        ollama_model=ollama_model,
        chunk_size=int(chunk_size),
        overlap=int(overlap),
    )

try:
    if not st.session_state.rag.validate_ollama():
        st.warning("Ollama does not appear to be running. Start Ollama and run: `ollama pull llama3.2`")
        st.stop()

    with st.spinner("Loading and indexing documents..."):
        st.session_state.rag.build_index()

    question = st.text_area("Ask a question", placeholder="What would you like to know?")

    if st.button("Ask") and question.strip():
        with st.spinner("Searching relevant context and generating an answer..."):
            answer = st.session_state.rag.answer_question(question)
        st.subheader("Answer")
        st.write(answer)

    with st.expander("View indexed files"):
        docs_path = Path(docs_dir)
        if docs_path.exists():
            for file in sorted(docs_path.iterdir()):
                if file.is_file():
                    st.write(file.name)
        else:
            st.write("No documents folder found yet.")

except Exception as exc:
    st.error(f"Something went wrong: {exc}")
    st.info("Make sure Ollama is installed and running locally. Also make sure the `data` folder contains text files or markdown files.")


if __name__ == "__main__":
    pass
