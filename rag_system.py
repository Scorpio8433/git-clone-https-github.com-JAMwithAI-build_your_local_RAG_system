from pathlib import Path

import streamlit as st

from rag_system import RAGSystem


st.set_page_config(page_title="Private Local RAG Assistant", page_icon="🧠", layout="wide")
st.title("Private Local RAG Assistant")
st.caption("Ask questions about your local documents using a local LLM.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "rag" not in st.session_state or st.session_state.rag is None:
    st.session_state.rag = RAGSystem()

with st.sidebar:
    st.header("Settings")
    docs_dir = st.text_input("Documents folder", value="data")
    index_dir = st.text_input("Index folder", value=".rag_index")

    default_models = ["llama3.2"]
    available_models = st.session_state.rag.list_ollama_models()
    model_choice = st.selectbox(
        "Ollama model",
        options=available_models or default_models,
        index=0,
    )

    uploaded_files = st.file_uploader(
        "Upload files",
        type=["txt", "md", "csv", "pdf"],
        accept_multiple_files=True,
    )

    if st.button("Apply settings"):
        st.session_state.rag = RAGSystem(
            docs_dir=docs_dir,
            index_dir=index_dir,
            ollama_model=model_choice,
        )
        st.session_state.chat_history = []
        st.rerun()

    if st.button("Rebuild index"):
        st.session_state.rag.build_index(force_rebuild=True)
        st.success("Index rebuilt.")

if uploaded_files:
    saved_files = st.session_state.rag.save_uploaded_files(uploaded_files)
    if saved_files:
        with st.spinner("Processing uploaded files..."):
            st.session_state.rag.build_index(force_rebuild=True)
        st.success(f"Processed {len(saved_files)} uploaded file(s).")

rag = st.session_state.rag

if not rag.validate_ollama():
    st.warning(
        "Ollama is not running or not reachable. Start Ollama and make sure your selected model is installed, for example: `ollama pull llama3.2`."
    )
    st.stop()

with st.spinner("Loading documents and index..."):
    rag.build_index()

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask a question about your local files")

if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Searching relevant context and generating an answer..."):
        answer = rag.answer_question(prompt, chat_history=st.session_state.chat_history)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

with st.expander("Available local files"):
    source_files = rag.list_source_files()
    if source_files:
        for file_path in source_files:
            st.write(file_path)
    else:
        st.write("No files found in the documents folder yet.")
