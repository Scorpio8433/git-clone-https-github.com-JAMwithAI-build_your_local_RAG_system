import os
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer


class RAGSystem:
    def __init__(
        self,
        docs_dir: str = "data",
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        ollama_model: str = "llama3.2",
        ollama_base_url: str = "http://localhost:11434",
        chunk_size: int = 400,
        overlap: int = 80,
    ):
        self.docs_dir = Path(docs_dir)
        self.embed_model_name = embed_model
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.embedder = SentenceTransformer(self.embed_model_name)
        self.documents: List[str] = []
        self.metadata: List[str] = []
        self.index = None
        self.dimension = None

    def _read_documents(self) -> List[Tuple[str, str]]:
        if not self.docs_dir.exists():
            self.docs_dir.mkdir(parents=True, exist_ok=True)

        files = []
        for path in sorted(self.docs_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in {".txt", ".md", ".csv"}:
                try:
                    files.append((path.name, path.read_text(encoding="utf-8", errors="ignore")))
                except Exception:
                    continue

        if not files:
            sample = self.docs_dir / "knowledge.md"
            sample.write_text(
                "# Sample knowledge\n\n"
                "This is a local RAG project for JAMwithAI.\n\n"
                "Use this app to chat with your own local documents.\n\n"
                "A local RAG system works by indexing text, retrieving relevant chunks, and sending those chunks to a language model for a grounded answer.",
                encoding="utf-8",
            )
            files.append((sample.name, sample.read_text(encoding="utf-8", errors="ignore")))

        return files

    def _chunk_text(self, text: str, source_name: str) -> List[str]:
        words = text.split()
        chunks: List[str] = []

        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = words[i : i + self.chunk_size]
            if not chunk:
                continue
            chunks.append(f"Source: {source_name}\n\n" + " ".join(chunk))

        return chunks

    def build_index(self) -> None:
        files = self._read_documents()
        self.documents = []
        self.metadata = []

        for source_name, content in files:
            chunks = self._chunk_text(content, source_name)
            self.documents.extend(chunks)
            self.metadata.extend([source_name] * len(chunks))

        if not self.documents:
            raise ValueError("No documents found to index.")

        embeddings = self.embedder.encode(self.documents, convert_to_numpy=True, show_progress_bar=False)
        embeddings = np.asarray(embeddings, dtype="float32")
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)

    def _search(self, query: str, top_k: int = 4) -> List[Tuple[str, float, str]]:
        if self.index is None:
            raise ValueError("Index has not been built yet. Call build_index() first.")

        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        query_embedding = np.asarray(query_embedding, dtype="float32")
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0:
                continue
            results.append((self.documents[int(idx)], float(distances[0][i]), self.metadata[int(idx)]))
        return results

    def answer_question(self, question: str, top_k: int = 4) -> str:
        results = self._search(question, top_k=top_k)

        if not results:
            return "I could not find relevant information in the indexed documents yet."

        context = "\n\n---\n\n".join(chunk for chunk, _, _ in results)
        prompt = (
            "You are a helpful assistant. Answer the user's question using the provided context. "
            "If the answer is not in the context, say so clearly.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=120,
            )
        except requests.RequestException as e:
            raise RuntimeError(f"Ollama request failed: {e}")

        if response.status_code != 200:
            raise RuntimeError(f"Ollama request failed: {response.status_code} - {response.text}")

        payload = response.json()
        # Ollama responses may vary; try common keys
        if isinstance(payload, dict):
            return payload.get("response") or payload.get("result") or payload.get("output") or str(payload)
        return str(payload)

    def validate_ollama(self) -> bool:
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=10)
            return response.status_code == 200
        except requests.RequestException:
            return False
