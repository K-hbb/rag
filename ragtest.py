# rag_backend.py

import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
import google.generativeai as genai
import fitz  

# === CONFIG ===
DATA_FOLDER = Path("procedures_text_output")
EMBED_MODEL_NAME = "all-mpnet-base-v2"
GEMINI_API_KEY = "AIzaSyBjRJ4EjTEmd5LI2-fY9_5wglwOfKVW4S0" 
TOP_K = 3


def chunk_text(text, max_chars=1000):
    """Split long text into fixed-size character chunks (for PDFs)."""
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

def extract_text_from_pdf(pdf_path):
    """Extract full text from a PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)

def load_documents(txt_folder="procedures_text_output", pdf_folder="docs"):
    """Load unchunked .txt files and chunked .pdf files into a document list."""
    documents = []

    # Load .txt files (one document per file)
    for txt_file in Path(txt_folder).rglob("*.txt"):
        with open(txt_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
        entite = txt_file.parent.name
        titre = txt_file.stem
        doc_id = f"{entite}__{titre}"
        documents.append({
            "content": content,
            "doc_id": doc_id
        })

    # Load and chunk PDF files
    for pdf_file in Path(pdf_folder).rglob("*.pdf"):
        try:
            full_text = extract_text_from_pdf(pdf_file)
            chunks = chunk_text(full_text, max_chars=1000)

            for i, chunk in enumerate(chunks, 1):
                doc_id = f"{pdf_file.stem}__chunk_{i}"
                documents.append({
                    "content": chunk.strip(),
                    "doc_id": doc_id
                })

        except Exception as e:
            print(f"❌ Failed to process {pdf_file.name}: {e}")

    return documents


def build_index(documents, model):
    """Generate embeddings and build FAISS index."""
    texts = [doc["content"] for doc in documents]
    embeddings = model.encode(texts, convert_to_tensor=False, show_progress_bar=True)

    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    return index

def retrieve(query, model, index, documents, top_k=TOP_K):
    """Find top_k most relevant documents."""
    q_embed = model.encode([query], convert_to_tensor=False)
    distances, indices = index.search(np.array(q_embed).astype("float32"), top_k)
    return [documents[i] for i in indices[0]]



reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
def rerank(query, retrieved_docs):
    pairs = [(query, doc["content"]) for doc in retrieved_docs]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked]


def ask_gemini(query, context_docs, history=None):
    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("models/gemini-2.5-flash")

    # Include context from retrieved documents
    context = "\n\n".join([doc["content"] for doc in context_docs])

    # Optional history
    dialogue = ""
    if history:
        for msg in history:
            role = "Utilisateur" if msg["role"] == "user" else "Assistant"
            dialogue += f"{role}: {msg['content']}\n"

    prompt = f"""
Tu es un assistant expert dans les procédures administratives en Mauritanie. Voici quelques documents utiles :

{context}

{dialogue}
Utilisateur: {query}
Assistant:"""

    response = model.generate_content(prompt)
    return response.text

