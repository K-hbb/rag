# evaluation/retriever_wrapper.py

import pickle, faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from pathlib import Path
import sys
import numpy as np

# ensure project root on path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ragtest import retrieve   # use your existing retrieve()

# — CONFIG —
BASE_MODEL    = "sentence-transformers/all-mpnet-base-v2"
RERANK_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
INDEX_PATH    = "evaluation/index.faiss"
DOCS_PATH     = "evaluation/documents.pkl"
FAISS_TOP_K   = 20
RETURN_TOP_K  = 5

# load index & docs
index     = faiss.read_index(INDEX_PATH)
with open(DOCS_PATH, "rb") as f:
    documents = pickle.load(f)

# init encoders
embedder  = SentenceTransformer(BASE_MODEL)     # only needed for query embedding
reranker  = CrossEncoder(RERANK_MODEL)

def rerank(query, docs):
    pairs = [(query, d["content"]) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d,_ in ranked]

def get_retrieved_docs(query):
    # 1) dense retrieve top-K
    q_embed = embedder.encode([query], convert_to_tensor=False)
    _, idxs = index.search(np.array(q_embed, dtype="float32"), FAISS_TOP_K)
    initial = [documents[i] for i in idxs[0]]
    # 2) rerank & trim
    best = rerank(query, initial)[:RETURN_TOP_K]
    return [d["doc_id"] for d in best]
