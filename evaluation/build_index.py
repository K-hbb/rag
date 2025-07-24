# evaluation/build_index.py

import sys
from pathlib import Path

# ensure parent dir (where ragtest.py lives) is on PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent.parent))

import faiss
import numpy as np
import pickle
from ragtest import load_documents
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
INDEX_PATH = "evaluation/index.faiss"
DOCS_PATH  = "evaluation/documents.pkl"

# 1. Load & embed
docs = load_documents()
texts = [d["content"] for d in docs]
model = SentenceTransformer(MODEL_NAME)
embs = model.encode(texts, show_progress_bar=True, convert_to_tensor=False)
embs = np.array(embs, dtype="float32")

# 2. Build FAISS index
index = faiss.IndexFlatL2(embs.shape[1])
index.add(embs)

# 3. Persist to disk
faiss.write_index(index, INDEX_PATH)
with open(DOCS_PATH, "wb") as f:
    pickle.dump(docs, f)

print(f"✔️ Index and documents saved to {INDEX_PATH}, {DOCS_PATH}")
