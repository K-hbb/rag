import streamlit as st
from ragtest import load_documents, build_index, retrieve, ask_gemini
from sentence_transformers import SentenceTransformer
import os
import numpy as np


if "stats" not in st.session_state:
    st.session_state.stats = {
        "queries": 0,
        "tokens_generated": 0,
        "docs_per_query": [],
        "retrieved_docs": {}
    }   


    

# === PAGE SETUP ===
st.set_page_config(page_title=" Assistant Administratif", layout="wide")


# === SIDEBAR ===
with st.sidebar:
    st.title(" Assistant Administratif ")
    st.write("Posez vos questions sur les procédures administratives en Mauritanie.")

    st.subheader("Paramètres")
    top_k = st.slider("Nombre de documents à utiliser", 1, 10, 3)
    st.markdown("---")
    if st.button("Effacer l'historique"):
        st.session_state.messages = []

with st.sidebar.expander("📊 Statistiques de session"):
    stats = st.session_state.stats
    st.write(f"Questions posées : {stats['queries']}")
    st.write(f"Tokens générés : {stats['tokens_generated']}")
    st.write(f"Moyenne de documents par requête : {np.mean(stats['docs_per_query']):.2f}")

    st.write("🔁 Documents les plus fréquemment retrouvés :")
    sorted_docs = sorted(stats["retrieved_docs"].items(), key=lambda x: x[1], reverse=True)
    for doc_id, count in sorted_docs[:5]:
        st.write(f"- {doc_id}: {count} fois")


# === CACHING ===
@st.cache_resource
def load_rag():
    docs = load_documents("procedures_text_output")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    index = build_index(docs, embedder)
    return docs, embedder, index

# === LOAD DATA ===
documents, embedder, index = load_rag()

# === SESSION STATE ===
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Bonjour! Posez-moi une question sur une procédure administrative."}
    ]

# === DISPLAY HISTORY ===
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# === USER INPUT ===
if prompt := st.chat_input("Posez votre question ici..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Recherche en cours..."):
            relevant_docs = retrieve(prompt, embedder, index, documents, top_k=top_k)
            response = ask_gemini(prompt, relevant_docs, history=st.session_state.messages)
            
            # After response = ask_gemini(...)
            st.session_state.stats["queries"] += 1
            st.session_state.stats["tokens_generated"] += len(response.split())
            
            # Track retrieved document IDs
            st.session_state.stats["docs_per_query"].append(len(relevant_docs))
            for doc in relevant_docs:
                doc_id = doc["doc_id"]
                if doc_id not in st.session_state.stats["retrieved_docs"]:
                    st.session_state.stats["retrieved_docs"][doc_id] = 0
                st.session_state.stats["retrieved_docs"][doc_id] += 1
                
        
        st.write(response)
        # Store the response in session state
        st.session_state.messages.append({"role": "assistant", "content": response})
