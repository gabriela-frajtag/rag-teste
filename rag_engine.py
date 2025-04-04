import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_documents():
    docs = []
    for fname in os.listdir("data"):
        with open(os.path.join("data", fname), "r", encoding="utf-8") as f:
            content = f.read()
            docs.append((fname, content))
    return docs

def create_or_load_index():
    index_path = "index/vec.index"
    if os.path.exists(index_path):
        return faiss.read_index(index_path), np.load("index/docs.npy", allow_pickle=True)
    docs = load_documents()
    texts = [doc[1] for doc in docs]
    embeddings = model.encode(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    faiss.write_index(index, index_path)
    np.save("index/docs.npy", np.array(texts, dtype=object))
    return index, texts

index, texts = create_or_load_index()

def get_relevant_chunks(query, k=2):
    q_vec = model.encode([query])
    D, I = index.search(np.array(q_vec), k)
    return "

".join([texts[i] for i in I[0]])
