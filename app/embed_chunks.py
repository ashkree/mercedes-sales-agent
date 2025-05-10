# scripts/embed_chunks.py

import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DOCSTORE_PATH = "data/raw/docstore.jsonl"
FAISS_INDEX_PATH = "data/faiss/index.bin"
ID_MAP_PATH = "data/faiss/id_map.json"
MODEL_NAME = "BAAI/bge-base-en-v1.5"

def load_chunks(docstore_path):
    with open(docstore_path, "r") as f:
        return [json.loads(line.strip()) for line in f]

def embed_texts(texts, model_name):
    model = SentenceTransformer(model_name)
    return model.encode(texts, show_progress_bar=True)

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def save_index(index, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)

def save_id_map(ids, path):
    with open(path, "w") as f:
        json.dump(ids, f)

def main():
    print("Loading chunks...")
    chunks = load_chunks(DOCSTORE_PATH)
    texts = [chunk["text"] for chunk in chunks]
    ids = [chunk["id"] for chunk in chunks]

    print("Embedding chunks...")
    embeddings = embed_texts(texts, MODEL_NAME)

    print("Building FAISS index...")
    index = build_faiss_index(np.array(embeddings))

    print(f"Saving index to {FAISS_INDEX_PATH}...")
    save_index(index, FAISS_INDEX_PATH)

    print(f"Saving ID map to {ID_MAP_PATH}...")
    save_id_map(ids, ID_MAP_PATH)

    print("✅ Embedding completed.")

if __name__ == "__main__":
    main()

