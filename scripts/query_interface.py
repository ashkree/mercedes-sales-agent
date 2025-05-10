# scripts/query_interface.py

import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

FAISS_INDEX_PATH = "data/faiss/index.bin"
ID_MAP_PATH = "data/faiss/id_map.json"
DOCSTORE_PATH = "data/raw/docstore.jsonl"
MODEL_NAME = "BAAI/bge-base-en-v1.5"

# Load all at once
model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(FAISS_INDEX_PATH)

with open(ID_MAP_PATH) as f:
    id_map = json.load(f)

with open(DOCSTORE_PATH) as f:
    docstore = {json.loads(line)['id']: json.loads(line)['text'] for line in f}

def search(query, top_k=5):
    q_vec = model.encode([query])
    D, I = index.search(np.array(q_vec), top_k)
    results = [docstore[id_map[i]] for i in I[0]]
    return results

if __name__ == "__main__":
    while True:
        q = input("❓ Ask a question (or type 'exit'): ")
        if q.lower() == "exit":
            break
        results = search(q)
        print("\n📄 Top Results:")
        for res in results:
            print("-", res)
        print()

