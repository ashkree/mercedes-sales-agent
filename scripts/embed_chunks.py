# scripts/embed_chunks.py

import json
import os
import sys
import numpy as np
import faiss
from tqdm import tqdm
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: sentence-transformers package not installed.")
    print("Please install with: pip install sentence-transformers")
    sys.exit(1)

# Use absolute paths based on the project root
BASE_DIR = Path(__file__).resolve().parent.parent
DOCSTORE_PATH = BASE_DIR / "data" / "raw" / "docstore.jsonl"
FAISS_INDEX_PATH = BASE_DIR / "data" / "faiss" / "index.bin"
ID_MAP_PATH = BASE_DIR / "data" / "faiss" / "id_map.json"
MODEL_NAME = "BAAI/bge-base-en-v1.5"


def load_chunks(docstore_path):
    """Load document chunks from JSONL file"""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(docstore_path), exist_ok=True)

    # Check if the file exists
    if not os.path.exists(docstore_path):
        print(f"Error: Docstore not found at {docstore_path}")
        print("Please run the csv2chunks.py script first.")
        sys.exit(1)

    # Load the chunks
    chunks = []
    with open(docstore_path, "r") as f:
        for line in f:
            try:
                chunk = json.loads(line.strip())
                chunks.append(chunk)
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON line: {line}")

    print(f"Loaded {len(chunks)} chunks from {docstore_path}")
    return chunks


def embed_texts(texts, model_name):
    """Create embeddings using sentence-transformers"""
    print(f"Initializing model: {model_name}")
    model = SentenceTransformer(model_name)

    print("Encoding texts...")
    return model.encode(texts, show_progress_bar=True)


def build_faiss_index(embeddings):
    """Build a FAISS L2 index from embeddings"""
    dim = embeddings.shape[1]
    print(
        f"Building FAISS index with {embeddings.shape[0]} vectors of dimension {dim}...")
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def save_index(index, path):
    """Save FAISS index to disk"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Saving index to {path}...")
    faiss.write_index(index, str(path))


def save_id_map(ids, path):
    """Save document ID mapping to disk"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Saving ID map to {path}...")
    with open(path, "w") as f:
        json.dump(ids, f)


def main():
    print("Starting embedding process...")
    chunks = load_chunks(DOCSTORE_PATH)

    if not chunks:
        print("No chunks found. Please run the csv2chunks.py script first.")
        return

    texts = [chunk["text"] for chunk in chunks]
    ids = [chunk["id"] for chunk in chunks]

    print(f"Embedding {len(texts)} chunks...")
    embeddings = embed_texts(texts, MODEL_NAME)

    print("Building FAISS index...")
    index = build_faiss_index(np.array(embeddings))

    print(f"Saving index to {FAISS_INDEX_PATH}...")
    save_index(index, FAISS_INDEX_PATH)

    print(f"Saving ID map to {ID_MAP_PATH}...")
    # Create a mapping from index position to document ID
    id_map = {str(i): str(id) for i, id in enumerate(ids)}
    save_id_map(id_map, ID_MAP_PATH)

    print("✅ Embedding completed.")


if __name__ == "__main__":
    main()
